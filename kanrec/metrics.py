import math

import numpy as np
import torch

SESSION_CUTOFFS = (10, 20)
BASKET_CUTOFFS = (10, 20)


def target_ranks(scores, targets):
    """Pessimistic rank of each target: number of items scoring >= the target (GRU4Rec 'conservative')."""
    t = scores.gather(1, targets.view(-1, 1))
    return (scores >= t).sum(1)


def session_metrics_from_ranks(ranks, cutoffs=SESSION_CUTOFFS):
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {}
    for k in cutoffs:
        hit = ranks <= k
        out[f"Recall@{k}"] = hit.mean()
        out[f"MRR@{k}"] = (hit / ranks).mean()
        out[f"NDCG@{k}"] = (hit / np.log2(ranks + 1)).mean()
    return out


def per_event_session_metric(ranks, name):
    metric, k = name.split("@")
    k = int(k)
    ranks = np.asarray(ranks, dtype=np.float64)
    hit = ranks <= k
    if metric == "Recall":
        return hit.astype(np.float64)
    if metric == "MRR":
        return hit / ranks
    if metric == "NDCG":
        return hit / np.log2(ranks + 1)
    raise ValueError(name)


def basket_user_metrics_loop(scores, truth, history=None, cutoffs=BASKET_CUTOFFS):
    """Per-user next-basket metrics.

    scores: (U, N) tensor; truth: list of U arrays of item indices; history: list of U arrays
    (items the user bought before) used for the repeat/explore split.
    Returns a dict of numpy arrays with one value per user.
    """
    U, N = scores.shape
    max_k = max(max(cutoffs), 2 * max(len(t) for t in truth))
    top = torch.topk(scores, min(max_k, N), dim=1).indices.cpu().numpy()
    res = {f"{m}@{k}": np.zeros(U) for k in cutoffs for m in ("Recall", "NDCG", "PHR")}
    for name in ("P@B/2", "R@B/2", "P@B", "P@2B", "R@2B", "AvgRank"):
        res[name] = np.zeros(U)
    if history is not None:
        for k in cutoffs:
            res[f"RepRecall@{k}"] = np.full(U, np.nan)
            res[f"ExplRecall@{k}"] = np.full(U, np.nan)
    for u in range(U):
        tset = set(int(i) for i in truth[u])
        nb = len(tset)
        rel = np.array([int(i) in tset for i in top[u]])
        for k in cutoffs:
            hits = rel[:k]
            res[f"Recall@{k}"][u] = hits.sum() / nb
            res[f"PHR@{k}"][u] = float(hits.any())
            dcg = (hits / np.log2(np.arange(2, k + 2))).sum()
            idcg = (1.0 / np.log2(np.arange(2, min(k, nb) + 2))).sum()
            res[f"NDCG@{k}"][u] = dcg / idcg
            if history is not None:
                hset = set(int(i) for i in history[u])
                rep, expl = tset & hset, tset - hset
                topk = set(int(i) for i in top[u][:k])
                if rep:
                    res[f"RepRecall@{k}"][u] = len(topk & rep) / len(rep)
                if expl:
                    res[f"ExplRecall@{k}"][u] = len(topk & expl) / len(expl)
        half = math.ceil(nb / 2)
        res["P@B/2"][u] = rel[:half].sum() / half
        res["R@B/2"][u] = rel[:half].sum() / nb
        res["P@B"][u] = rel[:nb].sum() / nb
        res["P@2B"][u] = rel[: 2 * nb].sum() / (2 * nb)
        res["R@2B"][u] = rel[: 2 * nb].sum() / nb
        s = scores[u]
        ts = s[torch.tensor(sorted(tset), device=s.device)]
        greater = (s[None, :] > ts[:, None]).sum(1).float()
        equal = (s[None, :] == ts[:, None]).sum(1).float()
        res["AvgRank"][u] = (greater + (equal + 1) / 2).mean().item()
    return res


def summarize(per_user):
    return {k: float(np.nanmean(v)) for k, v in per_user.items()}


def basket_user_metrics(scores, truth, history=None, cutoffs=BASKET_CUTOFFS):
    """Vectorised version of basket_user_metrics_loop (same definitions, same outputs)."""
    U, N = scores.shape
    dev = scores.device
    sizes = torch.tensor([len(set(int(i) for i in t)) for t in truth], device=dev)
    rows = torch.repeat_interleave(torch.arange(U, device=dev), torch.tensor([len(t) for t in truth], device=dev))
    cols = torch.as_tensor(np.concatenate([np.asarray(t, dtype=np.int64) for t in truth]), device=dev)
    tmask = torch.zeros(U, N, dtype=torch.bool, device=dev)
    tmask[rows, cols] = True
    k_max = min(N, max(max(cutoffs), 2 * int(sizes.max())))
    top = torch.topk(scores, k_max, dim=1).indices
    rel = tmask.gather(1, top).float()
    cum = torch.cumsum(rel, 1)
    disc = 1.0 / torch.log2(torch.arange(2, k_max + 2, device=dev, dtype=torch.float32))
    cdisc = torch.cumsum(disc, 0)
    res = {}
    nb = sizes.float()
    for k in cutoffs:
        hits = cum[:, k - 1]
        res[f"Recall@{k}"] = hits / nb
        res[f"PHR@{k}"] = (hits > 0).float()
        dcg = (rel[:, :k] * disc[:k]).sum(1)
        res[f"NDCG@{k}"] = dcg / cdisc[torch.clamp(sizes, max=k) - 1]
    half = torch.ceil(nb / 2).long()
    res["P@B/2"] = cum.gather(1, (half - 1).view(-1, 1)).squeeze(1) / half.float()
    res["R@B/2"] = cum.gather(1, (half - 1).view(-1, 1)).squeeze(1) / nb
    at_b = cum.gather(1, (sizes - 1).view(-1, 1)).squeeze(1)
    res["P@B"] = at_b / nb
    at_2b = cum.gather(1, (torch.clamp(2 * sizes, max=k_max) - 1).view(-1, 1)).squeeze(1)
    res["P@2B"] = at_2b / (2 * nb)
    res["R@2B"] = at_2b / nb
    srt = torch.sort(scores, dim=1).values
    lengths = torch.tensor([len(t) for t in truth], device=dev)
    width = int(lengths.max())
    slot = torch.arange(width, device=dev)[None, :]
    valid = slot < lengths[:, None]
    padded = torch.zeros(U, width, dtype=torch.long, device=dev)
    offsets = torch.cumsum(lengths, 0) - lengths
    padded[valid] = cols[(offsets[:, None] + slot)[valid]]
    vals = scores.gather(1, padded).contiguous()
    right = torch.searchsorted(srt, vals, right=True)
    left = torch.searchsorted(srt, vals, right=False)
    rank = (N - right).float() + ((right - left).float() + 1) / 2
    res["AvgRank"] = torch.where(valid, rank, torch.zeros_like(rank)).sum(1) / lengths.float()
    if history is not None:
        hrows = torch.repeat_interleave(torch.arange(U, device=dev), torch.tensor([len(h) for h in history], device=dev))
        hcols = torch.as_tensor(np.concatenate([np.asarray(h, dtype=np.int64) for h in history]), device=dev)
        hmask = torch.zeros(U, N, dtype=torch.bool, device=dev)
        hmask[hrows, hcols] = True
        rep, expl = tmask & hmask, tmask & ~hmask
        for k in cutoffs:
            kmask = torch.zeros(U, N, dtype=torch.bool, device=dev)
            kmask.scatter_(1, top[:, :k], True)
            nrep, nexpl = rep.sum(1).float(), expl.sum(1).float()
            res[f"RepRecall@{k}"] = torch.where(nrep > 0, (kmask & rep).sum(1) / nrep.clamp(min=1), torch.nan)
            res[f"ExplRecall@{k}"] = torch.where(nexpl > 0, (kmask & expl).sum(1) / nexpl.clamp(min=1), torch.nan)
    return {k: v.double().cpu().numpy() for k, v in res.items()}
