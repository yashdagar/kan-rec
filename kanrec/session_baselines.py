"""Non-neural session baselines evaluated on every next-item prediction, like the GRU4Rec models."""
import numpy as np
import scipy.sparse as sp
import torch


def _events(test, itemidmap):
    test = test[test["ItemId"].isin(itemidmap.index)]
    items = itemidmap[test["ItemId"].values].values.astype(np.int64)
    sess = test["SessionId"].values
    same = np.r_[False, sess[1:] == sess[:-1]]
    return items, sess, same


def _ranks(score_fn, items, same, n_items, device, batch=2048):
    pos = np.flatnonzero(same)
    out = []
    for i in range(0, len(pos), batch):
        p = pos[i:i + batch]
        scores = score_fn(p)
        t = torch.as_tensor(items[p], device=device).view(-1, 1)
        out.append((scores >= scores.gather(1, t)).sum(1).cpu())
    return torch.cat(out).numpy()


def popularity(train, itemidmap):
    pop = train.groupby("ItemId").size()
    return pop.reindex(itemidmap.index).fillna(0).to_numpy(np.float32)


def pop_ranks(train, test, itemidmap, device):
    items, _, same = _events(test, itemidmap)
    pop = torch.tensor(popularity(train, itemidmap), device=device)
    return _ranks(lambda p: pop.expand(len(p), -1), items, same, len(itemidmap), device)


def spop_ranks(train, test, itemidmap, device):
    """Session popularity: items already in the session ranked by in-session count, ties and the rest
    by global popularity."""
    items, sess, same = _events(test, itemidmap)
    pop = popularity(train, itemidmap)
    pop_t = torch.tensor(pop / (pop.max() + 1), device=device)
    start = np.zeros(len(items), dtype=np.int64)
    for i in range(1, len(items)):
        start[i] = start[i - 1] if same[i] else i

    def score(p):
        s = pop_t.expand(len(p), -1).clone()
        for r, j in enumerate(p):
            seen = torch.as_tensor(items[start[j]:j], device=device)
            s[r].index_add_(0, seen, torch.ones(len(seen), device=device))
        return s

    return _ranks(score, items, same, len(itemidmap), device)


def item_knn_ranks(train, test, itemidmap, device, lam=20.0, alpha=0.5):
    """Item-kNN baseline from Hidasi et al. (2016): similarity of the current item to every other
    item, sim(i, j) = n_ij / (n_i^alpha * n_j^(1-alpha) + lambda), from session co-occurrence."""
    tr = train[train["ItemId"].isin(itemidmap.index)]
    rows = tr["SessionId"].astype("category").cat.codes.to_numpy()
    cols = itemidmap[tr["ItemId"].values].values
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(rows.max() + 1, len(itemidmap)))
    m.data[:] = 1.0
    co = (m.T @ m).tocsr()
    n = np.asarray(m.sum(0)).ravel()
    co.setdiag(0)
    co.eliminate_zeros()
    items, _, same = _events(test, itemidmap)
    pop = popularity(train, itemidmap)
    tie = torch.tensor(pop / (pop.max() + 1) * 1e-6, device=device)

    def score(p):
        prev = items[p - 1]
        block = co[prev].toarray()
        denom = (n[prev, None] ** alpha) * (n[None, :] ** (1 - alpha)) + lam
        return torch.tensor(block / denom, device=device, dtype=torch.float32) + tie

    return _ranks(score, items, same, len(itemidmap), device)
