"""KAN-NBR: an interpretable repeat-aware next-basket model.

Each (user, candidate item) pair is described by a few scalar features computed from the user's
history; a small KAN maps the features to a purchase logit. With a single KAN layer the model is
additive, score = sum_f phi_f(feature_f), so every learned phi_f can be plotted. Candidates are the
user's previously bought items plus the most popular items, which covers repeat and explore items.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import KANLinear, matched_mlp, n_params

FEATURES = ["log baskets since last purchase", "log purchase count", "purchase share",
            "decayed frequency", "log item popularity", "log overdue ratio", "in history"]


def _features(history, pop_log, candidates, decay=0.9):
    n = len(history)
    last = np.full(pop_log.shape[0], -1, dtype=np.int64)
    count = np.zeros(pop_log.shape[0], dtype=np.float32)
    decayed = np.zeros(pop_log.shape[0], dtype=np.float32)
    first = np.full(pop_log.shape[0], -1, dtype=np.int64)
    for t, b in enumerate(history):
        last[b] = t
        count[b] += 1
        decayed[b] += decay ** (n - 1 - t)
        new = first[b] < 0
        first[b[new]] = t
    c = candidates
    seen = count[c] > 0
    since = np.where(seen, n - last[c], n + 1).astype(np.float32)
    span = np.where(seen, last[c] - first[c], 0).astype(np.float32)
    interval = np.where(count[c] > 1, span / np.maximum(count[c] - 1, 1), n + 1)
    overdue = since / np.maximum(interval, 1.0)
    X = np.stack([np.log1p(since), np.log1p(count[c]), count[c] / n, decayed[c], pop_log[c],
                  np.log1p(overdue), seen.astype(np.float32)], 1)
    return X.astype(np.float32)


class NBRFeaturizer:
    def __init__(self, data, n_popular=100):
        counts = np.zeros(data.n_items)
        total = 0
        for u in range(len(data.baskets)):
            for b in data.history(u):
                counts[b] += 1
                total += 1
        self.pop_log = np.log((counts + 1) / (total + data.n_items)).astype(np.float32)
        self.popular = np.argsort(-counts)[:n_popular]
        self.data = data
        self._cache = {}

    def for_user(self, u):
        """Candidates and features for predicting user u's held-out basket from the full history;
        computed once and cached, since they do not change during training."""
        if u not in self._cache:
            cand, X, _ = self.user(self.data.history(u))
            self._cache[u] = (cand, X)
        return self._cache[u]

    def candidates(self, history):
        return np.union1d(np.unique(np.concatenate(history)), self.popular)

    def user(self, history, target=None):
        cand = self.candidates(history)
        X = _features(history, self.pop_log, cand)
        y = None if target is None else np.isin(cand, target).astype(np.float32)
        return cand, X, y


class KANNBR(nn.Module):
    """kind="kan": KAN [F -> hidden -> 1] (hidden=0 gives the additive single-layer model);
    kind="mlp": parameter-matched MLP; kind="linear": logistic regression on the same features."""

    def __init__(self, n_features, kind="kan", hidden=0, basis="rbf", grid_size=8):
        super().__init__()
        self.norm = nn.BatchNorm1d(n_features, affine=False)
        if kind == "kan":
            if hidden:
                self.f = nn.Sequential(KANLinear(n_features, hidden, basis=basis, grid_size=grid_size),
                                       KANLinear(hidden, 1, basis=basis, grid_size=grid_size))
            else:
                self.f = KANLinear(n_features, 1, basis=basis, grid_size=grid_size)
        elif kind == "mlp":
            ref = KANNBR(n_features, "kan", hidden, basis, grid_size)
            self.f = matched_mlp(n_features, 1, n_params(ref.f))
        elif kind == "linear":
            self.f = nn.Linear(n_features, 1)
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.f(self.norm(x)).squeeze(-1)


def build_training_set(feat, data, users, cut_points=3):
    """Training pairs from earlier cut points of each user's history: predict basket t from baskets
    before t, for the last `cut_points` history positions. The held-out final basket is never used."""
    Xs, ys = [], []
    for u in users:
        hist = data.history(u)
        for k in range(1, cut_points + 1):
            t = len(hist) - k
            if t < 2:
                break
            _, X, y = feat.user(hist[:t], hist[t])
            Xs.append(X)
            ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


@torch.no_grad()
def kannbr_scores(model, feat, data, users, device):
    model.eval()
    parts = [feat.for_user(u) for u in users]
    X = torch.from_numpy(np.concatenate([p[1] for p in parts])).to(device)
    s = torch.cat([model(X[i:i + 262144]) for i in range(0, len(X), 262144)]).cpu()
    rows = torch.from_numpy(np.repeat(np.arange(len(users)), [len(p[0]) for p in parts]))
    cols = torch.from_numpy(np.concatenate([p[0] for p in parts]).astype(np.int64))
    out = torch.full((len(users), data.n_items), -1e9)
    out[rows, cols] = s
    return out


def train_kannbr(data, device, kind="kan", hidden=0, basis="rbf", grid_size=8, epochs=30, lr=1e-2,
                 batch_size=4096, patience=4, seed=42, log=print, eval_fn=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feat = NBRFeaturizer(data)
    users = np.arange(len(data.baskets))
    X, y = build_training_set(feat, data, users)
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    model = KANNBR(X.shape[1], kind, hidden, basis, grid_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(len(X))).to(device)
        losses = []
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_s = time.time() - t0
        m = eval_fn(model, feat)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_s": train_s, **m})
        log(f"epoch {epoch + 1}: loss {np.mean(losses):.5f} ({train_s:.1f}s) valid Recall@10 {m['Recall@10']:.4f} NDCG@10 {m['NDCG@10']:.4f}")
        if m["Recall@10"] > best:
            best, best_state, bad = m["Recall@10"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.n_params = n_params(model)
    return model, feat, history


@torch.no_grad()
def shape_functions(model, X_sample, n_points=100):
    """For the additive model: the learned function of each feature over the observed feature range,
    in original feature units, centred to mean zero."""
    layer = model.f
    mean = model.norm.running_mean.cpu()
    std = (model.norm.running_var.cpu() + model.norm.eps).sqrt()
    curves = {}
    for i, name in enumerate(FEATURES):
        lo, hi = np.percentile(X_sample[:, i], [1, 99])
        xs = torch.linspace(float(lo), float(hi), n_points)
        z = (xs - mean[i]) / std[i]
        y = layer.cpu().edge_function(0, i, z)
        curves[name] = (xs.numpy(), (y - y.mean()).numpy())
    return curves
