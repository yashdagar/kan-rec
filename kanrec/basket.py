"""Next-basket models: basket-GRU (van Maasakkers et al. 2023) with optional KAN / MLP modules,
and the standard frequency and TIFU-KNN baselines."""
import copy
import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import ResidualBlock, n_params
from .metrics import basket_user_metrics, summarize
from .session import CELLS


class BasketGRU(nn.Module):
    """Multi-hot basket -> linear input map (EmbeddingBag, sum) -> GRU -> sigmoid score per item.

    input_block / head_block: optional "kan" or "mlp" residual block on the d-dimensional input or
    hidden state. cell="kangru" replaces nn.GRU with a GRU whose candidate state is a KAN.
    """

    def __init__(self, n_items, hidden, dropout=0.0, input_block=None, head_block=None, cell="gru",
                 basis="rbf", grid_size=8):
        super().__init__()
        self.n_items, self.hidden, self.cell_type = n_items, hidden, cell
        self.inp = nn.EmbeddingBag(n_items, hidden, mode="sum")
        self.input_block = ResidualBlock(hidden, input_block, basis=basis, grid_size=grid_size) if input_block else None
        if cell == "gru":
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.rnn = CELLS[cell](hidden, hidden, basis=basis, grid_size=grid_size)
        self.drop = nn.Dropout(dropout)
        self.head_block = ResidualBlock(hidden, head_block, basis=basis, grid_size=grid_size) if head_block else None
        self.out = nn.Linear(hidden, n_items)

    def encode(self, flat_items, offsets, B, T):
        x = self.inp(flat_items, offsets).view(B, T, self.hidden)
        if self.input_block is not None:
            x = self.input_block(x)
        x = self.drop(x)
        if self.cell_type == "gru":
            h, _ = self.rnn(x)
        else:
            hs, ht = [], x.new_zeros(B, self.hidden)
            for t in range(T):
                ht = self.rnn(x[:, t], ht)
                hs.append(ht)
            h = torch.stack(hs, 1)
        h = self.drop(h)
        if self.head_block is not None:
            h = self.head_block(h)
        return h

    def forward(self, flat_items, offsets, B, T):
        return self.out(self.encode(flat_items, offsets, B, T))


def _pack(seqs, device):
    """seqs: list of lists of item arrays (one list of baskets per user). Pads with empty baskets on
    the right and returns EmbeddingBag inputs plus true lengths."""
    B, T = len(seqs), max(len(s) for s in seqs)
    flat, offsets, pos = [], [], 0
    for s in seqs:
        for t in range(T):
            offsets.append(pos)
            if t < len(s):
                flat.append(s[t])
                pos += len(s[t])
    flat = torch.from_numpy(np.concatenate(flat).astype(np.int64)).to(device)
    offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    lengths = torch.tensor([len(s) for s in seqs], device=device)
    return flat, offsets, lengths, B, T


def _target_index(baskets_per_step, T, device):
    """Flat (row, item) indices of the target items, where row = user * T + step."""
    rows = np.concatenate([np.full(len(items), b * T + t) for b, steps in enumerate(baskets_per_step)
                           for t, items in enumerate(steps)])
    cols = np.concatenate([items for steps in baskets_per_step for items in steps])
    return torch.from_numpy(rows).to(device), torch.from_numpy(cols.astype(np.int64)).to(device)


def sparse_bce(logits, rows, cols):
    """Per-step BCE averaged over items, identical to binary_cross_entropy_with_logits(logits, y)
    .mean(-1) for the multi-hot y given by (rows, cols), without building y:
    BCE(x, y) = softplus(x) - x * y."""
    B, T, N = logits.shape
    flat = logits.reshape(B * T, N)
    pos = torch.zeros(B * T, device=logits.device, dtype=logits.dtype).index_add_(0, rows, flat[rows, cols])
    return (F.softplus(logits).mean(-1).reshape(B * T) - pos / N).reshape(B, T)


def early_stopping_users(data, max_users=20000, sample=10000, seed=0):
    """All validation users, or a fixed random sample of them when there are very many (full
    Instacart), so that per-epoch validation stays cheap. Test evaluation always uses all users."""
    if len(data.valid_users) <= max_users:
        return data.valid_users
    return np.sort(np.random.default_rng(seed).choice(data.valid_users, sample, replace=False))


def _multi_hot(baskets_per_step, n_items, device):
    B, T = len(baskets_per_step), max(len(s) for s in baskets_per_step)
    rows = np.concatenate([np.full(len(items), b * T + t) for b, steps in enumerate(baskets_per_step)
                           for t, items in enumerate(steps)])
    cols = np.concatenate([items for steps in baskets_per_step for items in steps])
    y = torch.zeros(B * T, n_items, device=device)
    y[torch.from_numpy(rows).to(device), torch.from_numpy(cols.astype(np.int64)).to(device)] = 1.0
    return y.view(B, T, n_items)


@torch.no_grad()
def basket_gru_scores(model, data, users, device, batch_size=256, max_history=100):
    model.eval()
    out = []
    for i in range(0, len(users), batch_size):
        seqs = [data.history(u)[-max_history:] for u in users[i:i + batch_size]]
        flat, offsets, lengths, B, T = _pack(seqs, device)
        h = model.encode(flat, offsets, B, T)
        last = h[torch.arange(B, device=device), lengths - 1]
        out.append(torch.sigmoid(model.out(last)))
    return torch.cat(out)


def _chunked_scores_metrics(model, data, users, device, max_history, chunk=2048):
    parts = []
    for i in range(0, len(users), chunk):
        us = users[i:i + chunk]
        parts.append(evaluate_scores(basket_gru_scores(model, data, us, device, max_history=max_history), data, us))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def evaluate_scores(scores, data, users):
    truth = [data.target(u) for u in users]
    hist = [data.history_items(u) for u in users]
    return basket_user_metrics(scores, truth, hist)


def train_basket_gru(data, device, hidden=256, epochs=20, lr=1e-3, batch_size=64, dropout=0.3,
                     max_history=100, model_kwargs=None, seed=42, select_metric="Recall@10", patience=3,
                     log=print, amp=False):
    """BCE over every next basket in each user's history (the held-out final basket is never used
    for training). Early stopping on validation users' final basket. amp=True runs the forward pass
    in bfloat16 on CUDA (the loss is computed in float32)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BasketGRU(data.n_items, hidden, dropout, **(model_kwargs or {})).to(device)
    users = np.array([u for u in range(len(data.baskets)) if len(data.history(u)) >= 2])
    counts = np.zeros(data.n_items)
    n_baskets = 0
    for u in users:
        for b in data.history(u)[1:]:
            counts[b] += 1
            n_baskets += 1
    p = np.clip(counts / n_baskets, 1e-6, 1 - 1e-6)
    with torch.no_grad():
        model.out.bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = amp and device.type == "cuda"
    vusers = early_stopping_users(data)
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(users)
        loss_sum, n_batches = torch.zeros((), device=device), 0
        for i in range(0, len(perm), batch_size):
            seqs = [data.history(u)[-(max_history + 1):] for u in perm[i:i + batch_size]]
            flat, offsets, lengths, B, T = _pack([s[:-1] for s in seqs], device)
            rows, cols = _target_index([s[1:] for s in seqs], T, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(flat, offsets, B, T)
            mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()
            loss = (sparse_bce(logits.float(), rows, cols) * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.detach()
            n_batches += 1
        mean_loss = loss_sum.item() / n_batches
        train_s = time.time() - t0
        per_user = _chunked_scores_metrics(model, data, vusers, device, max_history)
        m = summarize(per_user)
        history.append({"epoch": epoch + 1, "loss": mean_loss, "train_s": train_s, **m})
        log(f"epoch {epoch + 1}: loss {mean_loss:.5f} ({train_s:.1f}s) valid Recall@10 {m['Recall@10']:.4f} NDCG@10 {m['NDCG@10']:.4f} P@B {m['P@B']:.4f}")
        if m[select_metric] > best:
            best, best_state, bad = m[select_metric], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.n_params = n_params(model)
    return model, history


def _history_matrix(data, users, weights=None):
    rows, cols, vals = [], [], []
    for r, u in enumerate(users):
        for t, b in enumerate(data.history(u)):
            rows.extend([r] * len(b))
            cols.extend(b.tolist())
            vals.extend([1.0 if weights is None else weights[r][t]] * len(b))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(users), data.n_items), dtype=np.float32)


def global_popularity(data):
    all_users = np.arange(len(data.baskets))
    return np.asarray(_history_matrix(data, all_users).sum(0)).ravel()


def frequency_scores(data, users, kind="gp"):
    """G-TopFreq ('g'), P-TopFreq ('p'), GP-TopFreq ('gp': personal, ties and remaining slots filled by
    global popularity) and last basket ('last')."""
    pop = global_popularity(data)
    pop = pop / (pop.max() + 1)
    if kind == "g":
        return torch.tensor(np.tile(pop, (len(users), 1)), dtype=torch.float32)
    if kind == "last":
        m = sp.lil_matrix((len(users), data.n_items), dtype=np.float32)
        for r, u in enumerate(users):
            m[r, data.history(u)[-1]] = 1.0
        personal = m.toarray()
    else:
        personal = _history_matrix(data, users).toarray()
    if kind == "p":
        return torch.tensor(personal, dtype=torch.float32)
    return torch.tensor(personal + pop[None, :], dtype=torch.float32)


def tifu_vectors(data, users, within_decay, group_decay, group_size):
    """TIFU-KNN user representation (Hu et al., SIGIR 2020): baskets are split into groups of
    `group_size` (aligned to the most recent basket); repeat frequency is time-decayed within and
    across groups."""
    weights = []
    for u in users:
        n = len(data.history(u))
        r = n - 1 - np.arange(n)
        group = r // group_size
        n_groups = group.max() + 1
        k = np.minimum(group_size, n - group * group_size)
        weights.append(within_decay ** (r % group_size) / k * group_decay ** group / n_groups)
    return _history_matrix(data, users, weights)


def tifu_knn_scores(data, users, n_neighbors=300, within_decay=0.9, group_decay=0.7, group_size=7, alpha=0.7,
                    chunk=512):
    all_users = np.arange(len(data.baskets))
    pool = tifu_vectors(data, all_users, within_decay, group_decay, group_size)
    norms = np.sqrt(np.asarray(pool.multiply(pool).sum(1)).ravel()) + 1e-12
    pool_n = sp.diags(1 / norms) @ pool
    q = pool[users]
    out = np.zeros((len(users), data.n_items), dtype=np.float32)
    for i in range(0, len(users), chunk):
        sims = (pool_n[users[i:i + chunk]] @ pool_n.T).toarray()
        sims[np.arange(sims.shape[0]), users[i:i + chunk]] = -np.inf
        k = min(n_neighbors, sims.shape[1] - 1)
        nn_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(len(nn_idx)), k)
        avg = sp.csr_matrix((np.full(rows.size, 1.0 / k, dtype=np.float32), (rows, nn_idx.ravel())),
                            shape=(len(nn_idx), pool.shape[0]))
        neigh = (avg @ pool).toarray()
        out[i:i + chunk] = alpha * q[i:i + chunk].toarray() + (1 - alpha) * neigh
    return torch.tensor(out)
