"""KAN variants of GRU4Rec built on the official PyTorch implementation.

The official SessionDataIterator (session-parallel mini-batches, shared popularity-based negative
samples), loss functions and IndexedAdagradM optimizer are reused unchanged; only the network is
extended with optional KAN or MLP modules.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "GRU4Rec_PyTorch_Official"))
from gru4rec_pytorch import GRU4Rec, GRU4RecModel, IndexedAdagradM, SessionDataIterator  # noqa: E402

from .kan import KANLinear, ResidualBlock, matched_mlp, n_params  # noqa: E402
from .metrics import session_metrics_from_ranks  # noqa: E402


class KANGRUCell(nn.Module):
    """GRU cell whose candidate state is computed by a KAN; update and reset gates stay linear + sigmoid."""

    def __init__(self, input_size, hidden_size, basis="rbf", grid_size=8):
        super().__init__()
        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.candidate = KANLinear(input_size + hidden_size, hidden_size, basis=basis, grid_size=grid_size)

    def forward(self, x, h):
        r, z = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.candidate(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class MLPGRUCell(KANGRUCell):
    """Control for KANGRUCell: the candidate state comes from an MLP with the same number of
    parameters as the KAN it replaces."""

    def __init__(self, input_size, hidden_size, basis="rbf", grid_size=8):
        super().__init__(input_size, hidden_size, basis, grid_size)
        self.candidate = matched_mlp(input_size + hidden_size, hidden_size, n_params(self.candidate))


class KANRNNCell(nn.Module):
    """The first draft's ungated cell, h = tanh(LN(KAN([x, h]))), with a correct B-spline basis."""

    def __init__(self, input_size, hidden_size, basis="bspline", grid_size=8):
        super().__init__()
        self.kan = KANLinear(input_size + hidden_size, hidden_size, basis=basis, grid_size=grid_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        return torch.tanh(self.norm(self.kan(torch.cat([x, h], 1))))


class MLPRNNCell(KANRNNCell):
    """Control for KANRNNCell: the same ungated cell with a parameter-matched MLP."""

    def __init__(self, input_size, hidden_size, basis="bspline", grid_size=8):
        super().__init__(input_size, hidden_size, basis, grid_size)
        self.kan = matched_mlp(input_size + hidden_size, hidden_size, n_params(self.kan))


CELLS = {"kangru": KANGRUCell, "mlpgru": MLPGRUCell, "kanrnn": KANRNNCell, "mlprnn": MLPRNNCell}

ADDED_PREFIXES = ("input_block.", "head_block.")


def added_parameter(name, model):
    """Parameters that are not part of the official GRU4Rec network: the residual blocks and the
    KAN parts of KAN cells (the KAN-GRU gates are ordinary linear GRU weights)."""
    if name.startswith(ADDED_PREFIXES):
        return True
    if name.startswith("G."):
        i, rest = name.split(".")[1], name.split(".", 2)[2]
        return model.cells[int(i)] != "gru" and not rest.startswith("gates.")
    return False


class KANGRU4RecModel(GRU4RecModel):
    """GRU4Rec network with optional modules.

    input_block: applied to the item embedding before the recurrent layers ("kan" or "mlp").
    head_block: applied to the final hidden state before dot-product scoring ("kan" or "mlp").
    cells: per-layer recurrent cell, "gru" (official nn.GRUCell), "kangru" or "kanrnn".
    """

    def __init__(self, n_items, layers, dropout_p_embed=0.0, dropout_p_hidden=0.0, embedding=0,
                 constrained_embedding=True, input_block=None, head_block=None, cells=None,
                 basis="rbf", grid_size=8):
        super().__init__(n_items, layers, dropout_p_embed, dropout_p_hidden, embedding, constrained_embedding)
        self.cells = cells or ["gru"] * len(layers)
        for i, cell in enumerate(self.cells):
            if cell != "gru":
                n_in = self.G[i].input_size
                self.G[i] = CELLS[cell](n_in, layers[i], basis=basis if cell in ("kangru", "mlpgru") else "bspline", grid_size=grid_size)
        n_input = layers[-1] if constrained_embedding else (embedding or n_items)
        self.input_block = ResidualBlock(n_input, input_block, basis=basis, grid_size=grid_size) if input_block else None
        self.head_block = ResidualBlock(layers[-1], head_block, basis=basis, grid_size=grid_size) if head_block else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        cells = getattr(self, "cells", None)
        if cells is None:
            return super().reset_parameters()
        saved = self.G
        self.G = nn.ModuleList([g for g, c in zip(saved, cells) if c == "gru"])
        super().reset_parameters()
        self.G = saved

    def forward(self, X, H, Y, training=False):
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)
        if self.input_block is not None:
            E = self.input_block(E)
        Xh = self.hidden_step(E, H, training=training)
        if self.head_block is not None:
            Xh = self.head_block(Xh)
        return self.score_items(Xh, O, B)


class DeviceSessionIterator(SessionDataIterator):
    """The official session-parallel iterator with one change: all item indices live on the device,
    and the items of each block of steps (until the shortest active session ends) are gathered with a
    single host-to-device transfer instead of one transfer per step. Yields exactly what the official
    iterator yields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_dev = torch.tensor(np.array(self.data_items), device=self.device)

    def __call__(self, enable_neg_samples, reset_hook=None):
        batch_size = self.batch_size
        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters] + 1]
        finished = False
        while not finished:
            minlen = (end - start).min()
            pos = torch.as_tensor(start[None, :] + np.arange(minlen)[:, None], device=self.device)
            block = self.items_dev[pos]
            out_idx = block[0]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = block[i + 1]
                if enable_neg_samples:
                    y = torch.cat([out_idx, self.sample_cache.get_sample()])
                else:
                    y = out_idx
                yield in_idx, y
            start = start + minlen - 1
            finished_mask = end - start <= 1
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = iters < len(self.offset_sessions) - 1
            n_valid = valid_mask.sum()
            if n_valid == 0:
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)


def _reset_hidden(H, finished_mask, valid_mask, n_valid):
    with torch.no_grad():
        for i in range(len(H)):
            H[i][finished_mask] = 0
    if n_valid < len(valid_mask):
        for i in range(len(H)):
            H[i] = H[i][valid_mask]
    return False


@torch.no_grad()
def evaluate(model, data, itemidmap, layers, device, batch_size=512):
    """Rank of the true next item for every event in `data` (conservative tie handling, as the
    official batch_eval). Returns a numpy array of ranks, one per prediction."""
    model.eval()
    batch_size = min(batch_size, data["SessionId"].nunique())
    H = [torch.zeros((batch_size, h), device=device) for h in layers]
    it = DeviceSessionIterator(data, batch_size, 0, 0, 0, device=device, itemidmap=itemidmap)
    ranks = []
    hook = lambda n_valid, finished_mask, valid_mask: _reset_hidden(H, finished_mask, valid_mask, n_valid)
    for in_idx, out_idx in it(enable_neg_samples=False, reset_hook=hook):
        scores = model.forward(in_idx, H, None, training=False)
        t = scores.gather(1, out_idx.view(-1, 1))
        ranks.append((scores >= t).sum(1))
    return torch.cat(ranks).cpu().numpy()


def train_session_model(train, valid, params, device, model_kwargs=None, n_epochs=None, seed=42,
                        select_metric="Recall@20", log=print, added_lr=1e-3):
    """Train a (KAN-)GRU4Rec model with the official training loop; evaluate on `valid` after each
    epoch and keep the best epoch. Parameters added to the official network use Adagrad with
    learning rate `added_lr`; with the official 0.07, unbounded residual blocks (KAN and MLP alike)
    diverge. Returns (model, itemidmap, history, layers)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = dict(params)
    n_epochs = n_epochs or params.pop("n_epochs", 10)
    params.pop("n_epochs", None)
    g = GRU4Rec(device=device, **params)
    it = DeviceSessionIterator(train, g.batch_size, n_sample=g.n_sample, sample_alpha=g.sample_alpha, device=device)
    if g.logq and g.loss == "cross-entropy":
        pop = train.groupby("ItemId").size()
        g.P0 = torch.tensor(pop[it.itemidmap.index.values].values, dtype=torch.float32, device=device)
    model = KANGRU4RecModel(it.n_items, g.layers, g.dropout_p_embed, g.dropout_p_hidden, g.embedding,
                            g.constrained_embedding, **(model_kwargs or {})).to(device)
    named = list(model.named_parameters())
    added = [p for n, p in named if added_parameter(n, model)]
    base = [p for n, p in named if not added_parameter(n, model)]
    groups = [{"params": base}] + ([{"params": added, "lr": added_lr}] if added else [])
    opt = IndexedAdagradM(groups, g.learning_rate, g.momentum)
    history, best, best_state = [], -1.0, None
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        H = [torch.zeros((g.batch_size, h), device=device) for h in g.layers]
        loss_sum = torch.zeros((), device=device)
        n_events = 0
        hook = lambda n_valid, finished_mask, valid_mask: _reset_hidden(H, finished_mask, valid_mask, n_valid)
        for in_idx, out_idx in it(enable_neg_samples=g.n_sample > 0, reset_hook=hook):
            for h in H:
                h.detach_()
            model.zero_grad()
            R = model.forward(in_idx, H, out_idx, training=True)
            n_valid = in_idx.shape[0]
            L = g.loss_function(R, out_idx, n_valid) / g.batch_size
            L.backward()
            opt.step()
            loss_sum += L.detach() * n_valid
            n_events += n_valid
        loss = loss_sum.item() / n_events
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
        train_s = time.time() - t0
        ranks = evaluate(model, valid, it.itemidmap, g.layers, device)
        m = session_metrics_from_ranks(ranks)
        history.append({"epoch": epoch + 1, "loss": loss, "train_s": train_s, **m})
        log(f"epoch {epoch + 1}: loss {loss:.4f} ({train_s:.1f}s) valid Recall@20 {m['Recall@20']:.4f} MRR@20 {m['MRR@20']:.4f}")
        if m[select_metric] > best:
            best, best_state = m[select_metric], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.n_params = n_params(model)
    return model, it.itemidmap, history, g.layers
