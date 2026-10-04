"""Resumable experiment runs. Each (dataset, model, seed) result is stored as JSON (plus per-event or
per-user metric arrays for significance tests) and reused if it already exists."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import basket as nb
from . import session as ns
from . import session_baselines as sb
from .metrics import session_metrics_from_ranks, summarize

SESSION_VARIANTS = {
    "GRU4Rec": {},
    "+KAN head": {"head_block": "kan"},
    "+MLP head": {"head_block": "mlp"},
    "+KAN input": {"input_block": "kan"},
    "+MLP input": {"input_block": "mlp"},
    "+KAN head+input": {"head_block": "kan", "input_block": "kan"},
    "KAN-GRU cell": {"cells": ["kangru"]},
    "MLP-GRU cell": {"cells": ["mlpgru"]},
    "Draft Pure KAN": {"cells": ["kanrnn"]},
    "Draft Hybrid": {"cells": ["gru", "kanrnn"], "_layers": 2},
    "Draft Hybrid (MLP)": {"cells": ["gru", "mlprnn"], "_layers": 2},
    "GRU4Rec (2 layers)": {"_layers": 2},
    "GRU4Rec (wide)": {"_hidden": 634},
}

BASKET_VARIANTS = {
    "Basket-GRU": {},
    "+KAN head": {"head_block": "kan"},
    "+MLP head": {"head_block": "mlp"},
    "+KAN input": {"input_block": "kan"},
    "+MLP input": {"input_block": "mlp"},
    "KAN-GRU cell": {"cell": "kangru"},
    "MLP-GRU cell": {"cell": "mlpgru"},
}


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def _paths(results_dir, dataset, model, seed):
    d = Path(results_dir) / dataset
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(model)}__s{seed}"
    return d / f"{stem}.json", d / f"{stem}.npz"


def free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save(jpath, npath, record, arrays):
    np.savez_compressed(npath, **arrays)
    jpath.write_text(json.dumps(record, indent=1, default=float))
    free_memory()
    return record


def run_session_baselines(splits, results_dir, dataset, device):
    ids = splits["train"]["ItemId"].unique()
    imap = pd.Series(np.arange(len(ids)), index=ids)
    out = []
    for name, fn in (("Pop", sb.pop_ranks), ("S-Pop", sb.spop_ranks), ("Item-kNN", sb.item_knn_ranks)):
        jpath, npath = _paths(results_dir, dataset, name, 0)
        if jpath.exists():
            out.append(json.loads(jpath.read_text()))
            continue
        ranks = fn(splits["train"], splits["test"], imap, device)
        rec = {"dataset": dataset, "model": name, "seed": 0, "test": session_metrics_from_ranks(ranks),
               "n_params": 0, "kind": "baseline"}
        out.append(_save(jpath, npath, rec, {"ranks": ranks}))
    return out


def run_session_model(splits, name, model_kwargs, params, results_dir, dataset, device, seed=42, n_epochs=None, log=print):
    jpath, npath = _paths(results_dir, dataset, name, seed)
    if jpath.exists():
        return json.loads(jpath.read_text())
    kwargs = dict(model_kwargs)
    params = dict(params)
    n_layers = kwargs.pop("_layers", 1)
    hidden = kwargs.pop("_hidden", None)
    if hidden is not None:
        params["layers"] = [hidden]
    params["layers"] = list(params["layers"]) * n_layers if len(params["layers"]) == 1 else params["layers"]
    log(f"[{dataset}] {name} seed {seed}")
    model, imap, hist, layers = ns.train_session_model(splits["train"], splits["valid"], params, device, kwargs,
                                                       n_epochs=n_epochs, seed=seed, log=log)
    t0 = time.time()
    ranks = ns.evaluate(model, splits["test"], imap, layers, device)
    infer_s = time.time() - t0
    best = max(hist, key=lambda h: h["Recall@20"])
    rec = {"dataset": dataset, "model": name, "seed": seed, "kind": "neural", "model_kwargs": model_kwargs,
           "params": {k: v for k, v in params.items()}, "history": hist, "best_epoch": best["epoch"],
           "valid": {k: best[k] for k in best if "@" in k}, "test": session_metrics_from_ranks(ranks),
           "n_params": model.n_params, "train_s_per_epoch": float(np.mean([h["train_s"] for h in hist])),
           "test_infer_s": infer_s, "n_test_events": int(len(ranks))}
    log(f"  test Recall@20 {rec['test']['Recall@20']:.4f} MRR@20 {rec['test']['MRR@20']:.4f} params {model.n_params:,}")
    mdir = npath.parent / "models"
    mdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), mdir / f"{jpath.stem}.pt")
    return _save(jpath, npath, rec, {"ranks": ranks})


def run_basket_baselines(data, results_dir, device, tifu_grid=None):
    out = []
    users = data.test_users
    specs = [("G-TopFreq", lambda us: nb.frequency_scores(data, us, "g")),
             ("P-TopFreq", lambda us: nb.frequency_scores(data, us, "p")),
             ("GP-TopFreq", lambda us: nb.frequency_scores(data, us, "gp")),
             ("Last basket", lambda us: nb.frequency_scores(data, us, "last"))]
    for name, fn in specs:
        jpath, npath = _paths(results_dir, data.name, name, 0)
        if jpath.exists():
            out.append(json.loads(jpath.read_text()))
            continue
        per_user = _chunked_eval(fn, data, users)
        rec = {"dataset": data.name, "model": name, "seed": 0, "kind": "baseline", "n_params": 0,
               "test": summarize(per_user)}
        out.append(_save(jpath, npath, rec, per_user))
    jpath, npath = _paths(results_dir, data.name, "TIFU-KNN", 0)
    if jpath.exists():
        out.append(json.loads(jpath.read_text()))
        return out
    grid = tifu_grid or [dict(n_neighbors=k, within_decay=rb, group_decay=rg, group_size=m, alpha=a)
                         for k in (100, 300) for rb in (0.9,) for rg in (0.6, 0.8) for m in (3, 7) for a in (0.5, 0.7, 0.9)]
    best, best_cfg = -1, None
    vusers = data.valid_users[:5000]
    for cfg in grid:
        m = summarize(_chunked_eval(lambda us: nb.tifu_knn_scores(data, us, **cfg), data, vusers))["Recall@10"]
        if m > best:
            best, best_cfg = m, cfg
    per_user = _chunked_eval(lambda us: nb.tifu_knn_scores(data, us, **best_cfg), data, users)
    rec = {"dataset": data.name, "model": "TIFU-KNN", "seed": 0, "kind": "baseline", "n_params": 0,
           "config": best_cfg, "valid_Recall@10": best, "test": summarize(per_user)}
    out.append(_save(jpath, npath, rec, per_user))
    return out


def _chunked_eval(score_fn, data, users, chunk=2048):
    parts = []
    for i in range(0, len(users), chunk):
        us = users[i:i + chunk]
        parts.append(nb.evaluate_scores(score_fn(us), data, us))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run_basket_model(data, name, model_kwargs, train_kwargs, results_dir, device, seed=42, log=print):
    jpath, npath = _paths(results_dir, data.name, name, seed)
    if jpath.exists():
        return json.loads(jpath.read_text())
    log(f"[{data.name}] {name} seed {seed}")
    model, hist = nb.train_basket_gru(data, device, model_kwargs=model_kwargs, seed=seed, log=log, **train_kwargs)
    t0 = time.time()
    per_user = _chunked_eval(lambda us: nb.basket_gru_scores(model, data, us, device), data, data.test_users)
    infer_s = time.time() - t0
    best = max(hist, key=lambda h: h["Recall@10"])
    rec = {"dataset": data.name, "model": name, "seed": seed, "kind": "neural", "model_kwargs": model_kwargs,
           "train_kwargs": train_kwargs, "history": hist, "best_epoch": best["epoch"],
           "valid": {k: best[k] for k in best if k not in ("epoch", "loss", "train_s")},
           "test": summarize(per_user), "n_params": model.n_params,
           "train_s_per_epoch": float(np.mean([h["train_s"] for h in hist])), "test_infer_s": infer_s}
    log(f"  test Recall@10 {rec['test']['Recall@10']:.4f} NDCG@10 {rec['test']['NDCG@10']:.4f} P@B {rec['test']['P@B']:.4f}")
    mdir = npath.parent / "models"
    mdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), mdir / f"{jpath.stem}.pt")
    return _save(jpath, npath, rec, per_user)


def load_results(results_dir):
    rows = []
    for p in Path(results_dir).glob("*/*.json"):
        rows.append(json.loads(p.read_text()))
    return rows


KANNBR_VARIANTS = {
    "Logistic (features)": dict(kind="linear", hidden=0),
    "KAN-NBR (additive)": dict(kind="kan", hidden=0),
    "MLP-NBR (matched)": dict(kind="mlp", hidden=0),
    "KAN-NBR [F-4-1]": dict(kind="kan", hidden=4),
    "MLP-NBR [F-4-1] (matched)": dict(kind="mlp", hidden=4),
}


def run_kannbr_model(data, name, kwargs, results_dir, device, seed=42, log=print):
    from . import kannbr as kn
    jpath, npath = _paths(results_dir, data.name, name, seed)
    if jpath.exists():
        return json.loads(jpath.read_text())
    log(f"[{data.name}] {name} seed {seed}")
    vusers = nb.early_stopping_users(data)
    ev = lambda m, f: summarize(_chunked_eval(lambda us: kn.kannbr_scores(m, f, data, us, device), data, vusers))
    model, feat, hist = kn.train_kannbr(data, device, seed=seed, eval_fn=ev, log=log, **kwargs)
    t0 = time.time()
    per_user = _chunked_eval(lambda us: kn.kannbr_scores(model, feat, data, us, device), data, data.test_users)
    infer_s = time.time() - t0
    best = max(hist, key=lambda h: h["Recall@10"])
    rec = {"dataset": data.name, "model": name, "seed": seed, "kind": "neural", "model_kwargs": kwargs,
           "history": hist, "best_epoch": best["epoch"],
           "valid": {k: best[k] for k in best if k not in ("epoch", "loss", "train_s")},
           "test": summarize(per_user), "n_params": model.n_params,
           "train_s_per_epoch": float(np.mean([h["train_s"] for h in hist])), "test_infer_s": infer_s}
    log(f"  test Recall@10 {rec['test']['Recall@10']:.4f} NDCG@10 {rec['test']['NDCG@10']:.4f} P@B {rec['test']['P@B']:.4f}")
    mdir = npath.parent / "models"
    mdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), mdir / f"{jpath.stem}.pt")
    return _save(jpath, npath, rec, per_user)
