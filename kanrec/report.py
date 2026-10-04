"""Tables and significance tests written for the LaTeX report."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .experiments import _paths
from .metrics import per_event_session_metric


def results_frame(records, dataset, metrics):
    rows = []
    for r in records:
        if r["dataset"] != dataset:
            continue
        row = {"Model": r["model"], "seed": r["seed"], "Params": r.get("n_params", 0),
               "s/epoch": r.get("train_s_per_epoch", np.nan), "Best epoch": r.get("best_epoch", np.nan)}
        row.update({m: r["test"].get(m, np.nan) for m in metrics})
        rows.append(row)
    df = pd.DataFrame(rows)
    agg = df.groupby("Model", sort=False).agg(["mean", "std", "count"])
    return df, agg


def mean_std_table(df, metrics, order):
    g = df.groupby("Model")
    out = pd.DataFrame(index=[m for m in order if m in g.groups])
    for m in metrics + ["Params", "s/epoch"]:
        mean, std, n = g[m].mean(), g[m].std(), g[m].count()
        if m == "Params":
            out[m] = [f"{int(mean[k]):,}" if mean[k] else "--" for k in out.index]
        elif m == "AvgRank":
            out[m] = [f"{mean[k]:.1f}" + (f" $\\pm$ {std[k]:.1f}" if n[k] > 1 else "") for k in out.index]
        elif m == "s/epoch":
            out[m] = [f"{mean[k]:.1f}" if np.isfinite(mean[k]) else "--" for k in out.index]
        else:
            out[m] = [f"{mean[k]:.4f}" + (f" $\\pm$ {std[k]:.4f}" if n[k] > 1 else "") for k in out.index]
    return out


def bold_best(table, metrics, lower_better=()):
    t = table.copy()
    for m in metrics:
        vals = t[m].str.split(" ").str[0].astype(float)
        best = vals.min() if m in lower_better else vals.max()
        t[m] = [f"\\textbf{{{v}}}" if float(v.split(' ')[0]) == best else v for v in t[m]]
    return t


def fmt_p(p):
    return "$<10^{-300}$" if p == 0 else f"{p:.1e}"


def write_latex(table, path, caption, label, col_format=None):
    if table.index.nlevels == 1:
        table = table.rename_axis(index=None)
    n_index = table.index.nlevels
    tex = table.to_latex(escape=False, column_format=col_format or "l" * n_index + "r" * len(table.columns),
                         caption=caption, label=label, position="H")
    tex = tex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\centering\\small")
    if len(table.columns) + n_index >= 7:
        tex = tex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace(
            "\\end{tabular}", "\\end{tabular}}")
    Path(path).write_text(tex)
    return tex


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def session_significance(results_dir, dataset, models, reference="GRU4Rec", seed=42, metrics=("Recall@20", "MRR@20")):
    _, ref_path = _paths(results_dir, dataset, reference, seed)
    ref = np.load(ref_path)["ranks"]
    rows = []
    for m in models:
        if m == reference:
            continue
        _, p = _paths(results_dir, dataset, m, seed)
        if not p.exists():
            continue
        ranks = np.load(p)["ranks"]
        for met in metrics:
            a, b = per_event_session_metric(ranks, met), per_event_session_metric(ref, met)
            diff = a.mean() - b.mean()
            pval = wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(a != b) else 1.0
            rows.append({"Model": m, "Metric": met, "Delta vs " + reference: diff, "p": pval})
    df = pd.DataFrame(rows)
    if len(df):
        df["p (Holm)"] = holm(df["p"].to_numpy())
    return df


def basket_significance(results_dir, dataset, models, reference="Basket-GRU", seed=42, metrics=("Recall@10", "NDCG@10")):
    _, ref_path = _paths(results_dir, dataset, reference, seed)
    ref = np.load(ref_path)
    rows = []
    for m in models:
        if m == reference:
            continue
        p = _paths(results_dir, dataset, m, seed if not m.endswith(("TopFreq", "basket", "KNN")) else 0)[1]
        if not p.exists():
            continue
        cur = np.load(p)
        for met in metrics:
            a, b = cur[met], ref[met]
            pval = wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(a != b) else 1.0
            rows.append({"Model": m, "Metric": met, "Delta vs " + reference: a.mean() - b.mean(), "p": pval})
    df = pd.DataFrame(rows)
    if len(df):
        df["p (Holm)"] = holm(df["p"].to_numpy())
    return df
