"""Builds kan_rec.ipynb. Edit the cells here and re-run: `uv run python tools/build_notebook.py`."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# KAN in Recurrent Recommenders

Session-based (YooChoose) and next-basket (Instacart, Dunnhumby) recommendation with Kolmogorov-Arnold Network (KAN) layers inside GRU4Rec and the basket-GRU.

**How to run:** set `PROFILE` in the next cell, then run all cells top to bottom.

- `"mac"`: Apple Silicon (MPS). Reduced sizes. Run this first to check the pipeline and record timings.
- `"upes"`: UPES HPC, Blackwell GPU (CUDA). Full sizes, three seeds.
- `"colab"`: Google Colab GPU. Copy the whole `kan-rec` folder to `MyDrive/kan-rec` first; results are written there, so a disconnected session resumes where it stopped.

Every training run saves its result to `results/<dataset>/`, and a finished run is never repeated. Every table and figure for the report is written to `report/generated/` and `report/figures/`."""))

C.append(code('PROFILE = "mac"'))

C.append(md("## 0. Configuration"))
C.append(code('''%matplotlib inline
import os, sys, json, time, math, random, warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/kan-rec")
    os.chdir(ROOT)
    !pip -q install kagglehub
else:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import subprocess
GRU4REC_DIR = ROOT / "third_party" / "GRU4Rec_PyTorch_Official"
GRU4REC_COMMIT = "d1fc31105577665d2f105de1c8d9a38d223a9a22"
if not (GRU4REC_DIR / "gru4rec_pytorch.py").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/hidasib/GRU4Rec_PyTorch_Official", str(GRU4REC_DIR)], check=True)
    subprocess.run(["git", "-C", str(GRU4REC_DIR), "checkout", "-q", GRU4REC_COMMIT], check=True)
    subprocess.run(["git", "-C", str(GRU4REC_DIR), "apply", str(ROOT / "third_party" / "gru4rec_compat.patch")], check=True)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from kanrec import data_session as ds, data_basket as db, experiments as ex, report as rp
from kanrec.kan import KANLinear, n_params
from kanrec.session import KANGRU4RecModel

OFFICIAL_YOOCHOOSE = dict(loss="cross-entropy", constrained_embedding=True, embedding=0, elu_param=0,
                          layers=[480], batch_size=48, dropout_p_embed=0.0, dropout_p_hidden=0.2,
                          learning_rate=0.07, momentum=0.0, n_sample=2048, sample_alpha=0.2, bpreg=0.0, logq=1.0)

KEY_SESSION = {
    "GRU4Rec": {},
    "GRU4Rec (wide)": {"_hidden": 634},
    "+KAN head": {"head_block": "kan"},
    "+MLP head": {"head_block": "mlp"},
    "KAN-GRU cell [rational, G=8]": {"cells": ["kangru"], "basis": "rational", "grid_size": 8},
    "MLP-GRU cell [matched to rational, G=8]": {"cells": ["mlpgru"], "basis": "rational", "grid_size": 8},
    "KAN-GRU cell [bspline, G=8]": {"cells": ["kangru"], "basis": "bspline", "grid_size": 8},
    "MLP-GRU cell [matched to bspline, G=8]": {"cells": ["mlpgru"], "basis": "bspline", "grid_size": 8},
}
KEY_SESSION_LARGE = {k: KEY_SESSION[k] for k in ("GRU4Rec", "GRU4Rec (wide)", "KAN-GRU cell [rational, G=8]",
                                                 "MLP-GRU cell [matched to rational, G=8]")}
KEY_BASKET = {
    "Basket-GRU": {},
    "KAN-GRU cell [rational, G=8]": {"cell": "kangru", "basis": "rational", "grid_size": 8},
    "MLP-GRU cell [matched to rational, G=8]": {"cell": "mlpgru", "basis": "rational", "grid_size": 8},
}
KEY_KANNBR = {k: ex.KANNBR_VARIANTS[k] for k in ("Logistic (features)", "KAN-NBR (additive)", "MLP-NBR (matched)")}

PROFILES = {
    "mac": dict(
        device="mps",
        yoochoose_fractions=[64],
        session_params=dict(OFFICIAL_YOOCHOOSE),
        session_epochs=10,
        ablation_bases=["bspline", "rbf", "cheby", "rational"],
        ablation_grids=[4, 8, 12],
        instacart_user_frac=0.10,
        dunnhumby_max_items=None,
        basket=dict(hidden=512, epochs=30, lr=3e-3, batch_size=64, dropout=0.3, max_history=100),
        seeds=[42],
        key_seeds=[43, 44],
    ),
    "smoke": dict(
        device="mps",
        yoochoose_fractions=[64],
        session_params=dict(OFFICIAL_YOOCHOOSE, layers=[32], n_sample=128),
        session_epochs=1,
        ablation_bases=["bspline", "rbf", "cheby", "rational"],
        ablation_grids=[4, 8],
        instacart_user_frac=0.02,
        dunnhumby_max_items=None,
        basket=dict(hidden=64, epochs=1, lr=1e-3, batch_size=64, dropout=0.3, max_history=100),
        seeds=[42],
        key_seeds=[],
    ),
    "upes": dict(
        device="cuda",
        yoochoose_fractions=[64, 4],
        session_params=dict(OFFICIAL_YOOCHOOSE),
        session_epochs=10,
        ablation_bases=["bspline", "rbf", "cheby", "rational"],
        ablation_grids=[4, 8, 12],
        instacart_user_frac=1.0,
        dunnhumby_max_items=None,
        basket=dict(hidden=512, epochs=20, lr=3e-3, batch_size=64, dropout=0.3, max_history=100, amp=True),
        seeds=[42, 43, 44],
        key_seeds=[],
        run_ablation=False,
        session_models=KEY_SESSION,
        session_models_large=KEY_SESSION_LARGE,
        basket_models=KEY_BASKET,
        kannbr_models=KEY_KANNBR,
        key_models=KEY_SESSION_LARGE,
        key_target="KAN-GRU cell [rational, G=8]",
    ),
    "colab": dict(
        device="cuda",
        yoochoose_fractions=[64],
        session_params=dict(OFFICIAL_YOOCHOOSE),
        session_epochs=10,
        ablation_bases=["bspline", "rbf", "cheby", "rational"],
        ablation_grids=[4, 8, 12],
        instacart_user_frac=0.25,
        dunnhumby_max_items=None,
        basket=dict(hidden=512, epochs=20, lr=3e-3, batch_size=64, dropout=0.3, max_history=100),
        seeds=[42, 43, 44],
        key_seeds=[],
        run_ablation=False,
        session_models=KEY_SESSION,
        session_models_large=KEY_SESSION_LARGE,
        basket_models=KEY_BASKET,
        kannbr_models=KEY_KANNBR,
        key_models=KEY_SESSION_LARGE,
        key_target="KAN-GRU cell [rational, G=8]",
    ),
}
CFG = PROFILES[PROFILE]
SESSION_MODELS = CFG.get("session_models") or ex.SESSION_VARIANTS
SESSION_MODELS_LARGE = CFG.get("session_models_large") or SESSION_MODELS
BASKET_MODELS = CFG.get("basket_models") or ex.BASKET_VARIANTS
KANNBR_MODELS = CFG.get("kannbr_models") or ex.KANNBR_VARIANTS
RUN_ABLATION = CFG.get("run_ablation", True)
ABLATION_NAMES = set()

DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / PROFILE
GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
for p in (DATA_PROC, RESULTS, GEN, FIG):
    p.mkdir(parents=True, exist_ok=True)


def pick_device(wanted):
    if wanted == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if wanted == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    warnings.warn(f"{wanted} not available, using cpu")
    return torch.device("cpu")


DEVICE = pick_device(CFG["device"])
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
LOG_FILE = RESULTS / "run.log"


def LOG(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\\n")

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})
print(f"profile={PROFILE} device={DEVICE} torch={torch.__version__} root={ROOT}")'''))

C.append(md("""## 1. Data download

All three datasets are public on Kaggle and download without an API token. Instacart is taken from a public copy of the competition files (`psparks/instacart-market-basket-analysis`), which has the same row counts as the competition data."""))
C.append(code('''import kagglehub

RAW = {
    "yoochoose": Path(kagglehub.dataset_download("chadgostopp/recsys-challenge-2015")),
    "instacart": Path(kagglehub.dataset_download("psparks/instacart-market-basket-analysis")),
    "dunnhumby": Path(kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")),
}
for k, v in RAW.items():
    print(k, v, sorted(os.listdir(v)))'''))

C.append(md("""## 2. Preprocessing

**YooChoose (session-based).** GRU4Rec protocol: drop length-1 sessions and items seen in fewer than 5 sessions; sessions ending in the last day form the test set; the most recent 1/64 (or 1/4) of the remaining sessions is the training data, and its last day is held out for validation. Validation and test keep only items seen in training. The split is by session and by time, so no session contributes to two splits.

**Instacart and Dunnhumby (next-basket).** Port of van Maasakkers, Fok and Donkers (2023). Each user's last basket is held out; users are split 50/50 into validation and test; all earlier baskets of all users are used for training.
- Instacart: Kaggle `eval_set == "test"` orders are dropped (their products are hidden); rare products (fewer than 500 purchases in the training history) are merged within their aisle by cosine similarity, as in the authors' R script.
- Dunnhumby (The Complete Journey): products bought at least 50 times, households with at least 3 baskets, the 100 most recent baskets per household (the published script kept the oldest 100)."""))
C.append(code('''clicks = ds.load_yoochoose_clicks(RAW["yoochoose"], DATA_PROC)
YC = {}
for frac in CFG["yoochoose_fractions"]:
    cache = DATA_PROC / f"yoochoose_{frac}_splits.pkl"
    if cache.exists():
        YC[frac] = pd.read_pickle(cache)
    else:
        YC[frac] = ds.prepare_yoochoose(clicks, frac)
        pd.to_pickle(YC[frac], cache)
    print(f"yoochoose 1/{frac}")
    display(ds.session_stats(YC[frac]))

DUNN = db.load_dunnhumby(RAW["dunnhumby"], DATA_PROC, max_items=CFG["dunnhumby_max_items"])
INSTA = db.load_instacart(RAW["instacart"], DATA_PROC, user_frac=CFG["instacart_user_frac"])
basket_table = pd.DataFrame({"Instacart": db.basket_stats(INSTA), "Dunnhumby": db.basket_stats(DUNN)})
display(basket_table)'''))

C.append(code('''raw_lens = clicks.groupby("SessionId").size()
item_sup = clicks.groupby("ItemId").size()
yc_rows = {"Raw clicks": len(clicks), "Raw sessions": raw_lens.size, "Raw items": item_sup.size}
with open(GEN / "dataset_yoochoose.tex", "w") as fh:
    t = ds.session_stats(YC[CFG["yoochoose_fractions"][0]]).copy()
    t.columns = ["Train", "Validation", "Test"]
    fh.write(t.to_latex(float_format=lambda v: f"{v:,.2f}".rstrip("0").rstrip("."), caption=(
        f"YooChoose 1/{CFG['yoochoose_fractions'][0]} after preprocessing (raw log: {len(clicks):,} clicks, "
        f"{raw_lens.size:,} sessions, {item_sup.size:,} items)."), label="tab:yc_stats", position="H"))
with open(GEN / "dataset_baskets.tex", "w") as fh:
    fh.write(basket_table.to_latex(caption="Next-basket datasets after preprocessing.", label="tab:basket_stats", position="H"))

fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
ax[0].hist(raw_lens.clip(upper=30), bins=29, color="#4C72B0")
ax[0].set(title="YooChoose: clicks per session (raw)", xlabel="session length (clipped at 30)", ylabel="sessions", yscale="log")
s = np.sort(item_sup.to_numpy())[::-1]
ax[1].loglog(np.arange(1, len(s) + 1), s, color="#4C72B0")
ax[1].set(title="YooChoose: item popularity (raw)", xlabel="item rank", ylabel="clicks")
for data, c in ((INSTA, "#DD8452"), (DUNN, "#55A868")):
    n = np.array([len(b) for b in data.baskets])
    ax[2].hist(n, bins=40, alpha=0.6, label=data.name, color=c, density=True)
ax[2].set(title="Baskets per user", xlabel="baskets", ylabel="density")
ax[2].legend()
fig.tight_layout()
fig.savefig(FIG / "eda.pdf")
plt.show()
del clicks, raw_lens, item_sup
ex.free_memory()'''))

C.append(md("""## 3. KAN layer

`kanrec/kan.py` implements one `KANLinear` with four basis families: B-spline (Cox-de Boor on an extended grid), Gaussian RBF (FastKAN), Chebyshev polynomials (tanh-bounded input), and group-rational functions (GR-KAN). Each layer is `W_base silu(x) + sum_i phi(x_i)`. `ResidualBlock` wraps it as `x + f(LayerNorm(x))`, and the MLP control uses the same wrapper with a hidden width chosen so the parameter count matches.

The unit tests (`tests/`) check the B-spline partition of unity, agreement with the closed-form cubic B-spline, gradients for every basis, parameter counts, and parameter matching of the MLP controls."""))
C.append(code('''print(subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT).stdout[-400:])

xs = torch.linspace(-2.5, 2.5, 400)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=False)
for ax, basis in zip(axes, ("bspline", "rbf", "cheby")):
    layer = KANLinear(1, 1, basis=basis, grid_size=8)
    phi = layer.expand(xs.view(-1, 1))[:, 0]
    ax.plot(xs, phi.numpy(), lw=1)
    ax.set_title({"bspline": "Cubic B-spline basis", "rbf": "Gaussian RBF basis", "cheby": "Chebyshev basis (tanh input)"}[basis])
    ax.set_xlabel("x")
fig.tight_layout()
fig.savefig(FIG / "kan_bases.pdf")
plt.show()'''))

C.append(md("""## 4. Session-based experiments (YooChoose)

All neural models use the official GRU4Rec PyTorch training loop (session-parallel mini-batches, 2048 shared popularity-sampled negatives, cross-entropy with logQ correction, Adagrad) through `kanrec/session.py`. Only the network changes:

- **GRU4Rec**: the official model.
- **+KAN head / +MLP head**: residual block on the final hidden state before dot-product scoring, `h + f(LN(h))`.
- **+KAN input / +MLP input**: residual block on the item embedding before the GRU.
- **KAN-GRU cell**: update and reset gates stay linear + sigmoid; the candidate state is `tanh(KAN([x, r * h]))`.
- **Draft Pure KAN / Draft Hybrid**: the first draft's ungated cell `tanh(LN(KAN([x, h])))` (with a correct B-spline), alone and stacked on a GRU layer.

The best epoch is chosen on validation Recall@20; test metrics are computed once with that model."""))
C.append(code('''SESSION_RECORDS = []
for frac in CFG["yoochoose_fractions"]:
    name = f"yoochoose{frac}"
    SESSION_RECORDS += ex.run_session_baselines(YC[frac], RESULTS, name, DEVICE)
    for seed in CFG["seeds"]:
        models = SESSION_MODELS if frac == CFG["yoochoose_fractions"][0] else SESSION_MODELS_LARGE
        for model, kwargs in models.items():
            SESSION_RECORDS.append(ex.run_session_model(YC[frac], model, kwargs, CFG["session_params"], RESULTS, name,
                                                        DEVICE, seed=seed, n_epochs=CFG["session_epochs"], log=LOG))'''))

C.append(md("""### Basis and grid ablation

On the KAN placement with the best validation Recall@20 among the KAN variants, compare the four basis families (grid 8) and grid sizes 4 / 8 / 12 for the RBF basis."""))
C.append(code('''if not RUN_ABLATION:
    print("ablation skipped for this profile")
else:
    frac = CFG["yoochoose_fractions"][0]
    name = f"yoochoose{frac}"
    kan_variants = ["+KAN head", "+KAN input", "+KAN head+input", "KAN-GRU cell"]
    val = {r["model"]: r["valid"]["Recall@20"] for r in ex.load_results(RESULTS) if r["dataset"] == name and r["model"] in kan_variants and r["seed"] == CFG["seeds"][0]}
    BEST_PLACEMENT = max(val, key=val.get)
    print("best KAN placement on validation:", BEST_PLACEMENT, {k: round(v, 4) for k, v in val.items()})
    base_kwargs = ex.SESSION_VARIANTS[BEST_PLACEMENT]
    ABLATION = []
    for basis in CFG["ablation_bases"]:
        for grid in ([8] if basis != "rbf" else CFG["ablation_grids"]):
            if basis == "rational" and grid != 8:
                continue
            label = f"{BEST_PLACEMENT} [{basis}, G={grid}]"
            if basis == "rbf" and grid == 8:
                src, dst = (ex._paths(RESULTS, name, m, CFG["seeds"][0]) for m in (BEST_PLACEMENT, label))
                rec = json.loads(src[0].read_text())
                rec["model"] = label
                dst[0].write_text(json.dumps(rec))
                dst[1].write_bytes(src[1].read_bytes())
                ABLATION.append(rec)
                ABLATION_NAMES.add(label)
                continue
            ABLATION_NAMES.add(label)
            ABLATION.append(ex.run_session_model(YC[frac], label, dict(base_kwargs, basis=basis, grid_size=grid),
                                                 CFG["session_params"], RESULTS, name, DEVICE, seed=CFG["seeds"][0],
                                                 n_epochs=CFG["session_epochs"], log=LOG))'''))

C.append(md("""### Extra seeds for the key session models

The best ablation configuration, its parameter-matched MLP-GRU cell control and the two GRU4Rec baselines (official size and parameter-matched width) are trained with additional seeds, so that the main comparison reports mean and standard deviation. On the GPU profiles every model already runs with three seeds."""))
C.append(code('''frac = CFG["yoochoose_fractions"][0]
name = f"yoochoose{frac}"
if CFG.get("key_models"):
    KEY_MODELS = CFG["key_models"]
    BEST_ABLATION = CFG["key_target"]
else:
    abl_val = {r["model"]: r["valid"]["Recall@20"] for r in ex.load_results(RESULTS)
               if r["dataset"] == name and "[" in r["model"] and r["model"].startswith(BEST_PLACEMENT) and r["seed"] == CFG["seeds"][0]}
    BEST_ABLATION = max(abl_val, key=abl_val.get)
    best_basis = BEST_ABLATION.split("[")[1].split(",")[0]
    best_grid = int(BEST_ABLATION.split("G=")[1].rstrip("]"))
    print("best ablation configuration on validation:", BEST_ABLATION)
    KEY_MODELS = {
        "GRU4Rec": ex.SESSION_VARIANTS["GRU4Rec"],
        "GRU4Rec (wide)": ex.SESSION_VARIANTS["GRU4Rec (wide)"],
        BEST_ABLATION: dict(base_kwargs, basis=best_basis, grid_size=best_grid),
    }
    if BEST_PLACEMENT == "KAN-GRU cell":
        KEY_MODELS[f"MLP-GRU cell [matched to {best_basis}, G={best_grid}]"] = {"cells": ["mlpgru"], "basis": best_basis, "grid_size": best_grid}
for seed_k in CFG["seeds"] + CFG["key_seeds"]:
    for model, kwargs in KEY_MODELS.items():
        ex.run_session_model(YC[frac], model, kwargs, CFG["session_params"], RESULTS, name, DEVICE, seed=seed_k,
                             n_epochs=CFG["session_epochs"], log=LOG)'''))

C.append(md("## 5. Next-basket experiments (Instacart, Dunnhumby)"))
C.append(code('''BASKET_RECORDS = []
for data in (DUNN, INSTA):
    BASKET_RECORDS += ex.run_basket_baselines(data, RESULTS, DEVICE)
    for seed in CFG["seeds"]:
        for model, kwargs in BASKET_MODELS.items():
            BASKET_RECORDS.append(ex.run_basket_model(data, model, kwargs, CFG["basket"], RESULTS, DEVICE, seed=seed, log=LOG))'''))

C.append(md("""### KAN-NBR: an interpretable repeat-aware model

Repeat purchases dominate next-basket accuracy, so this model works directly on the repeat signal. Every (user, candidate item) pair, where candidates are the user's previously bought items plus the 100 most popular items, is described by seven scalar features: log baskets since the last purchase, log purchase count, purchase share, exponentially decayed frequency (0.9 per basket), log item popularity, log overdue ratio (baskets since last purchase divided by the user's average interval between purchases of that item) and whether the item is in the history. A KAN maps the features to a purchase logit. With a single KAN layer the model is additive, `score = sum_f phi_f(feature_f)`, so every learned `phi_f` is a readable curve. Controls: logistic regression on the same features (linear `phi_f`) and parameter-matched MLPs. Training pairs come from the last three positions of every user's history; the held-out final basket is never used. These models are small, so they run on the CPU."""))
C.append(code('''CPU = torch.device("cpu")
for data in (DUNN, INSTA):
    for seed in CFG["seeds"]:
        for model, kwargs in KANNBR_MODELS.items():
            BASKET_RECORDS.append(ex.run_kannbr_model(data, model, kwargs, RESULTS, CPU, seed=seed, log=LOG))'''))

C.append(md("## 6. Results"))
C.append(code('''ALL = ex.load_results(RESULTS)
SEED_NOTE = "single seed" if len(CFG["seeds"]) == 1 else f"mean $\\\\pm$ std over {len(CFG['seeds'])} seeds"
S_METRICS = ["Recall@10", "Recall@20", "MRR@20", "NDCG@10", "NDCG@20"]
S_ORDER = ["Pop", "S-Pop", "Item-kNN"] + list(SESSION_MODELS)
for frac in CFG["yoochoose_fractions"]:
    name = f"yoochoose{frac}"
    df, _ = rp.results_frame(ALL, name, S_METRICS)
    table = rp.bold_best(rp.mean_std_table(df, S_METRICS, S_ORDER), S_METRICS)
    display(table)
    rp.write_latex(table, GEN / f"results_{name}.tex",
                   f"Test results on YooChoose 1/{frac} ({SEED_NOTE}; best in bold).",
                   f"tab:results_{name}")
    sig = rp.session_significance(RESULTS, name, list(SESSION_MODELS), seed=CFG["seeds"][0])
    display(sig)
    rp.write_latex(sig.assign(**{"p": sig["p"].map(rp.fmt_p), "p (Holm)": sig["p (Holm)"].map(rp.fmt_p)}).set_index("Model"),
                   GEN / f"significance_{name}.tex",
                   f"Paired Wilcoxon signed-rank tests against GRU4Rec on YooChoose 1/{frac} test events (Holm-corrected).",
                   f"tab:sig_{name}")

PUBLISHED_YC64 = {"Pop": (6.71, 1.65), "S-Pop": (30.44, 18.35), "Item-kNN": (51.60, 21.81), "GRU4Rec": (60.64, 22.89),
                  "NARM": (68.32, 28.63), "STAMP": (68.74, 29.67), "SR-GNN": (70.57, 30.94)}
if 64 in CFG["yoochoose_fractions"]:
    ours = {r["model"]: r["test"] for r in ALL if r["dataset"] == "yoochoose64" and r["seed"] == CFG["seeds"][0]}
    ref_path = ROOT / "results" / "official_gru4rec_yoochoose64.json"
    rows = []
    for m, (rec, mrr) in PUBLISHED_YC64.items():
        row = {"Model": m, "Published Recall@20": f"{rec:.2f}", "Published MRR@20": f"{mrr:.2f}"}
        if m in ours:
            row.update({"Ours Recall@20": f"{100 * ours[m]['Recall@20']:.2f}", "Ours MRR@20": f"{100 * ours[m]['MRR@20']:.2f}"})
        rows.append(row)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["test"]
        rows.append({"Model": "GRU4Rec (official code, epoch 10)", "Published Recall@20": "", "Published MRR@20": "",
                     "Ours Recall@20": f"{100 * ref['Recall@20']:.2f}", "Ours MRR@20": f"{100 * ref['MRR@20']:.2f}"})
    repro = pd.DataFrame(rows).set_index("Model").fillna("--").replace("", "--")
    display(repro)
    rp.write_latex(repro, GEN / "reproduction_yoochoose64.tex",
                   "Our yoochoose1/64 test results next to the numbers published by Wu et al. (SR-GNN, 2019), in \\\\%. "
                   "The published GRU4Rec row is the original 2016 version; ours uses the improved 2018 training.",
                   "tab:repro")

abl = [r for r in ALL if RUN_ABLATION and r["dataset"] == f"yoochoose{CFG['yoochoose_fractions'][0]}"
       and r["model"].startswith(BEST_PLACEMENT + " [") and r["seed"] == CFG["seeds"][0]]
if abl:
    adf = pd.DataFrame([{"Variant": r["model"].split("[")[1].rstrip("]"), "Valid Recall@20": r["valid"]["Recall@20"],
                         "Test Recall@20": r["test"]["Recall@20"], "Test MRR@20": r["test"]["MRR@20"],
                         "Params": f"{r['n_params']:,}", "s/epoch": round(r["train_s_per_epoch"], 1)} for r in abl]).set_index("Variant")
    display(adf)
    rp.write_latex(adf.round(4), GEN / "ablation_basis.tex", f"KAN basis and grid ablation for {BEST_PLACEMENT} on YooChoose.", "tab:ablation")'''))

C.append(code('''name = f"yoochoose{CFG['yoochoose_fractions'][0]}"
kdf, _ = rp.results_frame([r for r in ALL if r["model"] in KEY_MODELS], name, S_METRICS)
n_key = len(CFG["seeds"] + CFG["key_seeds"])
ktable = rp.bold_best(rp.mean_std_table(kdf, S_METRICS, list(KEY_MODELS)), S_METRICS)
display(ktable)
rp.write_latex(ktable, GEN / "key_comparison.tex",
               f"Main comparison on YooChoose 1/{CFG['yoochoose_fractions'][0]}: best KAN configuration against GRU4Rec, a GRU4Rec with "
               f"the same number of parameters, and a parameter-matched MLP-GRU cell (mean $\\\\pm$ std over {n_key} seeds).",
               "tab:key")
key_sig = []
for ref in [m for m in KEY_MODELS if m != BEST_ABLATION]:
    for sd in CFG["seeds"] + CFG["key_seeds"]:
        d = rp.session_significance(RESULTS, name, [BEST_ABLATION], reference=ref, seed=sd)
        if len(d):
            key_sig.append(d.rename(columns={f"Delta vs {ref}": "Delta"}).assign(Reference=ref, Seed=sd))
if key_sig:
    ks = pd.concat(key_sig)
    ks["p (Holm)"] = rp.holm(ks["p"].to_numpy())
    display(ks)
    rp.write_latex(ks[["Reference", "Seed", "Metric", "Delta", "p", "p (Holm)"]].assign(
                       Delta=ks["Delta"].map("{:+.4f}".format), p=ks["p"].map(rp.fmt_p),
                       **{"p (Holm)": ks["p (Holm)"].map(rp.fmt_p)}).set_index("Reference"),
                   GEN / "key_significance.tex",
                   f"Paired Wilcoxon tests of {BEST_ABLATION} against each reference model, per seed (Holm-corrected).", "tab:key_sig")

B_METRICS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "PHR@10", "P@B", "R@2B", "AvgRank", "RepRecall@10", "ExplRecall@10"]
B_ORDER = ["G-TopFreq", "P-TopFreq", "GP-TopFreq", "Last basket", "TIFU-KNN"] + list(BASKET_MODELS) + list(KANNBR_MODELS)
for data in (INSTA, DUNN):
    df, _ = rp.results_frame(ALL, data.name, B_METRICS)
    table = rp.bold_best(rp.mean_std_table(df, B_METRICS, B_ORDER), B_METRICS, lower_better=("AvgRank",))
    display(table)
    rp.write_latex(table[["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "P@B", "AvgRank", "Params"]], GEN / f"results_{data.name}.tex",
                   f"Test results on {data.name.capitalize()} ({SEED_NOTE}; best in bold). P@B is precision at the true basket size.", f"tab:results_{data.name}")
    rp.write_latex(table[["RepRecall@10", "ExplRecall@10", "PHR@10", "R@2B"]], GEN / f"repeat_{data.name}.tex",
                   f"Repeat vs explore recall on {data.name.capitalize()}.", f"tab:repeat_{data.name}")
    sig = rp.basket_significance(RESULTS, data.name, list(BASKET_MODELS) + list(KANNBR_MODELS) + ["GP-TopFreq", "TIFU-KNN"], seed=CFG["seeds"][0])
    display(sig)
    rp.write_latex(sig.assign(**{"p": sig["p"].map(rp.fmt_p), "p (Holm)": sig["p (Holm)"].map(rp.fmt_p)}).set_index("Model"),
                   GEN / f"significance_{data.name}.tex", f"Paired Wilcoxon tests against Basket-GRU on {data.name.capitalize()} test users (Holm-corrected).",
                   f"tab:sig_{data.name}")'''))

C.append(code('''def curves(dataset, models, metric, ax, seed):
    for r in ALL:
        if r["dataset"] == dataset and r["model"] in models and r["seed"] == seed and "history" in r:
            ax.plot([h["epoch"] for h in r["history"]], [h[metric] for h in r["history"]], marker="o", ms=3, lw=1.2, label=r["model"])
    ax.set(xlabel="epoch", ylabel=f"validation {metric}")

seed = CFG["seeds"][0]
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
curves(f"yoochoose{CFG['yoochoose_fractions'][0]}", list(SESSION_MODELS), "Recall@20", axes[0], seed)
axes[0].set_title("YooChoose")
curves("instacart", list(BASKET_MODELS), "Recall@10", axes[1], seed)
axes[1].set_title("Instacart")
curves("dunnhumby", list(BASKET_MODELS), "Recall@10", axes[2], seed)
axes[2].set_title("Dunnhumby")
axes[0].legend(fontsize=6)
axes[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "validation_curves.pdf")
plt.show()

name = f"yoochoose{CFG['yoochoose_fractions'][0]}"
pts = [r for r in ALL if r["dataset"] == name and r.get("kind") == "neural" and r["seed"] == seed]
def family(m):
    if "KAN" in m and "MLP" not in m:
        return "KAN", "#DD8452", "o"
    if "MLP" in m:
        return "MLP (matched)", "#4C72B0", "s"
    return "GRU only", "#55A868", "^"
fig, ax = plt.subplots(figsize=(6.4, 4))
seen = set()
for r in pts:
    lab, col, mk = family(r["model"])
    ax.scatter(r["n_params"] / 1e6, r["test"]["Recall@20"], color=col, marker=mk, s=36, label=None if lab in seen else lab)
    seen.add(lab)
    ax.annotate(r["model"].replace("KAN-GRU cell ", "").replace(" seed", ""), (r["n_params"] / 1e6, r["test"]["Recall@20"]),
                fontsize=6, xytext=(3, 2), textcoords="offset points")
ax.set(xlabel="parameters (millions)", ylabel="test Recall@20", title=f"Accuracy vs model size, YooChoose 1/{CFG['yoochoose_fractions'][0]}")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "accuracy_vs_params.pdf")
plt.show()

cost = pd.DataFrame([{"Dataset": r["dataset"], "Model": r["model"], "Params": r["n_params"],
                      "s/epoch": r["train_s_per_epoch"], "Test inference (s)": r["test_infer_s"]}
                     for r in ALL if r.get("kind") == "neural" and r["seed"] == seed and r["model"] not in ABLATION_NAMES])
display(cost)
rp.write_latex(cost.assign(Params=cost["Params"].map("{:,}".format)).round(1).set_index(["Dataset", "Model"]),
               GEN / "cost.tex", f"Model size and training cost on the {PROFILE} profile ({DEVICE.type}).", "tab:cost")'''))

C.append(md("""## 7. Interpretability: learned KAN functions

For the best KAN model on YooChoose, plot the learned univariate functions on the edges with the largest spline contribution, and measure how far each edge function is from linear (R² of a linear fit over the input range). An edge with R² near 1 is effectively linear, which is what an MLP layer already provides."""))
C.append(code('''frac = CFG["yoochoose_fractions"][0]
name = f"yoochoose{frac}"
rec = max((r for r in ALL if r["dataset"] == name and r["model"] in ("+KAN head", "+KAN input", "+KAN head+input") and r["seed"] == seed),
          key=lambda r: r["valid"]["Recall@20"])
kw = dict(SESSION_MODELS.get(rec["model"], ex.SESSION_VARIANTS.get(rec["model"], {})))
ids = YC[frac]["train"]["ItemId"].unique()
model = KANGRU4RecModel(len(ids), CFG["session_params"]["layers"], **kw)
model.load_state_dict(torch.load(RESULTS / name / "models" / f"{ex._slug(rec['model'])}__s{seed}.pt", map_location="cpu"))
block = model.head_block if model.head_block is not None else model.input_block
layer = block.f
xs = torch.linspace(-2, 2, 200)
strength = (layer.coef.abs().sum(-1)).flatten()
top = torch.topk(strength, 9).indices
fig, axes = plt.subplots(3, 3, figsize=(8, 6), sharex=True)
for ax, idx in zip(axes.flat, top.tolist()):
    o, i = divmod(idx, layer.in_features)
    ax.plot(xs, layer.edge_function(o, i, xs).numpy(), lw=1.3)
    ax.set_title(f"out {o} <- in {i}", fontsize=8)
fig.suptitle(f"Strongest learned edge functions, {rec['model']} (YooChoose)")
fig.tight_layout()
fig.savefig(FIG / "kan_edges.pdf")
plt.show()

def linear_r2(y, x):
    A = np.stack([x, np.ones_like(x)], 1)
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - resid.var() / (y.var() + 1e-12)

sample = torch.randperm(layer.in_features * layer.out_features)[:2000]
r2 = np.array([linear_r2(layer.edge_function(*divmod(int(k), layer.in_features), xs).numpy(), xs.numpy()) for k in sample])
print(f"edge functions with linear-fit R^2 > 0.99: {np.mean(r2 > 0.99):.1%}; median R^2 {np.median(r2):.3f}")
fig, ax = plt.subplots(figsize=(4, 2.6))
ax.hist(r2, bins=40, color="#4C72B0")
ax.set(xlabel="R$^2$ of linear fit", ylabel="edges", title="Non-linearity of learned edge functions")
fig.savefig(FIG / "kan_linearity.pdf")
plt.show()
(GEN / "kan_linearity.tex").write_text(f"\\\\newcommand{{\\\\kanLinearShare}}{{{np.mean(r2 > 0.99) * 100:.1f}}}\\n\\\\newcommand{{\\\\kanMedianRtwo}}{{{np.median(r2):.3f}}}\\n")'''))

C.append(md("""### Learned KAN-NBR feature functions

The additive KAN-NBR is a sum of one learned function per feature. Each curve shows the contribution of a feature to the purchase logit (centred), over the 1st to 99th percentile of the feature in the training pairs. The binary history indicator is shown as two points."""))
C.append(code('''from kanrec import kannbr as kn
fig, axes = plt.subplots(2, 7, figsize=(16, 4.4))
for row, data in enumerate((INSTA, DUNN)):
    kw = KANNBR_MODELS["KAN-NBR (additive)"]
    feat = kn.NBRFeaturizer(data)
    m = kn.KANNBR(len(kn.FEATURES), **{k: v for k, v in kw.items()})
    m.load_state_dict(torch.load(RESULTS / data.name / "models" / f"{ex._slug('KAN-NBR (additive)')}__s{seed}.pt", map_location="cpu"))
    m.eval()
    X, _ = kn.build_training_set(feat, data, np.arange(min(2000, len(data.baskets))))
    curves = kn.shape_functions(m, X)
    for ax, (name, (xs, ys)) in zip(axes[row], curves.items()):
        if name == "in history":
            pts = kn.shape_functions(m, X, n_points=2)[name]
            ax.plot([0, 1], [pts[1][0], pts[1][-1]], "o", color="#C44E52")
            ax.set_xticks([0, 1])
        else:
            ax.plot(xs, ys, lw=1.6, color="#4C72B0")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(name, fontsize=8)
    axes[row][0].set_ylabel(f"{data.name}\\nlogit contribution")
fig.tight_layout()
fig.savefig(FIG / "kannbr_shapes.pdf")
plt.show()'''))

C.append(md("""## 8. Decision rule

Following the plan: a KAN placement counts as helpful only if it beats both the baseline and its parameter-matched MLP control by more than the seed-to-seed standard deviation (or, with one seed, by a Holm-significant margin) on **validation**."""))
C.append(code('''def verdict(dataset, base, pairs, metric):
    v = {r["model"]: r["valid"][metric] for r in ALL if r["dataset"] == dataset and r["seed"] == seed and "valid" in r}
    for kan, mlp in pairs:
        if kan in v:
            print(f"{dataset:12s} {kan:18s} {metric} {v[kan]:.4f} | {base} {v.get(base, float('nan')):.4f} | {mlp} {v.get(mlp, float('nan')):.4f} -> "
                  + ("KAN helps" if v[kan] > v.get(base, 0) and v[kan] > v.get(mlp, 0) else "no gain over baseline/MLP"))

for frac in CFG["yoochoose_fractions"]:
    verdict(f"yoochoose{frac}", "GRU4Rec", [("+KAN head", "+MLP head"), ("+KAN input", "+MLP input"), ("KAN-GRU cell", "MLP-GRU cell")], "Recall@20")
    verdict(f"yoochoose{frac}", "GRU4Rec (2 layers)", [("Draft Hybrid", "Draft Hybrid (MLP)")], "Recall@20")
    verdict(f"yoochoose{frac}", "GRU4Rec (wide)", [("KAN-GRU cell", "MLP-GRU cell"),
             ("KAN-GRU cell [rational, G=8]", "MLP-GRU cell [matched to rational, G=8]"),
             ("KAN-GRU cell [bspline, G=8]", "MLP-GRU cell [matched to bspline, G=8]")], "Recall@20")
for data in (INSTA, DUNN):
    verdict(data.name, "Basket-GRU", [("+KAN head", "+MLP head"), ("+KAN input", "+MLP input"), ("KAN-GRU cell", "MLP-GRU cell")], "Recall@10")
    verdict(data.name, "Basket-GRU", [("KAN-GRU cell [rational, G=8]", "MLP-GRU cell [matched to rational, G=8]")], "Recall@10")
    verdict(data.name, "Logistic (features)", [("KAN-NBR (additive)", "MLP-NBR (matched)"), ("KAN-NBR [F-4-1]", "MLP-NBR [F-4-1] (matched)")], "Recall@10")'''))

C.append(md("""## Appendix A. Audit of the first draft's KAN layer

Runs the `KANLinear._bspline` code published in the first draft and compares it with a correct cubic B-spline basis. This is the evidence for the errata appendix of the report."""))
C.append(code('''import torch.nn as nn
import torch.nn.functional as F
from kanrec.kan import bspline_basis


class DraftKANLinear(nn.Module):
    def __init__(self, in_dim, out_dim, grid=8, order=3):
        super().__init__()
        self.grid, self.order = grid, order
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, grid + order) * 0.05)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.scale = nn.Parameter(torch.ones(out_dim, in_dim) * 0.1)
        self.register_buffer("knots", torch.linspace(-1, 1, grid + 1))

    def _bspline(self, x):
        k0, target = self.knots, self.grid + self.order
        xv = x.unsqueeze(-1)
        b = ((xv >= k0[:-1]) & (xv < k0[1:])).float()
        for k in range(1, self.order + 1):
            kext = torch.cat([k0[0].expand(k), k0, k0[-1].expand(k)])
            w = b.size(-1)
            denom = (kext[k: k + w] - kext[:w]).clamp(min=1e-6)
            alpha = ((xv - kext[:w]) / denom).clamp(0.0, 1.0)
            b_sh = F.pad(b[..., 1:], (0, 1))
            b = alpha * b + (1.0 - alpha) * b_sh
            if b.size(-1) < target:
                b = F.pad(b, (0, target - b.size(-1)))
        return b


xs = torch.linspace(-0.999, 0.999, 2001)
draft = DraftKANLinear(4, 3)._bspline(xs.unsqueeze(0)).squeeze(0)
grid = torch.arange(-3, 12, dtype=torch.float32) * 0.25 - 1
ref = bspline_basis(xs, grid, 3)
dead = (draft.abs().amax(0) == 0).nonzero().flatten().tolist()
print(f"draft basis takes only values 0/1 (degree-0 step functions): {bool(((draft == 0) | (draft == 1)).all())}")
print(f"draft coefficients that are never used: {dead} of {draft.shape[1]}")
print(f"max |draft - correct cubic B-spline|: {(draft - ref).abs().max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
for ax, basis, title in ((axes[0], draft, "First draft (as published)"), (axes[1], ref, "Correct cubic B-spline")):
    for i in range(basis.shape[1]):
        ax.plot(xs, basis[:, i], lw=1)
    ax.set_title(title)
    ax.set_xlabel("x")
axes[0].set_ylabel("basis value")
fig.tight_layout()
fig.savefig(FIG / "audit_bspline_basis.pdf")
plt.show()'''))

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "kan_rec.ipynb"
if len(sys.argv) > 2:
    C[1] = code(f'PROFILE = "{sys.argv[2]}"')
    nb["cells"] = C
nbf.write(nb, out)
print(f"wrote {out} with {len(C)} cells")
