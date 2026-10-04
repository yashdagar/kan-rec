# KAN in Recurrent Recommenders

KAN layers in GRU4Rec (session-based, YooChoose) and in the basket-GRU (next-basket, Instacart and Dunnhumby), plus KAN-NBR, an interpretable repeat-aware next-basket model.

- `kan_rec.ipynb`: runs everything (download, preprocessing, training, tables, figures)
- `kanrec/`: the code the notebook calls
- `report/main.tex`: the report; tables and figures come from `report/generated/` and `report/figures/`
- `tests/`: unit tests (`uv run pytest`)

## Run on the Mac

```
uv sync
uv run jupyter lab kan_rec.ipynb
```

Keep `PROFILE = "mac"` and run all cells. Every finished model is saved under `results/mac/`, so a stopped run resumes where it left off.

## Run on the UPES HPC (Blackwell GPU)

Use `kan_rec_upes.ipynb`. It is the same notebook with `PROFILE = "upes"` preset, which runs only the key models (the worst-performing variants from the Mac runs are left out):

- YooChoose 1/64, 3 seeds: GRU4Rec, GRU4Rec (wide, same parameter count as the KAN-GRU cell), +KAN head, +MLP head, KAN-GRU cell with rational and B-spline bases, and the parameter-matched MLP-GRU cell for each
- YooChoose 1/4, 3 seeds: GRU4Rec, GRU4Rec (wide), KAN-GRU cell [rational], MLP-GRU cell [matched to rational]
- Instacart (all 206k users) and Dunnhumby, 3 seeds: frequency baselines, TIFU-KNN, Basket-GRU, KAN-GRU cell [rational] and its MLP control, logistic regression / KAN-NBR / MLP-NBR

Steps:

1. Copy the whole `kan-rec` folder to the cluster (`data/` and `results/` are not needed).
2. Install the environment: `pip install uv && uv sync` (or `pip install torch numpy pandas scipy matplotlib kagglehub jupyter nbconvert pyarrow pytest`).
3. Run headless, for example in a batch job:

```
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 kan_rec_upes.ipynb
```

Progress is written to `results/upes/run.log`, one line per epoch and per finished model. If the job is stopped, run the same command again: finished models are loaded from `results/upes/` and not retrained. The datasets download automatically on first run (about 2 GB). On CUDA the notebook enables TF32 matrix multiplication and bf16 autocast for the basket-GRU.

To send results back, copy `results/upes/` (or just `report/generated/` and `report/figures/`) and the executed `kan_rec_upes.ipynb`.

## Run on Google Colab

1. Upload the `kan-rec` folder to `MyDrive/kan-rec`.
2. Open `kan_rec_upes.ipynb` from Drive in Colab, select a GPU runtime, set `PROFILE = "colab"` and run all cells. The first cell mounts Drive.
3. If the session disconnects, run all cells again: finished models are loaded from `MyDrive/kan-rec/results/colab/` instead of being retrained.

## Build the report

```
cd report
tectonic -X compile main.tex
```

or upload the `report/` folder to Overleaf (pdfLaTeX works; the preamble switches fonts automatically). The tables are the ones produced by the last profile that was run.

## Notes

- GRU4Rec comes from the official implementation (github.com/hidasib/GRU4Rec_PyTorch_Official, free for research and education). It is not included in this repository: the first notebook cell clones it into `third_party/` at a fixed commit and applies `third_party/gru4rec_compat.patch`, a one-line pandas compatibility fix. Run the notebook once (or clone it manually) before `uv run pytest`.
- Instacart is downloaded from a public copy of the competition files, so no Kaggle account is needed.
