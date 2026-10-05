# Pending runs

Deferred on 2026-10-05 so that the next-basket models could run first on the Mac.

## Mac profile, YooChoose 1/64: extra seeds for the key comparison

Done: seed 42 for all four models; seed 43 for GRU4Rec, GRU4Rec (wide) and KAN-GRU cell [rational, G=8].

Still to run:

- seed 43: MLP-GRU cell [matched to rational, G=8]
- seed 44: GRU4Rec, GRU4Rec (wide), KAN-GRU cell [rational, G=8], MLP-GRU cell [matched to rational, G=8]

How: in `tools/build_notebook.py` set `key_seeds=[43, 44]` in the `"mac"` profile, rebuild with `uv run python tools/build_notebook.py`, and re-run `kan_rec.ipynb`. Finished runs are reused, so only these five train (about 15 minutes each on the M1 Pro).

Until then, the key comparison table (`report/generated/key_comparison.tex`) mixes models with one and two seeds; its caption must not claim three seeds.
