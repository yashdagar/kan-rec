import sys, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from kanrec import kannbr as kn
out = sys.argv[1]
names = ["Baskets since last\npurchase (log)", "Purchase count (log)", "Purchase share", "Decayed frequency",
         "Item popularity (log)", "Overdue ratio (log)"]
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
for row, ds in enumerate(["dunnhumby", "instacart"]):
    m = kn.KANNBR(len(kn.FEATURES), kind="kan", hidden=0)
    m.load_state_dict(torch.load(f"results/mac/{ds}/models/KAN_NBR__additive__s42.pt", map_location="cpu"))
    m.eval()
    mean = m.norm.running_mean; std = (m.norm.running_var + m.norm.eps).sqrt()
    for i in range(6):
        z = torch.linspace(-1.8, 1.8, 120)
        xs = (z * std[i] + mean[i]).numpy()
        keep = xs >= 0
        if keep.sum() < 20: keep[:] = True
        with torch.no_grad():
            y = m.f.edge_function(0, i, z).numpy()
        y = y - y[keep].mean()
        ax = axes[row][i]
        ax.plot(xs[keep], y[keep], lw=2.4, color="#6A1B9A" if row else "#1565C0")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.tick_params(labelsize=9)
        if row == 0:
            ax.set_title(names[i], fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[row][0].set_ylabel(("Dunnhumby" if row == 0 else "Instacart") + "\nlogit contribution", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=200)
