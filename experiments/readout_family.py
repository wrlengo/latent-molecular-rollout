"""Readout-family and displacement-cone analysis on archived latents.

Fits a family of scalar readouts on the same frozen latents, then measures
(a) how well each readout's input gradient aligns with true displacement
directions and (b) how well it recovers property amplitude. Also characterises
the geometry of the displacement set itself.

Dependencies: numpy only.
Outputs: data/readout_family.json, figs/fig_readout_family.pdf/.png
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")


# ---------- small stats helpers (no scipy) ----------
def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    return pearson(ra, rb)


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return 0.0
    r = float(np.dot(a, b) / den)
    n = len(a)
    if n < 3 or abs(r) >= 1.0:
        return r, 0.0
    # two-sided t approximation without scipy
    from math import sqrt
    t = r * sqrt((n - 2) / max(1e-12, 1 - r * r))
    # rough normal approximation for p (n is large here)
    from math import erfc
    p = erfc(abs(t) / sqrt(2.0))
    return r, p


def spearman_p(a, b):
    r = spearman(a, b)
    a2, b2 = np.asarray(a, float), np.asarray(b, float)
    if isinstance(r, tuple):
        r = r[0]
    _, p = pearson(_rank(a2), _rank(b2))
    return float(r), float(p)


# ---------- models ----------
def ridge_fit(X, y, alpha):
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d, dtype=X.dtype)
    w = np.linalg.solve(A, X.T @ y)
    return w


class NumpyMLP:
    """One-hidden-layer ReLU regressor trained with Adam; exact input gradient."""

    def __init__(self, d_in, width, seed=0, lr=1e-3, epochs=120, batch=128, patience=10):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, width)).astype(np.float32)
        self.b1 = np.zeros(width, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / width), (width, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr, self.epochs, self.batch, self.patience = lr, epochs, batch, patience

    def forward(self, X, cache=False):
        pre = X @ self.W1 + self.b1
        act = np.maximum(pre, 0.0)
        out = act @ self.W2 + self.b2
        if cache:
            return out.ravel(), pre, act
        return out.ravel()

    def fit(self, X, y, Xv, yv):
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        rng = np.random.default_rng(1)
        best, best_params, bad = np.inf, None, 0
        b1, b2 = 0.9, 0.999
        for ep in range(1, self.epochs + 1):
            perm = rng.permutation(len(y))
            for s in range(0, len(y), self.batch):
                idx = perm[s:s + self.batch]
                Xb, yb = X[idx], y[idx]
                out, pre, act = self.forward(Xb, cache=True)
                err = (out - yb) / len(idx)
                dW2 = act.T @ err[:, None]
                db2 = np.array([err.sum()], dtype=np.float32)
                dact = err[:, None] * self.W2.T
                dpre = dact * (pre > 0)
                dW1 = Xb.T @ dpre
                db1 = dpre.sum(0)
                for p, g, mi, vi in zip(params, [dW1, db1, dW2, db2], m, v):
                    mi *= b1
                    mi += (1 - b1) * g
                    vi *= b2
                    vi += (1 - b2) * g * g
                    p -= self.lr * (mi / (1 - b1 ** ep)) / (np.sqrt(vi / (1 - b2 ** ep)) + 1e-8)
            vmae = float(np.mean(np.abs(self.forward(Xv) - yv)))
            if vmae < best - 1e-4:
                best, bad = vmae, 0
                best_params = [p.copy() for p in params]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2 = best_params
        self.val_mae = best
        return self

    def predict(self, X):
        return self.forward(X)

    def input_grad(self, X):
        """d output / d input, exact, shape (n, d)."""
        pre = X @ self.W1 + self.b1
        g = self.W2.T[0][None, :] * (pre > 0)          # (n, width)
        return g @ self.W1.T                            # (n, d)


def load_split(name):
    z = np.load(os.path.join(DATA, f"h_{name}.npz"), allow_pickle=True)
    return z["h"].astype(np.float32), z["gt"].astype(np.float32), z["smiles"]


def main():
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    Xte, yte, smi_te = load_split("test")

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ybar = float(ytr.mean())
    yc = ytr - ybar

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridges = {}
    for a in alphas:
        w = ridge_fit(Ztr, yc, a)
        ridges[a] = dict(
            w_raw=w / sd,
            val_mae=float(np.mean(np.abs(Zva @ w + ybar - yva))),
            test_mae=float(np.mean(np.abs(Zte @ w + ybar - yte))),
        )
        print(f"ridge alpha={a:<6} val_mae={ridges[a]['val_mae']:.4f} test_mae={ridges[a]['test_mae']:.4f}", flush=True)

    best_alpha = min(ridges, key=lambda a: ridges[a]["val_mae"])
    print("best alpha by val MAE:", best_alpha, flush=True)

    rng = np.random.default_rng(0)
    ysh = yc[rng.permutation(len(yc))]
    w_sh = ridge_fit(Ztr, ysh, best_alpha)
    shuffled = dict(w_raw=w_sh / sd,
                    val_mae=float(np.mean(np.abs(Zva @ w_sh + ybar - yva))),
                    test_mae=float(np.mean(np.abs(Zte @ w_sh + ybar - yte))))

    idx = rng.permutation(len(yc))[: len(yc) // 2]
    w_half = ridge_fit(Ztr[idx], yc[idx], best_alpha)
    half = dict(w_raw=w_half / sd,
                val_mae=float(np.mean(np.abs(Zva @ w_half + ybar - yva))),
                test_mae=float(np.mean(np.abs(Zte @ w_half + ybar - yte))))

    mlps = {}
    for width in (64, 256):
        print(f"training MLP width={width} ...", flush=True)
        m = NumpyMLP(Ztr.shape[1], width, seed=0).fit(Ztr, yc, Zva, yva - ybar)
        pred = m.predict(Zte) + ybar
        mlps[width] = dict(
            model=m,
            grads_raw=m.input_grad(Zte) / sd,
            val_mae=m.val_mae,
            test_mae=float(np.mean(np.abs(pred - yte))),
        )
        print(f"mlp width={width:<5} val_mae={mlps[width]['val_mae']:.4f} test_mae={mlps[width]['test_mae']:.4f}", flush=True)

    # ---- held-out pairs ----
    smi_index = {}
    for i, s in enumerate(smi_te):
        smi_index.setdefault(str(s), i)
    ia, ib, d_pka = [], [], []
    with open(os.path.join(DATA, "policy_amplitude_readout_pairs.csv")) as fh:
        head = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(",")
            a, b = smi_index.get(parts[0]), smi_index.get(parts[1])
            if a is None or b is None or a == b:
                continue
            ia.append(a); ib.append(b); d_pka.append(float(parts[2]))
    ia = np.array(ia, dtype=int); ib = np.array(ib, dtype=int); d_pka = np.array(d_pka, dtype=float)
    print("held-out pairs retained:", len(ia), flush=True)

    D = Xte[ib] - Xte[ia]
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    def align_of(gmat):
        return np.sum(gmat * Dn, axis=1) / (np.linalg.norm(gmat, axis=1) + 1e-12)

    rows = []
    for a in alphas:
        g = np.repeat(ridges[a]["w_raw"][None, :], len(ia), axis=0)
        al = align_of(g)
        dh = D @ ridges[a]["w_raw"]
        rho, p = spearman_p(dh, d_pka)
        rows.append(dict(name=f"ridge_alpha_{a}", grad_method="linear",
                         mean_alignment=float(al.mean()), median_alignment=float(np.median(al)),
                         rho_amplitude=rho, rho_p=p,
                         val_mae=ridges[a]["val_mae"], test_mae=ridges[a]["test_mae"]))
    for nm, obj in (("shuffled", shuffled), ("half", half)):
        g = np.repeat(obj["w_raw"][None, :], len(ia), axis=0)
        al = align_of(g)
        rho, p = spearman_p(D @ obj["w_raw"], d_pka)
        rows.append(dict(name=f"ridge_{nm}_alpha_{best_alpha}", grad_method="linear",
                         mean_alignment=float(al.mean()), median_alignment=float(np.median(al)),
                         rho_amplitude=rho, rho_p=p,
                         val_mae=obj["val_mae"], test_mae=obj["test_mae"]))
    for width, info in mlps.items():
        al = align_of(info["grads_raw"][ia])
        dh = info["model"].predict(Zte[ib]) - info["model"].predict(Zte[ia])
        rho, p = spearman_p(dh, d_pka)
        rows.append(dict(name=f"mlp_h{width}", grad_method="exact_backprop",
                         mean_alignment=float(al.mean()), median_alignment=float(np.median(al)),
                         rho_amplitude=rho, rho_p=p,
                         val_mae=info["val_mae"], test_mae=info["test_mae"]))

    al_all = np.array([r["mean_alignment"] for r in rows])
    rh_all = np.array([r["rho_amplitude"] for r in rows])
    pr, pp = pearson(al_all, rh_all)
    sr, sp = spearman_p(al_all, rh_all)
    print(f"alignment vs rho: pearson={pr:.3f} (p={pp:.3g}) spearman={sr:.3f} (p={sp:.3g})", flush=True)

    # ---- displacement cone ----
    Cos = Dn @ Dn.T
    off = Cos[np.triu_indices_from(Cos, k=1)]
    m = Dn.mean(0)
    m /= (np.linalg.norm(m) + 1e-12)
    sv = np.linalg.svd(Dn, compute_uv=False)
    lam = sv ** 2 / np.sum(sv ** 2)
    pr_dim = float(1.0 / np.sum(lam ** 2))
    dim90 = int(np.searchsorted(np.cumsum(lam), 0.90) + 1)
    wb = ridges[best_alpha]["w_raw"]
    ang_best = float(np.degrees(np.arccos(np.clip(wb @ m / (np.linalg.norm(wb) + 1e-12), -1, 1))))
    cone = dict(n_pairs=int(len(ia)),
                pairwise_cosine_mean=float(off.mean()),
                pairwise_cosine_median=float(np.median(off)),
                pairwise_cosine_q10=float(np.quantile(off, 0.10)),
                pairwise_cosine_q90=float(np.quantile(off, 0.90)),
                participation_ratio=pr_dim,
                dims_for_90pct_variance=dim90,
                angle_best_ridge_vs_mean_axis_deg=ang_best)
    print(json.dumps(cone, indent=1), flush=True)

    out = dict(
        protocol=dict(
            hidden_states="PkaHead input (d=4096), frozen backbone",
            splits="h_train 9202 / h_val 1151 / h_test 1151 (runs/fig on the cluster)",
            standardisation="train mean/std; ridge intercept via centring",
            alpha_selection="minimum validation MAE",
            best_alpha=best_alpha,
            mlp="one hidden layer ReLU, Adam, early stopping on validation MAE",
            pair_rule="pairs_train.jsonl rows with both endpoints in the h_test set",
        ),
        n_states=dict(train=int(len(ytr)), val=int(len(yva)), test=int(len(yte))),
        n_pairs=int(len(ia)),
        readouts=rows,
        corr_alignment_vs_rho=dict(pearson=float(pr), pearson_p=float(pp),
                                   spearman=float(sr), spearman_p=float(sp)),
        cone=cone,
    )
    with open(os.path.join(DATA, "readout_family.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    make_figure(rows, off, cone, sr, sp)
    print("wrote data/readout_family.json and figs/fig_readout_family.pdf", flush=True)


def make_figure(rows, off, cone, sr, sp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
    GREEN, GREY = "#2E7D5B", "#8A8A8A"

    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5))

    ridge = sorted([r for r in rows if r["name"].startswith("ridge_alpha_")],
                   key=lambda r: float(r["name"].split("_")[-1]))
    ax[0].plot([r["mean_alignment"] for r in ridge], [r["rho_amplitude"] for r in ridge],
               "-o", color=GREEN, ms=3.5, lw=1.0, label=r"ridge ($\alpha$ sweep)")
    for r in ridge:
        ax[0].annotate(r["name"].replace("ridge_alpha_", r"$\alpha$="),
                       (r["mean_alignment"], r["rho_amplitude"]),
                       textcoords="offset points", xytext=(3, -7), fontsize=5.2, color=GREY)
    mlp = [r for r in rows if r["name"].startswith("mlp")]
    if mlp:
        ax[0].scatter([r["mean_alignment"] for r in mlp], [r["rho_amplitude"] for r in mlp],
                      marker="^", s=26, color=GREEN, edgecolor="k", lw=0.3, label="MLP head", zorder=3)
    other = [r for r in rows if r["name"].startswith(("ridge_shuffled", "ridge_half"))]
    if other:
        ax[0].scatter([r["mean_alignment"] for r in other], [r["rho_amplitude"] for r in other],
                      marker="s", s=22, facecolor="none", edgecolor=GREY, lw=0.8, label="control", zorder=3)
    ax[0].axhline(0, color="k", lw=0.4, ls=":")
    ax[0].set_xlabel("gradient–displacement alignment (mean cosine)")
    ax[0].set_ylabel(r"amplitude correlation ($\rho$)")
    ax[0].legend(frameon=False, fontsize=5.8, loc="lower right")
    ax[0].set_title(f"(a) readout family (Spearman $r$={sr:.2f}, p={sp:.2g})", fontsize=7)

    ax[1].hist(off, bins=40, color=GREEN, alpha=0.75, edgecolor="white", lw=0.3)
    ax[1].axvline(cone["pairwise_cosine_median"], color="k", lw=0.8, ls="--",
                  label=f"median={cone['pairwise_cosine_median']:.2f}")
    ax[1].set_xlabel("cosine between true displacement directions")
    ax[1].set_ylabel("pair count")
    ax[1].legend(frameon=False, fontsize=5.8)
    ax[1].set_title(f"(b) displacement cone: eff. dim {cone['participation_ratio']:.1f}, "
                    f"90% var in {cone['dims_for_90pct_variance']}", fontsize=7)

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(width=0.5, length=2.5, labelsize=6.5)
    fig.tight_layout(pad=0.4)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "fig_readout_family.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig_readout_family.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
