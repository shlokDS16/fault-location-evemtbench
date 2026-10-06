"""Figures of the revamped manuscript (draft 3). Inputs: [V] result files and the benchmark's grid graphs only.
Style: IEEE column 3.5 in / page 7.16 in, 8 pt Times, vector PDF with embedded TrueType fonts. One fixed colour +
marker + line style per method and one per test set across all figures; the tables hold the exact numbers.
Usage: python src/paper/figs_v3.py [grids dreq types time cond]  -> paper/figs/f_<name>.pdf/.png"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "paper" / "figs"
COL, PAGE = 3.5, 7.16
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
M = {  # method -> (label, colour, marker, linestyle)
    "two": ("Two-ended TD", "#2a78d6", "o", "-"),
    "ha2": ("PFB (one-ended)", "#eb6834", "s", "-"),
    "E": ("Eriksson", "#1baf7a", "^", "-"),
    "T": ("Takagi", "#eda100", "v", "--"),
    "R": ("Reactance", "#e87ba4", "D", ":"),
    "gru": ("GRU+Z", "#4a3aa8", "X", "-."),
}
S = {  # test set -> (label, colour, marker, filled)
    "DL": ("DoubleLine", "#2a78d6", "o", True),
    "TG": ("TestGrid", "#eb6834", "s", True),
    "ADAPT": ("DL-adapt", "#1baf7a", "^", True),
    "MV": ("CIGRE MV", "#7b4fb3", "D", True),
    "MV_zid": ("CIGRE MV, identified $Z_1$", "#7b4fb3", "D", False),
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def line(ax, x, y, key, **kw):
    lab, c, mk, ls = M[key]
    return ax.plot(x, y, color=c, marker=mk, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.4, **kw)[0]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"f_{name}.pdf")
    fig.savefig(OUT / f"f_{name}.png", dpi=300)
    plt.close(fig)
    print(f"f_{name} -> {OUT / f'f_{name}.pdf'}")


# ---------------------------------------------------------------------------------------------------------------
def fig_grids():
    """Single-line diagrams of the three test systems (main lines only), from the benchmark's graph files."""
    topo = pd.read_csv(R / "paper_grid_topology.csv")
    length = {}
    for t in ("DL", "TG", "MV"):
        g = pd.read_parquet(R / f"c1_grid_{t}.parquet", columns=["line", "length_km", "td2_est"])
        for ln, gg in g.groupby("line"):
            length[(t, ln)] = (gg.length_km.iloc[0], gg.td2_est.notna().any())
    pos = {
        "DL": {1: (0, 0), 2: (1.3, 0), 3: (2.6, 0)},
        "TG": {1: (0, 1), 2: (1.1, 1), 3: (2.2, 1), 4: (2.2, 0), 5: (1.1, 0), 6: (1.7, 0.5)},
        "MV": {1: (0, 5), 2: (0, 4), 3: (0, 3), 4: (1, 3), 5: (2, 3), 6: (3, 3), 7: (3, 1.8), 8: (0, 1.8),
               9: (0.45, 2.05), 10: (0.75, 2.35), 11: (1.0, 2.65), 12: (-1.3, 5), 13: (-1.3, 3.4), 14: (-1.3, 1.8)},
    }
    # bus-label offsets (dx, dy) in data units
    lab_off = {
        "DL": {1: (0, 0.22), 2: (0, 0.22), 3: (0, 0.22)},
        "TG": {1: (0, 0.16), 2: (-0.13, 0.15), 3: (0.14, 0.15), 4: (0.14, -0.1), 5: (0, -0.17), 6: (0.14, 0.05)},
        "MV": {1: (-0.22, 0), 2: (-0.22, 0), 3: (-0.22, 0), 4: (0, 0.24), 5: (0, 0.24), 6: (0.2, 0.2), 7: (0.22, 0),
               8: (-0.22, -0.14), 9: (0.12, -0.2), 10: (0.17, -0.17), 11: (0.2, -0.12), 12: (-0.25, 0),
               13: (-0.25, 0), 14: (-0.27, 0.12)},
    }
    lim = {"DL": ((-0.6, 3.2), (-0.7, 0.7)), "TG": ((-0.55, 2.75), (-0.35, 1.5)), "MV": ((-1.9, 3.5), (0.95, 6.45))}
    ext = {"DL": {1: (-0.4, 0), 3: (0.4, 0)}, "TG": {1: (-0.4, 0), 3: (0.4, 0)}, "MV": {}}
    ibr = {"DL": {}, "TG": {2: (0, 0.33), 3: (0, 0.33)}, "MV": {3: (0.33, 0.33), 14: (-0.42, -0.3)}}
    overhead_mv = ("MainLn12-13", "MainLn13-14", "MainLn8-14")
    fig, axs = plt.subplots(1, 3, figsize=(PAGE, 1.45), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.05]))
    ttl = {"DL": "(a) DoubleLine, 110 kV", "TG": "(b) TestGrid, 110 kV", "MV": "(c) CIGRE MV, 20 kV"}
    for ax, t in zip(axs, ("DL", "TG", "MV")):
        P = pos[t]
        e = topo[topo.grid == t]
        pairs = {}
        for _, r in e.iterrows():
            a, b = int(r.u.replace("MainBus", "")), int(r.v.replace("MainBus", ""))
            pairs.setdefault(tuple(sorted((a, b))), []).append(r["key"])
        for (a, b), keys in pairs.items():
            (x0, y0), (x1, y1) = P[a], P[b]
            dx, dy = x1 - x0, y1 - y0
            nrm = np.hypot(dx, dy)
            off = [0.0] if len(keys) == 1 else [0.07, -0.07]
            for k, o in zip(sorted(keys), off):
                ox, oy = -dy / nrm * o, dx / nrm * o
                L, both = length[(t, k)]
                cable = t == "MV" and k not in overhead_mv
                ux, uy = dx / nrm * 0.14, dy / nrm * 0.14
                xs = [x0, x0 + ux + ox, x1 - ux + ox, x1] if o else [x0, x1]
                ys = [y0, y0 + uy + oy, y1 - uy + oy, y1] if o else [y0, y1]
                ax.plot(xs, ys, color="#2a78d6" if cable else INK,
                        lw=1.0, ls="-" if both else (0, (2, 1.5)), solid_capstyle="butt", zorder=1)
                if t != "MV":
                    f = 3.0 if len(keys) > 1 else 0.0
                    mx, my = (x0 + x1) / 2 + ox * f, (y0 + y1) / 2 + oy * f
                    ax.text(mx, my, f"{L:g}", fontsize=6, ha="center", va="center", color=INK2,
                            bbox=dict(fc="white", ec="none", pad=0.1), zorder=2)
        for n, (sx, sy) in ext[t].items():
            x, y = P[n]
            ax.plot([x, x + sx], [y, y + sy], color=INK, lw=0.8, zorder=1)
            ax.plot(x + sx, y + sy, "s", ms=6, color="#d9d8d4", mec=INK, mew=0.8, zorder=3)
        for n, (sx, sy) in ibr[t].items():
            x, y = P[n]
            ax.plot([x, x + sx], [y, y + sy], color=INK, lw=0.8, zorder=1)
            ax.plot(x + sx, y + sy, "^", ms=5.5, color="#f3c9a8", mec=INK, mew=0.8, zorder=3)
        if t == "MV":
            ax.plot([-0.65], [5.85], "o", ms=7, color="white", mec=INK, mew=0.9, zorder=3)
            ax.text(-0.38, 5.85, "HV/MV", fontsize=6, ha="left", va="center")
            for n in (1, 12):
                ax.plot([-0.65, P[n][0]], [5.85, P[n][1]], color="#9a9893", lw=1.4, zorder=1)
        for n, (x, y) in P.items():
            ax.plot(x, y, "o", ms=4.2, color="white", mec=INK, mew=0.9, zorder=3)
            ldx, ldy = lab_off[t][n]
            ax.text(x + ldx, y + ldy, str(n), fontsize=6.5, ha="center", va="center", color=INK, zorder=4)
        ax.set_title(ttl[t], loc="left", pad=1)
        ax.set_xlim(*lim[t][0])
        ax.set_ylim(*lim[t][1])
        ax.axis("off")
    h = [mpl.lines.Line2D([], [], color=INK, lw=1.0), mpl.lines.Line2D([], [], color="#2a78d6", lw=1.0),
         mpl.lines.Line2D([], [], color=INK, lw=1.0, ls=(0, (2, 1.5))),
         mpl.lines.Line2D([], [], ls="none", marker="s", ms=6, color="#d9d8d4", mec=INK),
         mpl.lines.Line2D([], [], ls="none", marker="^", ms=5.5, color="#f3c9a8", mec=INK)]
    fig.legend(h, ["overhead line (length in km)", "cable", "measured at one end only", "external grid",
                   "inverter-based source"], loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5,
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=0.2)
    save(fig, "grids")


# ---------------------------------------------------------------------------------------------------------------
def fig_dreq():
    """Error model: required polarising-angle accuracy vs rho (eq. dreq), with the per-grid operating points."""
    w = pd.read_parquet(R / "c1_cond_windows.parquet")
    rho = np.logspace(-2, 2.5, 400)
    eps = 0.05
    fig, axs = plt.subplots(2, 1, figsize=(COL, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1.7, 1]))
    ax = axs[0]
    for kap, ls, lab in ((0.5, ":", r"$\kappa=0.5$"), (1.0, "-", r"$\kappa=1$"), (2.0, "--", r"$\kappa=2$")):
        d = np.degrees(np.arcsin(np.minimum(1, eps / (rho * kap))))
        ax.plot(rho, d, color=INK2, ls=ls, lw=1.0, label=lab)
    for g in ("DL", "TG", "ADAPT", "MV"):
        lab, c, mk, _ = S[g]
        x = w[w.grid == g]
        ang = x.angerr_T_deg.abs()
        r50, a50 = x.rho.median(), ang.median()
        r_lo, r_hi = x.rho.quantile([0.25, 0.75])
        a_lo, a_hi = ang.quantile([0.25, 0.75])
        ax.errorbar(r50, a50, xerr=[[r50 - r_lo], [r_hi - r50]], yerr=[[a50 - a_lo], [a_hi - a50]], fmt=mk, color=c,
                    ms=5 if g != "DL" else 6.5, mec=c if g == "DL" else "white", mfc="white" if g == "DL" else c,
                    mew=1.0 if g == "DL" else 0.5, elinewidth=0.8, capsize=0, zorder=3 if g != "DL" else 4)
        dx, dy, ha = {"DL": (1.15, 0.55, "left"), "TG": (0.85, 1.9, "right"), "ADAPT": (0.85, 0.42, "right"),
                      "MV": (1.0, 2.2, "center")}[g]
        ax.text(r50 * dx, a50 * dy, {"ADAPT": "DL-adapt", "MV": "CIGRE MV"}.get(g, g), color=c, fontsize=6.5,
                ha=ha, va="center", zorder=4)
    ax.set(xscale="log", yscale="log", ylim=(0.02, 90), ylabel="Angle (deg)")
    ax.set_title("(a) required accuracy (lines) vs Takagi's angle error (markers)", loc="left", pad=2)
    ax.legend(loc="lower left", handletextpad=0.3, fontsize=6.5, borderaxespad=0.1, handlelength=1.8)
    ax = axs[1]
    for g in ("DL", "TG", "ADAPT", "MV"):
        lab, c, mk, _ = S[g]
        x = np.sort(w[w.grid == g].rho.values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.2, label=lab)
    ax.set(xscale="log", xlim=(0.01, 300), ylim=(0, 1.02), ylabel="ECDF", xlabel=r"$\rho = R_f/|Z_1|$")
    ax.set_title(r"(b) distribution of $\rho$ (single-phase-to-ground short circuits)", loc="left", pad=2)
    ax.legend(loc="upper left", fontsize=6.5, borderaxespad=0.1, handlelength=1.2, labelspacing=0.2)
    fig.tight_layout(h_pad=0.5)
    save(fig, "dreq")


# ---------------------------------------------------------------------------------------------------------------
TYPES = [("flt_1phg_shc", "1ph-g"), ("flt_1phg_shc_w_arc", "1ph-g, arc"), ("flt_2ph_shc", "2ph"),
         ("flt_2phg_shc", "2ph-g"), ("flt_3ph_shc", "3ph"), ("flt_1phg_hif", "HIF"), ("flt_1phg_hif_w_arc", "HIF, arc"),
         ("flt_1phg_incipient", "incipient"), ("flt_1phg_incipient_w_arc", "incipient, arc")]


def fig_types():
    """Two-ended TD locator: MAE by fault type and test set (horizontal dot plot)."""
    sets = {}
    for tag in ("DL", "TG", "MV", "MV_zid"):
        d = pd.read_parquet(R / f"c1_grid_{tag}.parquet", columns=["etype", "td2_est", "y"]).dropna(subset=["td2_est"])
        sets[tag] = ((d.td2_est.clip(0, 1) - d.y).abs() * 100).groupby(d.etype).mean()
    a = pd.read_csv(R / "c1_adapt_td.csv")
    a = a[a.split == "test"]
    sets["ADAPT"] = a.err.groupby(a.etype).mean()
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ys = np.arange(len(TYPES))[::-1]
    order = ("DL", "TG", "ADAPT", "MV", "MV_zid")
    off = np.linspace(0.28, -0.28, len(order))
    for tag, o in zip(order, off):
        lab, c, mk, filled = S[tag]
        v = [sets[tag].get(t, np.nan) for t, _ in TYPES]
        ax.plot(v, ys + o, ls="none", marker=mk, color=c, mfc=c if filled else "white", mec=c if not filled else "white",
                mew=0.9 if not filled else 0.4, ms=4.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 80)
    ax.set_yticks(ys, [l for _, l in TYPES])
    ax.axhline(3.5, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("MAE (% of line length)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8,
              fontsize=6.5)
    save(fig, "types")


# ---------------------------------------------------------------------------------------------------------------
def fig_time():
    """Cross-grid one-ended MAE vs time from inception to window end (both directions), two-ended as reference."""
    d = pd.read_csv(R / "c1_fig4_data.csv")
    d = d[d.t != "all"].copy()
    d["t"] = d.t.astype(int)
    cols = [("two", "two-ended"), ("ha2", "H-A2"), ("E", "Eriksson"), ("T", "Takagi"), ("R", "reactance"),
            ("gru", "GRU+Z")]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.25), sharex=True)
    for ax, dirn, ttl in zip(axs, ("TG_to_DL", "DL_to_TG"), ("(a) trained on TestGrid, tested on DoubleLine",
                                                              "(b) trained on DoubleLine, tested on TestGrid")):
        g = d[d.direction == dirn]
        ax.axvspan(2.5, 17.5, color="#f0efec", zorder=0)
        for k, c in cols:
            line(ax, g.t, g[c], k)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 70)
        ax.set_ylabel("MAE (%)")
        ax.set_title(ttl, loc="left", pad=2)
    axs[0].text(10, 0.065, "no full post-\nfault cycle", ha="center", fontsize=6.5, color=INK2)
    axs[1].set_xlabel("Time from fault inception to window end (ms)")
    axs[1].set_xticks(range(5, 81, 10))
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=3, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(h_pad=0.6)
    save(fig, "time")


# ---------------------------------------------------------------------------------------------------------------
def fig_cond():
    """(a) one-ended error model check (reactance, 1ph-g); (b) MAE vs rho on CIGRE MV short circuits."""
    w = pd.read_parquet(R / "c1_cond_windows.parquet")
    b = pd.read_csv(R / "c1_cond_rho_bins.csv")
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.95), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axs[0]
    lim = (1e-2, 300)
    for grid, nm in (("DL", "110 kV"), ("TG", None), ("MV", "CIGRE MV")):
        c = "#8a8884" if grid != "MV" else S["MV"][1]
        mk = "o" if grid != "MV" else "D"
        g = w[w.grid == grid].sample(min(1500, (w.grid == grid).sum()), random_state=0)
        x = np.clip(np.abs(g.e_pred_R) * 100, *lim)
        y = np.clip(np.abs(g.e_obs_R) * 100, *lim)
        ax.plot(x, y, ls="none", marker=mk, ms=1.6, alpha=0.45, color=c, label=nm, markeredgewidth=0, rasterized=True)
    ax.plot(lim, lim, color=INK2, lw=0.6)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted $|e|$ (%)", ylabel="Observed $|e|$ (%)")
    ax.set_xticks([1e-2, 1, 100])
    ax.set_yticks([1e-2, 1, 100])
    ax.set_title("(a) model check", loc="left", pad=2)
    ax = axs[1]
    order = ["(0.0, 0.3]", "(0.3, 1.0]", "(1.0, 3.0]", "(3.0, 10.0]", "(10.0, 30.0]", "(30.0, 1000000000.0]"]
    labs = ["$\\leq$0.3", "(0.3, 1]", "(1, 3]", "(3, 10]", "(10, 30]", ">30"]
    mv = b[b.grid == "MV"]
    for k, m in (("two", "two-ended"), ("E", "E"), ("R", "R"), ("ha2", "H-A2")):
        s = mv[mv.method == m].set_index("rho_bin").reindex(order)
        line(ax, range(len(order)), s.mae.values, k)
    ax.set_xticks(range(len(order)), labs, rotation=45, ha="right", fontsize=6.5)
    ax.set(yscale="log", ylim=(0.5, 80), xlabel="$\\rho = R_f/|Z_1|$", ylabel="MAE (%)")
    ax.set_title("(b) CIGRE MV", loc="left", pad=2)
    fig.tight_layout(w_pad=0.8)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.55, 1.14), ncol=4, handletextpad=0.3, columnspacing=0.7)
    save(fig, "cond")


if __name__ == "__main__":
    for k in sys.argv[1:] or ["grids", "dreq", "types", "time", "cond"]:
        globals()[f"fig_{k}"]()
