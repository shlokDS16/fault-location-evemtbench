"""Paper figures 3-7 (data figures; Figs 1-2 are TikZ in the LaTeX source). Inputs: [V] result files only.
Style: IEEE column width 3.5 in, 8 pt Times, vector PDF with embedded TrueType fonts. One fixed colour + marker +
line style per METHOD across all figures (validated categorical order; low-contrast hues always carry a marker and a
direct label or legend entry, and the tables hold the exact numbers).
Usage: python src/paper/figs.py [3 4 5 6 7]  -> paper/figs/fig<k>.pdf"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "paper" / "figs"
COL = 3.5
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
M = {  # method -> (label, colour, marker, linestyle)
    "two": ("Two-ended TD (ours)", "#2a78d6", "o", "-"),
    "ha2": ("H-A2 one-ended (ours)", "#eb6834", "s", "-"),
    "E": ("Eriksson", "#1baf7a", "^", "-"),
    "T": ("Takagi", "#eda100", "v", "--"),
    "R": ("Reactance", "#e87ba4", "D", ":"),
    "gru": ("GRU+Z (learned)", "#008300", "X", "-."),
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def line(ax, x, y, key, **kw):
    lab, c, mk, ls = M[key]
    return ax.plot(x, y, color=c, marker=mk, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.4, **kw)[0]


def save(fig, k):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"fig{k}.pdf")
    fig.savefig(OUT / f"fig{k}.png", dpi=200)
    plt.close(fig)
    print(f"fig{k} -> {OUT / f'fig{k}.pdf'}")


TYPES = [("flt_1phg_shc", "1ph-g"), ("flt_1phg_shc_w_arc", "1ph-g arc"), ("flt_2ph_shc", "2ph"), ("flt_2phg_shc", "2ph-g"),
         ("flt_3ph_shc", "3ph"), ("flt_1phg_hif", "HIF"), ("flt_1phg_hif_w_arc", "HIF arc"),
         ("flt_1phg_incipient", "incip."), ("flt_1phg_incipient_w_arc", "incip. arc")]


def fig3():
    """Two-ended TD locator MAE by fault type on four grids (+ CIGRE MV with identified Z)."""
    sets = []
    for tag, nm, mk, c in (("DL", "DoubleLine", "o", "#1c5cab"), ("TG", "TestGrid 110 kV", "s", "#2a78d6"),
                           ("MV", "CIGRE MV, nameplate Z", "D", "#5598e7"), ("MV_zid", "CIGRE MV, identified Z", "d", "#86b6ef")):
        d = pd.read_parquet(R / f"c1_grid_{tag}.parquet", columns=["etype", "td2_est", "y"]).dropna(subset=["td2_est"])
        sets.append((nm, mk, c, ((d.td2_est.clip(0, 1) - d.y).abs() * 100).groupby(d.etype).mean()))
    a = pd.read_csv(R / "c1_adapt_td.csv")
    a = a[a.split == "test"]
    sets.insert(2, ("adapt test (cont. $R_f$)", "^", "#184f95", a.err.groupby(a.etype).mean()))
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    xs = np.arange(len(TYPES))
    off = np.linspace(-0.28, 0.28, len(sets))
    for (nm, mk, c, s), o in zip(sets, off):
        v = [s.get(t, np.nan) for t, _ in TYPES]
        ax.plot(xs + o, v, ls="none", marker=mk, color=c, label=nm, markeredgecolor="white", markeredgewidth=0.4,
                markersize=4.5)
    ax.set_yscale("log")
    ax.set_xticks(xs, [l for _, l in TYPES], rotation=30, ha="right")
    ax.set_ylabel("MAE (% of line length)")
    ax.axvline(4.5, color=INK2, lw=0.6, ls="--")
    ax.text(4.4, 0.06, "short circuits", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(4.6, 0.06, "HIF / incipient", ha="left", va="bottom", fontsize=7, color=INK2)
    ax.set_ylim(0.05, 80)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2, handletextpad=0.3, columnspacing=0.8)
    save(fig, 3)


def fig4():
    """Zero-shot MAE vs time since inception (both directions)."""
    d = pd.read_csv(R / "c1_fig4_data.csv")
    d = d[d.t != "all"].copy()
    d["t"] = d.t.astype(int)
    cols = [("two", "two-ended"), ("ha2", "H-A2"), ("E", "Eriksson"), ("T", "Takagi"), ("R", "reactance"), ("gru", "GRU+Z")]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.9), sharex=True)
    for ax, dirn, ttl in zip(axs, ("TG_to_DL", "DL_to_TG"), ("(a) TestGrid $\\rightarrow$ DoubleLine",
                                                              "(b) DoubleLine $\\rightarrow$ TestGrid")):
        g = d[d.direction == dirn]
        ax.axvspan(2.5, 17.5, color="#f0efec", zorder=0)
        for k, c in cols:
            line(ax, g.t, g[c], k)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 70)
        ax.set_ylabel("MAE (%)")
        ax.set_title(ttl, loc="left", pad=2)
    axs[0].text(10, 0.07, "< 1 cycle", ha="center", fontsize=7, color=INK2)
    axs[1].set_xlabel("Time since fault inception at window end (ms)")
    axs[1].set_xticks(range(5, 81, 10))
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=3, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(h_pad=0.6)
    save(fig, 4)


def fig5():
    """(a) one-ended error identity e_obs vs e_pred (reactance, 1ph-g); (b) MAE vs rho = R_f/|Z_L| (short circuits)."""
    w = pd.read_parquet(R / "c1_cond_windows.parquet")
    b = pd.read_csv(R / "c1_cond_rho_bins.csv")
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    for grid, mk, c, nm in (("DL", "o", "#1c5cab", "110 kV"), ("TG", "o", "#1c5cab", None), ("MV", "D", "#86b6ef", "CIGRE MV")):
        g = w[w.grid == grid].sample(min(1500, (w.grid == grid).sum()), random_state=0)
        ax.plot(np.abs(g.e_pred_R) * 100, np.abs(g.e_obs_R) * 100, ls="none", marker=mk, ms=1.6, alpha=0.5, color=c,
                label=nm, markeredgewidth=0, rasterized=True)
    lim = (1e-2, 1e4)
    ax.plot(lim, lim, color=INK2, lw=0.6)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted $|e|$ (%)", ylabel="Observed $|e|$ (%)")
    ax.set_title("(a) reactance, 1ph-g", loc="left", pad=2)
    ax.legend(loc="upper left", handletextpad=0.1, markerscale=3)
    ax = axs[1]
    order = ["(0.0, 0.3]", "(0.3, 1.0]", "(1.0, 3.0]", "(3.0, 10.0]", "(10.0, 30.0]", "(30.0, 1000000000.0]"]
    labs = ["$\\leq$0.3", "0.3–1", "1–3", "3–10", "10–30", ">30"]
    mv = b[b.grid == "MV"]
    for k, m in (("two", "two-ended"), ("E", "E"), ("R", "R"), ("ha2", "H-A2")):
        s = mv[mv.method == m].set_index("rho_bin").reindex(order)
        line(ax, range(len(order)), s.mae.values, k)
    ax.set_xticks(range(len(order)), labs, rotation=30, ha="right")
    ax.set(yscale="log", ylim=(0.5, 80), xlabel="$\\rho = R_f/|Z_1|$", ylabel="MAE (%)")
    ax.set_title("(b) CIGRE MV, short circuits", loc="left", pad=2)
    fig.tight_layout(w_pad=0.8)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, handletextpad=0.3, columnspacing=0.7)
    save(fig, 5)


def fig6():
    """Measurement-chain example: CT-severe secondary current and CVT-15 voltage on one DoubleLine fault."""
    sys.path.insert(0, str(ROOT / "src" / "c1"))
    import grid_pass
    from meas_chain import ct, cvt, CT_SEVERE
    from stage_a import terminal_cols
    gdir = ROOT / "data" / "raw" / "evemt" / "DoubleLine"
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    flt = s[(s["events/event_type"] == "flt_1phg_shc") & s["events/event_target"].str.match(grid_pass.LINE_RE)
            & (s["events/event_flt_shc_resistance"] <= 1) & (s["events/event_flt_target_line_location"] <= 20)]
    best = None
    for _, cand in flt.head(40).iterrows():  # pick a clearly saturating case (largest CT error in 0-60 ms)
        ln = cand["events/event_target"]
        mm = grid_pass.LINE_RE.match(ln)
        ff = gdir / "data" / f"result{int(cand['general/sim_idx'])}.csv"
        cc = terminal_cols(pd.read_csv(ff, nrows=0).columns.tolist(), mm.group(1), ln)
        kk = "abc".index(str(cand["events/event_phase_select_1ph"]).lower()[0])
        x = pd.read_csv(ff, skiprows=[1], usecols=[cc[kk]])[cc[kk]].to_numpy(float)
        n0 = int(np.floor((cand["events/event_start"] - 1.0) * 9600))
        err = np.abs(ct(x, **CT_SEVERE)[0] - x)[n0:n0 + 576].max()
        if best is None or err > best[0]:
            best = (err, cand)
    row = best[1]
    line_ = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line_)
    f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
    cs = terminal_cols(pd.read_csv(f, nrows=0).columns.tolist(), m.group(1), line_)
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    p = str(row["events/event_phase_select_1ph"]).lower()[0]
    k = "abc".index(p)
    i1, v1 = d[cs[k]].to_numpy(float), d[cs[3 + k]].to_numpy(float)
    nf = int(np.floor((row["events/event_start"] - 1.0) * 9600))
    t = (np.arange(len(i1)) - nf) / 9.6
    i2 = ct(i1, **CT_SEVERE)[0]
    v2 = cvt(v1, 0.015)
    sel = (t >= -20) & (t <= 80)
    fig, axs = plt.subplots(2, 1, figsize=(COL, 2.6), sharex=True)
    axs[0].plot(t[sel], i1[sel] / 1e3, color=INK2, lw=1.0, label="primary (ideal CT)")
    axs[0].plot(t[sel], i2[sel] / 1e3, color=M["two"][1], lw=1.0, ls="--", label="CT-severe (referred)")
    axs[0].set_ylabel("Current (kA)")
    axs[0].set_title("(a) faulted-phase current", loc="left", pad=2)
    axs[1].plot(t[sel], v1[sel] / 1e3, color=INK2, lw=1.0, label="primary voltage")
    axs[1].plot(t[sel], v2[sel] / 1e3, color=M["ha2"][1], lw=1.0, ls="--", label="CVT-15 output")
    axs[1].set_ylabel("Voltage (kV)")
    axs[1].set_title("(b) faulted-phase voltage", loc="left", pad=2)
    axs[1].set_xlabel("Time since fault inception (ms)")
    for ax in axs:
        ax.axvline(0, color=INK2, lw=0.5, ls=":")
    fig.tight_layout(h_pad=0.4)
    h = axs[0].get_legend_handles_labels()[0] + axs[1].get_legend_handles_labels()[0][1:]
    l = ["true primary (ideal CT / VT)", "CT severe, referred", r"CVT $\tau$=15 ms"]
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, handlelength=1.6, columnspacing=0.8)
    save(fig, 6)
    print("fig6 episode", int(row["general/sim_idx"]), line_, "phase", p, "R_f", row["events/event_flt_shc_resistance"],
          "loc", row["events/event_flt_target_line_location"])


def fig7():
    """Measurement-chain robustness (dot plot, both directions)."""
    e = pd.read_csv(R / "c1_chain_eval.csv")
    scen = ["clean", "AA", "CT-mild", "CT-severe", "CVT-5", "CVT-15", "FULL"]
    labs = ["clean", "AA 2 kHz", "CT mild", "CT severe", "CVT $\\tau$=5 ms", "CVT $\\tau$=15 ms", "full chain"]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.1), sharey=True)
    y = np.arange(len(scen))[::-1]
    for ax, dirn, ttl in zip(axs, ("TG_to_DL", "DL_to_TG"), ("(a) TG$\\rightarrow$DL", "(b) DL$\\rightarrow$TG")):
        g = e[e.direction == dirn].set_index("scenario").reindex(scen)
        for k, col, dy in (("two", "two_ended", 0.15), ("ha2", "HA2_clean_trained", -0.15)):
            lab, c, mk, _ = M[k]
            ax.plot(g[col].values, y + dy, ls="none", marker=mk, color=c, label=lab, markeredgecolor="white",
                    markeredgewidth=0.4, markersize=4.5)
        ax.plot([g.loc["FULL", "HA2_matched_FULL"]], [y[-1] - 0.15], ls="none", marker="s", mfc="white",
                mec=M["ha2"][1], mew=1.0, ms=4.5, label="H-A2 trained with chain")
        ax.set_xscale("log")
        ax.set_xlim(0.2, 30)
        ax.set_title(ttl, loc="left", pad=2)
        ax.set_xlabel("MAE (%)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, labs)
    fig.tight_layout(w_pad=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    save(fig, 7)


if __name__ == "__main__":
    which = sys.argv[1:] or ["3", "4", "5", "6", "7"]
    for k in which:
        globals()[f"fig{k}"]()
