"""Per-type two-ended numbers and per-time-bin numbers for the results text (called by make_macros.py)."""
import numpy as np
import pandas as pd


def extra_macros(R, put):
    for k, f in (("DL", "c1_grid_DL"), ("TG", "c1_grid_TG"), ("MV", "c1_grid_MV"), ("MVid", "c1_grid_MV_zid")):
        d = pd.read_parquet(R / f"{f}.parquet", columns=["etype", "td2_est", "y"]).dropna(subset=["td2_est"])
        e = ((d.td2_est.clip(0, 1) - d.y).abs() * 100).groupby(d.etype).mean()
        sc = e[[t for t in e.index if not any(s in t for s in ("incipient", "hif"))]]
        put(f"teTypeMin{k}", float(sc.min())); put(f"teTypeMax{k}", float(sc.max()))
        inc = e[[t for t in e.index if "incipient" in t]]
        put(f"teIncMin{k}", float(inc.min())); put(f"teIncMax{k}", float(inc.max()))
        hif = e[[t for t in e.index if "hif" in t]]
        if len(hif):
            put(f"teHifMin{k}", float(hif.min())); put(f"teHifMax{k}", float(hif.max()))
    f4 = pd.read_csv(R / "c1_fig4_data.csv")
    f4 = f4[f4.t != "all"].assign(t=lambda x: x.t.astype(int))
    for di, k in (("TG_to_DL", "TGDL"), ("DL_to_TG", "DLTG")):
        g = f4[f4.direction == di].set_index("t")
        for col, ck in (("H-A2", "Ha"), ("Eriksson", "E"), ("GRU+Z", "Gru"), ("two-ended", "Two")):
            put(f"fb{ck}Five{k}", g.loc[5, col]); put(f"fb{ck}Fifteen{k}", g.loc[15, col])
            put(f"fb{ck}Twenty{k}", g.loc[20, col])
            late = g.loc[g.index >= 55, col]
            put(f"fb{ck}LateMin{k}", float(late.min())); put(f"fb{ck}LateMax{k}", float(late.max()))
        sub = g.loc[g.index <= 15, "Eriksson"]
        put(f"fbESubMin{k}", float(sub.min())); put(f"fbESubMax{k}", float(sub.max()))
    sub = f4[f4.t <= 15]
    learned = ["H-A2", "GRU+Z", "GRU-raw", "MLP+Z", "MLP-raw"]
    put("fbLearnLeMin", float(sub[learned].min().min())); put("fbLearnLeMax", float(sub[learned].max().max()))
    put("fbESubAllMin", float(sub["Eriksson"].min())); put("fbESubAllMax", float(sub["Eriksson"].max()))
    rat = sub["Eriksson"] / sub[learned].min(axis=1)  # Eriksson vs best learned locator, per direction and time bin
    put("fbRatioMin", float(rat.min()), 1); put("fbRatioMax", float(rat.max()), 1)
    ch = pd.read_csv(R / "c1_chain_eval.csv").set_index(["direction", "scenario"])
    aa = [abs(ch.loc[(di, "AA"), c] - ch.loc[(di, "clean"), c]) for di in ("TG_to_DL", "DL_to_TG")
          for c in ("two_ended", "HA2_clean_trained")]
    put("chAAmax", float(max(aa)))
    hc = pd.read_csv(R / "c1_cond_rho_bins.csv")
    mv = hc[hc.grid == "MV"].set_index(["method", "rho_bin"]).mae
    lo, hi = "(0.0, 0.3]", "(30.0, 1000000000.0]"
    for m, k in (("E", "E"), ("H-A2", "Ha"), ("R", "R"), ("two-ended", "Two")):
        put(f"rb{k}Lo", float(mv.loc[(m, lo)])); put(f"rb{k}Hi", float(mv.loc[(m, hi)]))
    two = hc[(hc.grid == "MV") & (hc.method == "two-ended")].mae
    put("rbTwoMax", float(two.max()))
