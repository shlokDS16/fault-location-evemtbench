"""CIGRE MV Part B (after results/validate_ha2_mv_frozen.sha256): compare with the lead's
results/c1_grid_MV.parquet, c1_zs_MV.csv, c1_zs_MV_pred.csv and the source tables c1_grid_{DL,TG}.parquet."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from audit_partB import lead_name  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402

R = C.RESULTS
TOK = LOCAL_TOKENS + TD_TOKENS


def key(df, col):
    return (df[col] * 9.6).round().astype(int)


def tok_diff(mine, lead, tag):
    rows = []
    for c in TOK:
        ln = lead_name(c)
        a = mine[c + "_v"] if c + "_v" in mine else mine[c]
        b = lead[ln + "_l"] if ln + "_l" in lead else lead[ln]
        d = np.abs(a.to_numpy() - b.to_numpy())
        rows.append({"set": tag, "token": c, "max_abs_diff": float(d.max()), "p99": float(np.quantile(d, .99)),
                     "n_gt_1e-3": int((d > 1e-3).sum())})
    return rows


def main():
    al, tdr = {}, []
    mine = pd.read_parquet(R / "validate_ha2_mv_tokens.parquet")
    lead = pd.read_parquet(R / "c1_grid_MV.parquet")
    print("lead MV columns:", [c for c in lead.columns if c not in {lead_name(t) for t in TOK}])
    mine["k"], lead["k"] = key(mine, "pft_ms"), key(lead, "post_ms")
    j = mine.merge(lead, on=["sim_idx", "k"], how="outer", suffixes=("_v", "_l"), indicator=True)
    al.update({"MV_n_mine": len(mine), "MV_n_lead": len(lead), **{f"MV_{k}": int(v) for k, v in j["_merge"].value_counts().items()}})
    j = j[j["_merge"] == "both"]
    al["MV_y_local_maxdiff"] = float(np.abs(j["y_local"] - j["y_l"]).max())
    al["MV_flip_mismatch"] = int((j["flipped"].astype(int) != j["flip"]).sum())
    al["MV_flip_lines_lead"] = ",".join(sorted(j.loc[j["flip"] == 1, "line_l"].unique()))
    al["MV_oracle_mismatch"] = int((j["oracle"] != j["oracle_loop"]).sum())
    both = j["two_ended"].notna() & j["td2_est"].notna()
    al["MV_two_ended_nan_mismatch"] = int((j["two_ended"].isna() != j["td2_est"].isna()).sum())
    al["MV_two_ended_maxabsdiff"] = float(np.abs(j.loc[both, "two_ended"] - j.loc[both, "td2_est"]).max())
    zl = j[["Z1r", "Z1x", "Z0r", "Z0x"]].to_numpy()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mv
    zs = np.array([[mv.line_z(ln)[0].real, mv.line_z(ln)[0].imag, mv.line_z(ln)[1].real, mv.line_z(ln)[1].imag]
                   for ln in j["line_v"]])
    al["MV_Z_maxreldiff_vs_spec"] = float(np.max(np.abs(zl - zs) / np.abs(zs)))
    al["MV_lead_nonfinite_ALLOW"] = int((~np.isfinite(lead[[lead_name(t) for t in TOK]].to_numpy())).sum())
    tdr += tok_diff(j, j, "MV")
    # raw windows of the lead vs my cache (local terminal)
    raw = np.load(R / "c1_grid_MV_raw.npz")
    RX = raw["X"]
    rk = pd.DataFrame({"sim_idx": raw["sim_idx"], "k": (raw["post_ms"] * 9.6).round().astype(int),
                       "i": np.arange(len(RX))}).merge(mine[["sim_idx", "k", "s1", "flipped"]], on=["sim_idx", "k"])
    dmax = 0.0
    for sid, g in rk.groupby("sim_idx"):
        sig = np.load(mv.CACHE / f"ep{sid}.npy")
        loc = sig[:, 6:12] if bool(g["flipped"].iloc[0]) else sig[:, 0:6]
        for i, s1 in zip(g["i"], g["s1"]):
            dmax = max(dmax, float(np.abs(RX[i] - loc[s1 - 480:s1].astype(np.float32)).max()))
    al["MV_raw_matched"], al["MV_raw_maxabsdiff"] = len(rk), dmax
    # source tables used by the lead's zs_eval
    for g in ("DL", "TG"):
        m = pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet")
        lg = pd.read_parquet(R / f"c1_grid_{g}.parquet")
        m["k"], lg["k"] = key(m, "tau"), key(lg, "post_ms")
        jj = m.merge(lg, on=["sim_idx", "k"], how="outer", suffixes=("_v", "_l"), indicator=True)
        al[f"{g}_src_n_lead"] = len(lg)
        al[f"{g}_src_only_one_side"] = int((jj["_merge"] != "both").sum())
        jj = jj[jj["_merge"] == "both"]
        al[f"{g}_src_y_maxdiff"] = float(np.abs(jj["y_v"] - jj["y_l"]).max())
        al[f"{g}_src_flipped"] = int(lg["flip"].sum()) if "flip" in lg else "no flip column"
        al[f"{g}_src_meta_cols"] = ",".join(c for c in lg.columns if c not in {lead_name(t) for t in TOK})
        tdr += tok_diff(jj, jj, g)
    # numbers
    me = pd.read_csv(R / "validate_ha2_mv_summary.csv").set_index("method")
    pr = pd.read_csv(R / "validate_ha2_mv_pred.csv")
    inc = pr["ftype"].str.contains("incipient")
    le = pd.read_csv(R / "c1_zs_MV.csv").set_index("method")
    nm = {"phys_react_oracle": "phys_react_oracle", "phys_tak2_oracle": "phys_tak2_oracle",
          "phys_react_minloop": "phys_react_minabsz", "td2_est": "two_ended_14seg", "H-A": "HA", "H-A2": "HA2"}
    rows = []
    for lm, mm in nm.items():
        if mm == "two_ended_14seg":
            sub = pr[pr["two_ended"].notna()]
            e = (sub["two_ended"].clip(0, 1) - sub["y"]).abs() * 100
            nonin = ~sub["ftype"].str.contains("incipient")
        else:
            e = (pr[mm] - pr["y_local"]).abs() * 100
            nonin = ~inc
        tol = 0.01 if lm.startswith(("phys", "td2")) else 0.3
        d = me.loc[mm, "MAE"] - le.loc[lm, "mae"]
        rows.append({"method": lm, "n_mine": int(me.loc[mm, "n"]), "n_lead": int(le.loc[lm, "n"]),
                     "MAE_mine": me.loc[mm, "MAE"], "MAE_lead": le.loc[lm, "mae"], "delta": d,
                     "median_mine": me.loc[mm, "median"], "median_lead": le.loc[lm, "median"],
                     "pft_le15_mine": me.loc[mm, "MAE_pft_le15"], "pft_le15_lead": le.loc[lm, "post_le15"],
                     "pft_ge21_mine": me.loc[mm, "MAE_pft_ge21"], "pft_ge21_lead": le.loc[lm, "post_ge21"],
                     "non_incipient_mine": float(e[nonin].mean()), "non_incipient_lead": le.loc[lm, "mae_shc_only"],
                     "verdict": "REPRODUCED" if abs(d) <= tol else "NOT REPRODUCED", "tol": tol})
    cmp_ = pd.DataFrame(rows)
    # per-window H-A2 agreement
    lp = pd.read_csv(R / "c1_zs_MV_pred.csv")
    lp["k"] = key(lp, "post_ms")
    pr["k"] = key(pr, "pft_ms")
    jj = pr.merge(lp, on=["sim_idx", "k"], validate="one_to_one")
    dp = (jj["HA2"] - jj["pred"]).abs() * 100
    pw = {"n": len(jj), "mean_abs_diff_pp": dp.mean(), "p95_pp": dp.quantile(.95),
          "corr": np.corrcoef(jj["HA2"], jj["pred"])[0, 1], "y_maxdiff": float((jj["y_local"] - jj["y_y"]).abs().max())}
    pd.Series(al).to_csv(R / "validate_ha2_mv_partB_alignment.csv", header=["value"])
    td = pd.DataFrame(tdr)
    td.to_csv(R / "validate_ha2_mv_partB_token_diff.csv", index=False)
    cmp_.to_csv(R / "validate_ha2_mv_partB_compare.csv", index=False)
    pd.Series(pw).to_csv(R / "validate_ha2_mv_partB_perwindow.csv", header=["value"])
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(pd.Series(al).to_string())
        print(td.groupby("set").agg(max_abs=("max_abs_diff", "max"), p99=("p99", "max"), n_gt=("n_gt_1e-3", "sum")))
        print(td.sort_values("max_abs_diff", ascending=False).head(6).to_string(index=False))
        print(cmp_.round(3).to_string(index=False))
        print(pd.Series(pw).round(4).to_string())


if __name__ == "__main__":
    main()
