"""In-grid Part B (run only after results/validate_ha2_ingrid_frozen.sha256 was written): compare with the lead's
results/c1_adapt_tokens.parquet, c1_adapt_raw.npz, c1_adapt_ingrid.csv and c1_adapt_ingrid_pred_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from audit_partB import lead_name  # noqa: E402
from physics_hgb import HGB_PARAMS  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402

R = C.RESULTS


def hgb_mae(tr, tes, cols):
    ms = [HistGradientBoostingRegressor(random_state=s, **HGB_PARAMS).fit(tr[cols].to_numpy(), tr["y"].to_numpy())
          for s in (0, 1, 2)]
    return {k: float(np.abs(np.clip(np.mean([m.predict(te[cols].to_numpy()) for m in ms], 0), 0, 1)
                            - te["y"].to_numpy()).mean() * 100) for k, te in tes.items()}


def main() -> None:
    mine = pd.read_parquet(R / "validate_ha2_ingrid_tokens.parquet")
    lead = pd.read_parquet(R / "c1_adapt_tokens.parquet")
    print("lead columns:", list(lead.columns)[:22], "...", len(lead.columns))
    mine["key"] = (mine["pft_ms"] * 9.6).round().astype(int)
    lead["key"] = (lead["post_ms"] * 9.6).round().astype(int)
    j = mine.merge(lead, on=["sim_idx", "key"], how="outer", suffixes=("_v", "_l"), indicator=True)
    al = {"n_mine": len(mine), "n_lead": len(lead), **{str(k): int(v) for k, v in j["_merge"].value_counts().items()}}
    j = j[j["_merge"] == "both"]
    al["split_mismatch"] = int((j["split_v"] != j["split_l"]).sum())
    al["y_maxdiff"] = float(np.abs(j["y_v"] - j["y_l"]).max())
    al["etype_mismatch"] = int((j["etype_v"] != j["etype_l"]).sum())
    al["oracle_mismatch"] = int((j["oracle"] != j["oracle_loop"]).sum())
    for sp in ("train", "test"):
        al[f"lead_{sp}_episodes"] = int(lead.loc[lead["split"] == sp, "sim_idx"].nunique())
        al[f"lead_{sp}_windows"] = int((lead["split"] == sp).sum())
    al["lead_train_test_overlap"] = len(set(lead.loc[lead.split == "train", "sim_idx"]) & set(lead.loc[lead.split == "test", "sim_idx"]))
    # Z used by the lead vs the spec values
    zl = j[["Z1r", "Z1x", "Z0r", "Z0x"]].to_numpy()
    zs = np.array([[C.line_z("AD", ln)[0].real, C.line_z("AD", ln)[0].imag, C.line_z("AD", ln)[1].real,
                    C.line_z("AD", ln)[1].imag] for ln in j["line_v"]])
    al["Z_maxreldiff_vs_spec"] = float(np.max(np.abs(zl - zs) / np.abs(zs)))
    rows = []
    for c in LOCAL_TOKENS + TD_TOKENS:
        ln = lead_name(c)
        a = j[c + "_v"] if c + "_v" in j else j[c]
        b = j[ln + "_l"] if ln + "_l" in j else j[ln]
        d = np.abs(a.to_numpy() - b.to_numpy())
        rows.append({"token": c, "lead_col": ln, "max_abs_diff": float(d.max()), "p99": float(np.quantile(d, 0.99)),
                     "n_gt_1e-3": int((d > 1e-3).sum())})
    td = pd.DataFrame(rows)
    # raw windows of the lead vs my episode cache
    raw = np.load(R / "c1_adapt_raw.npz")
    rk = pd.DataFrame({"sim_idx": raw["sim_idx"], "key": (raw["post_ms"] * 9.6).round().astype(int),
                       "i": np.arange(len(raw["sim_idx"]))}).merge(mine[["sim_idx", "key", "s1"]], on=["sim_idx", "key"])
    dmax = 0.0
    RX = raw["X"]
    for sid, g in rk.groupby("sim_idx"):
        sig = np.load(C.CACHE / "AD" / f"ep{sid}.npy").astype(np.float32)
        for i, s1 in zip(g["i"], g["s1"]):
            dmax = max(dmax, float(np.abs(RX[i] - sig[s1 - 480:s1]).max()))
    al["raw_n_lead"] = len(raw["sim_idx"])
    al["raw_matched"] = len(rk)
    al["raw_window_maxabsdiff"] = dmax
    al["raw_is_test_matches_split"] = bool((raw["is_test"][rk["i"]] == (mine.set_index(["sim_idx", "key"])
                                           .loc[list(zip(rk.sim_idx, rk.key)), "split"].to_numpy() == "test")).all())
    # numbers
    me = pd.read_csv(R / "validate_ha2_ingrid_summary.csv")
    me = me[me["seed"].isin(["avg", "na"])]
    le = pd.read_csv(R / "c1_adapt_ingrid.csv")
    nm = {"phys_react_minloop": "phys_react_minabsz", "H-A": "HA", "H-A2": "HA2"}
    ts = {"benchmark": "benchmark_DL", "adapt_test": "adapt_test"}
    cmp_ = []
    for r in le.to_dict("records"):
        m = me[(me["method"] == nm.get(r["method"], r["method"])) & (me["testset"] == ts[r["test"]])].iloc[0]
        tol = 0.01 if r["method"].startswith("phys") else 0.3
        cmp_.append({"method": r["method"], "test": r["test"], "n_mine": m["n"], "n_lead": r["n"],
                     "MAE_mine": m["MAE"], "MAE_lead": r["mae"], "delta": m["MAE"] - r["mae"],
                     "median_mine": m["median"], "median_lead": r["median"],
                     "pft_le15_mine": m["MAE_pft_le15"], "pft_le15_lead": r["post_le15"],
                     "pft_ge21_mine": m["MAE_pft_ge21"], "pft_ge21_lead": r["post_ge21"],
                     "shc_mine": m["MAE_shortcircuit"], "shc_lead": r["mae_shc_only"],
                     "verdict": "REPRODUCED" if abs(m["MAE"] - r["mae"]) <= tol else "NOT REPRODUCED"})
    cmp_ = pd.DataFrame(cmp_)
    # per-window H-A2 prediction agreement
    pw = []
    for k, lk in (("adapt_test", "adapt_test"), ("benchmark_DL", "benchmark")):
        a = pd.read_csv(R / f"validate_ha2_ingrid_pred_HA2_{k}.csv")
        b = pd.read_csv(R / f"c1_adapt_ingrid_pred_{lk}.csv")
        a["key"] = (a["pft_ms"] * 9.6).round().astype(int)
        b["key"] = (b["post_ms"] * 9.6).round().astype(int)
        jj = a.merge(b, on=["sim_idx", "key"], suffixes=("_v", "_l"), validate="one_to_one")
        dp = np.abs(jj["pred_v"] - jj["pred_l"]) * 100
        pw.append({"test": k, "n": len(jj), "mean_abs_diff_pp": dp.mean(), "p95_pp": dp.quantile(0.95),
                   "corr": np.corrcoef(jj["pred_v"], jj["pred_l"])[0, 1]})
    # diagnostic: my HGB on the lead's tokens (lead column order) and on my tokens in the lead's order
    from tokens import LOCAL_TOKENS as LT
    lead_local = [lead_name(c) for c in LT]
    lead_order_local = [c for c in lead.columns if c in set(lead_local)]
    lead_order_all = lead_order_local + [c for c in lead.columns if c in {lead_name(t) for t in TD_TOKENS}]
    inv = {lead_name(c): c for c in LOCAL_TOKENS + TD_TOKENS}
    bl = pd.read_parquet(R / "c1_local_feats.parquet").merge(pd.read_parquet(R / "c1_td_tokens.parquet"),
                                                             on=["sim_idx", "tau_ms"])
    bm = pd.read_parquet(R / "validate_ha2_tokens_DL.parquet")
    dg = []
    for meth, cols in (("HA", lead_order_local), ("HA2", lead_order_all)):
        r1 = hgb_mae(lead[lead.split == "train"], {"adapt_test": lead[lead.split == "test"], "benchmark": bl}, cols)
        r2 = hgb_mae(mine[mine.split == "train"], {"adapt_test": mine[mine.split == "test"], "benchmark": bm},
                     [inv[c] for c in cols])
        dg += [{"method": meth, "variant": "lead tokens, lead order", **r1},
               {"method": meth, "variant": "my tokens, lead order", **r2}]
    pd.Series(al).to_csv(R / "validate_ha2_ingrid_partB_alignment.csv", header=["value"])
    td.to_csv(R / "validate_ha2_ingrid_partB_token_diff.csv", index=False)
    cmp_.to_csv(R / "validate_ha2_ingrid_partB_compare.csv", index=False)
    pd.DataFrame(pw).to_csv(R / "validate_ha2_ingrid_partB_perwindow.csv", index=False)
    pd.DataFrame(dg).to_csv(R / "validate_ha2_ingrid_partB_diag.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(pd.Series(al).to_string())
        print(td.sort_values("max_abs_diff", ascending=False).head(8).to_string(index=False))
        print("tokens: max", td.max_abs_diff.max(), "p99 max", td.p99.max(), "n>1e-3", td["n_gt_1e-3"].sum())
        print(cmp_.round(3).to_string(index=False))
        print(pd.DataFrame(pw).round(4).to_string(index=False))
        print(pd.DataFrame(dg).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
