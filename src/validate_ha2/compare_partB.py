"""Part B: compare frozen Part A numbers with the lead's result files, plus two diagnostics:
 (a) my HGB on the lead's token tables with the lead's column order (should reproduce the lead's numbers),
 (b) my HGB on MY tokens but in the lead's column order (isolates the column-order effect on max_features).
Writes results/validate_ha2_partB_compare.csv and results/validate_ha2_partB_diag.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402
from physics_hgb import HGB_PARAMS  # noqa: E402
from audit_partB import lead_name, META  # noqa: E402

R = C.RESULTS
DIRS = {"TGtoDL": "TG_to_DL", "DLtoTG": "DL_to_TG"}
TAG = {"DL": "", "TG": "_TestGrid110kV"}
FT = ["1phg_incipient", "1phg_incipient_w_arc", "1phg_shc", "1phg_shc_w_arc", "2ph_shc", "2phg_shc", "3ph_shc"]


def metrics(err: np.ndarray, tau: np.ndarray, ftype: np.ndarray) -> dict:
    d = {"MAE": err.mean(), "median": np.median(err), "MAE_tau_le15": err[tau <= 15].mean(),
         "MAE_tau_ge21": err[tau >= 21].mean()}
    for f in FT:
        d[f"MAE_{f}"] = err[ftype == f].mean()
    return d


def lead_numbers() -> pd.DataFrame:
    rows = []
    for d, ld in DIRS.items():
        tgt = d[-2:]
        for meth, suf in (("HA", ""), ("HA2", "_td")):
            f = pd.read_csv(R / f"c1_hybrid_ha_{ld}{suf}.csv")
            ft = f["etype"].str[len("flt_"):].to_numpy()
            rows.append({"method": meth, "direction": d, "n": len(f),
                         **metrics(f["err"].to_numpy(), f["tau_ms"].to_numpy(), ft)})
        lf = pd.read_parquet(R / f"c1_local_feats{TAG[tgt]}.parquet")
        ft = lf["etype"].str[len("flt_"):].to_numpy()
        for mine, lc in (("phys_react_oracle", "phys_react_oracle"), ("phys_tak2_oracle", "phys_tak2_oracle"),
                         ("phys_react_minabsz", "phys_react_minloop")):
            e = (lf[lc].clip(0, 1) - lf["y"]).abs().to_numpy() * 100
            rows.append({"method": mine, "direction": d, "n": len(lf), **metrics(e, lf["tau_ms"].to_numpy(), ft)})
    ei = pd.read_csv(R / "c1_equal_info.csv")
    for (m, v, ld), g in ei.groupby(["model", "variant", "direction"]):
        d = {x: y for y, x in DIRS.items()}[ld]
        name = f"{m}_{'raw' if v == '-raw' else '+Z'}"
        rows.append({"method": name, "direction": d, "n_seeds": len(g), "MAE": g["mae"].mean(),
                     "MAE_sd": g["mae"].std(ddof=1), "median": g["median"].mean(),
                     "MAE_tau_le15": g["tau_le15"].mean(), "MAE_tau_ge21": g["tau_ge21"].mean()})
    return pd.DataFrame(rows)


def per_window_agreement() -> list[dict]:
    out = []
    for d, ld in DIRS.items():
        for meth, suf in (("HA", ""), ("HA2", "_td")):
            mine = pd.read_csv(R / f"validate_ha2_pred_{meth}_{d}.csv").rename(columns={"tau": "tau_ms"})
            lead = pd.read_csv(R / f"c1_hybrid_ha_{ld}{suf}.csv")
            j = mine.merge(lead, on=["sim_idx", "tau_ms"], suffixes=("_v", "_l"), validate="one_to_one")
            dp = np.abs(j["pred_v"] - j["pred_l"]) * 100
            out.append({"method": meth, "direction": d, "n": len(j), "pred_mean_abs_diff_pp": dp.mean(),
                        "pred_p95_abs_diff_pp": dp.quantile(0.95), "pred_corr": np.corrcoef(j["pred_v"], j["pred_l"])[0, 1]})
    return out


def hgb_mae(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> float:
    p = np.mean([HistGradientBoostingRegressor(random_state=s, **HGB_PARAMS).fit(tr[cols].to_numpy(), tr["y"].to_numpy())
                 .predict(te[cols].to_numpy()) for s in (0, 1, 2)], axis=0)
    return float(np.abs(np.clip(p, 0, 1) - te["y"].to_numpy()).mean() * 100)


def diagnostics() -> list[dict]:
    lead, mine = {}, {}
    for g in ("DL", "TG"):
        lf = pd.read_parquet(R / f"c1_local_feats{TAG[g]}.parquet")
        td = pd.read_parquet(R / f"c1_td_tokens{TAG[g]}.parquet")
        lead[g] = lf.merge(td, on=["sim_idx", "tau_ms"], validate="one_to_one")
        mine[g] = pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet")
    lf_cols = [c for c in pd.read_parquet(R / "c1_local_feats.parquet").columns
               if c not in META and not c.startswith("phys_")]
    td_cols = [c for c in pd.read_parquet(R / "c1_td_tokens.parquet").columns if c not in ("sim_idx", "tau_ms")]
    inv = {lead_name(c): c for c in LOCAL_TOKENS + TD_TOKENS}
    rows = []
    for d in DIRS:
        src, tgt = d[:2], d[-2:]
        for meth, lcols in (("HA", lf_cols), ("HA2", lf_cols + td_cols)):
            rows.append({"method": meth, "direction": d, "variant": "lead tokens, lead column order",
                         "MAE": hgb_mae(lead[src], lead[tgt], lcols)})
            rows.append({"method": meth, "direction": d, "variant": "my tokens, lead column order",
                         "MAE": hgb_mae(mine[src], mine[tgt], [inv[c] for c in lcols])})
            print(rows[-2], rows[-1], flush=True)
    return rows


def main() -> None:
    mine = pd.read_csv(R / "validate_ha2_summary.csv")
    lead = lead_numbers()
    tol = {"phys": 0.01, "HA": 0.3}
    rows = []
    for r in lead.to_dict("records"):
        m = mine[(mine["method"] == r["method"]) & (mine["direction"] == r["direction"])].iloc[0]
        delta = m["MAE"] - r["MAE"]
        if r["method"].startswith("phys"):
            ok = abs(delta) <= tol["phys"]
            crit = "|d|<=0.01"
        elif r["method"] in ("HA", "HA2"):
            ok = abs(delta) <= tol["HA"]
            crit = "|d|<=0.3"
        else:
            ok = abs(delta) <= 1.0 or abs(delta) <= 2 * max(m["MAE_sd"], r["MAE_sd"])
            crit = "|d|<=1.0 or <=2sd"
        row = {"method": r["method"], "direction": r["direction"], "MAE_mine": m["MAE"], "MAE_mine_sd": m.get("MAE_sd"),
               "MAE_lead": r["MAE"], "MAE_lead_sd": r.get("MAE_sd"), "delta": delta, "criterion": crit,
               "verdict": "REPRODUCED" if ok else "NOT REPRODUCED"}
        for k in ("median", "MAE_tau_le15", "MAE_tau_ge21"):
            row[f"{k}_mine"], row[f"{k}_lead"] = m[k], r[k]
        rows.append(row)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(R / "validate_ha2_partB_compare.csv", index=False)
    lead.to_csv(R / "validate_ha2_partB_lead_numbers.csv", index=False)
    pw = pd.DataFrame(per_window_agreement())
    pw.to_csv(R / "validate_ha2_partB_perwindow.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(cmp_.round(3).to_string(index=False))
        print(pw.round(4).to_string(index=False))
    dg = pd.DataFrame(diagnostics())
    dg.to_csv(R / "validate_ha2_partB_diag.csv", index=False)
    print(dg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
