"""Part 7 comparison (run AFTER results/validate_ha2_cond_frozen.sha256 was written): mine vs the lead's
results/c1_cond_windows.parquet, c1_cond_summary.csv, c1_cond_rho_bins.csv, c1_fig4_data.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.RESULTS
assert (R / "validate_ha2_cond_frozen.sha256").exists()
GMAP = {"ADAPT": "AD"}
out = []


def log(s: str) -> None:
    print(s, flush=True)
    out.append(s)


# 1. per-window conditioning quantities
me = pd.read_parquet(R / "validate_ha2_cond_perwindow_1phg.parquet")
le = pd.read_parquet(R / "c1_cond_windows.parquet")
le["grid"] = le["grid"].replace(GMAP)
me["k"], le["k"] = me["pft_ms"].round(3), le["post_ms"].round(3)
m = me.merge(le, on=["grid", "sim_idx", "k"], how="outer", suffixes=("", "_L"), indicator=True, validate="1:1")
log(f"window join: {m['_merge'].value_counts().to_dict()}")
m = m[m["_merge"] == "both"]
me_dreqT = np.degrees(np.arcsin(np.minimum(1.0, 0.05 / (m["rho"] * m["kappa_T"]))))
pairs = {"e_obs_R": ("eobs_R", "e_obs_R"), "e_pred_R": ("epred_R", "e_pred_R"), "e_obs_T": ("eobs_T", "e_obs_T"),
         "e_pred_T": ("epred_T", "e_pred_T"), "rho": ("rho", "rho_L"), "|IF/I|": ("absq", "ratio_IF_I"),
         "kappa_T": ("kappa_T", None), "angerr_T": ("tak_angle_err_deg", None), "Dreq(kappa_T)": (None, "Dreq_deg")}
rows = []
for g, d in m.groupby("grid"):
    idx = d.index
    for nm, (a, b) in pairs.items():
        if nm == "kappa_T":
            x, y = d["kappa_T"], d["kappa_T_L"].abs()
        elif nm == "angerr_T":
            x, y = d["tak_angle_err_deg"], d["angerr_T_deg"].abs()
        elif nm == "Dreq(kappa_T)":
            x, y = me_dreqT.loc[idx], d["Dreq_deg"]
        else:
            x, y = d[a], d[b]
        diff = (x - y).abs()
        scale = 100 if nm.startswith("e_") else 1
        rows.append({"grid": g, "quantity": nm, "n": len(d), "max_abs_diff": diff.max() * scale,
                     "p99_abs_diff": diff.quantile(0.99) * scale, "n_diff_gt_1e-6": int((diff * scale > 1e-6).sum())})
pw = pd.DataFrame(rows)
pw.to_csv(R / "validate_ha2_condcmp_perwindow.csv", index=False)
log(pw.to_string(index=False))

# 2. summary side by side
ms = pd.read_csv(R / "validate_ha2_cond_summary.csv").set_index("grid")
ls = pd.read_csv(R / "c1_cond_summary.csv")
ls["grid"] = ls["grid"].replace(GMAP)
ls = ls.set_index("grid")
# my kappa_T-based D_req median (the lead's definition) for a like-for-like cell
dq = me.assign(D=np.degrees(np.arcsin(np.minimum(1.0, 0.05 / (me["rho"] * me["kappa_T"])))))
my_dreqT_med = dq.groupby("grid")["D"].median()
my_obs_med = me.assign(R=me["eobs_R"].abs() * 100, T=me["eobs_T"].abs() * 100).groupby("grid")[["R", "T"]].median()
cmp = [("ident_R_med_pp", "id_R_med_pp"), ("ident_R_p90_pp", "id_R_p90_pp"), ("ident_T_med_pp", "id_T_med_pp"),
       ("ident_T_p90_pp", "id_T_p90_pp"), ("rho_med", "rho_med"), ("rho_p90", "rho_p90"), ("kappaT_med", "kappa_T_med"),
       ("angerr_T_med_deg", "tak_angle_err_med_deg"), ("angerr_T_p90_deg", "tak_angle_err_p90_deg"),
       ("share_Dreq_lt1deg", "share_Dreq5_lt1deg_kappaT")]
srows = []
for g in ("DL", "TG", "AD", "MV"):
    for lc, mc in cmp:
        srows.append({"grid": g, "quantity": lc, "lead": ls.loc[g, lc], "mine": ms.loc[g, mc],
                      "delta": ms.loc[g, mc] - ls.loc[g, lc]})
    srows.append({"grid": g, "quantity": "Dreq_med_deg (kappa_T)", "lead": ls.loc[g, "Dreq_med_deg"],
                  "mine": my_dreqT_med[g], "delta": my_dreqT_med[g] - ls.loc[g, "Dreq_med_deg"]})
    for nm in ("R", "T"):
        srows.append({"grid": g, "quantity": f"mae_obs_{nm} (lead col = MEDIAN |e|)", "lead": ls.loc[g, f"mae_obs_{nm}"],
                      "mine": my_obs_med.loc[g, nm], "delta": my_obs_med.loc[g, nm] - ls.loc[g, f"mae_obs_{nm}"]})
ss = pd.DataFrame(srows)
ss.to_csv(R / "validate_ha2_condcmp_summary.csv", index=False)
log(ss.round(4).to_string(index=False))

# 3. rho bins
mb = pd.read_csv(R / "validate_ha2_cond_rho_bins.csv")
lb = pd.read_csv(R / "c1_cond_rho_bins.csv")
bmap = {"(0.0, 0.3]": "[0,0.3)", "(0.3, 1.0]": "[0.3,1)", "(1.0, 3.0]": "[1,3)", "(3.0, 10.0]": "[3,10)",
        "(10.0, 30.0]": "[10,30)", "(30.0, 1000000000.0]": "[30,inf)"}
lb["rho_bin"] = lb["rho_bin"].map(bmap)
assert lb["rho_bin"].notna().all()
mmap = {"two-ended": "two_ended", "R": "R", "T": "T", "E": "E", "H-A2": "HA2"}
brow = []
for r in lb.itertuples():
    x = mb[(mb.grid == r.grid) & (mb.rho_bin == r.rho_bin)].iloc[0]
    col = mmap[r.method]
    n_me = x["n_two"] if col == "two_ended" else x["n"]
    brow.append({"grid": r.grid, "rho_bin": r.rho_bin, "method": r.method, "n_lead": r.n, "n_mine": n_me,
                 "lead": r.mae, "mine": x[col], "delta": x[col] - r.mae})
bb = pd.DataFrame(brow)
bb.to_csv(R / "validate_ha2_condcmp_rho_bins.csv", index=False)
log(bb.round(3).to_string(index=False))

# 4. Fig 4
mf = pd.read_csv(R / "validate_ha2_cond_fig4.csv")
lf = pd.read_csv(R / "c1_fig4_data.csv")
lmap = {"two-ended": "two_ended", "reactance": "R", "Takagi": "T", "Eriksson": "E", "H-A2": "HA2", "MLP-raw": "MLP_raw",
        "MLP+Z": "MLP_Z", "GRU-raw": "GRU_raw", "GRU+Z": "GRU_Z"}
lf = lf.rename(columns=lmap)
lf["grid"] = lf["direction"].map({"TG_to_DL": "DL", "DL_to_TG": "TG"})
mf["t"] = mf["post_ms"].astype(str)
lf["t"] = lf["t"].astype(str)
one = ["R", "T", "E", "HA2", "MLP_raw", "MLP_Z", "GRU_raw", "GRU_Z"]
frow = []
for r in lf.itertuples(index=False):
    rr = r._asdict()
    x = mf[(mf.grid == rr["grid"]) & (mf.t == rr["t"])].iloc[0]
    lead_one = {k: rr[k] for k in one}
    lb_ = sorted(lead_one, key=lead_one.get)
    row = {"grid": rr["grid"], "t": rr["t"], "n_lead": rr["n"], "n_mine": x["n"]}
    for k in ["two_ended"] + one:
        row[f"d_{k}"] = x[k] - rr[k]
    row.update(best_mine=x["best_one_ended"], best_lead=lb_[0], second_lead=lb_[1],
               margin_lead=lead_one[lb_[1]] - lead_one[lb_[0]], margin_mine=x["margin_pp"],
               best_match=x["best_one_ended"] == lb_[0])
    frow.append(row)
ff = pd.DataFrame(frow)
ff.to_csv(R / "validate_ha2_condcmp_fig4.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    log(ff.round(2).to_string(index=False))
    log(lf[["grid", "t", "n", "two_ended"] + one].round(2).to_string(index=False))
(R / "validate_ha2_condcmp.log").write_text("\n".join(out) + "\n")
