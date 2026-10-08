"""Single source of truth for every number in the paper: reads the [V] result files (and the benchmark's own published
aggregated_results.csv v1.1.0) and writes paper/macros_auto.tex. Never type a result into the .tex by hand.
Usage: python src/paper/make_macros.py"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
PUB = ROOT / "external" / "evemtbench-benchmark" / "results" / "1.1.0" / "aggregated_results.csv"
out = {}


def put(name, v, nd=2):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in out, name
    out[name] = f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else (f"{v:,}".replace(",", "{,}") if isinstance(v, (int, np.integer)) else str(v))


def mae(est, y):
    return float((np.clip(est, 0, 1) - y).abs().mean() * 100)


# ---- published learned baselines (EvEMTBench v1.1.0, fault location, MAE %) ----
p = pd.read_csv(PUB)
p = p[p.task.str.startswith("fault_location") & (p.metric == "mae")]
p_all = p
p = p[p.baseline != "majority"]  # published LEARNED models only; the constant predictor is reported separately
GRID = {"DL": "double_line", "TG": "testgrid_110kv", "MV": "cigre_mv"}
for k, g in GRID.items():
    q = p[(p.grid == g) & (p.protocol == "held_out") & (p.test_set == "benchmark")]
    put(f"pubBest{k}", q["mean"].min())                                   # best in-grid model, any view
    put(f"pubLocal{k}", q[q.task == "fault_location_local"]["mean"].min())
    put(f"pubLine{k}", q[q.task == "fault_location_line"]["mean"].min())
    z = p[(p.grid == g) & (p.protocol == "transfer_zeroshot") & (p.test_set == "benchmark") & (p.task == "fault_location_local")]
    put(f"pubZsLocal{k}", z["mean"].min())
q = p[(p.grid == "double_line") & (p.protocol == "held_out") & (p.test_set == "test")]
put("pubBestAdaptTest", q["mean"].min())
put("pubLocalAdaptTest", q[q.task == "fault_location_local"]["mean"].min())
put("pubMajorityMV", p_all[(p_all.grid == "cigre_mv") & (p_all.baseline == "majority") & (p_all.protocol == "held_out")
                           & (p_all.test_set == "benchmark") & (p_all.task == "fault_location_local")]["mean"].iloc[0])

# ---- two-ended TD locator ----
for k, f in (("DL", "DL"), ("TG", "TG"), ("MV", "MV"), ("MVid", "MV_zid")):
    d = pd.read_parquet(R / f"c1_grid_{f}.parquet", columns=["etype", "td2_est", "y"]).dropna(subset=["td2_est"])
    put(f"teMae{k}", mae(d.td2_est, d.y))
    sc = ~d.etype.str.contains("incipient|hif")
    put(f"teMaeSC{k}", mae(d.td2_est[sc], d.y[sc]))
    put(f"teN{k}", int(len(d)))
a = pd.read_csv(R / "c1_adapt_td.csv")
a = a[a.split == "test"]
put("teMaeAdapt", float(a.err.mean()))
put("teMaeSCAdapt", float(a.err[~a.etype.str.contains("incipient")].mean()))
put("teNAdapt", int(len(a)))

# ---- H-A2 / H-A (zero-shot, in-grid, MV) ----
h = pd.read_csv(R / "c1_ha2_eval.csv")
for m, tag in (("H-A2", "haTwo"), ("H-A", "haOne")):
    for dirn, k in (("TG_to_DL", "TGDL"), ("DL_to_TG", "DLTG")):
        r = h[(h.method == m) & (h.direction == dirn) & (h.test_noise == "clean")].iloc[0]
        put(f"{tag}Zs{k}", r.mae)
        if m == "H-A2":
            put(f"{tag}Zs{k}Le", r.tau_le15)
            put(f"{tag}Zs{k}Ge", r.tau_ge21)
            for nz, nk in (("_SNR40", "Forty"), ("_SNR30", "Thirty")):
                put(f"{tag}Zs{k}Snr{nk}", h[(h.method == m) & (h.direction == dirn) & (h.test_noise == nz)].mae.iloc[0])
ig = pd.read_csv(R / "c1_adapt_ingrid.csv")
for m, tag in (("H-A2", "haTwo"), ("H-A", "haOne"), ("phys_react_oracle", "react")):
    put(f"{tag}InAdapt", ig[(ig.method == m) & (ig.test == "adapt_test")].mae.iloc[0])
    put(f"{tag}InBench", ig[(ig.method == m) & (ig.test == "benchmark")].mae.iloc[0])
zm = pd.read_csv(R / "c1_zs_MV.csv").set_index("method")
put("haTwoMV", zm.loc["H-A2", "mae"])
put("haOneMV", zm.loc["H-A", "mae"])
put("reactMV", zm.loc["phys_react_oracle", "mae"])
_a3 = pd.read_csv(R / "c1_ablation_a3_fixed.csv").set_index("direction").mae  # design 19a (corrected A3)
put("aThreeMV", _a3["DL+TG_to_MV"])

# ---- equal-information learned baselines ----
e = pd.read_csv(R / "c1_equal_info.csv").groupby(["model", "variant", "direction"]).mae.agg(["mean", "std"])
for (mo, va, di), r in e.iterrows():
    k = mo.title() + ("Z" if va == "+Z" else "Raw") + ("TGDL" if di == "TG_to_DL" else "DLTG")
    put(f"eq{k}", r["mean"])
    put(f"eq{k}Sd", r["std"])
for di, k in (("TG_to_DL", "TGDL"), ("DL_to_TG", "DLTG")):
    best = e.xs(di, level="direction")["mean"].min()
    put(f"eqBest{k}", best)
    put(f"gainZs{k}", 100 * (1 - float(out[f"haTwoZs{k}"]) / best), 0)
ea = pd.read_csv(R / "c1_adapt_equal_info.csv").groupby(["model", "variant"])[["adapt_test_mae", "benchmark_mae"]].mean()
put("eqBestInAdapt", ea.adapt_test_mae.min())
put("eqBestInBench", ea.benchmark_mae.min())
put("gainInAdapt", 100 * (1 - float(out["haTwoInAdapt"]) / ea.adapt_test_mae.min()), 0)
put("gainInBench", 100 * (1 - float(out["haTwoInBench"]) / ea.benchmark_mae.min()), 0)
em = pd.read_csv(R / "c1_zs_equal_info_MV.csv").groupby(["model", "variant"]).mae.mean()
put("eqBestMV", em.min())

# ---- classical one-ended (registered rule 21b) ----
c = pd.read_csv(R / "c1_classical_report.csv")
SET = {"zero-shot TG->DL": "DL", "zero-shot DL->TG": "TG", "in-grid adapt test": "Adapt", "zero-shot DL+TG->MV": "MV"}
for s, k in SET.items():
    g = c[c.setting == s].set_index("method")
    for m in ("R", "T", "MT", "E"):
        put(f"cl{m}{k}", g.loc[m, "mae"])
    put(f"clELe{k}", g.loc["E", "post<=15"])
    put(f"clEGe{k}", g.loc["E", ">=55"])
    put(f"clEnan{k}", 100 * g.loc["E", "n_nan"] / g.loc["E", "n"], 0)

# ---- measurement chain ----
ch = pd.read_csv(R / "c1_chain_eval.csv").set_index(["direction", "scenario"])
for di, k in (("TG_to_DL", "TGDL"), ("DL_to_TG", "DLTG")):
    for sc, sk in (("clean", "Clean"), ("CT-severe", "CTsev"), ("CVT-5", "CVTfive"), ("CVT-15", "CVTfifteen"), ("FULL", "Full")):
        put(f"chTwo{sk}{k}", ch.loc[(di, sc), "two_ended"])
        put(f"chHa{sk}{k}", ch.loc[(di, sc), "HA2_clean_trained"])
    put(f"chSat{k}", ch.loc[(di, "CT-severe"), "sat_pct"], 1)
    put(f"chHaMatched{k}", ch.loc[(di, "FULL"), "HA2_matched_FULL"])

# ---- conditioning ----
cs = pd.read_csv(R / "c1_cond_summary.csv").set_index("grid")
for g, k in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
    put(f"rhoMed{k}", cs.loc[g, "rho_med"])
    put(f"angErr{k}", cs.loc[g, "angerr_T_med_deg"])
    put(f"dReq{k}", cs.loc[g, "Dreq_med_deg"])
    put(f"shareSub{k}", 100 * cs.loc[g, "share_Dreq_lt1deg"], 0)
    put(f"identMed{k}", cs.loc[g, "ident_R_med_pp"])

# ---- ablations ----
ab = pd.read_csv(R / "c1_ablations.csv").set_index(["ablation", "direction"])
put("abUnnormTGDL", _a3["TG_to_DL"]); put("abUnnormDLTG", _a3["DL_to_TG"])
for nm, k in (("A2_phasor_only", "Phasor"), ("A2_td_only", "Td"), ("A1_raw_window", "Raw"),
              ("A1_raw_window+Z", "RawZ")):
    put(f"ab{k}TGDL", ab.loc[(nm, "TG_to_DL"), "mae"])
    put(f"ab{k}DLTG", ab.loc[(nm, "DL_to_TG"), "mae"])


# ---- extra values for the tables ----
put("teMaeAdaptAll", float(a.err.mean()))
put("teNMVall", int(len(pd.read_parquet(R / "c1_grid_MV.parquet", columns=["y"]))))
for k, mo, va in (("MlpRaw", "MLP", "-raw"), ("MlpZ", "MLP", "+Z"), ("GruRaw", "GRU", "-raw"), ("GruZ", "GRU", "+Z")):
    put(f"eq{k}MV", em.loc[(mo, va)])
    put(f"eq{k}InAdapt", ea.loc[(mo, va), "adapt_test_mae"])
    put(f"eq{k}InBench", ea.loc[(mo, va), "benchmark_mae"])
put("abRawMV", zm.loc["A1_raw_window", "mae"])
put("abRawZMV", zm.loc["A1_raw_window+Z", "mae"])
for s_, k in SET.items():
    g = c[c.setting == s_].set_index("method")
    put(f"clTtwo{k}", g.loc["T2", "mae"])


# ---- grid facts for Table I (from the same window tables) ----
for k, f in (("DL", "c1_grid_DL"), ("TG", "c1_grid_TG"), ("MV", "c1_grid_MV")):
    t = pd.read_parquet(R / f"{f}.parquet", columns=["sim_idx", "line", "length_km", "Z1r", "Z1x"])
    u = t.drop_duplicates("line")
    z = np.hypot(u.Z1r, u.Z1x)
    put(f"gEp{k}", int(t.sim_idx.nunique())); put(f"gLines{k}", int(len(u)))
    nd = 2 if k == "MV" else 0  # 110 kV lines are whole kilometres
    put(f"gLenMin{k}", float(u.length_km.min()), nd); put(f"gLenMax{k}", float(u.length_km.max()), nd)
    put(f"gZMin{k}", float(z.min()), 2); put(f"gZMax{k}", float(z.max()), 2)
t = pd.read_parquet(R / "c1_adapt_tokens.parquet", columns=["sim_idx", "split", "R", "y"])
for sp, k in (("train", "Tr"), ("test", "Te")):
    g = t[t.split == sp]
    put(f"gEpAdapt{k}", int(g.sim_idx.nunique())); put(f"gWinAdapt{k}", int(len(g)))
put("gRfMaxAdapt", float(t.R.max()), 0)


# ---- CVT model residual after a bolted terminal collapse at voltage peak, 20 ms later (meas_chain.cvt_residual) ----
import sys as _s
_s.path.insert(0, str(ROOT / "src" / "c1"))
from meas_chain import cvt_residual  # noqa: E402
_res = {(r[0], r[1]): r for r in cvt_residual()}
put("cvtResFifteen", float(_res[(15.0, "peak")][3]), 0)
put("cvtResFive", float(_res[(5.0, "peak")][3]), 0)


# ---- bus-fault protocol finding ----
_s.path.insert(0, str(ROOT / "src" / "paper"))
from macros_bus import bus_macros  # noqa: E402
bus_macros(R, PUB, put)


# ---- per-type and per-time-bin numbers ----
from macros_extra import extra_macros  # noqa: E402
extra_macros(R, put)


from macros_more import more_macros, conservative_gains  # noqa: E402
more_macros(R, put)
conservative_gains(R, out, put)


_r = [float(out[f"pubBest{k}"]) / float(out[f"teMae{k}"]) for k in ("DL", "TG", "MV")]
put("ratioMin", min(_r), 1); put("ratioMax", max(_r), 0)

from macros_review import review_macros  # noqa: E402
review_macros(R, put)

from macros_revamp import revamp_macros  # noqa: E402
revamp_macros(R, PUB, put)
from macros_revamp import ratio_macros  # noqa: E402
ratio_macros(out, put)
from macros_revamp import cond_extra_macros  # noqa: E402
cond_extra_macros(R, put)

from macros_prof import td_noise_macros  # noqa: E402
td_noise_macros(R, put, out)
from macros_prof import learn2_macros, m6_macros  # noqa: E402
learn2_macros(R, put)
m6_macros(R, put)

for _d, _k in (("TG_to_DL", "TGDL"), ("DL_to_TG", "DLTG")):
    put(f"haSingle{_k}", h[(h.method == "H-A2") & (h.direction == _d) & (h.test_noise == "clean")].single_seed_mean.iloc[0])

lines = ["% AUTO-GENERATED by src/paper/make_macros.py from results/ and the EvEMTBench v1.1.0 aggregated results.",
         "% Do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
(ROOT / "paper").mkdir(exist_ok=True)
(ROOT / "paper" / "macros_auto.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(out)} macros -> paper/macros_auto.tex")
for k in ("pubBestDL", "pubBestTG", "pubBestMV", "pubBestAdaptTest", "teMaeDL", "teMaeTG", "teMaeSCAdapt", "teMaeMV",
          "teMaeMVid", "haTwoZsTGDL", "haTwoZsDLTG", "gainZsTGDL", "gainZsDLTG", "gainInAdapt", "gainInBench", "clEDL", "clETG",
          "clEAdapt", "clEMV", "haTwoMV"):
    print(k, out[k])
