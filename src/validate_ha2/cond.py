"""Part 7: one-ended conditioning (design section 23) and the Fig 4 per-time table.

Written from claudedocs/validator_spec_cond.md + c1_hybrid_design.md sections 21b / 23 only (no src/c1 code, no
results/c1_* file opened before the freeze).

Derivation (mine): lumped line, loop quantities at terminal S, V = d Z_L I + R_f I_F.  A one-ended locator that assumes
angle(I_F / I) = b^ multiplies V/I by exp(-j b^) and keeps the imaginary part (the R_f term then vanishes):
    d^ = Im(V/I e^{-j b^}) / Im(Z_L e^{-j b^}) = d + R_f |I_F/I| sin(b - b^) / (|Z_L| sin(theta_L - b^)).
So e = d^ - d = rho * |I_F/I| * sin(b - b^) / sin(theta_L - b^), rho = R_f / |Z_L|, b = angle(I_F / I).
Reactance: b^ = 0.  Takagi: d = Im(V conj(dI)) / Im(Z I conj(dI)) = the same with b^ = angle(dI / I).
D_req (eps): |e| <= eps  <=>  |sin(b - b^)| <= eps / (rho kappa); linearised at b^ -> b, kappa = |I_F/I| / |sin(theta_L - b)|
(my choice; kappa at the Takagi and reactance b^ is also reported).

Part A data: 1phg_shc windows, post-fault >= 25 ms, lines measured at both ends (DL, TG tau windows; adapt TEST and
CIGRE MV rule windows; MV MainLn8-14 has no remote end and is excluded). I_F = I_Sp + I_Rp (faulted phase, both ends,
last-cycle DFT phasors; line charging ignored). Loop V, I with k0 compensation (as classical.py / tokens.py).
Part A(e) and Part B use the frozen part-6 per-window classical table, my guarded H-A2 predictions, my two-ended
locator (chain table 'clean' for DL/TG, mv_pred for MV) and my part-1 neural npy predictions.
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv as MVM  # noqa: E402
from ingrid import adapt_episodes, rule_windows  # noqa: E402
from tokens import PH, phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
N = C.N_CYC
PRE_GAP = 10
EPS = 0.05
RHO_EDGES = [0, 0.3, 1, 3, 10, 30, np.inf]
RHO_LAB = ["[0,0.3)", "[0.3,1)", "[1,3)", "[3,10)", "[10,30)", "[30,inf)"]


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def synthetic_check(n: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.uniform(0, 1, n)
    Rf = rng.uniform(0, 50, n)
    Z = rng.uniform(1, 10, n) * np.exp(1j * rng.uniform(np.deg2rad(60), np.deg2rad(88), n))
    I = rng.uniform(0.1, 10, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    IF = rng.uniform(0.1, 10, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    P = rng.uniform(0.1, 10, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))  # any polarising current
    V = d * Z * I + Rf * IF
    tak = (V * np.conj(P)).imag / (Z * I * np.conj(P)).imag
    bh = np.angle(P / I)
    b = np.angle(IF / I)
    e_formula = Rf / np.abs(Z) * np.abs(IF / I) * np.sin(b - bh) / np.sin(np.angle(Z) - bh)
    react = (V / I).imag / Z.imag
    e_r = Rf * (IF / I).imag / Z.imag
    return {"n": n, "max|tak-d-e_formula|": float(np.max(np.abs(tak - d - e_formula))),
            "max|react-d-e_R|": float(np.max(np.abs(react - d - e_r)))}


def work(task: tuple) -> list[dict]:
    grid, r = task
    idx = int(r["sim_idx"])
    if grid in ("DL", "TG"):
        loc = np.load(C.CACHE / grid / f"ep{idx}.npy")
        rem = np.load(C.CACHE / f"{grid}_R" / f"ep{idx}.npy")
        Z1, Z0 = C.line_z(grid, r["line"])
        nf = int(round((r["start"] - 1.0) * C.FS))
        ends = [nf + int(round(t / 1000.0 * C.FS)) for t in C.taus_for(r["etype"])]
    elif grid == "AD":
        loc = np.load(C.CACHE / "AD" / f"ep{idx}.npy")
        rem = np.load(C.CACHE / "AD_R" / f"ep{idx}.npy")
        Z1, Z0 = C.line_z("AD", r["line"])
        nf = math.floor((r["start"] - 1.0) * C.FS)
        ends = rule_windows(r["start"], r["etype"], r["dur"], len(loc))
    else:
        sig = np.load(MVM.CACHE / f"ep{idx}.npy")
        if np.isnan(sig[:, 0]).any() or np.isnan(sig[:, 6]).any():
            return []  # MainLn8-14: not measured at both ends
        loc, rem = sig[:, 0:6], sig[:, 6:12]
        Z1, Z0 = MVM.line_z(r["line"])
        nf = math.floor((r["start"] - 1.0) * C.FS)
        ends = [s1 for s1 in range(C.WIN, len(sig) + 1, 48) if s1 > nf and s1 - C.WIN < nf + 289]
    lp = C.oracle_loop(r["etype"], r["ph1"], r["ph2"])
    assert lp.endswith("g")
    p = PH[lp[0]]
    k0 = (Z0 - Z1) / (3.0 * Z1)
    y = r["loc"] / 100.0
    Rf = float(r["res"])
    pre = nf - PRE_GAP - N
    P, PR = phasor(loc, pre), phasor(rem, pre)
    rows = []
    for s1 in ends:
        pft = (s1 - nf) / 9.6
        if pft < 25 - 1e-9:
            continue
        L, LR = phasor(loc, s1 - N), phasor(rem, s1 - N)
        LI, LV = L[0:3], L[3:6]
        dI = LI - P[0:3]
        I0, dI0 = LI.sum() / 3, dI.sum() / 3
        V = LV[p]
        I = LI[p] + k0 * 3 * I0
        dIl = dI[p] + k0 * 3 * dI0
        IF = LI[p] + LR[p]
        q = IF / I
        thL = np.angle(Z1)
        b = np.angle(q)
        bT = np.angle(dIl / I)
        rho = Rf / abs(Z1)
        est_R = (V / I).imag / Z1.imag
        est_T = (V * np.conj(dIl)).imag / (Z1 * I * np.conj(dIl)).imag
        resid = V - y * Z1 * I - Rf * IF
        rows.append({
            "grid": grid, "sim_idx": idx, "s1": s1, "pft_ms": pft, "line": r["line"], "res": Rf, "y": y,
            "rho": rho, "absq": abs(q), "b_deg": np.degrees(b), "bT_deg": np.degrees(bT), "thL_deg": np.degrees(thL),
            "kappa": abs(q) / abs(math.sin(thL - b)), "kappa_T": abs(q) / abs(math.sin(thL - bT)),
            "kappa_R": abs(q) / math.sin(thL),
            "est_R": est_R, "est_T": est_T, "eobs_R": est_R - y, "eobs_T": est_T - y,
            "epred_R": rho * abs(q) * math.sin(b) / math.sin(thL),
            "epred_T": rho * abs(q) * math.sin(b - bT) / math.sin(thL - bT),
            "tak_angle_err_deg": abs(np.degrees(wrap(b - bT))),
            "model_resid_rel": abs(resid) / abs(V) if abs(V) > 0 else np.nan,
            "prefault_kcl_rel": abs(P[p] + PR[p]) / max(abs(P[p]), 1e-12),
        })
    return rows


def tasks() -> list[tuple]:
    t = []
    for g in ("DL", "TG"):
        e = C.episodes(g)
        for r in e[e["etype"] == "flt_1phg_shc"].sort_values("sim_idx").to_dict("records"):
            t.append((g, r))
    a = adapt_episodes()
    a = a[(a["split"] == "test") & (a["etype"] == "flt_1phg_shc")]
    for r in a.to_dict("records"):
        t.append(("AD", r))
    m = MVM.episodes()
    for r in m[m["etype"] == "flt_1phg_shc"].to_dict("records"):
        t.append(("MV", r))
    return t


def q(x: pd.Series, p: float) -> float:
    return float(np.nanpercentile(x, p))


def part_a() -> pd.DataFrame:
    out = R / "validate_ha2_cond_perwindow_1phg.parquet"
    if out.exists():
        df = pd.read_parquet(out)
    else:
        with ProcessPoolExecutor(max_workers=NW) as ex:
            res = list(ex.map(work, tasks(), chunksize=4))
        df = pd.DataFrame([x for rr in res for x in rr])
        df.to_parquet(out, index=False)
    # cross-check est_R / est_T against the frozen part-6 classical table
    cl = pd.read_parquet(R / "validate_ha2_step3_classical_perwindow.parquet", columns=["grid", "sim_idx", "s1", "R", "T"])
    m = df.merge(cl, on=["grid", "sim_idx", "s1"], how="left", validate="1:1")
    print("est_R vs part 6 max diff", float(np.nanmax(np.abs(m["est_R"] - m["R"]))),
          "est_T vs part 6 max diff", float(np.nanmax(np.abs(m["est_T"] - m["T"]))), flush=True)
    rows = []
    for g in ("DL", "TG", "AD", "MV"):
        d = df[df["grid"] == g]
        dR = (d["eobs_R"] - d["epred_R"]).abs() * 100
        dT = (d["eobs_T"] - d["epred_T"]).abs() * 100
        dreq = np.degrees(np.arcsin(np.minimum(1.0, EPS / (d["rho"] * d["kappa"]))))
        dreqT = np.degrees(np.arcsin(np.minimum(1.0, EPS / (d["rho"] * d["kappa_T"]))))
        rows.append({
            "grid": g, "n": len(d), "episodes": d["sim_idx"].nunique(),
            "id_R_med_pp": dR.median(), "id_R_p90_pp": q(dR, 90), "id_T_med_pp": dT.median(), "id_T_p90_pp": q(dT, 90),
            "eobs_R_MAE_unclipped": d["eobs_R"].abs().mean() * 100, "eobs_T_MAE_unclipped": d["eobs_T"].abs().mean() * 100,
            "model_resid_rel_med": d["model_resid_rel"].median(), "model_resid_rel_p90": q(d["model_resid_rel"], 90),
            "prefault_kcl_rel_med": d["prefault_kcl_rel"].median(),
            "rho_med": d["rho"].median(), "rho_p90": q(d["rho"], 90), "rho_min": d["rho"].min(), "rho_max": d["rho"].max(),
            "kappa_med": d["kappa"].median(), "kappa_p90": q(d["kappa"], 90),
            "kappa_T_med": d["kappa_T"].median(), "kappa_T_p90": q(d["kappa_T"], 90),
            "kappa_R_med": d["kappa_R"].median(), "kappa_R_p90": q(d["kappa_R"], 90),
            "rho_kappa_med": (d["rho"] * d["kappa"]).median(),
            "Dreq5_med_deg": dreq.median(), "Dreq5_p10_deg": q(dreq, 10), "share_Dreq5_lt1deg": float((dreq < 1).mean()),
            "share_Dreq5_lt1deg_kappaT": float((dreqT < 1).mean()),
            "tak_angle_err_med_deg": d["tak_angle_err_deg"].median(), "tak_angle_err_p90_deg": q(d["tak_angle_err_deg"], 90),
            "share_tak_err_gt_Dreq": float((d["tak_angle_err_deg"] > dreq).mean()),
        })
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_cond_summary.csv", index=False)
    # per R_f value (labelled R_f is discrete on DL/TG/MV)
    by = []
    for (g, rf), d in df[df["grid"] != "AD"].groupby(["grid", "res"]):
        dreq = np.degrees(np.arcsin(np.minimum(1.0, EPS / (d["rho"] * d["kappa"]))))
        by.append({"grid": g, "res": rf, "n": len(d), "rho_med": d["rho"].median(), "kappa_med": d["kappa"].median(),
                   "share_Dreq5_lt1deg": float((dreq < 1).mean()), "tak_err_med_deg": d["tak_angle_err_deg"].median(),
                   "eobs_T_MAE": d["eobs_T"].abs().mean() * 100, "id_T_med_pp": ((d["eobs_T"] - d["epred_T"]).abs() * 100).median()})
    pd.DataFrame(by).to_csv(R / "validate_ha2_cond_by_rf.csv", index=False)
    return s


def perwindow_all() -> pd.DataFrame:
    """All DL / TG / MV windows with R, T, E (registered rule 21b: NaN -> 0.5), H-A2 and two-ended."""
    cl = pd.read_parquet(R / "validate_ha2_step3_classical_perwindow.parquet")
    cl = cl[cl["grid"].isin(["DL", "TG", "MV"])].copy()
    ha2 = {"DL": "validate_ha2_guard_pred_zeroshot_HA2_TGtoDL.csv", "TG": "validate_ha2_guard_pred_zeroshot_HA2_DLtoTG.csv",
           "MV": "validate_ha2_guard_pred_mv_HA2_MV.csv"}
    parts = []
    for g, f in ha2.items():
        p = pd.read_csv(R / f)[["sim_idx", "s1", "pred"]].rename(columns={"pred": "HA2"})
        d = cl[cl["grid"] == g].merge(p, on=["sim_idx", "s1"], how="left", validate="1:1")
        assert d["HA2"].notna().all() and len(d) == len(p), g
        if g in ("DL", "TG"):
            t = pd.read_parquet(R / f"validate_ha2_step3_chain_tokens_{g}.parquet",
                                columns=["scen", "sim_idx", "s1", "tau", "y", "two_ended"])
            t = t[t["scen"] == "clean"].drop(columns="scen")
            d = d.merge(t, on=["sim_idx", "s1"], how="left", validate="1:1")
            assert d["two_ended"].notna().all() and np.allclose(d["y"], d["y_local"])
            assert np.allclose(d["tau"], d["pft_ms"])
        else:
            t = pd.read_csv(R / "validate_ha2_mv_pred.csv", usecols=["sim_idx", "s1", "y", "y_local", "two_ended"])
            d = d.merge(t.rename(columns={"y_local": "yl_chk"}), on=["sim_idx", "s1"], how="left", validate="1:1")
            assert np.allclose(d["yl_chk"], d["y_local"])
            d = d.drop(columns="yl_chk")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    for m in ("R", "T", "E"):
        v = df[m].to_numpy()
        df[f"e_{m}"] = np.abs(np.where(np.isfinite(v), np.clip(v, 0, 1), 0.5) - df["y_local"]) * 100
    df["e_HA2"] = np.abs(df["HA2"] - df["y_local"]) * 100
    df["e_two"] = np.abs(np.clip(df["two_ended"], 0, 1) - df["y"]) * 100  # NaN on MV 8-14 (no remote end)
    zabs = np.array([abs(C.line_z(g, ln)[0]) if g in ("DL", "TG") else abs(MVM.line_z(ln)[0])
                     for g, ln in zip(df["grid"], df["line"])])
    df["rho"] = df["res"] / zabs
    df["rhobin"] = pd.cut(df["rho"], RHO_EDGES, right=False, labels=RHO_LAB)
    return df


def part_e(df: pd.DataFrame) -> pd.DataFrame:
    sc = df[df["ftype"].str.contains("shc") & (df["pft_ms"] >= 25 - 1e-9)]
    rows = []
    for g in ("DL", "TG", "MV"):
        for b in RHO_LAB + ["all"]:
            d = sc[sc["grid"] == g] if b == "all" else sc[(sc["grid"] == g) & (sc["rhobin"] == b)]
            row = {"grid": g, "rho_bin": b, "n": len(d), "n_two": int(d["e_two"].notna().sum()),
                   "n_E_undef": int((~np.isfinite(d["E"])).sum())}
            for m, c in (("two_ended", "e_two"), ("R", "e_R"), ("T", "e_T"), ("E", "e_E"), ("HA2", "e_HA2")):
                row[m] = d[c].mean() if len(d) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(R / "validate_ha2_cond_rho_bins.csv", index=False)
    return out


def part_b(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, direc in (("DL", "TGtoDL"), ("TG", "DLtoTG")):
        d = df[df["grid"] == g].copy()
        meta = pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet", columns=["sim_idx", "tau", "s1", "y"])
        nn = {}
        for mdl in ("MLP", "GRU"):
            for var in ("raw", "Z"):
                P = np.stack([np.load(R / f"validate_ha2_nnpred_{mdl}_{var}_{direc}_s{s}.npy") for s in range(3)])
                assert P.shape[1] == len(meta)
                E = np.abs(np.clip(P, 0, 1) - meta["y"].to_numpy()[None]) * 100
                for s in range(3):
                    meta[f"e_{mdl}_{var}_s{s}"] = E[s]
                meta[f"e_{mdl}_{var}_ens"] = np.abs(np.clip(P.mean(0), 0, 1) - meta["y"].to_numpy()) * 100
                nn[f"{mdl}_{var}"] = [f"e_{mdl}_{var}_s{s}" for s in range(3)]
        d = d.merge(meta.drop(columns=["y", "tau"]), on=["sim_idx", "s1"], how="left", validate="1:1")
        assert d[[c for v in nn.values() for c in v]].notna().all().all()
        for t in list(range(5, 81, 5)) + ["all"]:
            dd = d if t == "all" else d[np.isclose(d["pft_ms"], t)]
            row = {"grid": g, "direction": direc, "post_ms": t, "n": len(dd), "two_ended": dd["e_two"].mean(),
                   "R": dd["e_R"].mean(), "T": dd["e_T"].mean(), "E": dd["e_E"].mean(), "HA2": dd["e_HA2"].mean()}
            for k, cols in nn.items():
                v = [dd[c].mean() for c in cols]
                row[k] = float(np.mean(v))
                row[k + "_sd"] = float(np.std(v, ddof=1))
                row[k + "_ens"] = dd[f"e_{k}_ens"].mean()
            one = {m: row[m] for m in ("R", "T", "E", "HA2", "MLP_raw", "MLP_Z", "GRU_raw", "GRU_Z")}
            srt = sorted(one, key=one.get)
            row["best_one_ended"], row["second"] = srt[0], srt[1]
            row["margin_pp"] = one[srt[1]] - one[srt[0]]
            noe = {m: v for m, v in one.items() if m != "E"}
            row["best_excl_E"] = min(noe, key=noe.get)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(R / "validate_ha2_cond_fig4.csv", index=False)
    return out


def main() -> None:
    sc = synthetic_check()
    print("synthetic derivation check", sc, flush=True)
    pd.DataFrame([sc]).to_csv(R / "validate_ha2_cond_synthetic_check.csv", index=False)
    s = part_a()
    df = perwindow_all()
    e = part_e(df)
    b = part_b(df)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(s.T.to_string())
        print(pd.read_csv(R / "validate_ha2_cond_by_rf.csv").round(3).to_string(index=False))
        print(e.round(2).to_string(index=False))
        cols = ["grid", "post_ms", "n", "two_ended", "R", "T", "E", "HA2", "MLP_raw", "MLP_Z", "GRU_raw", "GRU_Z",
                "best_one_ended", "second", "margin_pp", "best_excl_E"]
        print(b[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
