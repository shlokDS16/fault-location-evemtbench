"""Part 8, after the freeze (results/validate_ha2_part8_frozen.sha256): comparison with the lead's files and
diagnostics. Does not modify any frozen file. Outputs results/validate_ha2_part8cmp_*.csv.
  1. A / C / D1 by post-fault bin with the LEAD's edges (0, 17.5], (17.5, 32.5], (32.5, 52.5], > 52.5 ms.
  2. MV phasor locator: windows with |Z1 (I1S + I1R)| <= 1e-9 (lead -> NaN -> 0.5; mine -> raw value).
  3. 24(d): two-ended TD MAE for post <= 50 ms and post > 52.5 ms windows, all grids.
  4. Lead's extra cells (not in my spec): phasor locator under fractional delays (DL, TG); integer shifts and
     parameter scaling on ADAPT test.
  5. A3 with the lead's definition (clipped per-unit tokens x Z1) on my guarded tables, HGB seeds 0-2.
  6. adapt TD per window: mine vs results/c1_adapt_td.csv (test split).
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import part8 as P8  # noqa: E402
from ingrid import adapt_episodes, rule_windows  # noqa: E402
from mv import two_ended  # noqa: E402
from physics_hgb import HGB_PARAMS  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, TD_TOKENS, phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
LBINS = [0, 17.5, 32.5, 52.5, 1e9]
LAB = ["<=15", "20-30", "35-50", ">=55"]


def lbin(p) -> pd.Series:
    return pd.cut(pd.Series(p), LBINS, labels=LAB)


def sc(e, y):
    return np.abs(np.where(np.isfinite(e), np.clip(e, 0, 1), 0.5) - y) * 100


def mv_den(task):
    import mv as MVM
    r = task
    sig = np.load(MVM.CACHE / f"ep{r['sim_idx']}.npy")
    if np.isnan(sig[:, 0]).any() or np.isnan(sig[:, 6]).any():
        return []
    Z1, _ = MVM.line_z(r["line"])
    nf = math.floor((r["start"] - 1.0) * C.FS)
    D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
    out = []
    for s1 in [s for s in range(C.WIN, len(sig) + 1, 48) if s > nf and s - C.WIN < nf + D]:
        pS, pR = phasor(sig[:, 0:6], s1 - C.N_CYC), phasor(sig[:, 6:12], s1 - C.N_CYC)
        out.append({"sim_idx": r["sim_idx"], "s1": s1, "absden": abs(Z1 * (P8.pos(pS[0:3]) + P8.pos(pR[0:3])))})
    return out


def extra_work(task):
    grid, r = task
    idx = int(r["sim_idx"])
    out = []
    if grid in ("DL", "TG"):
        S = np.load(C.CACHE / grid / f"ep{idx}.npy")
        Rr = np.load(C.CACHE / f"{grid}_R" / f"ep{idx}.npy")
        Z1, _ = C.line_z(grid, r["line"])
        nf = int(round((r["start"] - 1.0) * C.FS))
        ends = [nf + int(round(t / 1000.0 * C.FS)) for t in C.taus_for(r["etype"])]
        RF = {us: P8.frac_delay(Rr, us * 1e-6) for us in (1, 10, 50)}
        for s1 in ends:
            row = {"grid": grid, "sim_idx": idx, "s1": s1, "y": r["loc"] / 100.0}
            for us, Rf in RF.items():
                row[f"ph_{us}us"] = P8.phasor_two_ended(S, Rf, s1, Z1)
            out.append(row)
        return out
    S = np.load(C.CACHE / "AD" / f"ep{idx}.npy")
    Rr = np.load(C.CACHE / "AD_R" / f"ep{idx}.npy")
    Z1, _ = C.line_z("AD", r["line"])
    R1, L1 = Z1.real, Z1.imag / P8.W0
    dS, dR = P8.sg(S[:, 0:3]), P8.sg(Rr[:, 0:3])
    for s1 in rule_windows(r["start"], r["etype"], r["dur"], len(S)):
        sl = slice(s1 - C.WIN, s1)
        row = {"grid": "AD", "sim_idx": idx, "s1": s1, "y": r["loc"] / 100.0}
        for k in P8.SHIFTS:
            row[f"sh{k:+d}"] = two_ended(S, np.roll(Rr, k, axis=0), dS, np.roll(dR, k, axis=0), R1, L1, sl)
        for s in P8.SCALE_RL:
            row[f"RL{s:.2f}"] = two_ended(S, Rr, dS, dR, R1 * s, L1 * s, sl)
        for s in P8.SCALE_R:
            row[f"R{s:.1f}"] = two_ended(S, Rr, dS, dR, R1 * s, L1, sl)
        out.append(row)
    return out


def main() -> None:
    pw = pd.read_parquet(R / "validate_ha2_part8_perwindow.parquet")
    rows = []
    # 1 + 3
    for g, d in pw.groupby("grid", sort=False):
        b = lbin(d["pft_ms"].to_numpy())
        y = d["y"].to_numpy()
        for m in ("ph2e", "td2e"):
            e = sc(d[m].to_numpy(), y)
            for k in LAB:
                rows.append({"item": f"bin_{m}", "grid": g, "bin": k, "n": int((b == k).sum()), "MAE": e[(b == k).to_numpy()].mean()})
        e = sc(d["td2e"].to_numpy(), y)
        p = d["pft_ms"].to_numpy()
        rows.append({"item": "td_with_prefault(<=50)", "grid": g, "n": int((p <= 50).sum()), "MAE": e[p <= 50].mean()})
        rows.append({"item": "td_post_only(>52.5)", "grid": g, "n": int((p > 52.5).sum()), "MAE": e[p > 52.5].mean()})
    tabs = {"DL": (pd.read_parquet(R / "validate_ha2_tokens_DL.parquet"), "y", "tau"),
            "TG": (pd.read_parquet(R / "validate_ha2_tokens_TG.parquet"), "y", "tau"),
            "AD": (pd.read_parquet(R / "validate_ha2_ingrid_tokens.parquet").query("split == 'test'"), "y", "pft_ms"),
            "MV": (pd.read_parquet(R / "validate_ha2_mv_tokens.parquet"), "y_local", "pft_ms")}
    for g, (t, yc, pc) in tabs.items():
        t = t.reset_index(drop=True)
        ii = t["oracle"].map({lp: k for k, lp in enumerate(LOOPS)}).to_numpy().astype(int)
        est = t[[f"dtd_{lp}" for lp in LOOPS]].to_numpy()[np.arange(len(t)), ii]
        e = sc(est, t[yc].to_numpy())
        p = t[pc].to_numpy(np.float64)
        rows.append({"item": "dtd_le17.5", "grid": g, "n": int((p <= 17.5).sum()), "MAE": e[p <= 17.5].mean()})
        rows.append({"item": "dtd_gt52.5", "grid": g, "n": int((p > 52.5).sum()), "MAE": e[p > 52.5].mean()})
    # 2. MV small denominators
    import mv as MVM
    with ProcessPoolExecutor(max_workers=NW) as ex:
        den = pd.DataFrame([x for rr in ex.map(mv_den, MVM.episodes().to_dict("records"), chunksize=8) for x in rr])
    d = pw[pw["grid"] == "MV"].merge(den, on=["sim_idx", "s1"], validate="1:1")
    small = d["absden"] <= 1e-9
    e_mine = sc(d["ph2e"].to_numpy(), d["y"].to_numpy())
    e_lead = sc(np.where(small, np.nan, d["ph2e"].to_numpy()), d["y"].to_numpy())
    rows.append({"item": "MV_ph2_absden<=1e-9", "grid": "MV", "n": int(small.sum()), "MAE": e_mine.mean(),
                 "MAE_leadrule": e_lead.mean(), "ftypes": ",".join(sorted(d.loc[small, "ftype"].unique()))})
    for k in LAB:
        m = (lbin(d["pft_ms"].to_numpy()) == k).to_numpy()
        rows.append({"item": "MV_ph2_leadrule_bin", "grid": "MV", "bin": k, "n": int(m.sum()), "MAE": e_lead[m].mean()})
    # 4. lead's extra cells
    t = [(g, r) for g in ("DL", "TG") for r in C.episodes(g).sort_values("sim_idx").to_dict("records")]
    a = adapt_episodes()
    t += [("AD", r) for r in a[a["split"] == "test"].to_dict("records")]
    with ProcessPoolExecutor(max_workers=NW) as ex:
        ex_rows = pd.DataFrame([x for rr in ex.map(extra_work, t, chunksize=4) for x in rr])
    for g, dd in ex_rows.groupby("grid", sort=False):
        for c in [c for c in dd.columns if c not in ("grid", "sim_idx", "s1", "y")]:
            if dd[c].notna().any():
                rows.append({"item": f"extra_{c}", "grid": g, "n": int(dd[c].notna().sum()),
                             "MAE": sc(dd[c].to_numpy(), dd["y"].to_numpy())[dd[c].notna().to_numpy()].mean()})
    # 6. adapt TD per window vs lead
    lt = pd.read_csv(R / "c1_adapt_td.csv").query("split == 'test'")
    mine = pw[pw["grid"] == "AD"].copy()
    mine["k"] = mine["pft_ms"].round(4)
    lt["k"] = lt["post_ms"].round(4)
    j = mine.merge(lt, on=["sim_idx", "k"], how="inner", validate="1:1")
    rows.append({"item": "adapt_td_perwindow_vs_lead", "grid": "AD", "n": len(j), "n_mine": len(mine), "n_lead": len(lt),
                 "MAE": float(np.abs(j["td2e"] * 100 - j["loc_est"]).max()),
                 "ftypes": f"max|y-loc_true/100| {float(np.abs(j['y'] - j['loc_true'] / 100).max()):.2e}"})
    # 5. A3 with the lead's definition
    T = {g: pd.read_parquet(R / f"validate_ha2_guard_tokens_{g}.parquet").sort_values(["sim_idx", "tau"])
         .reset_index(drop=True) for g in ("DL", "TG")}
    allc = list(LOCAL_TOKENS) + list(TD_TOKENS)

    def unnorm(df):
        d2 = df.copy()
        for lp in LOOPS:
            zr, zi = df[f"zr_{lp}"], df[f"zi_{lp}"]
            d2[f"zr_{lp}"] = zr * df.R1 - zi * df.X1
            d2[f"zi_{lp}"] = zr * df.X1 + zi * df.R1
            d2[f"absz_{lp}"] = df[f"absz_{lp}"] * np.hypot(df.R1, df.X1)
        return d2
    clipped = {g: float(((T[g][[f"zr_{lp}" for lp in LOOPS] + [f"zi_{lp}" for lp in LOOPS]].abs() >= 5 - 1e-12).any(axis=1)
                         | (T[g][[f"absz_{lp}" for lp in LOOPS]] >= 10 - 1e-12).any(axis=1)).mean()) for g in T}
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        tr, te = unnorm(T[src]), unnorm(T[tgt])
        P = [HistGradientBoostingRegressor(random_state=s, **HGB_PARAMS).fit(tr[allc].to_numpy(np.float64), tr["y"].to_numpy())
             .predict(te[allc].to_numpy(np.float64)) for s in (0, 1, 2)]
        e = np.abs(np.clip(np.mean(P, axis=0), 0, 1) - te["y"].to_numpy()) * 100
        rows.append({"item": "A3_leaddef_on_my_tokens", "grid": f"{src}to{tgt}", "n": len(te), "MAE": e.mean(),
                     "ftypes": f"windows with a clipped z token: {src} {clipped[src]:.3f}, {tgt} {clipped[tgt]:.3f}"})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R / "validate_ha2_part8cmp_diag.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 80):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
