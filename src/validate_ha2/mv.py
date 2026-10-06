"""Part 4: CIGRE MV (benchmark family), claudedocs/validator_spec_cigre.md. CPU only.

Steps (cached / resumable):
  1. extract the S-terminal (6 ch) and, where it exists, the R-terminal (6 ch) of every line-fault episode
     -> results/validate_ha2_epcache/MV/ep<idx>.npy, shape (4801, 12); R part NaN for MainLn8-14;
  2. windows by the reconstructed rule (nf = 960 here, so identical to tau = 5..80 / 5..50 ms);
     59 + 18 tokens on the LOCAL terminal (bus X; bus 14 for MainLn8-14 with y_local = 1 - y);
     two-ended aerial-mode R-L estimate on the 14 two-terminal segments;
  3. Z1 identification from pre-fault positive-sequence phasors (both ends) per segment;
  4. H-A / H-A2 zero-shot (train on DL + TG official windows pooled, my cached tokens), physics baselines,
     R_f diagnosis, 8-14 orientation check.
"""
from __future__ import annotations

import csv
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, TD_TOKENS, local_tokens, phasor, td_tokens  # noqa: E402

R = C.RESULTS
MV = C.RAW / "CigreMVGrid"
CACHE = C.CACHE / "MV"
W0 = 2 * np.pi * C.F0
NW = int(__import__("os").environ.get("VAL_WORKERS", "6"))
LUMPED = (0.501, 0.716, 0.817, 1.598)
DISTRIB = (0.4738, 0.5075, 0.6186, 1.7561)
LEN = {"1-2": 2.82, "2-3": 4.42, "3-4": 0.61, "3-8": 1.30, "4-5": 0.56, "5-6": 1.54, "6-7": 0.24, "7-8": 1.67,
       "8-9": 0.32, "9-10": 0.77, "10-11": 0.33, "4-11": 0.49, "12-13": 4.89, "13-14": 2.99, "8-14": 2.00}
DIST_LINES = {"12-13", "13-14", "8-14"}
FT = ["1phg_incipient", "1phg_incipient_w_arc", "1phg_hif", "1phg_hif_w_arc", "1phg_shc", "1phg_shc_w_arc",
      "2ph_shc", "2phg_shc", "3ph_shc"]


def line_z(line: str) -> tuple[complex, complex]:
    seg = line[len("MainLn"):]
    r1, x1, r0, x0 = DISTRIB if seg in DIST_LINES else LUMPED
    L = LEN[seg]
    return complex(r1 * L, x1 * L), complex(r0 * L, x0 * L)


def buses(line: str) -> tuple[str, str]:
    m = re.fullmatch(r"MainLn(\d+)-(\d+)", line)
    return m.group(1), m.group(2)


def episodes() -> pd.DataFrame:
    df = pd.read_csv(MV / "labels" / "settings_clean.csv")
    m = (df[C.C_TYPE].astype(str).str.startswith("flt_") & df[C.C_TARGET].astype(str).str.fullmatch(r"MainLn(\d+)-(\d+)")
         & df[C.C_LOC].notna())
    e = df.loc[m, [C.C_IDX, C.C_TYPE, C.C_START, C.C_TARGET, C.C_LOC, C.C_RES, C.C_PH1, C.C_PH2, C.C_DUR]].copy()
    e.columns = ["sim_idx", "etype", "start", "line", "loc", "res", "ph1", "ph2", "dur"]
    e["sim_idx"] = e["sim_idx"].astype(int)
    return e.sort_values("sim_idx").reset_index(drop=True)


def cub_cols(names: list[str], units: list[str], bus: str, line: str) -> list[int] | None:
    tgt = f"pex_MainBus{bus}_{line}"
    cols = [j for j, nm in enumerate(names) if nm.endswith("\\" + tgt) or nm == tgt]
    if not cols:
        return None
    if len(cols) != 6 or cols != list(range(cols[0], cols[0] + 6)):
        raise ValueError(f"{line} bus {bus}: {cols}")
    for k, ch in enumerate(C.CHANNELS):
        if ch not in units[cols[k]]:
            raise ValueError(f"unit {units[cols[k]]} != {ch}")
    return cols


def extract(sim_idx: int, line: str) -> str:
    out = CACHE / f"ep{sim_idx}.npy"
    if out.exists():
        return "cached"
    path = MV / "data" / f"result{sim_idx}.csv"
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        names, units = next(rd), next(rd)
    bx, by = buses(line)
    cx, cy = cub_cols(names, units, bx, line), cub_cols(names, units, by, line)
    if cx is None and cy is None:
        raise ValueError(f"{sim_idx}: no cubicle for {line}")
    use = [0] + (cx or []) + (cy or [])
    raw = pd.read_csv(path, skiprows=2, header=None, usecols=use, dtype=np.float64, engine="c")
    t = raw[0].to_numpy()
    assert len(t) == C.N_SAMPLES and abs(t[0] - 1.0) < 1e-9, (sim_idx, len(t), t[0])
    sig = np.full((len(t), 12), np.nan)
    if cx is not None:
        sig[:, 0:6] = raw[cx].to_numpy()
    if cy is not None:
        sig[:, 6:12] = raw[cy].to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, sig)
    tmp.replace(out)
    return "parsed"


def clarke(x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = x3[:, 0], x3[:, 1], x3[:, 2]
    return (2 * a - b - c) / 3.0, (b - c) / np.sqrt(3.0)


def two_ended(S: np.ndarray, Rr: np.ndarray, dS: np.ndarray, dR: np.ndarray, R1: float, L1: float, sl: slice) -> float:
    """d from S: vS - vR + R iR + L iR' = d [R (iS + iR) + L (iS' + iR')], aerial modes alpha, beta."""
    num = den = 0.0
    for k in (0, 1):
        iS, iR = clarke(S[:, 0:3])[k][sl], clarke(Rr[:, 0:3])[k][sl]
        vS, vR = clarke(S[:, 3:6])[k][sl], clarke(Rr[:, 3:6])[k][sl]
        diS, diR = clarke(dS)[k][sl], clarke(dR)[k][sl]
        a = vS - vR + R1 * iR + L1 * diR
        b = R1 * (iS + iR) + L1 * (diS + diR)
        num += float(a @ b)
        den += float(b @ b)
    return num / den


def pos_seq(p: np.ndarray) -> complex:
    return (p[0] + C.A_OP * p[1] + C.A_OP ** 2 * p[2]) / 3.0


def work(r: dict) -> tuple[list[dict], dict]:
    sig = np.load(CACHE / f"ep{r['sim_idx']}.npy")
    line = r["line"]
    bx, by = buses(line)
    Z1, Z0 = line_z(line)
    has_x = not np.isnan(sig[:, 0]).any()
    has_y = not np.isnan(sig[:, 6]).any()
    if has_x:
        loc_sig, flip, local_bus = sig[:, 0:6], False, bx
    else:  # MainLn8-14: no cubicle at bus 8 -> local terminal bus 14, y_local = 1 - y
        loc_sig, flip, local_bus = sig[:, 6:12], True, by
    y = r["loc"] / 100.0
    y_local = 1.0 - y if flip else y
    dloc = savgol_filter(loc_sig, 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)
    two = has_x and has_y
    if two:
        S, Rr = sig[:, 0:6], sig[:, 6:12]
        dS = savgol_filter(S[:, 0:3], 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)
        dR = savgol_filter(Rr[:, 0:3], 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)
    nf = math.floor((r["start"] - 1.0) * C.FS)
    D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
    ends = [s1 for s1 in range(C.WIN, len(sig) + 1, 48) if s1 > nf and s1 - C.WIN < nf + D]
    rows = []
    for s1 in ends:
        row = {"sim_idx": r["sim_idx"], "s1": s1, "nf": nf, "pft_ms": (s1 - nf) / 9.6, "etype": r["etype"],
               "ftype": C.ftype(r["etype"]), "line": line, "local_bus": local_bus, "flipped": flip, "res": r["res"],
               "y": y, "y_local": y_local, "oracle": C.oracle_loop(r["etype"], r["ph1"], r["ph2"])}
        row.update(local_tokens(loc_sig, s1, Z1, Z0))
        row.update(td_tokens(loc_sig, dloc, s1, Z1, Z0))
        row["two_ended"] = two_ended(S, Rr, dS, dR, Z1.real, Z1.imag / W0, slice(s1 - C.WIN, s1)) if two else np.nan
        rows.append(row)
    zid = {"sim_idx": r["sim_idx"], "line": line}
    if two:
        st = nf - 2 * C.N_CYC  # pre-fault cycle ending one cycle before inception
        pS, pR = phasor(sig[:, 0:6], st), phasor(sig[:, 6:12], st)
        I1S, V1S, I1R, V1R = pos_seq(pS[0:3]), pos_seq(pS[3:6]), pos_seq(pR[0:3]), pos_seq(pR[3:6])
        Zi = (V1S - V1R) / ((I1S - I1R) / 2.0)
        zid.update(R1_id=Zi.real, X1_id=Zi.imag, I1S=abs(I1S), I1R=abs(I1R), through_ratio=abs(I1S + I1R) / (abs(I1S) + 1e-9))
    return rows, zid


def summarise(err: np.ndarray, d: pd.DataFrame, method: str) -> dict:
    pft = d["pft_ms"].to_numpy()
    sc = d["ftype"].str.endswith("shc").to_numpy() | d["ftype"].str.endswith("shc_w_arc").to_numpy()
    row = {"method": method, "n": len(d), "MAE": err.mean(), "median": np.median(err),
           "MAE_pft_le15": err[pft <= 15].mean(), "MAE_pft_ge21": err[pft >= 21].mean(),
           "MAE_shortcircuit": err[sc].mean()}
    for f in FT:
        row[f"MAE_{f}"] = err[(d["ftype"] == f).to_numpy()].mean()
    return row


def main() -> None:
    e = episodes()
    print("episodes", len(e), flush=True)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        st = list(ex.map(extract, e["sim_idx"].tolist(), e["line"].tolist(), chunksize=4))
    print("extract:", pd.Series(st).value_counts().to_dict(), flush=True)
    tok_path = R / "validate_ha2_mv_tokens.parquet"
    if tok_path.exists():
        df = pd.read_parquet(tok_path)
        zid = pd.read_csv(R / "validate_ha2_mv_zid_episodes.csv")
    else:
        with ProcessPoolExecutor(max_workers=NW) as ex:
            res = list(ex.map(work, e.to_dict("records"), chunksize=8))
        df = pd.DataFrame([x for rr, _ in res for x in rr])
        zid = pd.DataFrame([z for _, z in res])
        assert np.isfinite(df[LOCAL_TOKENS + TD_TOKENS].to_numpy()).all()
        df.to_parquet(tok_path, index=False)
        zid.to_csv(R / "validate_ha2_mv_zid_episodes.csv", index=False)
    counts = {"episodes": e["sim_idx"].nunique(), "windows": len(df),
              "windows_two_ended": int(df["two_ended"].notna().sum()),
              "windows_8_14": int((df["line"] == "MainLn8-14").sum()),
              "episodes_flipped": int(df.loc[df["flipped"], "sim_idx"].nunique())}
    counts.update({f"windows_{f}": int((df["ftype"] == f).sum()) for f in FT})
    print(counts, flush=True)
    rows = []
    # one-ended physics vs y_local
    idx = df["oracle"].map({lp: k for k, lp in enumerate(LOOPS)}).to_numpy().astype(int)
    react = df[[f"react_{lp}" for lp in LOOPS]].to_numpy()
    tak2 = df[[f"tak2_{lp}" for lp in LOOPS]].to_numpy()
    absz = df[[f"absz_{lp}" for lp in LOOPS]].to_numpy()
    rr = np.arange(len(df))
    phys = {"phys_react_oracle": react[rr, idx], "phys_tak2_oracle": tak2[rr, idx],
            "phys_react_minabsz": react[rr, absz.argmin(axis=1)]}
    yl = df["y_local"].to_numpy()
    preds = {}
    for m, p in phys.items():
        preds[m] = np.clip(p, 0, 1)
        rows.append(summarise(np.abs(preds[m] - yl) * 100, df, m))
    # two-ended (14 segments), vs y from bus X
    te = df["two_ended"].notna().to_numpy()
    d2 = df[te]
    p2 = np.clip(d2["two_ended"].to_numpy(), 0, 1)
    rows.append(summarise(np.abs(p2 - d2["y"].to_numpy()) * 100, d2, "two_ended_14seg"))
    # H-A / H-A2 zero-shot, train on DL + TG pooled
    tr = pd.concat([pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet") for g in ("DL", "TG")], ignore_index=True)
    print("train windows", len(tr), flush=True)
    for meth, cols in (("HA", list(LOCAL_TOKENS)), ("HA2", list(LOCAL_TOKENS) + list(TD_TOKENS))):
        assert_allow_list(cols)
        P = []
        for seed in (0, 1, 2):
            mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(
                tr[cols].to_numpy(np.float64), tr["y"].to_numpy())
            P.append(mdl.predict(df[cols].to_numpy(np.float64)))
        single = [float(np.abs(np.clip(p, 0, 1) - yl).mean() * 100) for p in P]
        preds[meth] = np.clip(np.mean(P, axis=0), 0, 1)
        rows.append({**summarise(np.abs(preds[meth] - yl) * 100, df, meth),
                     "single_seed_mean": np.mean(single), "single_seed_sd": np.std(single, ddof=1)})
        print(meth, rows[-1]["MAE"], flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_mv_summary.csv", index=False)
    pr = df[["sim_idx", "s1", "pft_ms", "ftype", "line", "res", "flipped", "y", "y_local", "two_ended"]].copy()
    for k, v in preds.items():
        pr[k] = v
    pr.to_csv(R / "validate_ha2_mv_pred.csv", index=False)
    pd.Series(counts).to_csv(R / "validate_ha2_mv_counts.csv", header=["value"])

    # 8-14 orientation check: bolted (1 ohm) short circuits, pft >= 25 ms, oracle reactance vs y_local and vs y
    o = (df["line"] == "MainLn8-14") & (df["res"] == 1.0) & df["ftype"].str.contains("shc") & (df["pft_ms"] >= 25)
    pr_o = preds["phys_react_oracle"][o.to_numpy()]
    orient = {"n_windows": int(o.sum()),
              "MAE_vs_y_local(1-y)": float(np.abs(pr_o - df.loc[o, "y_local"]).mean() * 100),
              "MAE_vs_y_unflipped": float(np.abs(pr_o - df.loc[o, "y"]).mean() * 100)}
    o2 = (df["line"] != "MainLn8-14") & (df["res"] == 1.0) & df["ftype"].str.contains("shc") & (df["pft_ms"] >= 25)
    orient["reference_other_lines_MAE_vs_y"] = float(np.abs(preds["phys_react_oracle"][o2.to_numpy()]
                                                            - df.loc[o2, "y_local"]).mean() * 100)
    pd.Series(orient).to_csv(R / "validate_ha2_mv_orientation.csv", header=["value"])

    # R_f diagnosis: short circuits (not hif / incipient), pft >= 25 ms
    sc = df["ftype"].str.endswith("shc") | df["ftype"].str.endswith("shc_w_arc")
    q = (sc & (df["pft_ms"] >= 25)).to_numpy()
    dq = pr[q].copy()
    dq["e_HA2"] = np.abs(dq["HA2"] - dq["y_local"]) * 100
    dq["e_react"] = np.abs(dq["phys_react_oracle"] - dq["y_local"]) * 100
    dq["e_two"] = np.abs(dq["two_ended"].clip(0, 1) - dq["y"]) * 100
    diag = dq.groupby("res").agg(n=("e_HA2", "size"), HA2=("e_HA2", "mean"), react_oracle=("e_react", "mean"),
                                 two_ended=("e_two", "mean"), n_two=("e_two", "count"))
    diag.to_csv(R / "validate_ha2_mv_rf_diag.csv")

    # Z1 identification per segment
    zid = zid.dropna(subset=["R1_id"])
    zs = zid.groupby("line").agg(n=("R1_id", "size"), R1_id_med=("R1_id", "median"), X1_id_med=("X1_id", "median"),
                                 R1_id_iqr=("R1_id", lambda v: v.quantile(.75) - v.quantile(.25)),
                                 X1_id_iqr=("X1_id", lambda v: v.quantile(.75) - v.quantile(.25)),
                                 I1S_med=("I1S", "median"), through_ratio_med=("through_ratio", "median"))
    zs["R1_np"] = [line_z(ln)[0].real for ln in zs.index]
    zs["X1_np"] = [line_z(ln)[0].imag for ln in zs.index]
    zs["R1_ratio"] = zs["R1_id_med"] / zs["R1_np"]
    zs["X1_ratio"] = zs["X1_id_med"] / zs["X1_np"]
    zs.to_csv(R / "validate_ha2_mv_zid.csv")
    # per-line two-ended
    bl = pr[te].assign(e=np.abs(pr.loc[te, "two_ended"].clip(0, 1) - pr.loc[te, "y"]) * 100).groupby("line")["e"].mean()
    bl.to_csv(R / "validate_ha2_mv_two_ended_by_line.csv")
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(s.round(3).to_string(index=False))
        print(pd.Series(orient).round(3).to_string())
        print(diag.round(3).to_string())
        print(zs.round(4).to_string())
        print(bl.round(3).to_string())


if __name__ == "__main__":
    main()
