"""In-grid protocol on adapt_grid-DoubleLine (claudedocs/validator_spec_ingrid.md), CPU only.

Steps (each cached / resumable):
  1. window-rule counts: benchmark family (all faults vs line faults) and adapt test (all faults vs line faults);
     check that the rule reproduces my 14,640 benchmark windows window-for-window;
  2. cache the 6 local channels of every TRAIN/TEST line-fault episode (results/validate_ha2_epcache/AD/);
  3. 77 tokens at every rule window end s1 -> results/validate_ha2_ingrid_tokens.parquet;
  4. H-A (59) / H-A2 (77) HGB trained on TRAIN line-fault windows only; physics baselines;
     tested on (a) adapt TEST line-fault windows, (b) benchmark-family 14,640 windows (cached tokens).
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from parse_episodes import extract  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, TD_TOKENS, local_tokens, td_tokens  # noqa: E402

R = C.RESULTS
SPLITS = C.ROOT / "external" / "evemtbench-benchmark" / "src" / "evemtbench" / "splits" / "v1.0" / "held_out" / "double_line"
SC_D = 289
FT = ["1phg_incipient", "1phg_incipient_w_arc", "1phg_shc", "1phg_shc_w_arc", "2ph_shc", "2phg_shc", "3ph_shc"]


def rule_windows(start: float, etype: str, dur: float, n: int = C.N_SAMPLES) -> list[int]:
    nf = math.floor((start - 1.0) * C.FS)
    D = math.floor(dur * C.FS) if "incipient" in etype else SC_D
    return [s1 for s1 in range(C.WIN, n + 1, 48) if s1 > nf and s1 - C.WIN < nf + D]


def labels(grid: str) -> pd.DataFrame:
    return pd.read_csv(C.RAW / C.GRIDS[grid] / "labels" / "settings_clean.csv")


def split_ids(name: str) -> set[int]:
    ids = set()
    for ln in (SPLITS / f"{name}.txt").read_text().split():
        fam, grid, idx = ln.strip().split(":")
        assert fam == "adapt_grid" and grid == "double_line", ln
        ids.add(int(idx))
    return ids


def count_windows(df: pd.DataFrame) -> int:
    return int(sum(len(rule_windows(r[C.C_START], r[C.C_TYPE], r[C.C_DUR])) for _, r in df.iterrows()))


def step1_counts() -> dict:
    out = {}
    b = labels("DL")
    flt = b[C.C_TYPE].astype(str).str.startswith("flt_")
    line = flt & b[C.C_TARGET].astype(str).str.startswith("MainLn") & b[C.C_LOC].notna()
    out["bench_all_fault_episodes"] = int(flt.sum())
    out["bench_all_fault_windows"] = count_windows(b[flt])
    out["bench_line_episodes"] = int(line.sum())
    out["bench_line_windows"] = count_windows(b[line])
    # window-for-window check against my cached tau windows
    mine = pd.read_parquet(R / "validate_ha2_tokens_DL.parquet", columns=["sim_idx", "s1"])
    rule = {(int(r[C.C_IDX]), s1) for _, r in b[line].iterrows()
            for s1 in rule_windows(r[C.C_START], r[C.C_TYPE], r[C.C_DUR])}
    cached = set(zip(mine["sim_idx"].astype(int), mine["s1"].astype(int)))
    out["bench_rule_eq_cached_windows"] = rule == cached
    a = labels("AD")
    test, train, val = split_ids("test"), split_ids("train"), split_ids("val")
    out["split_sizes"] = f"train {len(train)} val {len(val)} test {len(test)}"
    out["splits_disjoint"] = not (train & test or train & val or val & test)
    aflt = a[C.C_TYPE].astype(str).str.startswith("flt_")
    aline = aflt & a[C.C_TARGET].astype(str).str.startswith("MainLn") & a[C.C_LOC].notna()
    for nm, ids in (("test", test), ("train", train)):
        ins = a[C.C_IDX].isin(ids)
        out[f"adapt_{nm}_all_fault_episodes"] = int((ins & aflt).sum())
        out[f"adapt_{nm}_all_fault_windows"] = count_windows(a[ins & aflt])
        out[f"adapt_{nm}_line_episodes"] = int((ins & aline).sum())
        out[f"adapt_{nm}_line_windows"] = count_windows(a[ins & aline])
    out["adapt_bus_faults_with_location"] = int((aflt & ~a[C.C_TARGET].astype(str).str.startswith("MainLn")
                                                 & a[C.C_LOC].notna()).sum())
    return out


def adapt_episodes() -> pd.DataFrame:
    a = labels("AD")
    test, train = split_ids("test"), split_ids("train")
    m = (a[C.C_TYPE].astype(str).str.startswith("flt_") & a[C.C_TARGET].astype(str).str.startswith("MainLn")
         & a[C.C_LOC].notna())
    e = a.loc[m & a[C.C_IDX].isin(train | test),
              [C.C_IDX, C.C_TYPE, C.C_START, C.C_TARGET, C.C_LOC, C.C_RES, C.C_PH1, C.C_PH2, C.C_DUR]].copy()
    e.columns = ["sim_idx", "etype", "start", "line", "loc", "res", "ph1", "ph2", "dur"]
    e["sim_idx"] = e["sim_idx"].astype(int)
    e["split"] = np.where(e["sim_idx"].isin(train), "train", "test")
    assert set(e["line"]) <= set(C.LINE_LEN["AD"]), set(e["line"])
    return e.sort_values("sim_idx").reset_index(drop=True)


def step2_parse(e: pd.DataFrame) -> None:
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(extract, "AD", int(r.sim_idx), r.line) for r in e.itertuples()]
        for f in as_completed(futs):
            f.result()


def tok_work(r: dict) -> list[dict]:
    sig = np.load(C.CACHE / "AD" / f"ep{r['sim_idx']}.npy")
    dsig = savgol_filter(sig, window_length=11, polyorder=2, deriv=1, delta=1.0 / C.FS, axis=0)
    Z1, Z0 = C.line_z("AD", r["line"])
    nf = math.floor((r["start"] - 1.0) * C.FS)
    rows = []
    for s1 in rule_windows(r["start"], r["etype"], r["dur"], len(sig)):
        row = {"sim_idx": r["sim_idx"], "split": r["split"], "s1": s1, "nf": nf, "pft_ms": (s1 - nf) / 9.6,
               "etype": r["etype"], "ftype": C.ftype(r["etype"]), "line": r["line"], "res": r["res"],
               "y": r["loc"] / 100.0, "oracle": C.oracle_loop(r["etype"], r["ph1"], r["ph2"])}
        row.update(local_tokens(sig, s1, Z1, Z0))
        row.update(td_tokens(sig, dsig, s1, Z1, Z0))
        rows.append(row)
    return rows


def step3_tokens(e: pd.DataFrame) -> pd.DataFrame:
    path = R / "validate_ha2_ingrid_tokens.parquet"
    if path.exists():
        return pd.read_parquet(path)
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(tok_work, e.to_dict("records"), chunksize=8))
    df = pd.DataFrame([x for rr in res for x in rr])
    assert np.isfinite(df[LOCAL_TOKENS + TD_TOKENS].to_numpy()).all()
    df.to_parquet(path, index=False)
    return df


def summarise(te: pd.DataFrame, pred: np.ndarray, method: str, testset: str) -> dict:
    err = np.abs(np.clip(pred, 0, 1) - te["y"].to_numpy()) * 100
    pft = te["pft_ms"].to_numpy()
    sc = ~te["ftype"].str.contains("incipient").to_numpy()
    row = {"method": method, "testset": testset, "n": len(te), "MAE": err.mean(), "median": np.median(err),
           "MAE_pft_le15": err[pft <= 15].mean(), "MAE_pft_ge21": err[pft >= 21].mean(),
           "MAE_shortcircuit": err[sc].mean(), "n_pft_le15": int((pft <= 15).sum()), "n_pft_ge21": int((pft >= 21).sum())}
    for f in FT:
        row[f"MAE_{f}"] = err[(te["ftype"] == f).to_numpy()].mean()
    return row


def physics_preds(te: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = te["oracle"].map({lp: k for k, lp in enumerate(LOOPS)}).to_numpy().astype(int)
    react = te[[f"react_{lp}" for lp in LOOPS]].to_numpy()
    tak2 = te[[f"tak2_{lp}" for lp in LOOPS]].to_numpy()
    absz = te[[f"absz_{lp}" for lp in LOOPS]].to_numpy()
    r = np.arange(len(te))
    return {"phys_react_oracle": react[r, idx], "phys_tak2_oracle": tak2[r, idx],
            "phys_react_minabsz": react[r, absz.argmin(axis=1)]}


def main() -> None:
    cnt = step1_counts()
    pd.Series(cnt).to_csv(R / "validate_ha2_ingrid_counts.csv", header=["value"])
    print(pd.Series(cnt).to_string(), flush=True)
    e = adapt_episodes()
    print("adapt line-fault episodes:", e["split"].value_counts().to_dict(), flush=True)
    step2_parse(e)
    df = step3_tokens(e)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te_a = df[df["split"] == "test"].reset_index(drop=True)
    assert not set(tr["sim_idx"]) & set(te_a["sim_idx"])
    te_b = pd.read_parquet(R / "validate_ha2_tokens_DL.parquet")
    te_b["pft_ms"] = (te_b["s1"] - te_b["nf"]) / 9.6
    assert np.allclose(te_b["pft_ms"], te_b["tau"])
    print("windows: train", len(tr), "test(adapt)", len(te_a), "test(benchmark)", len(te_b), flush=True)
    tests = {"adapt_test": te_a, "benchmark_DL": te_b}
    rows = []
    for name, te in tests.items():
        for m, p in physics_preds(te).items():
            rows.append(summarise(te, p, m, name))
    for meth, cols in (("HA", list(LOCAL_TOKENS)), ("HA2", list(LOCAL_TOKENS) + list(TD_TOKENS))):
        assert_allow_list(cols)
        preds = {k: [] for k in tests}
        for seed in (0, 1, 2):
            mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(
                tr[cols].to_numpy(np.float64), tr["y"].to_numpy())
            for k, te in tests.items():
                p = mdl.predict(te[cols].to_numpy(np.float64))
                preds[k].append(p)
                rows.append({**summarise(te, p, meth, k), "seed": seed})
        for k, te in tests.items():
            p = np.clip(np.mean(preds[k], axis=0), 0, 1)
            rows.append({**summarise(te, p, meth, k), "seed": "avg"})
            out = te[["sim_idx", "s1", "pft_ms", "ftype", "y"]].copy()
            out["pred"] = p
            out["err"] = np.abs(p - out["y"]) * 100
            out.to_csv(R / f"validate_ha2_ingrid_pred_{meth}_{k}.csv", index=False)
            print(meth, k, f"MAE {out['err'].mean():.3f}", flush=True)
    s = pd.DataFrame(rows)
    s["seed"] = s["seed"].fillna("na").astype(str)
    s.to_csv(R / "validate_ha2_ingrid_summary.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(s[s["seed"].isin(["avg", "na"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
