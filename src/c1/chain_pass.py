"""Design section 22: measurement-chain robustness on DL and TG (110 kV).
Pass: grid_pass.one with a meas_chain.Chain applied to every terminal record before windowing
  -> results/c1_chain_<scen>_<DL|TG>.parquet (same columns as c1_grid_*, plus 'sat' for CT scenarios).
Eval: two-ended TD locator, physics one-ended (oracle reactance, Takagi-I2), H-A2 trained on the CLEAN source grid
  (c1_grid_<src>.parquet) and tested on the distorted target; H-A2 trained and tested on FULL (matched).
Usage: python src/c1/chain_pass.py [pass|eval|all]  -> results/c1_chain_eval.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from meas_chain import Chain, ct, CT_MILD, CT_SEVERE  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from tokens_core import ALLOW  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
SCEN = ["AA", "CT-mild", "CT-severe", "CVT-5", "CVT-15", "FULL"]
GRIDS = {"DL": "DoubleLine", "TG": "TestGrid110kV"}


def sat_flags(row, gdir, ends, p):
    """Per window: max|i_e| > 0.05 max|i_s| on any local phase (design 22)."""
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
    cs = terminal_cols(pd.read_csv(f, nrows=0).columns.tolist(), m.group(1), line)[:3]
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    res = [ct(d[c].to_numpy(float), **p)[1:] for c in cs]
    out = []
    for s1 in ends:
        a = s1 - grid_pass.WLEN
        out.append(int(any(np.abs(ie[a:s1]).max() > 0.05 * np.abs(is_[a:s1]).max() for ie, is_ in res)))
    return out


def one(args):
    row, LP, gdir, scen = args
    recs, _ = grid_pass.one((row, LP, gdir, Chain(scen)))
    if scen in ("CT-mild", "CT-severe", "FULL"):
        nf = int(np.floor((row["events/event_start"] - 1.0) * grid_pass.FS))
        ends = [nf + int(round(r["post_ms"] / 1000 * grid_pass.FS)) for r in recs]
        for r, s in zip(recs, sat_flags(row, gdir, ends, CT_MILD if scen == "CT-mild" else CT_SEVERE)):
            r["sat"] = s
    return recs


def run_pass():
    for tag, grid in GRIDS.items():
        gdir = ROOT / "data" / "raw" / "evemt" / grid
        LP = grid_pass.line_params(gdir)
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
        fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(grid_pass.LINE_RE)
               & s["events/event_flt_target_line_location"].notna()]
        clean = pd.read_parquet(R / f"c1_grid_{tag}.parquet")
        for scen in SCEN:
            out = R / f"c1_chain_{scen}_{tag}.parquet"
            if out.exists():
                continue
            with ProcessPoolExecutor(6) as ex:
                df = pd.DataFrame([r for rr in ex.map(one, [(r, LP, gdir, scen) for _, r in fl.iterrows()], chunksize=4)
                                   for r in rr])
            # alignment with the clean table (same windows, same order)
            assert len(df) == len(clean) and (df.sim_idx.values == clean.sim_idx.values).all()
            assert np.allclose(df.post_ms.values, clean.post_ms.values)
            df.to_parquet(out, index=False)
            print(f"{tag} {scen}: {len(df)} windows" + (f", saturated {df.sat.mean() * 100:.1f} %" if "sat" in df else ""),
                  flush=True)


def mae(p, y):
    return float(np.mean(np.abs(np.clip(p, 0, 1) - y)) * 100)


def hgb(tr, te_list):
    check_allow(ALLOW)
    models = [HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[ALLOW], tr.y) for s in (0, 1, 2)]
    return [np.mean([m.predict(te[ALLOW]) for m in models], 0) for te in te_list]


def run_eval():
    rows = []
    tabs = {(sc, g): pd.read_parquet(R / f"c1_chain_{sc}_{g}.parquet") for sc in SCEN for g in GRIDS}
    for g in GRIDS:
        tabs[("clean", g)] = pd.read_parquet(R / f"c1_grid_{g}.parquet")
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        allsc = ["clean"] + SCEN
        preds = dict(zip(allsc, hgb(tabs[("clean", src)], [tabs[(sc, tgt)] for sc in allsc])))
        matched = hgb(tabs[("FULL", src)], [tabs[("FULL", tgt)]])[0]
        for sc in allsc:
            te = tabs[(sc, tgt)]
            y, pm = te.y.values, te.post_ms.values
            r = dict(direction=f"{src}_to_{tgt}", scenario=sc, n=len(te),
                     sat_pct=float(te.sat.mean() * 100) if "sat" in te else np.nan,
                     two_ended=mae(te.td2_est.values, y), react_oracle=mae(te.phys_react_oracle.values, y),
                     tak2_oracle=mae(te.phys_tak2_oracle.values, y), HA2_clean_trained=mae(preds[sc], y),
                     HA2_post_le15=mae(preds[sc][pm <= 15], y[pm <= 15]),
                     HA2_post_ge21=mae(preds[sc][pm >= 21], y[pm >= 21]))
            if sc == "FULL":
                r["HA2_matched_FULL"] = mae(matched, y)
            if "sat" in te:
                k = te.sat.values == 1
                r["HA2_sat_windows"] = mae(preds[sc][k], y[k]) if k.any() else np.nan
                r["two_ended_sat_windows"] = mae(te.td2_est.values[k], y[k]) if k.any() else np.nan
            rows.append(r)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(R / "c1_chain_eval.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("pass", "all"):
        run_pass()
    if what in ("eval", "all"):
        run_eval()
