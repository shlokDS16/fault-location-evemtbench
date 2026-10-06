"""Design section 19 ablations (report-only), zero-shot TG->DL and DL->TG, fixed HGB settings, seeds 0-2.
A1 HGB on raw 480x6 local windows (-raw / +Z); A2 phasor-only (59) / TD-only (18) / both (77);
A3 un-normalised apparent-impedance tokens (zr, zi, absz in ohm = z * Z1). Inputs from grid_pass.py outputs.
Resumable: one row per (ablation, direction) appended to results/c1_ablations.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import ALLOW, LOCAL_TOKENS, TD_TOKENS  # noqa: E402
from local_features import LOOPS  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "c1_ablations.csv"


def unnormalise(df):
    d = df.copy()
    for lp in LOOPS:
        zr, zi = d[f"{lp}_zr"], d[f"{lp}_zi"]
        d[f"{lp}_zr"] = zr * d.Z1r - zi * d.Z1x
        d[f"{lp}_zi"] = zr * d.Z1x + zi * d.Z1r
        d[f"{lp}_absz"] = d[f"{lp}_absz"] * np.hypot(d.Z1r, d.Z1x)
    return d


def fit_eval(Xtr, ytr, Xte, yte, pm):
    P = np.stack([HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(Xtr, ytr).predict(Xte) for s in (0, 1, 2)])
    err = np.abs(np.clip(P.mean(0), 0, 1) - yte) * 100
    return dict(mae=err.mean(), median=np.median(err), post_le15=err[pm <= 15].mean(), post_ge21=err[pm >= 21].mean())


def main():
    tab = {g: pd.read_parquet(R / f"c1_grid_{g}.parquet") for g in ("DL", "TG")}
    raw = {g: np.load(R / f"c1_grid_{g}_raw.npz") for g in ("DL", "TG")}
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.ablation, prev.direction))
    jobs = [("A2_phasor_only", LOCAL_TOKENS), ("A2_td_only", TD_TOKENS), ("A2_both_HA2", ALLOW),
            ("A3_unnormalised", ALLOW), ("A1_raw_window", None), ("A1_raw_window+Z", None)]
    for name, cols in jobs:
        for src, tgt in (("TG", "DL"), ("DL", "TG")):
            key = (name, f"{src}_to_{tgt}")
            if key in done:
                continue
            t0 = time.time()
            tr, te = tab[src], tab[tgt]
            pm = te.post_ms.values
            if cols is not None:
                check_allow(cols)
                if name == "A3_unnormalised":
                    tr, te = unnormalise(tr), unnormalise(te)
                r = fit_eval(tr[cols], tr.y.values, te[cols], te.y.values, pm)
            else:
                Xtr = raw[src]["X"].reshape(len(raw[src]["X"]), -1)
                Xte = raw[tgt]["X"].reshape(len(raw[tgt]["X"]), -1)
                if name.endswith("+Z"):
                    Xtr, Xte = np.hstack([Xtr, raw[src]["Z"]]), np.hstack([Xte, raw[tgt]["Z"]])
                assert np.array_equal(raw[tgt]["y"], te.y.values)
                r = fit_eval(Xtr, raw[src]["y"], Xte, raw[tgt]["y"], pm)
            row = dict(ablation=name, direction=key[1], **r, sec=time.time() - t0)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(row, flush=True)
    print(pd.read_csv(OUT).pivot(index="ablation", columns="direction", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
