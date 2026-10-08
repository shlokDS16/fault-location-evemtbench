"""Design section 26(c) (report-only, professor M6): main-table MAE recomputed without the windows whose true location
is d = 0.5 (all incipient and HIF episodes sit there), from the existing per-window predictions (no retraining).
Phasor two-ended estimates are recomputed per window with review_checks.one (its stored MAE is the paper's).
Usage: python src/c1/prof_m6.py  -> results/c1_prof_m6.csv, results/c1_prof_m6_extra.csv (+ checks vs paper macros)"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revamp_stats import load_sets, err, key  # noqa: E402
import review_checks as rc  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
MAC = Path(__file__).resolve().parents[2] / "paper" / "macros_auto.tex"
HALF = 1e-9


def phasor(tag):
    with ProcessPoolExecutor(3) as ex:
        d = pd.DataFrame([r for rr in ex.map(rc.one, rc.jobs(tag), chunksize=4) for r in rr])
    return key(d)[["sim_idx", "k", "ph2"]]


def rl_id(tag):
    f = "c1_adapt_tokens" if tag == "ADAPT" else f"c1_grid_{tag}"
    t = pd.read_parquet(R / f"{f}.parquet")
    if tag == "ADAPT":
        t = t[t.split == "test"]
    est = np.array([t[f"{ol}_dtd"].iloc[i] for i, ol in enumerate(t.oracle_loop.values)])
    return key(pd.DataFrame(dict(sim_idx=t.sim_idx.values, post_ms=t.post_ms.values, rl=est)))[["sim_idx", "k", "rl"]]


def main():
    sets = load_sets()
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", MAC.read_text()))
    rows, extra = [], []
    for tag, mk in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
        s = key(sets[tag])
        c = key(pd.read_parquet(R / f"c1_classical_{tag}.parquet"))[["sim_idx", "k", "etype", "R", "T"]]
        s = s.merge(c, on=["sim_idx", "k"], validate="one_to_one")
        s = s.merge(rl_id(tag), on=["sim_idx", "k"], how="left", validate="one_to_one")
        s = s.merge(phasor(tag), on=["sim_idx", "k"], how="left", validate="one_to_one")
        y = s.y.values
        cols = {"TD two-ended": s.two.values,
                "phasor two-ended": np.where(s.ph2.notna() | s.two.notna(), err(s.ph2, y), np.nan),
                "reactance": err(s.R, y), "Takagi": err(s["T"], y), "Eriksson": s.E.values,
                "R-L identification": err(s.rl, y), "PFB": s.ha2.values}
        if "eq" in s:
            cols["best MLP/GRU equal inform."] = s["eq"].values
        if tag == "MV":  # two-ended rows only on the sections measured at both ends
            cols["phasor two-ended"] = np.where(np.isfinite(s.two.values), cols["phasor two-ended"], np.nan)
        half = np.abs(y - 0.5) < HALF
        for m, e in cols.items():
            ok = np.isfinite(e)
            rows.append(dict(set=tag, method=m, n_all=int(ok.sum()), mae_all=float(e[ok].mean()),
                             n_excl=int((ok & ~half).sum()), mae_excl=float(e[ok & ~half].mean())))
        hifinc = s.etype.str.contains("incipient|hif").values
        extra.append(dict(set=tag, n=len(s), share_half=100 * half.mean(),
                          share_hifinc_at_half=100 * (hifinc & half).sum() / max(hifinc.sum(), 1),
                          share_E_undef_and_half=100 * ((~s.E_def.values) & half).mean(),
                          pfb_mae_inc_hif=float(s.ha2[hifinc].mean()), pfb_mae_sc=float(s.ha2[~hifinc].mean()),
                          const_half_mae_inc_hif=float(np.abs(0.5 - y[hifinc]).mean() * 100)))
    out = pd.DataFrame(rows)
    out.to_csv(R / "c1_prof_m6.csv", index=False)
    pd.DataFrame(extra).to_csv(R / "c1_prof_m6_extra.csv", index=False)
    # reproduction check of the full-set values against the paper macros
    ref = {("DL", "TD two-ended"): "teMaeDL", ("TG", "TD two-ended"): "teMaeTG", ("ADAPT", "TD two-ended"): "teMaeAdaptAll",
           ("MV", "TD two-ended"): "teMaeMV", ("ADAPT", "PFB"): "haTwoInAdapt", ("DL", "PFB"): "haTwoZsTGDL",
           ("TG", "PFB"): "haTwoZsDLTG", ("MV", "PFB"): "haTwoMV", ("DL", "best MLP/GRU equal inform."): "eqBestTGDL",
           ("TG", "best MLP/GRU equal inform."): "eqBestDLTG"}
    for t, k in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
        ref.update({(t, "phasor two-ended"): f"phMae{k}", (t, "reactance"): f"clR{k}", (t, "Takagi"): f"clT{k}",
                    (t, "Eriksson"): f"clE{k}", (t, "R-L identification"): f"suonan{k}"})
    o = out.set_index(["set", "method"])
    for (t, m), k in ref.items():
        if (t, m) in o.index and k in mac:
            print(f"check {t:5s} {m:28s} recomputed {o.loc[(t, m), 'mae_all']:7.2f}  macro {k}={mac[k]}")
    print(out.round(2).to_string(index=False))
    print(pd.DataFrame(extra).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
