"""Design section 21 report: classical one-ended locators vs H-A2 on the same windows, by post-fault time.
H-A2 predictions: zero-shot TG->DL / DL->TG (re-fitted here, same settings as ha2_eval), in-grid (adapt_ingrid preds),
CIGRE MV zero-shot (zs_eval preds). NaN classical estimates are scored as 0.5 (design 21a).
Usage: python src/c1/classical_report.py  -> results/c1_classical_report.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import ALLOW  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
M = ["R", "T", "T2", "MT", "E"]
BINS = [(0, 17.5, "post<=15"), (17.5, 32.5, "20-30"), (32.5, 52.5, "35-50"), (52.5, 1e9, ">=55")]


def zs_pred(src, tgt):
    check_allow(ALLOW)
    tr, te = pd.read_parquet(R / f"c1_grid_{src}.parquet"), pd.read_parquet(R / f"c1_grid_{tgt}.parquet")
    p = np.mean([HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[ALLOW], tr.y).predict(te[ALLOW])
                 for s in (0, 1, 2)], 0)
    return te[["sim_idx", "post_ms"]].assign(pred=p)


def join(cl, pr):
    a, b = cl.assign(k=cl.post_ms.round(3)), pr.assign(k=pr.post_ms.round(3))[["sim_idx", "k", "pred"]]
    out = a.merge(b, on=["sim_idx", "k"], how="inner", validate="one_to_one")
    assert len(out) == len(cl), (len(out), len(cl))
    return out


def summarise(d, setting):
    rows = []
    sc = ~d.etype.str.contains("incipient|hif").values
    for m in M + ["pred"]:
        p = d[m].fillna(0.5).clip(0, 1).values
        e = np.abs(p - d.y.values) * 100
        r = dict(setting=setting, method="H-A2" if m == "pred" else m, n=len(d), n_nan=int(d[m].isna().sum()),
                 mae=e.mean(), median=np.median(e), mae_sc=e[sc].mean())
        for lo, hi, nm in BINS:
            k = (d.post_ms.values > lo) & (d.post_ms.values <= hi)
            r[nm] = e[k].mean()
        rows.append(r)
    return rows


def main():
    rows = []
    for tgt, src in (("DL", "TG"), ("TG", "DL")):
        rows += summarise(join(pd.read_parquet(R / f"c1_classical_{tgt}.parquet"), zs_pred(src, tgt)), f"zero-shot {src}->{tgt}")
    rows += summarise(join(pd.read_parquet(R / "c1_classical_DL.parquet"), pd.read_csv(R / "c1_adapt_ingrid_pred_benchmark.csv")),
                      "in-grid benchmark DL")
    rows += summarise(join(pd.read_parquet(R / "c1_classical_ADAPT.parquet"), pd.read_csv(R / "c1_adapt_ingrid_pred_adapt_test.csv")),
                      "in-grid adapt test")
    rows += summarise(join(pd.read_parquet(R / "c1_classical_MV.parquet"), pd.read_csv(R / "c1_zs_MV_pred.csv")),
                      "zero-shot DL+TG->MV")
    df = pd.DataFrame(rows)
    df.to_csv(R / "c1_classical_report.csv", index=False)
    for s, g in df.groupby("setting", sort=False):
        best = g[g.method != "H-A2"].sort_values("mae").iloc[0]
        ha2 = g[g.method == "H-A2"].iloc[0]
        print(f"\n== {s}: best classical {best.method} {best.mae:.2f}; H-A2 {ha2.mae:.2f}; ratio {ha2.mae / best.mae:.2f} "
              f"-> {'CLAIM (<= 0.8)' if ha2.mae <= 0.8 * best.mae else 'no claim'}")
        print(g.drop(columns="setting").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
