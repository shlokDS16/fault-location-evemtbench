"""Macros for the professor-revision analyses (design 26; called by make_macros.py): (b) white-noise sweep of the
two-ended time-domain locator (Savitzky-Golay derivative, official windows)."""
import numpy as np
import pandas as pd


def td_noise_macros(R, put, out):
    for f, k in (("c1_td_noise", "DL"), ("c1_td_noise_TestGrid110kV", "TG")):
        d = pd.read_csv(R / f"{f}.csv")
        d = d[d.deriv == "sg"]
        e = (np.clip(d.loc_est, 0, 100) - d.loc_true).abs()  # locations in % of length
        m = e.groupby(d.snr).mean()
        assert f"{m.loc[999]:.2f}" == out[f"teMae{k}"], (k, m.loc[999], out[f"teMae{k}"])  # clean = Table II
        put(f"tdSnrForty{k}", m.loc[40]); put(f"tdSnrThirty{k}", m.loc[30])


def learn2_macros(R, put):
    """(a) two-ended learners: best seed-averaged cell per input and test set, full and d != 0.5 (design 26, 26a)."""
    p = pd.read_parquet(R / "c1_two_end_learn_preds.parquet", engine="fastparquet")
    p["e"] = (np.clip(p.pred, 0, 1) - p.y).abs() * 100
    p["keep"] = (p.y - 0.5).abs() >= 1e-9
    g = p.groupby(["input", "model", "direction", "seed"])
    s = pd.DataFrame(dict(all=g.e.mean(), excl=p[p.keep].groupby(["input", "model", "direction", "seed"]).e.mean()))
    s = s.groupby(["input", "model", "direction"]).mean()
    csv = pd.read_csv(R / "c1_two_end_learn.csv").groupby(["input", "model", "direction"]).mae.mean()
    assert np.allclose(csv.loc[s.index], s["all"], atol=0.01)
    for d, k in (("TG_to_DL", "DL"), ("DL_to_TG", "TG"), ("DL+TG_to_MV", "MV")):
        for inp in ("P", "Z"):
            c = s.xs((inp, d), level=("input", "direction"))
            best = c["all"].idxmin()
            put(f"lrn{inp}{k}", c.loc[best, "all"]); put(f"lrn{inp}x{k}", c.loc[best, "excl"])
            put(f"lrn{inp}Model{k}", best)


def m6_macros(R, put):
    """(c) d = 0.5 exclusion (design 26c): results/c1_prof_m6.csv, c1_prof_m6_extra.csv."""
    m = pd.read_csv(R / "c1_prof_m6.csv").set_index(["set", "method"]).mae_excl
    x = pd.read_csv(R / "c1_prof_m6_extra.csv").set_index("set")
    for st, k in (("DL", "DL"), ("TG", "TG"), ("MV", "MV")):
        put(f"teX{k}", m[(st, "TD two-ended")]); put(f"eX{k}", m[(st, "Eriksson")]); put(f"haX{k}", m[(st, "PFB")])
        put(f"halfShare{k}", x.loc[st, "share_half"], 0); put(f"eUndefHalf{k}", x.loc[st, "share_E_undef_and_half"], 1)
        put(f"pfbIncHif{k}", x.loc[st, "pfb_mae_inc_hif"]); put(f"pfbSc{k}", x.loc[st, "pfb_mae_sc"])
    for st, k in (("DL", "DL"), ("TG", "TG")):
        eq = m[(st, "best MLP/GRU equal inform.")]
        put(f"eqX{k}", eq); put(f"redX{k}", 100 * (1 - m[(st, "PFB")] / eq), 0)
