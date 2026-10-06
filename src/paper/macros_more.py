"""Remaining text macros: identified-vs-nameplate R on CIGRE MV and the seed standard-deviation range of the
equal-information baselines (called by make_macros.py)."""
import numpy as np
import pandas as pd


def more_macros(R, put):
    a = pd.read_parquet(R / "c1_grid_MV.parquet", columns=["line", "Z1r", "Z1x"]).drop_duplicates("line").set_index("line")
    b = pd.read_parquet(R / "c1_grid_MV_zid.parquet", columns=["line", "Z1r", "Z1x"]).drop_duplicates("line").set_index("line")
    rr, rx = (b.Z1r / a.Z1r), (b.Z1x / a.Z1x)
    changed = rr[(rr - 1).abs() > 0.05]
    put("zidNsec", int(len(changed)))
    put("zidRpct", float(100 * (changed.median() - 1)), 0)
    put("zidXdevMax", float(100 * (rx - 1).abs().max()), 1)
    e = pd.read_csv(R / "c1_equal_info.csv").groupby(["model", "variant", "direction"]).mae.std()
    m = pd.read_csv(R / "c1_zs_equal_info_MV.csv").groupby(["model", "variant"]).mae.std()
    put("eqSdMin", float(min(e.min(), m.min())), 1)
    put("eqSdMax", float(max(e.max(), m.max())), 1)


def conservative_gains(R, out, put):
    """Ledger rule (C1_GATE, I0 GUARD [V]): gain = 1 - worse H-A2 (lead, validator) / better best equal-information
    baseline (lead, validator)."""
    v = pd.read_csv(R / "validate_ha2_guard_oldnew.csv")
    vn = pd.read_csv(R / "validate_ha2_neural_runs.csv").groupby(["model", "variant", "direction"]).MAE.mean()
    vi = pd.read_csv(R / "validate_ha2_ingrid_neural_summary.csv", header=[0, 1], index_col=[0, 1, 2])[("MAE", "mean")]
    for k, vk in (("TGDL", "TGtoDL"), ("DLTG", "DLtoTG")):
        ha = max(float(out[f"haTwoZs{k}"]), float(v[(v.setting == "zeroshot") & (v.testset == vk) & (v.method == "HA2")].MAE_new.iloc[0]))
        base = min(float(out[f"eqBest{k}"]), float(vn.xs(vk, level="direction").min()))
        put(f"gainC{k}", 100 * (1 - ha / base), 0)
    for k, vk, ok in (("Adapt", "adapt_test", "InAdapt"), ("Bench", "benchmark_DL", "InBench")):
        ha = max(float(out[f"haTwo{ok}"]), float(v[(v.setting == "ingrid") & (v.testset == vk) & (v.method == "HA2")].MAE_new.iloc[0]))
        base = min(float(out[f"eqBest{ok}"]), float(vi.xs(vk, level=2).min()))
        put(f"gainCIn{k}", 100 * (1 - ha / base), 0)
