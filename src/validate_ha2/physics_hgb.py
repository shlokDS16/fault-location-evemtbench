"""Step 3: physics-only one-ended baselines and the H-A / H-A2 learners (zero-shot, both directions).

Writes results/validate_ha2_pred_{method}_{src}to{tgt}.csv (per-window predictions) and
results/validate_ha2_summary.csv (all summary metrics; neural rows are appended by neural.py).
Resumable: an existing prediction file is not recomputed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS, LOOPS  # noqa: E402

FORBIDDEN_EXACT = {"y", "loc", "sim_idx", "etype", "ftype", "res", "tau", "line", "oracle", "grid",
                   "nf", "s1", "start", "ph1", "ph2", "R1", "X1", "R0", "X0"}


def assert_allow_list(cols: list[str]) -> None:
    """No label, identifier, fault type, resistance, tau, line name/length or remote quantity."""
    allowed = set(LOCAL_TOKENS) | set(TD_TOKENS)
    bad = [c for c in cols if c in FORBIDDEN_EXACT or c not in allowed]
    # token names were defined in tokens.py from local-terminal phasors / samples only
    assert not bad, f"forbidden or unknown columns in allow-list: {bad}"
    assert len(set(cols)) == len(cols)


HGB_PARAMS = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=31,
                  min_samples_leaf=50, l2_regularization=1.0, max_features=0.8,
                  early_stopping=False)


def load(grid: str) -> pd.DataFrame:
    return pd.read_parquet(C.RESULTS / f"validate_ha2_tokens_{grid}.parquet")


def summarise(df: pd.DataFrame, method: str, direction: str, seed: str = "avg") -> list[dict]:
    e = df["err"]
    row = {"method": method, "direction": direction, "seed": seed, "n": len(df),
           "MAE": e.mean(), "median": e.median(),
           "MAE_tau_le15": e[df["tau"] <= 15].mean(), "MAE_tau_ge21": e[df["tau"] >= 21].mean()}
    for ft, g in df.groupby("ftype"):
        row[f"MAE_{ft}"] = g["err"].mean()
    return [row]


def save_pred(te: pd.DataFrame, pred: np.ndarray, method: str, direction: str) -> pd.DataFrame:
    out = te[["sim_idx", "tau", "ftype", "line", "res", "y"]].copy()
    out["pred"] = pred
    out["err"] = np.abs(out["pred"] - out["y"]) * 100.0
    out.to_csv(C.RESULTS / f"validate_ha2_pred_{method}_{direction}.csv", index=False)
    return out


def physics(te: pd.DataFrame, direction: str) -> list[dict]:
    rows = []
    idx = te["oracle"].map({lp: k for k, lp in enumerate(LOOPS)}).to_numpy()
    assert not np.isnan(idx.astype(float)).any()
    react = te[[f"react_{lp}" for lp in LOOPS]].to_numpy()
    tak2 = te[[f"tak2_{lp}" for lp in LOOPS]].to_numpy()
    absz = te[[f"absz_{lp}" for lp in LOOPS]].to_numpy()
    r = np.arange(len(te))
    preds = {
        "phys_react_oracle": react[r, idx],
        "phys_tak2_oracle": tak2[r, idx],
        "phys_react_minabsz": react[r, absz.argmin(axis=1)],
    }
    for m, p in preds.items():
        out = save_pred(te, np.clip(p, 0, 1), m, direction)
        rows += summarise(out, m, direction)
    return rows


def hgb(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], method: str, direction: str) -> list[dict]:
    assert_allow_list(cols)
    path = C.RESULTS / f"validate_ha2_pred_{method}_{direction}.csv"
    rows = []
    if path.exists():
        out = pd.read_csv(path)
        print(f"  {method} {direction}: cached")
    else:
        Xtr, ytr = tr[cols].to_numpy(np.float64), tr["y"].to_numpy()
        Xte = te[cols].to_numpy(np.float64)
        preds = []
        for seed in (0, 1, 2):
            t0 = time.time()
            m = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(Xtr, ytr)
            p = m.predict(Xte)
            preds.append(p)
            np.save(C.RESULTS / f"validate_ha2_seedpred_{method}_{direction}_s{seed}.npy", p)
            print(f"  {method} {direction} seed {seed}: "
                  f"MAE {np.abs(np.clip(p, 0, 1) - te['y'].to_numpy()).mean() * 100:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out = save_pred(te, np.clip(np.mean(preds, axis=0), 0, 1), method, direction)
    rows += summarise(out, method, direction)
    for seed in (0, 1, 2):
        f = C.RESULTS / f"validate_ha2_seedpred_{method}_{direction}_s{seed}.npy"
        if f.exists():
            p = np.clip(np.load(f), 0, 1)
            o = te[["tau", "ftype", "y"]].copy()
            o["err"] = np.abs(p - o["y"].to_numpy()) * 100
            rows += summarise(o, method, direction, seed=str(seed))
    return rows


def main() -> None:
    data = {g: load(g) for g in ("DL", "TG")}
    rows: list[dict] = []
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        d = f"{src}to{tgt}"
        tr, te = data[src], data[tgt]
        print(d, "train", len(tr), "test", len(te), flush=True)
        rows += physics(te, d)
        rows += hgb(tr, te, list(LOCAL_TOKENS), "HA", d)
        rows += hgb(tr, te, list(LOCAL_TOKENS) + list(TD_TOKENS), "HA2", d)
    s = pd.DataFrame(rows)
    s.to_csv(C.RESULTS / "validate_ha2_summary_hgb_phys.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s[s["seed"] == "avg"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
