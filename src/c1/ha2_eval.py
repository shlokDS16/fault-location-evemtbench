"""Consolidated, allow-list-only evaluation of H-A and H-A2 (zero-shot, both directions), clean and under
test-time noise (models trained on clean source data). Replaces hybrid_ha2.py's H-A2 step and is the
generating script of results/c1_ha2_noise.csv (validator finding 9, research/VALIDATION_HA2.md).
Also reports: short-circuit-only MAE (incipient episodes all sit at 50 % / 10 ohm, so their low error is a
type shortcut, validator finding 4) and the mean/sd of single-seed MAEs next to the 3-seed ensemble.
Inputs: results/c1_local_feats<TAG>.parquet + results/c1_td_tokens<TAG>.parquet; learner columns = ALLOW only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import ALLOW, LOCAL_TOKENS  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
PARAMS = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=50,
              l2_regularization=1.0, max_features=0.8, early_stopping=False)
FORBIDDEN = ("sim_idx", "line", "etype", "tau", "R", "y", "oracle", "phys_", "theta", "sin_t", "cos_t", "teacher")
GRID_TAG = {"DL": "", "TG": "_TestGrid110kV"}


def check_allow(cols):
    assert len(cols) == len(set(cols))
    for c in cols:
        assert not any(c == f or c.startswith(f) for f in FORBIDDEN), c


def load(grid, noise=""):
    t = GRID_TAG[grid] + noise
    f = pd.read_parquet(R / f"c1_local_feats{t}.parquet")
    d = pd.read_parquet(R / f"c1_td_tokens{t}.parquet")
    return f.merge(d, on=["sim_idx", "tau_ms"], validate="one_to_one").sort_values(["sim_idx", "tau_ms"]).reset_index(drop=True)


def summary(err, df):
    tau, shc = df.tau_ms.values, ~df.etype.str.contains("incipient").values
    return dict(mae=err.mean(), median=np.median(err), tau_le15=err[tau <= 15].mean(), tau_ge21=err[tau >= 21].mean(),
                mae_shc_only=err[shc].mean())


def main():
    rows = []
    for name, cols in (("H-A", LOCAL_TOKENS), ("H-A2", ALLOW)):
        check_allow(cols)
        for src, tgt in (("TG", "DL"), ("DL", "TG")):
            tr = load(src)
            models = [HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[cols], tr.y) for s in (0, 1, 2)]
            for noise in ("", "_SNR40", "_SNR30"):
                te = load(tgt, noise)
                P = np.stack([m.predict(te[cols]) for m in models])
                y = te.y.values
                err = np.abs(np.clip(P.mean(0), 0, 1) - y) * 100
                single = [(np.abs(np.clip(p, 0, 1) - y) * 100).mean() for p in P]
                rows.append(dict(method=name, direction=f"{src}_to_{tgt}", test_noise=noise or "clean", n_test=len(te),
                                 **summary(err, te), single_seed_mean=np.mean(single), single_seed_sd=np.std(single, ddof=1)))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R / "c1_ha2_eval.csv", index=False)
    out[out.method == "H-A2"][["direction", "test_noise", "mae", "median", "tau_le15", "tau_ge21"]].to_csv(
        R / "c1_ha2_noise.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
