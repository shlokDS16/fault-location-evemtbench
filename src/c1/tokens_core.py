"""Local-view (terminal S) H-A2 tokens for ARBITRARY window ends, shared by the adapt_grid and CIGRE MV runs.
Local phasor tokens = local_features.window_feats (unchanged). TD tokens = the same computation as td_tokens.py
(design section 15), refactored so that window ends and the data directory are arguments. Equality with the
benchmark-family tables is checked by check_tokens_core() before any new grid is processed."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_features import N, WLEN, LOOPS, window_feats  # noqa: E402

FS, F0 = 9600.0, 50.0
W0 = 2 * np.pi * F0
PH = {"a": 0, "b": 1, "c": 2}
LOCAL_TOKENS = [f"{lp}_{k}" for lp in LOOPS for k in ("zr", "zi", "react", "tak2", "takd") + (("tak0",) if lp.endswith("g") else ()) + ("absz",)]
LOCAL_TOKENS += ["i0r", "i2r", "v0r", "v2r", "i2c", "i2s", "i0c", "i0s"]
LOCAL_TOKENS += [f"{k}{nm}_{s}" for nm in "abc" for k, s in (("i", "onset"), ("v", "drop"), ("i", "share"), ("di", "rel"))]
TD_TOKENS = [f"{lp}_{k}" for lp in LOOPS for k in ("dtd", "drl", "tdres")]
ALLOW = LOCAL_TOKENS + TD_TOKENS  # the ONLY learner inputs (59 + 18 = 77)


def der(x):
    return savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def td_loops(X, Z1, Z0):
    i, v = X[:3], X[3:]
    R1, L1, R0, L0 = Z1.real, Z1.imag / W0, Z0.real, Z0.imag / W0
    i0 = (i[0] + i[1] + i[2]) / 3
    di = [der(x) for x in i]
    di0 = der(i0)
    loops = {}
    for lp in LOOPS:
        if lp.endswith("g"):
            p = PH[lp[0]]
            loops[lp] = (v[p], R1 * i[p] + L1 * di[p] + (R0 - R1) * i0 + (L0 - L1) * di0, i[p])
        else:
            p, q = PH[lp[0]], PH[lp[1]]
            loops[lp] = (v[p] - v[q], R1 * (i[p] - i[q]) + L1 * (di[p] - di[q]), i[p] - i[q])
    return loops


def td_feats(loops, s1):
    a, b = s1 - WLEN + N, s1
    rec = {}
    for lp, (vl, ul, il) in loops.items():
        vv, uu = vl[a:b], ul[a:b]
        gg = il[a:b] - il[a - N:b - N]
        Xm = np.stack([uu, gg], 1)
        beta, *_ = np.linalg.lstsq(Xm, vv, rcond=None)
        res = np.linalg.norm(vv - Xm @ beta) / (np.linalg.norm(vv) + 1e-12)
        drl = float(np.dot(vv, uu) / (np.dot(uu, uu) + 1e-12))
        rec[f"{lp}_dtd"] = float(np.clip(beta[0], -1, 2))
        rec[f"{lp}_drl"] = float(np.clip(drl, -1, 2))
        rec[f"{lp}_tdres"] = float(np.clip(res, 0, 5))
    return rec


def episode_tokens(X, Z1, Z0, ends):
    """X: list of 6 arrays (Ia Ib Ic Va Vb Vc) of the local terminal; ends: window-end sample indices (exclusive).
    Returns a list of dicts with the 77 allow-listed tokens plus the physics-only one-ended estimates per loop."""
    loops = td_loops(X, Z1, Z0)
    out = []
    for s1 in ends:
        f, _ = window_feats(X, s1, Z1, Z0)
        f.update(td_feats(loops, s1))
        out.append(f)
    return out


def check_tokens_core():
    """Recompute the DoubleLine benchmark tokens with this module and compare to the stored tables."""
    import pandas as pd
    from local_features import line_params
    from stage_a import D, T0, terminal_cols
    import re
    R = Path(__file__).resolve().parents[2] / "results"
    ref = pd.read_parquet(R / "c1_local_feats.parquet").merge(pd.read_parquet(R / "c1_td_tokens.parquet"),
                                                               on=["sim_idx", "tau_ms"], validate="one_to_one")
    assert set(ALLOW) <= set(ref.columns), set(ALLOW) - set(ref.columns)
    assert len(ALLOW) == 77 and len(set(ALLOW)) == 77
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    LP = line_params()
    worst = 0.0
    for sid in ref.sim_idx.drop_duplicates().sample(12, random_state=0):
        row = s[s["general/sim_idx"] == sid].iloc[0]
        line = row["events/event_target"]
        bus = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line).group(1)
        f = D / "data" / f"result{sid}.csv"
        cs = terminal_cols(pd.read_csv(f, nrows=0).columns.tolist(), bus, line)
        d = pd.read_csv(f, skiprows=[1], usecols=cs)
        X = [d[c].to_numpy(float) for c in cs]
        nf = int(round((row["events/event_start"] - T0) * FS))
        r = ref[ref.sim_idx == sid].sort_values("tau_ms")
        ends = [nf + int(round(t / 1000 * FS)) for t in r.tau_ms]
        new = pd.DataFrame(episode_tokens(X, *LP[line], ends))[ALLOW].to_numpy()
        worst = max(worst, float(np.nanmax(np.abs(new - r[ALLOW].to_numpy()))))
    print(f"tokens_core vs stored tables, 12 episodes: max abs diff {worst:.3e}")
    return worst


if __name__ == "__main__":
    assert check_tokens_core() < 1e-9
