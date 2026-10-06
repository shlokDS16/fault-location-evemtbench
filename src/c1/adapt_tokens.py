"""In-grid protocol (design section 17 / 17a): H-A2 tokens and raw local windows for the adapt_grid-DoubleLine
TRAIN and TEST splits (line faults only), windows by the reconstructed benchmark rule:
  window ends s1 = 480 + 48 k (5 ms grid from t = 1.0 s), fault-active iff s1 > nf and s1 - 480 < nf + D,
  nf = floor((event_start - 1.0) * 9600), D = 289 samples (short circuits) or floor(duration * 9600) (incipient).
Outputs: results/c1_adapt_tokens.parquet (meta + 77 ALLOW tokens + physics-only one-ended estimates) and
results/c1_adapt_raw.npz (480 x 6 local windows + nameplate Z for the equal-information baselines).
Resumable per episode chunk is unnecessary (about 10 min on 12 workers)."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import episode_tokens, ALLOW, FS  # noqa: E402
from local_features import LOOPS, WLEN, oracle_loop  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from read_graph import load as load_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AD = ROOT / "data" / "raw" / "evemt" / "adapt" / "DoubleLine"
SPLITS = ROOT / "external" / "evemtbench-benchmark" / "src" / "evemtbench" / "splits" / "v1.0" / "held_out" / "double_line"
D_SC = 289


def line_params():
    G = load_graph(AD / "graphs" / "graph_grid0.pickle")
    out = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if str(k).startswith("MainLn"):
            p = d[f"{k}_param_dict"]
            L = p["length"]
            out[k] = (L * complex(p["rline"], p["xline"]), L * complex(p["rline0"], p["xline0"]))
    return out


def window_ends(row, n_samples):
    nf = int(np.floor((row["events/event_start"] - 1.0) * FS))
    inc = "incipient" in row["events/event_type"]
    D = int(np.floor(row["events/event_iflt_duration"] * FS)) if inc else D_SC
    ends = 480 + 48 * np.arange((n_samples - 480) // 48 + 1)
    return nf, [int(e) for e in ends if e > nf and e - 480 < nf + D]


def one(args):
    row, LP, split = args
    line = row["events/event_target"]
    bus = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line).group(1)
    sid = int(row["general/sim_idx"])
    f = AD / "data" / f"result{sid}.csv"
    cs = terminal_cols(pd.read_csv(f, nrows=0).columns.tolist(), bus, line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    X = [d[c].to_numpy(float) for c in cs]
    nf, ends = window_ends(row, len(X[0]))
    Z1, Z0 = LP[line]
    toks = episode_tokens(X, Z1, Z0, ends)
    ol = oracle_loop(row)
    recs, wins = [], []
    Xa = np.stack(X, 1).astype(np.float32)
    for s1, t in zip(ends, toks):
        minloop = min(LOOPS, key=lambda l: t[f"{l}_absz"])
        rec = dict(sim_idx=sid, split=split, line=line, etype=row["events/event_type"], post_ms=(s1 - nf) / FS * 1000,
                   R=row["events/event_flt_shc_resistance"], y=row["events/event_flt_target_line_location"] / 100.0,
                   oracle_loop=ol, phys_react_oracle=t[f"{ol}_react"], phys_tak2_oracle=t[f"{ol}_tak2"],
                   phys_react_minloop=t[f"{minloop}_react"], Z1r=Z1.real, Z1x=Z1.imag, Z0r=Z0.real, Z0x=Z0.imag)
        rec.update({k: t[k] for k in ALLOW})
        recs.append(rec)
        wins.append(Xa[s1 - WLEN:s1])
    return recs, wins


def main():
    LP = line_params()
    s = pd.read_csv(AD / "labels" / "settings_clean.csv")
    jobs = []
    for split in ("train", "test"):
        ids = {int(l.strip().split(":")[2]) for l in open(SPLITS / f"{split}.txt") if l.strip()}
        fl = s[s["general/sim_idx"].isin(ids) & s["events/event_type"].str.startswith("flt_")
               & s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()]
        jobs += [(r, LP, split) for _, r in fl.iterrows()]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, jobs, chunksize=8))
    df = pd.DataFrame([r for rr, _ in res for r in rr])
    W = np.stack([w for _, ww in res for w in ww])
    df.to_parquet(ROOT / "results" / "c1_adapt_tokens.parquet", index=False)
    np.savez(ROOT / "results" / "c1_adapt_raw.npz", X=W, sim_idx=df.sim_idx.values, is_test=(df.split == "test").to_numpy(np.int8),
             post_ms=df.post_ms.values, y=df.y.values, Z=df[["Z1r", "Z1x", "Z0r", "Z0x"]].to_numpy(np.float32))
    print(df.groupby("split").agg(episodes=("sim_idx", "nunique"), windows=("y", "size")))
    print(W.shape)


if __name__ == "__main__":
    main()
