"""Part 8 D2 (GPU): CIGRE MV zero-shot equal-information baselines, MLP / GRU, -raw / +Z, trained on POOLED DL + TG
(my cached official windows, 14,640 + 32,940), 3 seeds, tested on all 56,340 MV rule windows (LOCAL terminal: bus X,
bus 14 for MainLn8-14; target y_local). Same architecture / training as my part-1 neural.py (its run() is reused
unchanged). Episode key for the 10 % validation split: (grid, sim_idx), made unique by offsetting TG ids by 1e6.
'+Z' = nameplate (R1, X1, R0, X0) of the faulted line (MV: mv.line_z), standardised with training statistics.
Outputs: results/validate_ha2_part8_nn_mv_runs.csv, validate_ha2_nnpred_{MLP,GRU}_{raw,Z}_DLTGtoMV_s{0,1,2}.npy,
results/validate_ha2_part8_mv_windows.npz (raw MV windows, row-aligned with validate_ha2_mv_tokens.parquet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv as MVM  # noqa: E402
import neural as NN  # noqa: E402

R = C.RESULTS
RUNS = R / "validate_ha2_part8_nn_mv_runs.csv"
WIN_MV = R / "validate_ha2_part8_mv_windows.npz"


def mv_windows() -> tuple[np.ndarray, pd.DataFrame]:
    meta = pd.read_parquet(R / "validate_ha2_mv_tokens.parquet",
                           columns=["sim_idx", "s1", "pft_ms", "ftype", "line", "flipped", "y", "y_local"])
    if not WIN_MV.exists():
        X = np.empty((len(meta), C.WIN, 6), np.float32)
        for idx, g in meta.groupby("sim_idx", sort=False):
            sig = np.load(MVM.CACHE / f"ep{idx}.npy")
            fl = bool(g["flipped"].iloc[0])
            loc = sig[:, 6:12] if fl else sig[:, 0:6]
            assert np.isfinite(loc).all()
            for i, s1 in zip(g.index, g["s1"]):
                X[i] = loc[s1 - C.WIN:s1]
        np.savez(WIN_MV, X=X, sim_idx=meta["sim_idx"].to_numpy(), s1=meta["s1"].to_numpy())
    w = np.load(WIN_MV)
    assert (w["sim_idx"] == meta["sim_idx"].to_numpy()).all() and (w["s1"] == meta["s1"].to_numpy()).all()
    z = np.array([[*(lambda z1, z0: (z1.real, z1.imag, z0.real, z0.imag))(*MVM.line_z(ln))] for ln in meta["line"]])
    mt = pd.DataFrame({"sim_idx": meta["sim_idx"], "tau": meta["pft_ms"], "ftype": meta["ftype"],
                       "y": meta["y_local"], "R1": z[:, 0], "X1": z[:, 1], "R0": z[:, 2], "X0": z[:, 3]})
    return w["X"], mt


def pooled() -> tuple[np.ndarray, pd.DataFrame]:
    Xs, ms = [], []
    for k, g in enumerate(("DL", "TG")):
        X, m = NN.load(g)
        m = m.copy()
        m["sim_idx"] = m["sim_idx"].astype(np.int64) + k * 1_000_000
        Xs.append(X)
        ms.append(m)
    return np.concatenate(Xs), pd.concat(ms, ignore_index=True)


def main() -> None:
    torch.backends.cudnn.benchmark = True
    print("device", NN.DEV, flush=True)
    data = {"DLTG": pooled(), "MV": mv_windows()}
    print({k: v[0].shape for k, v in data.items()}, flush=True)
    done = set()
    if RUNS.exists():
        prev = pd.read_csv(RUNS)
        done = {(r.model, r.variant, int(r.seed)) for r in prev.itertuples()}
    for model_name in ("MLP", "GRU"):
        for variant in ("raw", "Z"):
            for seed in (0, 1, 2):
                if (model_name, variant, seed) in done:
                    continue
                row = NN.run(model_name, variant, "DLTG", "MV", seed, data)
                pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                print(f"{model_name} {variant} s{seed}: MAE {row['MAE']:.3f} best_ep {row['best_epoch']} "
                      f"({row['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
