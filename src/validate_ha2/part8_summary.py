"""Part 8 summary of A, B, C, D1 (from validate_ha2_part8_perwindow.parquet and my stored token tables).
Scoring: estimate clipped to [0, 1]; non-finite -> 0.5 (counted); MAE in % of line length.
Post-fault bins: <=15, 20-30, 35-50, >=55 ms (p <= 15, p <= 30, p <= 50, else; all window ends are on the 5 ms grid).
Output: results/validate_ha2_part8_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.RESULTS
BINS = ["<=15", "20-30", "35-50", ">=55"]


def pbin(p: np.ndarray) -> np.ndarray:
    return np.select([p <= 15, p <= 30, p <= 50], BINS[:3], BINS[3])


def score(est: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    ok = np.isfinite(est)
    return np.abs(np.where(ok, np.clip(est, 0, 1), 0.5) - y) * 100, int((~ok).sum())


def row(part: str, grid: str, method: str, subset: str, e: np.ndarray, pft: np.ndarray, nund: int) -> dict:
    b = pbin(pft)
    r = {"part": part, "grid": grid, "method": method, "subset": subset, "n": len(e), "MAE": e.mean(),
         "median": np.median(e), "p95": np.percentile(e, 95), "n_undefined": nund}
    for k in BINS:
        r[f"MAE_{k}"] = e[b == k].mean() if (b == k).any() else np.nan
        r[f"n_{k}"] = int((b == k).sum())
    return r


def main() -> None:
    df = pd.read_parquet(R / "validate_ha2_part8_perwindow.parquet")
    rows = []
    for g in ("DL", "TG", "AD", "MV"):
        d = df[df["grid"] == g]
        y, pft = d["y"].to_numpy(), d["pft_ms"].to_numpy()
        shc = d["ftype"].str.contains("shc").to_numpy()
        for m, lab in (("ph2e", "A_phasor_two_ended"), ("td2e", "TD_two_ended")):
            e, nu = score(d[m].to_numpy(), y)
            rows.append(row("A" if m == "ph2e" else ("D1" if g == "AD" else "ref"), g, lab, "all", e, pft, nu))
            rows.append(row("A" if m == "ph2e" else ("D1" if g == "AD" else "ref"), g, lab, "shc", e[shc], pft[shc],
                            int((~np.isfinite(d[m].to_numpy()[shc])).sum())))
        if g in ("DL", "TG"):
            for c in [c for c in d.columns if c.startswith("td2e_")]:
                e, nu = score(d[c].to_numpy(), y)
                rows.append(row("B", g, c, "all", e, pft, nu))
    # C: oracle-loop dtd token (one-ended TD R-L identification), my stored tokens
    tabs = {"DL": (pd.read_parquet(R / "validate_ha2_tokens_DL.parquet"), "y", "tau"),
            "TG": (pd.read_parquet(R / "validate_ha2_tokens_TG.parquet"), "y", "tau"),
            "AD": (pd.read_parquet(R / "validate_ha2_ingrid_tokens.parquet").query("split == 'test'"), "y", "pft_ms"),
            "MV": (pd.read_parquet(R / "validate_ha2_mv_tokens.parquet"), "y_local", "pft_ms")}
    for g, (t, yc, pc) in tabs.items():
        t = t.reset_index(drop=True)
        dtd = t[[f"dtd_{lp}" for lp in ("ag", "bg", "cg", "ab", "bc", "ca")]].to_numpy()
        ii = t["oracle"].map({lp: k for k, lp in enumerate(("ag", "bg", "cg", "ab", "bc", "ca"))}).to_numpy().astype(int)
        est = dtd[np.arange(len(t)), ii]
        e, nu = score(est, t[yc].to_numpy())
        pft = t[pc].to_numpy(np.float64)
        rows.append(row("C", g, "dtd_oracle", "all", e, pft, nu))
        shc = t["ftype"].str.contains("shc").to_numpy()
        rows.append(row("C", g, "dtd_oracle", "shc", e[shc], pft[shc], 0))
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_part8_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 30):
        print(s[["part", "grid", "method", "subset", "n", "MAE", "median", "n_undefined"] +
                [f"MAE_{k}" for k in BINS]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
