"""Design 19a: corrected A3 ablation. Loop apparent impedance tokens (zr, zi, absz) in OHM, unclipped, recomputed from the
raw local-terminal records (window_feats loop phasors); all other ALLOW tokens taken from the stored (guarded) tables.
Usage: python src/c1/ablation_a3_fix.py  -> results/c1_ablation_a3_fixed.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from local_features import window_feats, LOOPS  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from tokens_core import ALLOW  # noqa: E402
from ablations import fit_eval  # noqa: E402
from ha2_eval import check_allow  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
FS = 9600.0
GRID = {"DL": "DoubleLine", "TG": "TestGrid110kV", "MV": "CigreMVGrid"}


def one(args):
    row, LP, gdir = args
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = gdir / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    flip = not grid_pass.has_term(hdr, m.group(1), line)
    cs = terminal_cols(hdr, m.group(2) if flip else m.group(1), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    X = [d[c].to_numpy(float) for c in cs]
    nf, ends = grid_pass.window_ends(row, len(X[0]))
    Z1, Z0 = LP[line][0], LP[line][1]
    out = []
    for s1 in ends:
        _, vi = window_feats(X, s1, Z1, Z0)
        r = dict(sim_idx=sid, k=round((s1 - nf) / FS * 1000, 3))
        for lp in LOOPS:
            V, I = vi[lp]
            z = V / I if abs(I) > 1e-9 else 0j
            r[f"{lp}_zr"], r[f"{lp}_zi"], r[f"{lp}_absz"] = z.real, z.imag, abs(z)
        out.append(r)
    return out


def ohm_table(tag):
    gdir = ROOT / "data" / "raw" / "evemt" / GRID[tag]
    LP = grid_pass.line_params(gdir)
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(grid_pass.LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    with ProcessPoolExecutor(6) as ex:
        z = pd.DataFrame([r for rr in ex.map(one, [(r, LP, gdir) for _, r in fl.iterrows()], chunksize=4) for r in rr])
    t = pd.read_parquet(R / f"c1_grid_{tag}.parquet")
    t["k"] = t.post_ms.round(3)
    zc = [c for c in z.columns if c not in ("sim_idx", "k")]
    u = t.drop(columns=zc).merge(z, on=["sim_idx", "k"], how="left", validate="one_to_one")
    assert len(u) == len(t) and u[zc].notna().all().all(), tag
    return u.set_index(t.index)


def main():
    check_allow(ALLOW)
    tab = {g: ohm_table(g) for g in GRID}
    rows = []
    for name, src, tgt in (("A3fix", ["TG"], "DL"), ("A3fix", ["DL"], "TG"), ("A3fix", ["DL", "TG"], "MV")):
        tr = pd.concat([tab[s] for s in src], ignore_index=True)
        te = tab[tgt]
        r = fit_eval(tr[ALLOW], tr.y.values, te[ALLOW], te.y.values, te.post_ms.values)
        rows.append(dict(ablation=name, direction="+".join(src) + "_to_" + tgt, **r))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "c1_ablation_a3_fixed.csv", index=False)


if __name__ == "__main__":
    main()
