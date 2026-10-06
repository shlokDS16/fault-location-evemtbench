"""Design section 20: apply the I0 guard (local_features.guard_i0) to every stored token table and prove that the
patched tables equal a regeneration from the raw CSVs with the guarded code.
Old tables are kept once as results/<name>_preguard.parquet; the patch always starts from that copy (idempotent).
Usage: python src/c1/apply_i0_guard.py  -> results/c1_i0_guard_check.csv"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_features import guard_i0  # noqa: E402
from tokens_core import ALLOW  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
GUARDED = ["i0c", "i0s", "ag_tak0", "bg_tak0", "cg_tak0"]
TABLES = ["c1_local_feats", "c1_local_feats_SNR30", "c1_local_feats_SNR40", "c1_local_feats_TestGrid110kV",
          "c1_local_feats_TestGrid110kV_SNR30", "c1_local_feats_TestGrid110kV_SNR40",
          "c1_grid_DL", "c1_grid_TG", "c1_grid_MV", "c1_grid_MV_zid", "c1_adapt_tokens"]


def patch():
    rows = []
    for name in TABLES:
        cur, bak = R / f"{name}.parquet", R / f"{name}_preguard.parquet"
        if not bak.exists():
            shutil.copy2(cur, bak)
        old = pd.read_parquet(bak)
        new = guard_i0(old.copy())
        # structural check: only the 5 guarded columns change, and only where i0r < 1e-6
        other = [c for c in old.columns if c not in GUARDED]
        assert old[other].equals(new[other]), name
        z = (old.i0r < 1e-6).to_numpy()
        changed = (old[GUARDED].to_numpy() != new[GUARDED].to_numpy()).any(1)
        assert not changed[~z].any(), name
        new.to_parquet(cur, index=False)
        et = new.etype.str.replace("flt_", "", regex=False)
        rows.append(dict(table=name, n=len(new), guarded=int(z.sum()), changed=int(changed.sum()),
                         **{f"guarded_{k}": int(z[(et == k).to_numpy()].sum()) for k in sorted(et.unique())}))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def pick(df, k=4, seed=0):
    """12 episodes: 4 two-phase, 4 three-phase, 4 others (the guard acts on the first two groups)."""
    ep = df.drop_duplicates("sim_idx")[["sim_idx", "etype"]]
    groups = [ep[ep.etype == "flt_2ph_shc"], ep[ep.etype == "flt_3ph_shc"],
              ep[~ep.etype.isin(["flt_2ph_shc", "flt_3ph_shc"])]]
    return [int(s) for g in groups for s in g.sample(min(k, len(g)), random_state=seed).sim_idx]


def regen_check():
    import grid_pass
    import adapt_tokens
    out = []
    for tag, grid in (("DL", "DoubleLine"), ("TG", "TestGrid110kV"), ("MV", "CigreMVGrid")):
        gdir = ROOT / "data" / "raw" / "evemt" / grid
        tab = pd.read_parquet(R / f"c1_grid_{tag}.parquet")
        LP = grid_pass.line_params(gdir)
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv").set_index("general/sim_idx", drop=False)
        worst = 0.0
        for sid in pick(tab):
            recs, _ = grid_pass.one((s.loc[sid], LP, gdir))
            got = pd.DataFrame(recs).sort_values("post_ms")
            ref = tab[tab.sim_idx == sid].sort_values("post_ms")
            assert np.allclose(got.post_ms.values, ref.post_ms.values)
            worst = max(worst, float(np.nanmax(np.abs(got[ALLOW].to_numpy() - ref[ALLOW].to_numpy()))))
        out.append(dict(grid=tag, episodes=12, max_abs_diff=worst))
        print(out[-1], flush=True)
    tab = pd.read_parquet(R / "c1_adapt_tokens.parquet")
    LP = adapt_tokens.line_params()
    s = pd.read_csv(adapt_tokens.AD / "labels" / "settings_clean.csv").set_index("general/sim_idx", drop=False)
    worst = 0.0
    for sid in pick(tab):
        split = tab.loc[tab.sim_idx == sid, "split"].iloc[0]
        recs, _ = adapt_tokens.one((s.loc[sid], LP, split))
        got = pd.DataFrame(recs).sort_values("post_ms")
        ref = tab[tab.sim_idx == sid].sort_values("post_ms")
        worst = max(worst, float(np.nanmax(np.abs(got[ALLOW].to_numpy() - ref[ALLOW].to_numpy()))))
    out.append(dict(grid="adapt", episodes=12, max_abs_diff=worst))
    print(out[-1], flush=True)
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = patch()
    p.to_csv(R / "c1_i0_guard_patch.csv", index=False)
    c = regen_check()
    c.to_csv(R / "c1_i0_guard_check.csv", index=False)
    assert (c.max_abs_diff < 1e-9).all(), c
    print("PATCH == REGENERATION on all grids")
