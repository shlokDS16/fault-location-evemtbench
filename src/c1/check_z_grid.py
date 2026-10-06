"""Check: nameplate positive-sequence Z1 of each line vs the two-ended identified Z1 from PRE-FAULT phasors
(as stage_a.py; no labels except the line name). Confirms the signals are primary-referred and the nameplate data
match the simulation. Usage: python src/c1/check_z_grid.py <grid_dir> [episodes_per_line]"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import terminal_cols, phasor, pos_seq  # noqa: E402
from grid_pass import line_params, LINE_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
N = 192


def main():
    gdir = ROOT / "data" / "raw" / "evemt" / sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    LP = line_params(gdir)
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(LINE_RE)]
    rows = []
    for line, g in fl.groupby("events/event_target"):
        for _, row in g.head(k).iterrows():
            m = LINE_RE.match(line)
            f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
            hdr = pd.read_csv(f, nrows=0).columns.tolist()
            if not any(h.endswith(f"pex_MainBus{m.group(1)}_{line}") for h in hdr) or \
                    not any(h.endswith(f"pex_MainBus{m.group(2)}_{line}") for h in hdr):
                print(f"{line}: only one terminal measured -> skipped")
                break
            cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
            d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
            pre = int(round((row["events/event_start"] - 1.0) * 9600)) - 2 * N
            V = lambda cols: pos_seq(*(phasor(d[c].to_numpy(float), pre, N) for c in cols))  # noqa: E731
            Zid = (V(cs[3:]) - V(cr[3:])) / ((V(cs[:3]) - V(cr[:3])) / 2)
            Z1 = LP[line][0]
            rows.append(dict(line=line, km=LP[line][2], Z1_np=Z1, Z_id=Zid, ratio_abs=abs(Zid) / abs(Z1),
                             ang_np=np.degrees(np.angle(Z1)), ang_id=np.degrees(np.angle(Zid)),
                             I_through=abs(V(cs[:3]))))
    df = pd.DataFrame(rows)
    print(df.groupby("line")[["km", "ratio_abs", "ang_np", "ang_id", "I_through"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
