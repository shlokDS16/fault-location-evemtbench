"""Design 18a(2): label-orientation convention check for CIGRE MV line MainLn8-14, measured only at bus 14.
Oracle-loop reactance estimate from the bus-14 terminal on bolted (1 ohm) short circuits, complete windows
(post-fault 25-80 ms): compare MAE against y (location from bus 8) and against 1 - y (location from bus 14).
Same kind of check as stage_a.py's reverse-orientation row on the 110 kV grids."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import episode_tokens  # noqa: E402
from local_features import oracle_loop  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from grid_pass import line_params, window_ends  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data" / "raw" / "evemt" / "CigreMVGrid"


def main():
    LP = line_params(G)
    s = pd.read_csv(G / "labels" / "settings_clean.csv")
    fl = s[(s["events/event_target"] == "MainLn8-14") & s["events/event_type"].str.contains("shc")
           & (s["events/event_flt_shc_resistance"] == 1.0)]
    rows = []
    for _, row in fl.iterrows():
        f = G / "data" / f"result{int(row['general/sim_idx'])}.csv"
        cs = terminal_cols(pd.read_csv(f, nrows=0).columns.tolist(), "14", "MainLn8-14")
        d = pd.read_csv(f, skiprows=[1], usecols=cs)
        X = [d[c].to_numpy(float) for c in cs]
        nf, ends = window_ends(row, len(X[0]))
        ends = [e for e in ends if 25 <= (e - nf) / 9.6 <= 80]
        Z1, Z0, _ = LP["MainLn8-14"]
        ol = oracle_loop(row)
        for t in episode_tokens(X, Z1, Z0, ends):
            rows.append(dict(y=row["events/event_flt_target_line_location"] / 100, est=np.clip(t[f"{ol}_react"], 0, 1)))
    df = pd.DataFrame(rows)
    print(f"n windows {len(df)}; MAE vs y (from bus 8): {(df.est - df.y).abs().mean() * 100:.2f} %;"
          f" MAE vs 1 - y (from bus 14): {(df.est - (1 - df.y)).abs().mean() * 100:.2f} %")
    print(df.groupby("y").est.median().round(3).to_string())


if __name__ == "__main__":
    main()
