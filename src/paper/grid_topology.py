"""Extract the main-line topology (MainBus nodes and the lines between them) of the three EvEMTBench grids used in
the paper from the benchmark's graph files, for the test-system figure. The pickles are loaded with a restricted
unpickler that admits only networkx graph classes.
Usage: python src/paper/grid_topology.py  -> results/paper_grid_topology.csv"""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "evemt"
GRIDS = {"DL": "DoubleLine", "TG": "TestGrid110kV", "MV": "CigreMVGrid"}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("networkx.classes") or (module, name) in (("builtins", "dict"), ("builtins", "list")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main():
    rows = []
    for key, folder in GRIDS.items():
        with open(RAW / folder / "graphs" / "graph_benchmark.pickle", "rb") as fh:
            g = SafeUnpickler(fh).load()
        for u, v, k, data in g.edges(keys=True, data=True):
            if not (str(u).startswith("MainBus") and str(v).startswith("MainBus")):
                continue
            prm = {kk: vv for kk, vv in data.items() if not isinstance(vv, dict)}
            sub = {kk: vv for kk, vv in data.items() if isinstance(vv, dict)}
            rows.append(dict(grid=key, u=u, v=v, key=k, **prm,
                             **{f"{a}.{b}": c for a, d in sub.items() for b, c in d.items()
                                if not isinstance(c, (dict, list))}))
        srcs = [n for n in g.nodes if "ExtGrid" in str(n) or "Gen" in str(n) or "PV" in str(n) or "WT" in str(n)
                or "IBR" in str(n)]
        print(key, "main buses:", sorted(n for n in g.nodes if str(n).startswith("MainBus")))
        print(key, "source-like nodes:", srcs[:20])
        for n in g.nodes:
            if str(n).startswith("MainBus"):
                nb = [m for m in g.neighbors(n) if not str(m).startswith("MainBus")]
                print("   ", n, "->", nb)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "paper_grid_topology.csv", index=False)
    print(d.iloc[:, :12].to_string())


if __name__ == "__main__":
    main()
