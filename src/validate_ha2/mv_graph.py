"""Read the CIGRE MV grid pickle with a restricted unpickler (no code execution) and print main-line parameters."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ALLOWED = ("networkx", "numpy", "collections", "builtins", "pandas")
BLOCK = {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "input", "breakpoint"}


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ALLOWED and not (module == "builtins" and name in BLOCK):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


with open(C.RAW / "CigreMVGrid" / "graphs" / "graph_benchmark.pickle", "rb") as fh:
    G = Safe(fh).load()
for u, v, k, d in G.edges(keys=True, data=True):
    if str(k).startswith("MainLn"):
        p = d.get(f"{k}_param_dict", d)
        print(k, {kk: p[kk] for kk in p if not isinstance(p[kk], (dict, list))})
