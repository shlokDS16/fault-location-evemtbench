"""Read the official grid-model pickle with a RESTRICTED unpickler (no arbitrary code execution):
only networkx / numpy / builtin container classes may be constructed. Prints line edge attributes."""
import pickle
import sys
from pathlib import Path

ALLOWED_PREFIXES = ("networkx", "numpy", "collections", "builtins", "pandas")


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ALLOWED_PREFIXES and not (module == "builtins" and name in
                                                              {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr"}):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked class {module}.{name}")


def load(p):
    with open(p, "rb") as f:
        return SafeUnpickler(f).load()


if __name__ == "__main__":
    p = Path(sys.argv[1])
    G = load(p)
    print(type(G), G.number_of_nodes(), "nodes", G.number_of_edges(), "edges")
    for u, v, k, d in list(G.edges(keys=True, data=True))[:40]:
        name = d.get("name") or d.get("loc_name") or k
        keys = {kk: d[kk] for kk in d if any(s in kk.lower() for s in ("r1", "x1", "r0", "x0", "len", "type", "name", "r_", "x_", "b1", "c1"))}
        print(u, v, name, keys if keys else list(d.keys())[:25])
