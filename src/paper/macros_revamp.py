"""Macros for the revamp analyses (design section 25): episode-bootstrap CIs, errors in metres and P95, Eriksson on
defined windows, the exploratory switch rule, indicative runtime, and the names/views of the published best models
(called by make_macros.py)."""
import numpy as np
import pandas as pd

SETS = {"DL": "DL", "TG": "TG", "MV": "MV", "ADAPT": "Adapt"}
METH = {"two": "Two", "two_zid": "TwoId", "E": "E", "ha2": "Ha", "eq": "Eq"}
NAME = {"mlp": "MLP", "gru": "GRU", "cnn": "CNN", "resnet": "ResNet", "random_forest": "random forest"}


def revamp_macros(R, PUB, put):
    s = pd.read_csv(R / "c1_revamp_stats.csv").set_index(["set", "method", "stat"]).value
    for st, sk in SETS.items():
        for m, mk in METH.items():
            if (st, m, "mae") not in s:
                continue
            put(f"ci{mk}{sk}lo", s[(st, m, "ci_lo")])
            put(f"ci{mk}{sk}hi", s[(st, m, "ci_hi")])
            put(f"cx{mk}{sk}lo", s[(st, m, "ci_lo")], 1)  # one-decimal copies for narrow tables
            put(f"cx{mk}{sk}hi", s[(st, m, "ci_hi")], 1)
            put(f"pn{mk}{sk}", s[(st, m, "p95")], 1)
            put(f"m{mk}{sk}", int(round(s[(st, m, "mae_m")])))
            put(f"pm{mk}{sk}", int(round(s[(st, m, "p95_m")])))
            put(f"km{mk}{sk}", s[(st, m, "mae_m")] / 1000, 1)
        if (st, "ha2_vs_eq", "reduction") in s:
            put(f"red{sk}", s[(st, "ha2_vs_eq", "reduction")], 0)
            put(f"red{sk}lo", s[(st, "ha2_vs_eq", "ci_lo")], 0)
            put(f"red{sk}hi", s[(st, "ha2_vs_eq", "ci_hi")], 0)
        for nm, nk in (("all", ""), ("le15", "Le"), ("ge20", "Ge")):
            put(f"eDef{nk}{sk}", s[(st, "E_defined", f"mae_{nm}")])
            put(f"eCov{nk}{sk}", s[(st, "E_defined", f"coverage_{nm}")], 0)
        put(f"sw{sk}", s[(st, "switch", "mae")])
        put(f"sw{sk}lo", s[(st, "switch", "ci_lo")])
        put(f"sw{sk}hi", s[(st, "switch", "ci_hi")])
    # conservative CI bound of the gain: the lower of the two implementations' lower bounds ([V] part 9)
    v = pd.read_csv(R / "validate_ha2_part9_summary.csv").set_index(["set", "method", "stat"]).value
    for st, sk in (("DL", "DL"), ("TG", "TG")):
        lo = min(s[(st, "ha2_vs_eq", "ci_lo")], v[(st, "ha2_vs_eq", "ci_lo")] * 100)
        put(f"redC{sk}lo", float(np.floor(lo * 10) / 10), 1)  # floored: printed as "at least"
    rt = pd.read_csv(R / "c1_runtime.csv").set_index("item").ms_per_window
    put("rtTwo", rt["two-ended TD estimate"], 2)
    put("rtFeat", rt["77 local features"], 1)
    put("rtClass", rt["classical one-ended (R, T, T2, MT, E)"], 1)
    put("rtInf", rt["H-A2 inference (3 models, batch)"], 2)
    put("rtTrain", rt["H-A2 training on TG (3 models), s"], 0)
    p = pd.read_csv(PUB)
    p = p[p.task.str.startswith("fault_location") & (p.metric == "mae") & (p.protocol == "held_out")
          & (p.test_set == "benchmark")]
    for k, g in (("DL", "double_line"), ("TG", "testgrid_110kv"), ("MV", "cigre_mv")):
        q = p[p.grid == g]
        b = q.loc[q["mean"].idxmin()]
        put(f"pubBestModel{k}", NAME[b.baseline])
        put(f"pubBestView{k}", b.task.replace("fault_location_", ""))
        ql = q[q.task == "fault_location_line"]
        put(f"pubLineModel{k}", NAME[ql.loc[ql["mean"].idxmin()].baseline])
        qg = q[q.task == "fault_location_global"]
        put(f"pubGlobal{k}", qg["mean"].min())
        put(f"pubGlobalModel{k}", NAME[qg.loc[qg["mean"].idxmin()].baseline])


def ratio_macros(out, put):
    """Two-ended TD error against the best published LINE-view model (equal information), as factors."""
    r = [float(out[f"pubLine{k}"]) / float(out[f"teMae{k}"]) for k in ("DL", "TG", "MV")]
    put("ratioLineMin", min(r), 0)
    put("ratioLineMax", max(r), 0)
    for k, v in zip(("DL", "TG", "MV"), r):
        put(f"ratioLine{k}", v, 0)
    rp = [float(out[f"pubLine{k}"]) / float(out[f"phMae{k}"]) for k in ("DL", "TG", "MV")]  # phasor two-ended
    put("ratioPhLineMin", min(rp), 0)
    put("ratioPhLineMax", max(rp), 0)


def cond_extra_macros(R, put):
    """Median one-ended errors on the conditioning subset (1ph-g short circuits, tau >= 25 ms; [V] part 7)."""
    c = pd.read_csv(R / "c1_cond_summary.csv").set_index("grid")
    for g, k in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
        put(f"medObsR{k}", c.loc[g, "med_abs_eobs_R"], 1)
        put(f"medObsT{k}", c.loc[g, "med_abs_eobs_T"], 1)
        put(f"identPninety{k}", c.loc[g, "ident_R_p90_pp"], 1)
