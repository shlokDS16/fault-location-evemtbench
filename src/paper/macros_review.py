"""Macros for the review-driven checks (design 24/24a, [V] validator part 8): phasor two-ended locator, synchronization
and line-parameter sensitivity, Suonan-type one-ended TD baseline, inception-free cost (called by make_macros.py)."""
import pandas as pd


def review_macros(R, put):
    rc = pd.read_csv(R / "c1_review_checks.csv").set_index("grid")
    for g, k in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
        put(f"phMae{k}", rc.loc[g, "ph2"]); put(f"phMaeSC{k}", rc.loc[g, "ph2_sc"])
    for g in ("DL", "TG"):
        put(f"tePre{g}", rc.loc[g, "td_with_prefault"]); put(f"tePost{g}", rc.loc[g, "td_post_only"])
        put(f"syncOne{g}", (rc.loc[g, "sync+1"] + rc.loc[g, "sync-1"]) / 2)
        put(f"parFive{g}", (rc.loc[g, "RLx0.95"] + rc.loc[g, "RLx1.05"]) / 2)
        put(f"parTen{g}", (rc.loc[g, "RLx0.90"] + rc.loc[g, "RLx1.10"]) / 2)
        put(f"parR{g}", (rc.loc[g, "Rx0.80"] + rc.loc[g, "Rx1.20"]) / 2)
    bt = pd.read_csv(R / "c1_review_phasor_bytime.csv").set_index(["grid", "tbin"])
    for g, k in (("DL", "DL"), ("TG", "TG"), ("MV", "MV"), ("ADAPT", "Adapt")):
        for b, bk in (("<=15", "Le"), ("20-30", "Mid"), ("35-50", "Late"), (">=55", "Post")):
            put(f"btTe{bk}{k}", bt.loc[(g, b), "td"]); put(f"btPh{bk}{k}", bt.loc[(g, b), "ph2"])
    su = pd.read_csv(R / "c1_review_suonan.csv").set_index("grid")
    for g, k in (("DL", "DL"), ("TG", "TG"), ("ADAPT", "Adapt"), ("MV", "MV")):
        put(f"suonan{k}", su.loc[g, "suonan_td_oracle"])
    us = pd.read_csv(R / "c1_review_sync_us.csv").set_index("grid")
    for g in ("DL", "TG"):
        for u, uk in ((1, "One"), (10, "Ten"), (50, "Fifty")):
            put(f"usTe{uk}{g}", us.loc[g, f"td_{u}us"]); put(f"usPh{uk}{g}", us.loc[g, f"ph_{u}us"])
