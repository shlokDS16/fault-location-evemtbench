"""Bus-fault protocol finding macros (appended to paper/macros_auto.tex by make_macros.py).
Sources: the validator's independent window counts (results/validate_ha2_ingrid_counts.csv) and the published n_test of
the EvEMTBench v1.1.0 aggregated results."""
import pandas as pd


def bus_macros(R, PUB, put):
    c = pd.read_csv(R / "validate_ha2_ingrid_counts.csv", index_col=0)["value"]
    allw, linew = int(c["adapt_test_all_fault_windows"]), int(c["adapt_test_line_windows"])
    put("busAllWin", allw)
    put("busLineWin", linew)
    put("busShare", 100 * (1 - linew / allw), 0)
    put("busEpWithLoc", int(c["adapt_bus_faults_with_location"]))
    put("busAllEp", int(c["adapt_test_all_fault_episodes"]))
    put("busLineEp", int(c["adapt_test_line_episodes"]))
    p = pd.read_csv(PUB)
    q = p[(p.task == "fault_location_local") & (p.grid == "double_line") & (p.protocol == "held_out") & (p.test_set == "test")]
    put("pubNAdaptTest", int(q.n_test.iloc[0]))
