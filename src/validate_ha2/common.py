"""Shared constants and helpers for the independent H-A2 validation.

Written from claudedocs/validator_spec_HA2.md only (no code from src/c1/ was consulted).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "evemt"
RESULTS = ROOT / "results"
CACHE = RESULTS / "validate_ha2_epcache"

GRIDS = {"DL": "DoubleLine", "TG": "TestGrid110kV", "AD": "adapt/DoubleLine"}  # AD = adapt_grid family

FS = 9600.0
F0 = 50.0
N_CYC = 192
WIN = 480
N_SAMPLES = 4801
CHANNELS = ("Isec:A", "Isec:B", "Isec:C", "Usec:A", "Usec:B", "Usec:C")
A_OP = np.exp(2j * np.pi / 3)

C_IDX = "general/sim_idx"
C_TYPE = "events/event_type"
C_START = "events/event_start"
C_TARGET = "events/event_target"
C_LOC = "events/event_flt_target_line_location"
C_RES = "events/event_flt_shc_resistance"
C_PH1 = "events/event_phase_select_1ph"
C_PH2 = "events/event_phase_select_2ph"
C_DUR = "events/event_iflt_duration"

# Total line impedances (ohm), from the specification section 1.
_Z_BY_LEN = {
    20.0: (0.66785 + 5.20049j, 2.3629 + 19.57238j),
    25.0: (0.83481 + 6.50061j, 2.95362 + 24.46547j),
    30.0: (1.00177 + 7.80074j, 3.54435 + 29.35857j),
    40.0: (1.3357 + 10.40098j, 4.7258 + 39.14475j),
    33.5: (1.11865 + 8.71082j, 3.95785 + 32.78373j),
    35.0: (1.16874 + 9.10086j, 4.13507 + 34.25166j),
}
LINE_LEN = {
    "DL": {"MainLn1-2A": 20.0, "MainLn1-2B": 25.0, "MainLn2-3A": 30.0, "MainLn2-3B": 40.0},
    "TG": {"MainLn1-2A": 25.0, "MainLn1-2B": 25.0, "MainLn1-5": 20.0, "MainLn2-3": 40.0,
           "MainLn2-5": 40.0, "MainLn2-6": 30.0, "MainLn4-5": 30.0, "MainLn3-4": 33.5,
           "MainLn5-6": 35.0},
}
LINE_LEN["AD"] = dict(LINE_LEN["DL"])  # adapt grid model: same 4 lines as DoubleLine


def line_z(grid: str, line: str) -> tuple[complex, complex]:
    return _Z_BY_LEN[LINE_LEN[grid][line]]


def taus_for(event_type: str) -> list[int]:
    stop = 50 if "incipient" in event_type else 80
    return list(range(5, stop + 1, 5))


def local_bus(line: str) -> str:
    """MainLnX-Y[A|B] -> 'MainBusX' (local terminal S)."""
    x = line[len("MainLn"):].split("-")[0]
    return f"MainBus{x}"


def episodes(grid: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / GRIDS[grid] / "labels" / "settings_clean.csv")
    m = (df[C_TYPE].astype(str).str.startswith("flt_")
         & df[C_TARGET].astype(str).str.startswith("MainLn")
         & df[C_LOC].notna())
    ep = df.loc[m, [C_IDX, C_TYPE, C_START, C_TARGET, C_LOC, C_RES, C_PH1, C_PH2]].copy()
    ep.columns = ["sim_idx", "etype", "start", "line", "loc", "res", "ph1", "ph2"]
    ep["sim_idx"] = ep["sim_idx"].astype(int)
    ep["grid"] = grid
    return ep.reset_index(drop=True)


def oracle_loop(etype: str, ph1, ph2) -> str:
    if etype.startswith("flt_1phg"):
        return str(ph1).strip().lower() + "g"
    if etype.startswith("flt_2ph"):  # covers 2ph and 2phg
        return str(ph2).strip().lower()
    if etype.startswith("flt_3ph"):
        return "ab"
    raise ValueError(etype)


def ftype(etype: str) -> str:
    return etype[len("flt_"):]
