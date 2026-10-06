# Independent validation of H-A, H-A2 and the equal-information baselines (2026-10-04)

Specification: `claudedocs/validator_spec_HA2.md`. Validator code: `src/validate_ha2/`. Results: `results/validate_ha2_*`.

## 0. Independence
- Part A (re-implementation) was written from the specification only. During Part A, nothing in `src/c1/`, no
  `results/c1_*` file and not `research/C1_GATE.md` was opened. I read only the spec, the raw data, the label
  files and the header of `src/validate_c1/common.py` / `step3_locate.py` (file-format reference).
- The Part A numbers were frozen before Part B started. SHA-256 of the frozen files and a timestamp
  (11:31:51 IST) are in `results/validate_ha2_partA_frozen.sha256`. All Part B scripts read those files and
  do not change them.
- No number from any `*_INVALID_LEAK.csv` file was used.
- Learner: sklearn 1.8 `HistGradientBoostingRegressor`. LightGBM was not attempted. Neural nets: torch 2.11
  + CUDA, one GPU job at a time.

Pipeline (each step can be resumed):
1. `parse_episodes.py`: for each line-fault episode, extracts the 6 local channels (cubicle `pex_MainBusX_<line>`,
   channel order checked against the units row; time axis checked: 4801 samples, t0 = 1.0 s) and caches them
   to `results/validate_ha2_epcache/`.
2. `tokens.py`: 59 local tokens, 18 TD tokens and the raw 480 x 6 windows, written to
   `results/validate_ha2_tokens_{DL,TG}.parquet` and `results/validate_ha2_windows_{DL,TG}.npz`.
3. `physics_hgb.py`: the 3 physics baselines plus H-A and H-A2. Each learner gets an explicit allow-list
   (`LOCAL_TOKENS`, `LOCAL_TOKENS + TD_TOKENS`), and an assertion rejects any label, id, type, resistance,
   tau, line or Z column.
4. `neural.py`: the 24 equal-information runs. `summarise.py`: freezes the numbers.
5. Part B: `audit_partB.py` (alignment and per-token diffs) and `compare_partB.py` (numbers and diagnostics).

Choices the spec left open, with the option I took:
- Z for lines not listed explicitly: I used the listed total of a DL line of the same length. These lines
  are 25, 20, 40 and 30 km and all have the same line type.
- Neural nets: the validation episodes are a random ceil(10 %) of source sim_idx values (rng = seed).
  Standardisation uses statistics from the 90 % training part only. In the GRU '+Z' variant, dropout is
  applied to the hidden state only, and Z is concatenated after it. Predictions are clipped to [0, 1]. The
  unclipped MAE is also in `results/validate_ha2_summary.csv`.

## 1. Window counts (exact match to the spec)
| grid | episodes | windows |
|---|---|---|
| DoubleLine (DL) | 924 | 14,640 |
| TestGrid110kV (TG) | 2,079 | 32,940 |

Inception is at t = 1.1 s in every episode (nf = 960). No token is non-finite.

## 2. Part A results (mine). MAE in % of line length
H-A and H-A2 are each the average of 3 seed predictions, clipped to [0, 1]. Neural rows are the mean +- sd
of 3 single-seed runs. The physics baselines are deterministic.

### TG -> DL (train TG 32,940 windows, test DL 14,640)
| method | MAE | median | tau<=15 | tau>=21 |
|---|---|---|---|---|
| phys react (oracle loop) | 19.950 | 3.829 | 41.905 | 14.638 |
| phys tak2 (oracle loop) | 24.009 | 10.032 | 41.600 | 19.836 |
| phys react (min abs z loop) | 23.324 | 7.194 | 41.430 | 18.948 |
| MLP -raw | 13.08 +- 0.14 | 9.95 +- 0.15 | 15.73 +- 0.13 | 12.43 +- 0.13 |
| MLP +Z | 11.76 +- 0.16 | 8.10 +- 0.19 | 14.46 +- 0.10 | 11.11 +- 0.17 |
| GRU -raw | 11.53 +- 0.62 | 8.51 +- 0.43 | 12.58 +- 0.92 | 11.29 +- 0.68 |
| GRU +Z | 11.04 +- 0.73 | 7.61 +- 0.60 | 12.34 +- 0.68 | 10.71 +- 0.77 |
| H-A | 9.012 | 4.801 | 20.455 | 6.179 |
| H-A2 | 7.808 | 4.072 | 13.905 | 6.287 |

| method | 1phg incip. | 1phg incip. arc | 1phg | 1phg arc | 2ph | 2phg | 3ph |
|---|---|---|---|---|---|---|---|
| phys react oracle | 50.00 | 50.00 | 15.27 | 15.59 | 17.37 | 23.82 | 25.19 |
| phys tak2 oracle | 50.00 | 50.00 | 16.65 | 16.63 | 16.72 | 24.87 | 43.01 |
| phys react min abs z | 50.00 | 50.00 | 14.26 | 16.02 | 22.61 | 37.77 | 23.74 |
| MLP -raw | 32.48 | 32.49 | 13.39 | 13.78 | 10.42 | 12.88 | 13.31 |
| MLP +Z | 29.36 | 29.37 | 12.41 | 13.36 | 8.27 | 11.69 | 11.59 |
| GRU -raw | 24.21 | 24.21 | 12.33 | 10.02 | 9.93 | 11.25 | 13.07 |
| GRU +Z | 20.86 | 20.86 | 11.58 | 9.40 | 9.85 | 11.04 | 12.48 |
| H-A | 3.87 | 3.77 | 8.78 | 8.91 | 8.96 | 10.12 | 8.72 |
| H-A2 | 4.41 | 4.45 | 6.98 | 7.33 | 6.92 | 9.07 | 9.02 |

### DL -> TG (train DL 14,640 windows, test TG 32,940)
| method | MAE | median | tau<=15 | tau>=21 |
|---|---|---|---|---|
| phys react (oracle loop) | 14.451 | 1.984 | 38.357 | 8.661 |
| phys tak2 (oracle loop) | 19.789 | 6.043 | 39.468 | 15.149 |
| phys react (min abs z loop) | 19.098 | 4.143 | 37.431 | 14.632 |
| MLP -raw | 17.78 +- 0.19 | 12.40 +- 0.31 | 19.50 +- 0.24 | 17.36 +- 0.21 |
| MLP +Z | 15.95 +- 0.17 | 10.31 +- 0.17 | 17.96 +- 0.14 | 15.46 +- 0.19 |
| GRU -raw | 18.07 +- 0.37 | 11.96 +- 0.30 | 19.17 +- 0.63 | 17.78 +- 0.31 |
| GRU +Z | 17.48 +- 0.69 | 10.59 +- 0.34 | 19.58 +- 0.92 | 16.93 +- 0.56 |
| H-A | 12.039 | 7.388 | 20.261 | 10.043 |
| H-A2 | 10.518 | 6.316 | 15.504 | 9.357 |

| method | 1phg incip. | 1phg incip. arc | 1phg | 1phg arc | 2ph | 2phg | 3ph |
|---|---|---|---|---|---|---|---|
| phys react oracle | 50.00 | 50.00 | 10.91 | 11.52 | 11.77 | 16.95 | 18.14 |
| phys tak2 oracle | 49.97 | 49.43 | 12.54 | 13.67 | 11.85 | 17.64 | 40.75 |
| phys react min abs z | 50.00 | 50.00 | 11.16 | 12.17 | 17.24 | 35.75 | 16.60 |
| MLP -raw | 42.07 | 42.07 | 18.53 | 19.82 | 15.45 | 16.84 | 16.25 |
| MLP +Z | 40.88 | 40.89 | 16.73 | 17.84 | 13.89 | 14.86 | 14.36 |
| GRU -raw | 42.97 | 42.97 | 18.83 | 21.56 | 15.31 | 16.83 | 15.76 |
| GRU +Z | 45.79 | 45.80 | 18.22 | 20.50 | 14.65 | 16.38 | 15.30 |
| H-A | 2.05 | 1.91 | 13.23 | 13.27 | 12.84 | 12.42 | 9.27 |
| H-A2 | 3.85 | 3.65 | 11.65 | 11.73 | 10.91 | 10.43 | 8.43 |

### Relative gain of H-A2 over the best equal-information baseline (mine)
| direction | best equal-info | its MAE | H-A2 (3-seed ensemble) | gain | gain with H-A2 single-seed mean |
|---|---|---|---|---|---|
| TG -> DL | GRU +Z | 11.04 +- 0.73 | 7.81 | 29.2 % | 27.9 % (7.95 +- 0.13) |
| DL -> TG | MLP +Z | 15.95 +- 0.17 | 10.52 | 34.1 % | 33.4 % (10.63 +- 0.19) |

H-A2 against H-A on tau <= 15 ms: relative gain 0.320 (TG -> DL) and 0.235 (DL -> TG), mean 0.278. The lead
reports 0.279.

Sanity check of the phasor tokens: with the oracle-loop reactance estimate, on bolted (1 ohm) faults with
tau >= 25 ms, the TG -> DL MAE is 0.9 - 1.7 % across fault types (median 0.24 - 1.09 %).

## 3. Comparison with the lead (Part B)
| method | direction | mine | lead | delta | criterion | verdict |
|---|---|---|---|---|---|---|
| H-A | TG -> DL | 9.012 | 8.937 | +0.075 | abs(d) <= 0.3 | REPRODUCED |
| H-A2 | TG -> DL | 7.808 | 7.747 | +0.060 | abs(d) <= 0.3 | REPRODUCED |
| phys react oracle | TG -> DL | 19.950 | 19.950 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| phys tak2 oracle | TG -> DL | 24.009 | 24.009 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| phys react min abs z | TG -> DL | 23.324 | 23.324 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| MLP -raw | TG -> DL | 13.08 +- 0.14 | 13.18 +- 0.24 | -0.10 | abs(d) <= 1.0 or 2 sd | REPRODUCED |
| MLP +Z | TG -> DL | 11.76 +- 0.16 | 11.86 +- 0.16 | -0.10 | same | REPRODUCED |
| GRU -raw | TG -> DL | 11.53 +- 0.62 | 11.83 +- 0.99 | -0.30 | same | REPRODUCED |
| GRU +Z | TG -> DL | 11.04 +- 0.73 | 11.43 +- 0.35 | -0.39 | same | REPRODUCED |
| H-A | DL -> TG | 12.039 | 12.018 | +0.021 | abs(d) <= 0.3 | REPRODUCED |
| H-A2 | DL -> TG | 10.518 | 10.584 | -0.066 | abs(d) <= 0.3 | REPRODUCED |
| phys react oracle | DL -> TG | 14.451 | 14.451 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| phys tak2 oracle | DL -> TG | 19.789 | 19.789 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| phys react min abs z | DL -> TG | 19.098 | 19.098 | 0.000 | abs(d) <= 0.01 | REPRODUCED |
| MLP -raw | DL -> TG | 17.78 +- 0.19 | 17.83 +- 0.09 | -0.04 | same | REPRODUCED |
| MLP +Z | DL -> TG | 15.95 +- 0.17 | 16.20 +- 0.12 | -0.25 | same | REPRODUCED |
| GRU -raw | DL -> TG | 18.07 +- 0.37 | 18.40 +- 0.33 | -0.32 | same | REPRODUCED |
| GRU +Z | DL -> TG | 17.48 +- 0.69 | 17.91 +- 0.47 | -0.43 | same | REPRODUCED |

All 18 cells pass. Medians and the tau splits also agree; details are in
`results/validate_ha2_partB_compare.csv`. My neural baselines come out 0.04 - 0.43 pp lower than the lead's in
all 8 cells. That direction slightly favours the baselines, so it does not inflate the lead's claimed gain.
The relative gains (mine 29 % / 34 %, lead 32 % / 35 %) agree in substance.

Per-window agreement of the H-A / H-A2 predictions with the lead's: mean absolute difference 1.3 - 1.5 pp,
95th percentile 3.8 - 4.4 pp, Pearson r 0.997 - 0.998 (`results/validate_ha2_partB_perwindow.csv`).

Diagnostic (`results/validate_ha2_partB_diag.csv`). My HGB code, run on the LEAD's token tables in the lead's
column order, reproduces the lead's numbers exactly: 8.937 / 7.747 / 12.018 / 10.584. On MY tokens in the
lead's column order it gives 8.964 / 7.742 / 12.258 / 10.537. With my own column order it gives
9.012 / 7.808 / 12.039 / 10.518. So the remaining small H-A / H-A2 differences come from `max_features = 0.8`
sampling, which depends on column order, combined with token differences of about 1e-4. They are not a
method difference. Implication: H-A differences below about 0.25 pp are inside the numerical noise of this
learner and should not be claimed.

## 4. Audit of the lead's code (Part B)
(1) Columns that actually enter each learner. `hybrid_ha.feats()` excludes {sim_idx, line, etype, tau_ms, R,
y, oracle_loop} and every `phys_*` column. On `c1_local_feats*.parquet` (70 columns) that leaves exactly
59 columns:
ag/bg/cg/ab/bc/ca x {zr, zi, react, tak2, takd, absz}, ag/bg/cg tak0, i0r, i2r, v0r, v2r, i2c, i2s, i0c, i0s,
and for each phase a/b/c: i_onset, v_drop, i_share, di_rel. After the merge with `c1_td_tokens*.parquet`
(sim_idx, tau_ms plus 18 tokens), H-A2 sees those 59 plus ag..ca x {dtd, drl, tdres} = 77 columns. These map
one-to-one onto my allow-list: none missing, none extra. None of them is a label, identifier, fault type,
resistance, tau, line name or length, or an oracle-derived estimate.
- The neural baselines take `c1_raw*.npz` (X = local 480 x 6 window; Z = R1, X1, R0, X0). y, sim_idx and tau
  are used only for the target, the validation grouping and scoring.

(2) Test-grid information in training. None found in the code. HGB is fitted on the source frame only. The
neural nets take the channel, Z and target standardisation and the validation split from the source grid
only. Test labels are used only for scoring and for the physics comparison numbers.
Protocol caveats:
- (a) H-A2 was adopted over H-A by a pre-registered rule (design section 15: at least 20 % gain on
  tau <= 15 ms averaged over both directions). That rule is evaluated on the two TEST grids. It is a model
  selection made with test labels, even though it is pre-registered and binary. There is no third grid.
- (b) The HGB hyper-parameters are described as "fixed a priori", but their values do not appear in the
  pre-registered design doc, which says only "LightGBM regressor (L1 loss)". The code contains only one
  setting and shows no tuning, but the a-priori claim cannot be verified from the record.

(3) Remote-terminal quantities in the token tables. None. `local_features.py`, `td_tokens.py` and
`raw_windows.py` read only `terminal_cols(hdr, <first bus number>, line)`, i.e. the S cubicle. Independent
confirmation: my tokens, computed from local-only cache files, match the lead's tables per window (point 5),
and the lead's raw windows equal mine bit for bit. Line Z comes from the grid pickle (nameplate). It matches
the spec values to within a relative 5e-6.

(4) Window count and alignment. Both grids match exactly: 14,640 / 32,940 windows matched on (sim_idx, tau).
y, event type, line and oracle loop show 0 mismatches. The raw windows are identical (max abs diff 0.0).

(5) Per-token agreement (`results/validate_ha2_partB_token_diff.csv`). Over all 77 tokens x 47,580 windows,
the largest absolute difference is 0.0050 (TG tak0_ag). DL max is 0.00065. The 99th percentile is <= 8e-6 for
every token. Only 5 (token, window) cells differ by more than 1e-3. These differences match the 5e-6 relative
difference in Z (graph pickle against the rounded spec values), amplified where a denominator is near zero.

(6) Other problems found:
- `results/c1_ha2_noise.csv` (the H-A2 test-time-noise table in C1_GATE.md) has no generating script in the
  repository: grep for "ha2_noise" over src/ finds nothing. It cannot be reproduced as things stand.
- `hybrid_ha2.py`'s `__main__` goes on to `combine()`, which reads `results/c1_hybrid_hc_<dir>.csv`. That
  file is now renamed `*_INVALID_LEAK.csv`, so a rerun crashes after the H-A2 files are written. If someone
  restored the file, `combine()` would bring the leaky H-C-P2 numbers back. The H-A2 numbers themselves do
  not depend on it.
- C1_GATE.md says "63 dimensionless tokens" (H-A) and "63 local tokens + 18 TD tokens". The learners actually
  use 59 and 59 + 18 = 77. `src/c1/tokens_core.py` says 59 + 77 correctly. Text error.
- C1_GATE.md says the noise run used "an explicit allow-list". `hybrid_ha.py` / `hybrid_ha2.py`, which produced
  the headline H-A / H-A2 numbers, use the exclusion-list `feats()`. I verified that this list gives exactly
  the 59 / 77 legitimate columns, so the numbers are clean. The wording should still match the code, or the
  scripts should switch to the `ALLOW` list in `tokens_core.py`.
- Implementation differences in the lead's equal-info code (harmless; the results reproduce): validation
  episodes = floor(10 %) instead of ceil; the GRU head applies dropout to [h, Z] jointly; the validation loss
  is an unweighted mean of chunk losses; almost every run reaches the 60-epoch cap (early stopping rarely
  fires; mine too).

## 5. Leakage verdict per learner
| learner | inputs (verified) | verdict |
|---|---|---|
| H-A | 59 local dimensionless tokens | CLEAN: no label, id, type, R_f, tau, line or remote quantity |
| H-A2 | 59 + 18 local TD tokens | CLEAN (as above) |
| physics baselines | react / tak2 of the oracle loop and of the min abs z loop | No learning. The oracle-loop variants use the fault-type / phase labels to pick the loop (an advantage to the baseline, disclosed) |
| MLP / GRU -raw | local 480 x 6 window | CLEAN |
| MLP / GRU +Z | local window plus faulted-line R1, X1, R0, X0 | CLEAN (nameplate only, as allowed) |

## 6. Disclosure points for the paper / what a reviewer will raise
1. Two grids with the same 110 kV line type (identical per-km parameters). Transfer to a different line type
   is untested. Both "directions" are test sets, and the H-A2 adoption rule was evaluated on them (4.2a).
2. Labels are discrete: locations {1, 20, 50, 80, 99} %, R_f in {1, 10, 40} ohm. Every incipient episode sits
   at 50 % with 10 ohm, in both grids. The low incipient MAE of H-A / H-A2 (1.9 - 4.5 %) therefore comes from
   recognising the incipient signature plus the label prior (predictions 0.52 - 0.53 +- 0.03 - 0.06). It is not
   evidence of locating ability. The physics estimators give exactly 50 % error on incipient windows: their
   estimates always saturate at 0 or 1. Incipient windows are 1.6 % of each test set. Report them separately
   or leave them out of per-type claims.
3. With tau <= 15 ms the last DFT cycle still holds pre-fault samples, so all phasor methods are poor there:
   physics about 40 %, H-A about 20 %. H-A2's TD tokens cut the H-A error on these windows (20.5 to 13.9 % and
   20.3 to 15.5 %). On tau >= 21 ms H-A2 does not beat H-A in TG -> DL (6.29 against 6.18; lead 6.25 against
   6.11). It is also worse than H-A on incipient and on 3ph in TG -> DL.
4. Asymmetric aggregation: H-A / H-A2 are reported as 3-seed ensembles, the neural baselines as means of single
   seeds. Using single-seed H-A2 lowers the gain only slightly (29 to 28 %, 34 to 33 %). State it.
5. The equal-information baselines use the fixed published recipe and are not tuned for cross-grid transfer.
   Most runs end at the 60-epoch cap. A reviewer may ask for a stronger generic learner, e.g. HGB on generic
   waveform statistics, or a CNN.
6. HGB numbers move by up to about 0.25 pp with column order or 1e-4 token perturbations (Section 3
   diagnostic). Do not claim differences of that size.
7. The oracle-loop physics baselines use fault-type and phase labels. The only label-free physics baseline is
   min abs z (23.3 / 19.1 %).
8. Text and reproducibility fixes in C1_GATE.md: token count (59, not 63); allow-list wording; the missing
   noise script; `hybrid_ha2.py`'s dependence on the renamed leaky file.

## 7. Files
- Code: `src/validate_ha2/common.py`, `parse_episodes.py`, `tokens.py`, `physics_hgb.py`, `neural.py`,
  `summarise.py`, `audit_partB.py`, `compare_partB.py`.
- Part A results: `results/validate_ha2_summary.csv`, `validate_ha2_gain.csv`,
  `validate_ha2_summary_hgb_phys.csv`, `validate_ha2_neural_runs.csv`, `validate_ha2_pred_*.csv`,
  `validate_ha2_seedpred_*.npy`, `validate_ha2_nnpred_*.npy`, `validate_ha2_tokens_{DL,TG}.parquet`,
  `validate_ha2_windows_{DL,TG}.npz`, `validate_ha2_partA_frozen.sha256`.
- Part B results: `results/validate_ha2_partB_alignment.csv`, `validate_ha2_partB_token_diff.csv`,
  `validate_ha2_partB_compare.csv`, `validate_ha2_partB_lead_numbers.csv`, `validate_ha2_partB_perwindow.csv`,
  `validate_ha2_partB_diag.csv`, plus the logs `validate_ha2_*.log`.

---

## In-grid (adapt_grid)
Spec: `claudedocs/validator_spec_ingrid.md`. Code: `src/validate_ha2/ingrid.py` (Part A) and `compare_ingrid.py`
(Part B). I changed `common.py` to add the `AD` grid key. This change does not affect the DL or TG paths.
CPU only; no neural or GPU job was run.

Independence: Part A was written without opening `src/c1/` or `results/c1_adapt_*`. Its outputs were frozen
before Part B started; SHA-256 hashes and the timestamp (12:07:46 IST) are in
`results/validate_ha2_ingrid_frozen.sha256`.

### Window counts
Reconstructed rule: s1 = 480 + 48k. A window is scored iff s1 > nf and s1 - 480 < nf + D, with D = 289 for
short circuits and floor(duration x 9600) for incipient faults.

| set | episodes | windows |
|---|---|---|
| benchmark family, line faults | 924 | **14,640**. Identical window for window (same (sim_idx, s1) set) to my tau-based windows of Part A |
| benchmark family, all faults (incl. 153 MainBus faults) | 1,077 | 16,980 |
| adapt TEST, all faults (incl. MainBus) | 701 | **11,674** (published n_test 11,673: off by one) |
| adapt TEST, line faults only | 411 | **6,883** |
| adapt TRAIN, line faults only (training set) | 1,832 | 30,260 |
| adapt TRAIN, all faults | 3,266 | 54,005 |

- Split sizes: train 6,999, val 1,500, test 1,500, pairwise disjoint.
- In adapt_grid, all 2,030 MainBus faults carry a location value (1 - 99 %, mean 49), even though the fault
  is on a bus. The published in-grid test set therefore includes 11,674 - 6,883 = 4,791 bus-fault windows
  scored against a location label with no line meaning.

### My MAE (% of line length)
Trained on adapt TRAIN line-fault windows. H-A and H-A2 are the average of 3 seeds, clipped to [0, 1].
pft = post-fault time (s1 - nf) / 9.6 ms.

(a) adapt TEST, line faults (6,883 windows; 1,233 with pft <= 15, 5,151 with pft >= 21)

| method | MAE | median | pft<=15 | pft>=21 | short-circuit only | 1phg inc | 1phg inc arc | 1phg | 1phg arc | 2ph | 2phg | 3ph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phys react oracle | 32.20 | 23.49 | 46.66 | 28.54 | 25.78 | 46.34 | 47.07 | 17.77 | 23.72 | 19.83 | 39.48 | 28.19 |
| phys tak2 oracle | 35.00 | 26.65 | 47.01 | 32.04 | 29.72 | 48.28 | 45.23 | 20.12 | 22.72 | 17.53 | 41.07 | 46.25 |
| phys react min abs z | 34.98 | 27.27 | 46.96 | 32.09 | 29.74 | 46.76 | 46.85 | 18.35 | 21.89 | 26.06 | 55.10 | 27.17 |
| H-A | 12.68 | 8.16 | 21.39 | 10.49 | 9.23 | 21.08 | 19.70 | 8.94 | 8.75 | 8.91 | 11.11 | 8.47 |
| H-A2 | 12.05 | 7.71 | 18.59 | 10.47 | 8.28 | 20.88 | 20.18 | 8.15 | 7.68 | 7.77 | 10.50 | 7.35 |

(b) benchmark family DoubleLine (14,640 windows; 2,772 with pft <= 15, 10,944 with pft >= 21)

| method | MAE | median | pft<=15 | pft>=21 | short-circuit only | 1phg inc | 1phg inc arc | 1phg | 1phg arc | 2ph | 2phg | 3ph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phys react oracle | 19.95 | 3.83 | 41.91 | 14.64 | 19.45 | 50.00 | 50.00 | 15.27 | 15.59 | 17.37 | 23.82 | 25.19 |
| phys tak2 oracle | 24.01 | 10.03 | 41.60 | 19.84 | 23.58 | 50.00 | 50.00 | 16.65 | 16.63 | 16.72 | 24.87 | 43.01 |
| phys react min abs z | 23.32 | 7.19 | 41.43 | 18.95 | 22.88 | 50.00 | 50.00 | 14.26 | 16.02 | 22.61 | 37.77 | 23.74 |
| H-A | 10.59 | 7.50 | 22.85 | 7.60 | 10.67 | 5.46 | 5.48 | 10.45 | 10.27 | 11.02 | 11.13 | 10.50 |
| H-A2 | 9.23 | 6.72 | 17.30 | 7.27 | 9.30 | 5.29 | 5.28 | 9.32 | 9.17 | 9.49 | 10.26 | 8.24 |

Single-seed mean +- sd: H-A 12.73 +- 0.06 / 10.64 +- 0.02; H-A2 12.12 +- 0.06 / 9.29 +- 0.10 (adapt test / benchmark).

Why physics is much worse on adapt: fault resistance in adapt_grid is continuous on 0 - 50 ohm, against
{1, 10, 40} ohm in the benchmark family. The grounding is also mixed (solid / resistive / resonant, about 1/3
each). On adapt TEST short circuits with pft >= 25 ms, oracle-loop reactance error grows with R_f:
0.96 % (R_f <= 1 ohm), 1.45 % (1 - 5), 2.70 % (5 - 10), 8.66 % (10 - 20), 29.6 % (20 - 50). This confirms that
the tokens are correct and that the gap comes from the resistive-fault regime.

### Comparison with the lead (`results/c1_adapt_ingrid.csv`)
| method | test | mine | lead | delta | verdict |
|---|---|---|---|---|---|
| phys react oracle | adapt test | 32.197 | 32.197 | 0.000 | REPRODUCED |
| phys tak2 oracle | adapt test | 34.999 | 34.999 | 0.000 | REPRODUCED |
| phys react min abs z | adapt test | 34.981 | 34.981 | 0.000 | REPRODUCED |
| phys react oracle | benchmark | 19.950 | 19.950 | 0.000 | REPRODUCED |
| phys tak2 oracle | benchmark | 24.009 | 24.009 | 0.000 | REPRODUCED |
| phys react min abs z | benchmark | 23.324 | 23.324 | 0.000 | REPRODUCED |
| H-A | adapt test | 12.677 | 12.777 | -0.101 | REPRODUCED (abs(d) <= 0.3) |
| H-A | benchmark | 10.589 | 10.570 | +0.019 | REPRODUCED |
| H-A2 | adapt test | 12.055 | 11.982 | +0.072 | REPRODUCED |
| H-A2 | benchmark | 9.231 | 9.139 | +0.092 | REPRODUCED |

- Window counts and medians match. The pft splits and short-circuit-only MAE agree to within 0.17 pp; see
  `results/validate_ha2_ingrid_partB_compare.csv`.
- Per-window H-A2 predictions: mean absolute difference 1.17 pp (adapt) and 1.09 pp (benchmark), r 0.997 / 0.999.
- Diagnostic: my HGB on the lead's adapt tokens reproduces the lead exactly (12.777 / 10.570 / 11.982 / 9.139).
  On my tokens in the lead's column order it gives 12.705 / 10.603 / 12.002 / 9.177. The residual differences
  therefore come again from column-order sampling under `max_features = 0.8`.
- The lead's own gate G2-A2 (benchmark H-A2 MAE <= 5.85 and <= 0.8 x best physics) prints FAIL. My numbers agree:
  9.23 > 5.85, although the ratio to physics is 0.46.

### Leakage and protocol audit (`src/c1/adapt_tokens.py`, `adapt_ingrid.py`, `tokens_core.py`)
- Train/test disjointness: the lead's train and test episode sets have 0 overlapping sim_idx (1,832 / 411
  episodes, 30,260 / 6,883 windows; identical to mine). Splits come from the official `train.txt` / `test.txt`;
  `val.txt` is not used. The benchmark-family test set is a different simulation family, held out entirely.
- Windows and labels: all 37,143 lead windows match mine on (sim_idx, s1). Split, y, fault type and oracle
  loop show 0 mismatches. The lead's raw windows (`c1_adapt_raw.npz`) equal my local-channel slices bit for
  bit, and `is_test` agrees with the split. y = location / 100 of the line-fault episode. The MainLn filter
  is applied, so the uninformative bus-fault locations are excluded.
- Allow-list: `adapt_ingrid.py` trains on `tokens_core.LOCAL_TOKENS` (59) and `ALLOW` (77) by explicit list,
  with `check_allow`. The extra table columns (R, y, oracle_loop, phys_*, Z1r..Z0x, post_ms, line, split)
  never reach the learner. `check_allow` is a prefix blacklist and could be stricter (assert membership in
  ALLOW), but the lists are fixed constants, so no leak results.
- Remote quantities: tokens come from `terminal_cols(..., bus X, line)` only. Line Z is read from
  `graph_grid0.pickle` (the only graph in adapt_grid; every episode references it). It matches the spec to
  within a relative 5e-6. Token agreement with mine over 37,143 x 77: max abs diff 0.0022, 99th percentile
  <= 8.3e-6, 3 cells > 1e-3.
- No test-split statistic is used: HGB has no preprocessing, the hyper-parameters are fixed, and there is no
  early stopping.
- Verdict: **CLEAN** for H-A, H-A2 and the physics baselines in the in-grid protocol.

### Disclosure points (in-grid)
1. The published in-grid test count (11,673, matching the rule's 11,674 to within 1) includes all fault
   episodes. 4,791 of those windows are MainBus faults whose "location" label has no line meaning. Our
   line-fault-only test set (6,883 windows) is therefore a different window set. A comparison with the
   published in-grid MAE is not like-for-like and must say so.
2. The 1-window difference between 11,674 and the published 11,673 is unexplained: the rule is a
   reconstruction.
3. adapt_grid differs from the benchmark family in R_f range (0 - 50 ohm continuous) and in grounding. The
   physics baselines collapse there (32 - 35 %), while H-A2 still beats them by about 2.7x. Incipient
   locations vary in adapt (no 50 % shortcut), and H-A/H-A2 incipient MAE there is about 20 %.
4. Pre-registered gate G2-A2 FAILS (benchmark H-A2 9.1 - 9.2 % against a bar of 5.85 %).
5. Neural equal-information baselines for the in-grid protocol were not run (GPU reserved for the lead).

### In-grid equal-information neural baselines (part 3, GPU)
Code: `src/validate_ha2/neural_ingrid.py`. It uses the same `MLP` / `GRUNet` classes and the same fixed recipe
as Part A (`neural.py`):
- AdamW lr 1e-3, weight decay 1e-4, cosine schedule over 60 epochs, batch 256, MSE on the standardised target.
- Early stopping with patience 12, on 10 % of TRAIN episodes grouped by sim_idx (184 of 1,832 episodes).
- Channel, Z and target standardisation from the 90 % training part only.

Data:
- Train: 30,260 adapt TRAIN line-fault windows (480 x 6 local raw windows + nameplate R1, X1, R0, X0).
- Test: (a) 6,883 adapt TEST line-fault windows; (b) 14,640 benchmark-family windows. Predictions are
  clipped to [0, 1].

Frozen before opening `results/c1_adapt_equal_info.csv` or `src/c1/adapt_equal_info.py`:
`results/validate_ha2_ingrid_neural_frozen.sha256` (12:31:14 IST).

My numbers (mean +- sd over seeds 0-2):

| model | test | MAE | median | pft<=15 | pft>=21 | short-circuit only |
|---|---|---|---|---|---|---|
| MLP -raw | adapt test | 17.36 +- 0.12 | 14.43 +- 0.16 | 20.74 +- 0.13 | 16.71 +- 0.13 | 13.29 +- 0.12 |
| MLP +Z | adapt test | 16.69 +- 0.06 | 13.42 +- 0.27 | 20.53 +- 0.04 | 15.92 +- 0.08 | 12.50 +- 0.10 |
| GRU -raw | adapt test | 14.88 +- 0.43 | 10.42 +- 0.93 | 18.56 +- 0.38 | 14.09 +- 0.46 | 9.80 +- 0.67 |
| GRU +Z | adapt test | 14.59 +- 0.83 | 9.85 +- 1.30 | 18.87 +- 1.22 | 13.65 +- 0.72 | 9.39 +- 1.24 |
| MLP -raw | benchmark | 16.24 +- 0.30 | 14.26 +- 0.36 | 21.23 +- 0.30 | 15.08 +- 0.34 | 16.43 +- 0.30 |
| MLP +Z | benchmark | 14.75 +- 0.37 | 12.37 +- 0.53 | 20.49 +- 0.41 | 13.43 +- 0.36 | 14.93 +- 0.38 |
| GRU -raw | benchmark | 11.90 +- 0.76 | 8.62 +- 0.37 | 18.57 +- 1.56 | 10.23 +- 0.65 | 12.05 +- 0.77 |
| GRU +Z | benchmark | 12.19 +- 0.53 | 8.80 +- 0.73 | 19.26 +- 0.40 | 10.44 +- 0.57 | 12.35 +- 0.55 |

Comparison with `results/c1_adapt_equal_info.csv` (criterion: abs(d) <= 1.0 pp or <= 2 sd):

| model | test | mine | lead | delta | verdict |
|---|---|---|---|---|---|
| MLP -raw | adapt test | 17.36 +- 0.12 | 17.35 +- 0.01 | +0.01 | REPRODUCED |
| MLP +Z | adapt test | 16.69 +- 0.06 | 16.68 +- 0.04 | +0.01 | REPRODUCED |
| GRU -raw | adapt test | 14.88 +- 0.43 | 14.46 +- 0.18 | +0.42 | REPRODUCED |
| GRU +Z | adapt test | 14.59 +- 0.83 | 14.21 +- 0.09 | +0.38 | REPRODUCED |
| MLP -raw | benchmark | 16.24 +- 0.30 | 16.04 +- 0.15 | +0.20 | REPRODUCED |
| MLP +Z | benchmark | 14.75 +- 0.37 | 14.73 +- 0.06 | +0.03 | REPRODUCED |
| GRU -raw | benchmark | 11.90 +- 0.76 | 11.97 +- 0.67 | -0.07 | REPRODUCED |
| GRU +Z | benchmark | 12.19 +- 0.53 | 11.95 +- 0.26 | +0.24 | REPRODUCED |

- Medians and the pft splits agree within 0.66 pp; see `results/validate_ha2_ingrid_neural_partB_compare.csv`.
- My GRU runs stopped early: best epochs 20 - 55, and seed 2 stopped at epoch 32 / 38. The lead's GRU runs
  went to 57 - 60 epochs. This explains my larger GRU sd and the slightly higher means on adapt test. The most
  likely causes are the different validation episode draw and the early-stopping noise; the recipe itself is
  the same.

Relative gain of H-A2 (in-grid) over the best in-grid equal-information baseline:
- adapt test: mine 12.05 against GRU +Z 14.59, a 17.4 % gain (lead: 11.98 against 14.21, 15.7 %).
  Short-circuit only, the gap is smaller: H-A2 8.28 against GRU +Z 9.39, about 12 %.
- benchmark: mine 9.23 against GRU -raw 11.90, a 22.4 % gain (lead: 9.14 against GRU -raw 11.97 / GRU +Z 11.95,
  about 23.5 %).
- The in-grid gain over equal-information baselines is about half the cross-grid gain (29 - 34 %).

Audit of `src/c1/adapt_equal_info.py` (and the `equal_info.fit_predict` it reuses):
- Train/test disjointness: train = `~is_test` rows of `c1_adapt_raw.npz`. That file holds only TRAIN and
  TEST line-fault windows, and its `is_test` flag matches the official split for all 37,143 windows (checked
  above). The TRAIN and TEST sim_idx sets do not overlap. The benchmark test set (`c1_raw.npz`) is a separate
  family.
- Standardisation: channel mean/std, Z mean/std and target mean/std are computed on `Xtr[~vm]`, i.e. the
  training windows minus the validation episodes. No test statistic is used. The validation split is 10 % of
  TRAIN episodes (floor), grouped by sim_idx. Test inputs are only transformed with the training statistics.
- Labels: y from the npz (location / 100). Test y is used only for scoring. Predictions are clipped to [0, 1].
- Benchmark-test Z comes from the DoubleLine graph and train Z from `graph_grid0.pickle`; they agree to
  within 5e-6 relative.
- Verdict: **CLEAN**.

Brief re-audit of `src/c1/ha2_eval.py`:
- It trains on `tokens_core.LOCAL_TOKENS` (59) and `ALLOW` (77) by explicit list. `check_allow` is a prefix
  blacklist over {sim_idx, line, etype, tau, R, y, oracle, phys_, theta, sin_t, cos_t, teacher}; no ALLOW
  column matches it.
- Its clean rows (`results/c1_ha2_eval.csv`) reproduce the earlier headline exactly: H-A 8.937 / 12.018 and
  H-A2 7.747 / 10.584. Its short-circuit-only MAE (H-A2 7.81 / 10.70) matches mine (7.86 / 10.63) to within
  0.07 pp.
- It is now the generating script of `c1_ha2_noise.csv`: the models are trained on clean source data, and the
  noisy target tables are used only at test time.
- This closes earlier findings 9 (missing noise script) and the allow-list wording point. It does not change
  the hybrid_ha2.py dependence on the renamed H-C file; that script is simply superseded.
- Verdict: **CLEAN**.

---

## CIGRE MV
Spec: `claudedocs/validator_spec_cigre.md`. CPU only.

Code (`src/validate_ha2/`):
- `mv.py`: extraction, tokens, two-ended locator, physics, H-A/H-A2, diagnoses.
- `mv_two_ended_zid.py`: diagnostic with the identified R1.
- `mv_graph.py`: restricted-unpickler read of the grid pickle; it allows only networkx/numpy/builtin
  containers and executes no code.
- `compare_mv.py`: Part B.

Independence: frozen before opening `src/c1/`, `results/c1_grid_MV*` or `results/c1_zs_*`.
`results/validate_ha2_mv_frozen.sha256` (17:50:03 IST). The first extraction attempt with 12 workers failed
with a Windows paging-file error (the lead's GPU job held memory). The rerun with 6 workers was clean.

### Counts
- 3,555 line-fault episodes on 15 lines (237 each). With the reconstructed rule (nf = 960 in every episode),
  this gives **56,340 windows**: 3,465 non-incipient episodes x 16 (incl. 90 hif episodes, D = 289) plus
  90 incipient episodes x 10.
- Per type: 10,800 windows each for 1phg, 1phg arc, 2ph, 2phg and 3ph; 720 each for hif and hif arc; 450 each
  for incipient and incipient arc.
- Two-ended locator: defined on the 14 two-terminal segments, **52,584 windows (3,318 episodes)**. MainLn8-14
  (3,756 windows, 237 episodes) has no bus-8 cubicle and is excluded.
- All hif and incipient episodes sit at 50 % with R_f 10 ohm (the same label shortcut as in DL/TG). The hif
  episodes have hif resistance 5,000 ohm.

### 8-14 orientation check
MainLn8-14 is measured only at bus 14. On bolted (1 ohm) short circuits with pft >= 25 ms (900 windows),
the oracle-loop reactance estimate from bus 14 gives:

| compared against | MAE |
|---|---|
| y_local = 1 - y | **6.70 %** |
| unflipped y | 62.84 % |
| reference: the other 14 lines, measured from bus X | 5.60 % |

**The 1 - y mapping is confirmed.**

### My numbers (% of line length)
One-ended methods are scored against y_local. The two-ended locator is scored against y from bus X.
H-A and H-A2 are zero-shot: trained on all DL + TG official windows pooled (47,580; my cached tokens), using
the CIGRE nameplate Z for the tokens, 3 seeds averaged.

| method | n | MAE | median | pft<=15 | pft>=21 | short circuits only (5 shc types) | hif | hif arc | incip. | incip. arc |
|---|---|---|---|---|---|---|---|---|---|---|
| two-ended TD R-L (14 seg) | 52,584 | **4.003** | 1.255 | 5.904 | 3.456 | **2.409** | 36.31 | 37.19 | 47.51 | 47.00 |
| phys react oracle | 56,340 | 32.975 | 20.000 | 48.355 | 29.078 | 32.237 | 50.00 | 50.00 | 50.00 | 50.00 |
| phys tak2 oracle | 56,340 | 34.556 | 20.000 | 49.101 | 30.851 | 33.948 | 45.58 | 49.80 | 50.00 | 50.00 |
| phys react min abs z | 56,340 | 38.606 | 20.803 | 48.267 | 36.299 | 38.113 | 50.00 | 50.00 | 50.00 | 50.00 |
| H-A (zero-shot) | 56,340 | 31.849 | 25.412 | 38.761 | 30.172 | 32.475 | 21.78 | 21.14 | 11.08 | 10.76 |
| H-A2 (zero-shot) | 56,340 | 32.901 | 24.876 | 36.800 | 32.005 | 33.274 | 28.94 | 29.22 | 16.79 | 16.52 |

- Single-seed mean +- sd: H-A 31.91 +- 0.48; H-A2 32.95 +- 0.16.
- Two-ended, per short-circuit type: 1phg 2.56, 1phg arc 2.45, 2ph 2.05, 2phg 2.55, 3ph 2.44.
  Per line: 2.5 - 6.1 %.
- **H-A2 does not transfer to CIGRE MV.** It is no better than the oracle reactance baseline and worse than
  H-A. H-A/H-A2's lower MAE on hif and incipient comes from the 50 % label shortcut.

### R_f diagnosis
Short-circuit windows (5 shc types), pft >= 25 ms; MAE in %:

| R_f | n | H-A2 | react oracle | two-ended (n) |
|---|---|---|---|---|
| 1 ohm | 13,500 | 18.92 | **5.67** | 1.59 (12,600) |
| 10 ohm | 13,500 | 37.30 | 32.39 | 1.67 (12,600) |
| 40 ohm | 13,500 | 40.66 | 46.66 | 2.94 (12,600) |

The MV line impedances (abs Z1) are 0.21 - 3.9 ohm, so R_f of 10 - 40 ohm is about 3 - 190 x the line impedance. That
destroys every one-ended estimate, because remote infeed makes the apparent reactance move with R_f. The
two-ended estimate is almost independent of R_f. Even on bolted faults the zero-shot H-A2 (18.9 %) is far
worse than the plain reactance estimate (5.7 %). The learner was trained on 110 kV grids, where R_f is small
relative to Z_line, and does not extrapolate.

### Z1 identification check
Pre-fault positive-sequence, two-ended: Z1 = (V1S - V1R) / ((I1S - I1R) / 2), taken from the cycle ending
one cycle before inception. Median over all 237 episodes per segment; the IQR is about 1e-4 ohm or less.
Identified / nameplate ratios:

| segments | R1 ratio | X1 ratio |
|---|---|---|
| 12 cable segments (NA2XS2Y 120, r1 0.501 ohm/km) | **1.239 - 1.244** | 0.996 - 1.006 |
| 2 overhead distributed segments (12-13, 13-14; 8-14 not identifiable) | 1.001 | 1.000 |

- The grid pickle confirms the nameplate values (rline 0.501, xline 0.716 for the cables), so the
  simulation's cable R1 is about 24 % above nameplate. This fits an operating-temperature correction:
  aluminium at about 80 deg C gives about 1.24. That explanation is my hypothesis and is not verified.
- Diagnostic: the two-ended locator with the identified R1 (nameplate X1) improves from 4.003 to 1.810 % MAE.
  Short circuits go from 2.409 to 1.209. By R_f at pft >= 25 ms: 1 ohm 1.59 to 0.28, 10 ohm 1.67 to 0.49,
  40 ohm 2.94 to 1.28.
- The identification uses pre-fault data of the same episodes. It is label-free but transductive; disclose it.
- The lead's `c1_grid_MV_zid.log` lists identified Z1 values identical to mine to 4 digits, and its
  identified-Z two-ended MAE is 1.804.

### Comparison with the lead (`results/c1_zs_MV.csv`, `c1_zs_MV_pred.csv`, `c1_grid_MV.parquet`)
| method | n | mine | lead | delta | tolerance | verdict |
|---|---|---|---|---|---|---|
| two-ended TD (td2_est) | 52,584 | 4.003 | 4.003 | 0.000 | 0.01 | REPRODUCED |
| phys react oracle | 56,340 | 32.975 | 32.975 | 0.000 | 0.01 | REPRODUCED |
| phys tak2 oracle | 56,340 | 34.556 | 34.556 | 0.000 | 0.01 | REPRODUCED |
| phys react min abs z | 56,340 | 38.606 | 38.606 | 0.000 | 0.01 | REPRODUCED |
| H-A | 56,340 | 31.849 | 32.030 | -0.181 | 0.3 | REPRODUCED |
| H-A2 | 56,340 | 32.901 | 32.977 | -0.076 | 0.3 | REPRODUCED |

- Medians, pft splits and the lead's non-incipient MAE (which includes hif) all agree. Per-window H-A2
  predictions: mean absolute difference 2.07 pp, r 0.992.
- Note: the lead's `mae_shc_only` column is "all non-incipient", so it includes the hif windows. My
  "short circuits only" column excludes hif. Name the columns accordingly in the paper.

### Audit (`src/c1/grid_pass.py`, `zs_eval.py`)
- **Window alignment:** all 56,340 MV windows match on (sim_idx, window end), with 0 mismatches in y_local,
  flip and oracle loop. The lead flips only MainLn8-14. The lead's raw MV windows equal my local-terminal
  slices bit for bit (bus 14 for 8-14). The DL / TG source tables (`c1_grid_{DL,TG}.parquet`) match my
  14,640 / 32,940 windows with identical y.
- **The 8-14 mapping:** `flip = not has_term(bus X)`, local terminal = bus Y, y = 1 - y_label, and
  td2_est = NaN. My physics check confirms that this orientation is right.
- **NaN handling:** `zs_eval.py` scores each physics / two-ended method on `te[c].notna()`, which gives
  52,584 windows for td2_est. H-A/H-A2 are scored on all 56,340. The ALLOW token columns contain no
  non-finite values.
- **Leakage:**
  - The learners use `tokens_core.LOCAL_TOKENS` / `ALLOW` explicitly, checked by `check_allow`.
  - Training uses only the DL + TG tables; no CIGRE data, label or statistic enters training.
  - The y, y_label, R, flip, td2_est, phys_*, Z, line and length_km columns stay out of the learner.
  - The tokens come from the local terminal only; the remote terminal is read only for td2_est.
  - Z comes from the grid pickle (nameplate) and agrees with the spec within 9.4e-5 relative.
  - Verdict: **CLEAN**.
- **Token agreement** (my tokens against the lead's MV table):
  - Most tokens agree to about 1e-5.
  - Large differences (max 3.0 on tak0_*, about 1.3 on cos/sin(I0/I1)) occur only in 2ph and 3ph windows,
    where I0 is numerically zero (i0r about 1e-14). There the I0 angle and the 3I0-polarised Takagi estimate
    are pure floating-point noise, and their value depends on implementation details (my DFT uses n mod N,
    the lead's uses absolute n).
  - This is not leakage, but it is a robustness defect of the token set. The same holds in DL / TG, where
    the noise happened to agree. Guard these tokens, e.g. set them to 0 when i0r < 1e-6, and disclose it.
    It probably explains part of the H-A / H-A2 differences.
- **Minor:** the two-ended estimate differs per window by up to 0.016 (float order), and the MAE is identical.
  zs_eval's ablation rows (A1 raw-window HGB, A3 un-normalised) were not reproduced (not in the spec). The
  lead's A3 un-normalised tokens (29.69) beat H-A2 (32.98) on MV.

### Disclosure points (CIGRE MV)
1. The one-ended learned method fails zero-shot on MV: H-A2 32.9 %, no better than oracle reactance (33.0 %),
   and on bolted faults far worse than reactance (18.9 against 5.7 %). Only the two-ended locator works
   (4.0 %; 2.4 % on short circuits).
2. The R_f / Z_line ratio of the MV feeders (R_f 10 - 40 ohm on lines of 0.21 - 3.9 ohm) is outside the
   training distribution.
3. The cable nameplate R1 is about 24 % below the simulated value. With pre-fault-identified R1 the two-ended
   MAE halves (4.00 to 1.81 %). This identification is label-free but uses the test episodes' own pre-fault
   data.
4. MainLn8-14 is measured at one end only. y is mapped to 1 - y (verified), and the line is excluded from
   two-ended scoring (n = 52,584 of 56,340).
5. hif and incipient episodes all sit at 50 %, so their per-type MAE reflects a label shortcut.
6. The I0-angle and tak0 tokens are numerical noise for 2ph / 3ph faults (I0 about 0). Guard them.

---

## I0 guard
Spec: `claudedocs/validator_spec_guard.md` (design section 20). CPU only, sklearn HGB with the same settings,
seeds (0, 1, 2), 77-token allow-list and column order as parts 1-4.

Code (`src/validate_ha2/`):
- `guard.py` applies the rule to copies of my cached tables (`results/validate_ha2_guard_tokens_{DL,TG,AD,MV}.parquet`;
  the old files are unchanged) and re-runs H-A / H-A2.
- `freeze_guard.py` writes the hashes.
- `compare_guard.py` compares with the lead and audits the patch.

Independence: my numbers were frozen at 2026-10-04 20:20:43 +05:30, before I opened `src/c1/`, `results/c1_*` or
`results/rerun_guard*`. The hashes are in `results/validate_ha2_guard_frozen.sha256`.

### Guarded windows (i0r < 1e-6)
| set | n | guarded | 2ph | 3ph | other types |
|---|---|---|---|---|---|
| DL | 14,640 | 5,722 | 2,842 / 2,880 | 2,880 / 2,880 | 0 |
| TG | 32,940 | 12,960 | 6,455 / 6,480 | 6,480 / 6,480 | 25 incipient |
| adapt train | 30,260 | 9,367 | 4,336 | 3,968 | 1,043 incipient, 15 1phg shc (+arc), 5 2phg |
| adapt test | 6,883 | 2,259 | 992 | 1,008 | 257 incipient, 2 1phg shc arc |
| CIGRE MV | 56,340 | 17,449 | 8,770 / 10,800 | 8,640 / 10,800 | 39 incipient |

- My counts equal the lead's (`c1_i0_guard_patch.csv`) per grid and per type, and the per-window guard masks are
  identical: 0 mismatches. i0r agrees to within 1.4e-12.
- 4,190 MV 2ph / 3ph windows are not guarded: every one is on MainLn12-13, 13-14 or 8-14, the overhead segments.
  Their i0r is 1e-6 to 1.1e-2, i.e. physical capacitive I0, so leaving them unguarded is correct.

### Old vs new (MAE, % of line length; 3-seed ensemble)
| setting | method | mine old | mine new | lead old | lead new | new delta (mine - lead) |
|---|---|---|---|---|---|---|
| TG -> DL | H-A | 9.012 | 8.964 | 8.937 | 9.041 | -0.077 |
| TG -> DL | H-A2 | 7.808 | 7.754 | 7.747 | 7.689 | +0.065 |
| DL -> TG | H-A | 12.039 | 12.020 | 12.018 | 12.349 | **-0.329** |
| DL -> TG | H-A2 | 10.518 | 10.614 | 10.584 | 10.515 | +0.099 |
| in-grid, adapt test | H-A | 12.677 | 12.766 | 12.777 | 12.677 | +0.089 |
| in-grid, adapt test | H-A2 | 12.055 | 12.048 | 11.982 | 11.977 | +0.071 |
| in-grid, benchmark | H-A | 10.589 | 10.877 | 10.570 | 10.794 | +0.083 |
| in-grid, benchmark | H-A2 | 9.231 | 9.414 | 9.139 | 9.305 | +0.109 |
| CIGRE MV zero-shot | H-A | 31.849 | 31.598 | 32.030 | 31.749 | -0.151 |
| CIGRE MV zero-shot | H-A2 | 32.901 | 32.486 | 32.977 | 32.602 | -0.116 |

Medians and post-fault splits after the guard, mine (lead):

| setting | method | median | post <= 15 ms | post >= 21 ms |
|---|---|---|---|---|
| TG -> DL | H-A2 | 4.01 (3.92) | 13.79 (13.82) | 6.26 (6.16) |
| DL -> TG | H-A2 | 6.38 (6.30) | 15.57 (15.35) | 9.47 (9.40) |
| in-grid, benchmark | H-A2 | 6.94 (6.71) | 17.65 (17.61) | 7.40 (7.27) |
| CIGRE MV | H-A2 | 24.44 (24.79) | 36.54 (36.74) | 31.54 (31.65) |

- The guard moves every cell by at most 0.42 pp (mine) and 0.38 pp (lead). There is no systematic gain. Both of us
  get H-A2 on the in-grid benchmark slightly worse (+0.18 / +0.17).
- Per-window H-A2 agreement with the lead improves:
  - in-grid adapt: mean absolute difference 1.17 to 1.07 pp
  - in-grid benchmark: 1.09 to 1.06 pp
  - MV: 2.07 to 1.87 pp (r 0.992 to 0.993)
- **9 of 10 cells are within the 0.3 pp tolerance. H-A DL -> TG is not (-0.33).** Diagnostic
  (`results/validate_ha2_guard_cmp_hgb_diag.csv`):
  - My HGB on the lead's guarded tables in the lead's column order reproduces all four zero-shot lead numbers
    exactly (9.041 / 7.689 / 12.349 / 10.515).
  - On MY guarded tokens in the lead's column order, the same code gives 8.965 / 7.709 / **12.501** / 10.618.
  - So for H-A DL -> TG, column order alone moves my result by 0.48 pp (12.020 against 12.501), through
    `max_features = 0.8`. Token differences of about 1e-5 move it by 0.15 pp. The lead's single-seed sd for this
    cell is 0.29.
  - This is learner noise, not a method difference. The noise band is about 0.5 pp, wider than the 0.25 pp I
    estimated in Part B. Do not claim H-A or H-A2 differences below about 0.5 pp.

### Token agreement after the guard (mine against the lead's patched tables)
| grid | five guarded tokens: max abs diff, before -> after | max over 2ph / 3ph windows, after |
|---|---|---|
| DL | 5.8e-4 -> 5e-6 | <= 5e-6 |
| TG | 5.0e-3 -> 3e-6 | <= 3e-6 |
| adapt | 1.6e-3 -> 1.4e-5 | 0 |
| MV | **3.0 (tak0), 1.34 (cos I0) -> 5.1e-3** | 5.1e-3 (unguarded overhead-segment windows only) |

- The five tokens now agree. cos / sin are exactly 0 and tak0 is 0.5 on every guarded window.
- The remaining MV differences are pre-existing, unchanged by the guard, and do not involve I0:
  - tak2_bc: 1 window with a clip flip (-1 against 2)
  - react_*: up to 0.35 on about 100 non-2ph/3ph windows (denominator near zero)
  - MV p99 <= 1.2e-3 for every token.
- DL / TG / adapt: max <= 2.2e-3, p99 <= 8e-6 for all 77 tokens.

### Gate verdicts (unchanged)
- **G2-A2** (benchmark-family H-A2 <= 5.85 % and <= 0.8 x best physics 19.95): mine 9.41, lead 9.30 -> **FAIL**
  (as before; the ratio criterion 0.47 is met).
- **G3-MV(b)** (H-A2 <= 0.8 x best equal-information 41.59 = 33.27 AND <= 24.3 %): mine 32.49, lead 32.60. The first
  part passes (as before), the second fails -> **FAIL** (as before). The A3 un-normalised tokens remain better on MV
  (lead 29.69 -> 29.53).

### Conservative H-A2 gain over the best equal-information baseline
Each row uses the worse of my and the lead's H-A2 and the better of my and the lead's baseline. The baselines do not
use tokens, so they are unchanged.

| setting | H-A2 used | baseline used | gain old -> new |
|---|---|---|---|
| TG -> DL | 7.754 (mine) | GRU +Z 11.04 (mine) | 29.3 % -> 29.8 % |
| DL -> TG | 10.614 (mine) | MLP +Z 15.95 (mine) | 33.6 % -> 33.5 % |
| in-grid, adapt test | 12.048 (mine) | GRU +Z 14.21 (lead) | 15.2 % -> 15.2 % |
| in-grid, benchmark | 9.414 (mine) | GRU -raw 11.90 (mine) | 22.4 % -> 20.9 % |
| CIGRE MV | 32.602 (lead) | MLP -raw 41.59 (lead only; not validated by me) | 20.7 % -> 21.6 %, but at the level of the majority predictor (30.35) |

### Audit of the lead's guard
**`local_features.guard_i0`.** It implements the rule exactly:
- strict `i0r < 1e-6`
- i0c = i0s = 0 and ag/bg/cg_tak0 = 0.5
- it reads only i0r, which it does not change
- the dict path (per window, called at the end of `window_feats`) and the DataFrame path (stored tables) are
  equivalent.

**`apply_i0_guard.py`.** Patch equals regeneration:
- The patch always starts from `*_preguard.parquet` (idempotent).
- It asserts that only the 5 columns change and only on guarded rows.
- My independent re-application of the rule to all 11 `*_preguard` tables equals the lead's patched tables exactly
  (`DataFrame.equals`). Only i0c, i0s and ag/bg/cg_tak0 differ from the preguard tables
  (`results/validate_ha2_guard_cmp_patch_audit.csv`).
- `c1_i0_guard_check.csv` regenerates 12 episodes per grid from raw (DL / TG / MV via `grid_pass`, adapt via
  `adapt_tokens`): max diff 0.0.
- It does not regenerate `c1_local_feats*`, which `ha2_eval` uses. I checked that `c1_local_feats` /
  `_TestGrid110kV` equal `c1_grid_DL` / `c1_grid_TG` on all 59 local tokens (max diff 0.0), so the check covers
  them indirectly.

**`c1_i0_guard_patch.csv`.** The counts match mine. The lead's ablation rows (`c1_ablations.csv`) are internally
consistent:
- A2 phasor-only and A2 both equal the `ha2_eval` H-A / H-A2 rows.
- TD-only and A1 are unchanged, as expected, because they do not use the 5 tokens.
- A3 moves 9.91 -> 9.79 (TG -> DL) and 10.13 -> 10.19 (DL -> TG). A3 still beats H-A2 on DL -> TG (10.19 against
  10.52).

**Defect / disclosure (noise table).** On the SNR30 / SNR40 tables the guard fires on only 1 - 7 windows, because
noise makes I0 non-zero. So the noise experiment trains on guarded clean data, where the 5 tokens are constant on
2ph / 3ph, and tests on noisy data, where they carry noise values the model never saw. This is a train/test
mismatch that the guard introduces:
- H-A2 TG -> DL noise penalty (noisy minus clean MAE): from +1.17 / +1.22 pp (SNR40 / 30) before the guard to
  +1.48 / +1.57 pp after (`c1_ha2_eval.csv`).
- The other three cells change by <= 0.1 pp.

Either disclose this, or use a noise-aware guard threshold.

**Process note.** `c1_zs_MV.csv` was rewritten at 20:29, after my freeze; the lead's `zs_eval` rerun was still
running when I first opened the files. The numbers above use the final file. Its H-A2 row equals the MAE of the
`c1_zs_MV_pred.csv` written at 20:19.

Verdict: the guard is implemented correctly, the patched tables equal regeneration, the results reproduce (the
single H-A DL -> TG breach of 0.3 pp is explained by column-order learner noise), and no gate verdict changes.

Files:
- `results/validate_ha2_guard_counts.csv`, `_summary.csv`, `_oldnew.csv`, `_pred_*.csv`, `_tokens_*.parquet`,
  `_frozen.sha256`, `validate_ha2_guard.log`
- after the freeze: `results/validate_ha2_guard_cmp_{patch_audit,masks,token_diff,perwindow,hgb_diag}.csv`,
  `validate_ha2_guard_cmp.log`

---

## Classical baselines + measurement chain
Spec: `claudedocs/validator_spec_step3.md` (design sections 21, 21a, 22). CPU only; sklearn HGB with the part 1-5
settings, seeds 0-2, 77-token allow-list, I0 guard applied to every table.

Code (`src/validate_ha2/`): `remote.py` (caches the remote-terminal channels of DL / TG / adapt-test episodes),
`classical.py` (part A), `chain.py` (part B), `freeze_step3.py`, and `compare_step3.py` (after the freeze).

Independence: my numbers were frozen at **2026-10-04 21:37:34 +05:30**, before I opened `src/c1/`,
`results/c1_classical*` or `results/c1_chain*`. Hashes are in `results/validate_ha2_step3_frozen.sha256`.
Every H-A2 cell was run on all windows; no subset was used.

### A. Classical one-ended locators (oracle loop, pre-fault memory, remote source impedance as an oracle)
**Eriksson derivation.** Start from V = d ZL I + Rf If and dI = If (ZSR + (1-d) ZL) / (ZSL + ZL + ZSR). This gives
(V/(ZL I) - d)(1 + ZSR/ZL - d) = Rf dI/(ZL I)(1 + (ZSL + ZSR)/ZL), which is exactly the registered K1 / K2 / K3
equation. Synthetic two-source check over 2,000 random cases (`validate_ha2_step3_eriksson_check.log`):
- residual at the true (d, Rf): 3e-14
- the true d is always a root (within 5e-12)
- the registered selection rule (root in [-0.1, 1.1] closest to 0.5) picks the wrong root in 20 / 2,000 cases, all of
  them cases with both roots inside the interval. This is an inherent ambiguity of the rule.

**Bolted check** (DL, R_f <= 1 ohm, short circuits, post-fault >= 25 ms, n = 3,600). MAE in %:

| | R | T | T2 (3ph = R) | MT | E |
|---|---|---|---|---|---|
| mine | 1.128 | 1.540 | 1.516 | 1.531 | 2.012 |
| lead | 1.13 | 1.54 | (5.61 before 21a) | 1.53 | 1.08 |

All methods pass the 3 % limit. My E is higher because 65 / 720 bolted 1phg-arc windows have no admissible root.
Under 21a those windows score 0.5. Defined windows only give an E MAE of 1.78 on 1phg arc.

**MAE (% of line length), all windows. Undefined estimates are scored as 0.5. Lead values in brackets.**

| grid (n) | R | T | T2 | MT | E | H-A2 (mine) | best classical | H-A2 / best | claim (<= 0.8) |
|---|---|---|---|---|---|---|---|---|---|
| DL (14,640) | 19.95 (19.95) | 16.37 (16.37) | 20.50 (20.50) | 16.34 (16.36) | 10.55 (10.46) | 7.75 (7.69) | E | 0.735 (0.735) | yes (yes) |
| TG (32,940) | 14.45 (14.45) | 12.61 (12.61) | 15.34 (15.34) | 12.32 (12.45) | 8.83 (9.55) | 10.61 (10.52) | E | 1.201 (1.101) | no (no) |
| adapt test (6,883) | 32.20 (32.20) | 26.79 (26.79) | 32.35 (32.35) | 25.92 (26.81) | 16.13 (22.88) | 12.05 (11.98) | E | **0.747** (0.523) | yes (yes) |
| CIGRE MV (56,340) | 32.98 (32.98) | 34.56 (34.56) | 31.31 (31.31) | 28.69 (28.83) | **23.75** (30.87) | 32.49 (32.60) | E (lead: MT) | 1.368 (1.131) | no (no) |

Undefined estimates (mine / lead):
- E: DL 2,282 / 0, TG 4,132 / 0, adapt 2,473 / 0, MV 31,466 / 3,756
- MT: TG 31 / 0, adapt 268 / 5, MV 1,898 / 1,596

**Eriksson against H-A2 by post-fault time (mine; lead in brackets)**

| grid | method | <= 15 ms | 20-30 | 35-50 | >= 55 |
|---|---|---|---|---|---|
| DL | E | 31.27 (35.55) | 6.83 (6.49) | 5.42 (4.84) | 5.33 (3.45) |
| DL | H-A2 | 13.79 (13.82) | 6.53 (6.48) | 5.90 (5.85) | 6.56 (6.43) |
| TG | E | 30.90 (34.52) | 4.97 (5.66) | 3.40 (3.94) | 3.21 (2.56) |
| TG | H-A2 | 15.57 (15.35) | 9.05 (8.91) | 9.00 (8.93) | 9.98 (9.95) |
| adapt | E | 27.48 (40.98) | 16.67 (17.01) | 12.02 (15.39) | 13.29 (20.34) |
| adapt | H-A2 | 18.80 (17.87) | 10.39 (9.55) | 9.20 (9.22) | 11.47 (11.60) |
| MV | E | 30.39 (46.45) | 22.38 (27.37) | 21.71 (27.29) | 22.43 (27.12) |
| MV | H-A2 | 36.54 (36.74) | 31.69 (31.73) | 31.61 (31.72) | 31.42 (31.53) |

- In both versions, on the 110 kV benchmark grids **H-A2's advantage over Eriksson comes entirely from the first
  15 ms.** From 20 ms on, E is equal to or better than H-A2 on DL, and 2-3x better on TG.
- On adapt test, H-A2 is better than E in every bin.
- R_f bins and short-circuit-only rows are in `validate_ha2_step3_classical_summary.csv`.

**Per-window agreement** (`validate_ha2_step3cmp_classical.csv`):
- R, T and T2 match the lead on DL, TG and adapt within 0.004 pp per window.
- On MV, R differs by up to 13 pp on 287 windows, with the same MAE. These windows have a near-zero loop current
  (same phenomenon as the part-4 react tokens).

**Cause of the E differences (verified):**
- `classical_oe.eriksson` adds two fallbacks that the registered rule does not contain:
  - negative discriminant: returns p/2, the real part of the complex roots
  - no root in [-0.1, 1.1]: returns the out-of-range root closest to 0.5
- My code with only these two fallbacks added reproduces the lead's E exactly: 10.455 / 9.546 / 22.884 / 30.867, DL
  bolted 1.078, 0 undefined.
- The MT differences come from the dI0 threshold and from in-iteration NaN / clip handling:
  - lead: absolute |dI0| > 1e-9, NaN set to 0.5 and clipped to [-1, 2] inside the 3 iterations
  - mine: |dI0| > 1e-6 |I1| + 1e-9, NaN propagates
  - effect: up to 0.89 pp (adapt).

**Verdict on the comparison claim.** It is the same under both E variants:
- claim on DL (0.735) and adapt (0.747 mine, 0.523 lead)
- no claim on TG or MV.

But the adapt margin depends on E's undefined-handling rule: 0.747 against the 0.8 threshold under the registered
rule, 0.523 under the lead's fallbacks. On MV the spec-literal E (23.75) is the best classical method, partly because
of the 0.5 mid-line prior on 31,466 undefined windows.

### B. Measurement chain (DL = TG -> DL, TG = DL -> TG; all windows; mine, lead in brackets)
| grid | scenario | sat % | two-ended | react oracle | Takagi-I2, no 3ph fix | H-A2 trained on clean | H-A2 matched FULL |
|---|---|---|---|---|---|---|---|
| DL | clean | - | 0.612 (0.612) | 19.95 (19.95) | 24.01 (24.01) | 7.754 (7.689) | |
| DL | AA | - | 0.589 (0.589) | 20.00 (20.00) | 24.00 (24.00) | 7.754 (7.692) | |
| DL | CT-mild | 0.0 (0.0) | 0.612 (0.612) | 19.95 (19.95) | 24.01 (24.01) | 7.757 (7.690) | |
| DL | CT-severe | **18.0 (20.4)** | 0.655 (0.662) | 19.99 (19.98) | 24.08 (24.15) | 9.331 (9.392) | |
| DL | CVT-5 | - | 2.474 (2.474) | 21.17 (21.17) | 24.89 (24.89) | 9.007 (8.975) | |
| DL | CVT-15 | - | 4.900 (4.900) | 24.57 (24.57) | 26.83 (26.83) | 11.679 (11.662) | |
| DL | FULL | 18.0 (20.4) | 4.857 (4.858) | 24.52 (24.48) | 26.99 (27.04) | 12.534 (12.262) | 8.183 (8.169) |
| TG | clean | - | 0.411 (0.411) | 14.45 (14.45) | 19.79 (19.79) | 10.614 (10.515) | |
| TG | AA | - | 0.385 (0.385) | 14.50 (14.49) | 19.75 (19.75) | 10.649 (10.547) | |
| TG | CT-mild | 0.0 (0.0) | 0.411 (0.411) | 14.45 (14.45) | 19.79 (19.79) | 10.621 (10.521) | |
| TG | CT-severe | **18.1 (21.2)** | 0.468 (0.483) | 14.50 (14.48) | 19.64 (19.78) | 12.267 (12.105) | |
| TG | CVT-5 | - | 3.209 (3.209) | 16.01 (16.01) | 21.02 (21.02) | 12.001 (11.916) | |
| TG | CVT-15 | - | 6.704 (6.704) | 20.29 (20.29) | 23.90 (23.90) | 14.572 (14.500) | |
| TG | FULL | 18.1 (21.2) | 6.723 (6.729) | 20.26 (20.22) | 23.85 (23.97) | 15.571 (15.283) | 11.856 (11.783) |

- **Robustness wording (rise over clean; two-ended <= 1.0 pp, one-ended <= 2.0 pp).** My verdicts equal the lead's in
  every cell:
  - two-ended: robust to AA, CT-mild and CT-severe; not robust to CVT-5 (+1.86 / +2.80), CVT-15 (+4.29 / +6.29) or
    FULL (+4.24 / +6.31)
  - H-A2 trained on clean: robust to AA, CT-mild, CT-severe (+1.58 / +1.65) and CVT-5 (+1.25 / +1.39); not robust to
    CVT-15 (+3.92 / +3.96) or FULL (+4.78 / +4.96)
  - H-A2 matched FULL: +0.43 / +1.24
  - reactance: robust except CVT-15 and FULL
- Takagi-I2 with the 21a 3ph fix (the definition in part A) is 20.50 / 15.34 clean. The lead's chain table uses the
  unfixed token (24.01 / 19.79), which is inconsistent with 21a. The verdicts do not change.
- CVT residual voltage, % of the pre-collapse peak:
  - mine (worst over collapse angle, peak in the half-cycle after t): CVT-5 20.7 / 2.9 / 0.03; CVT-15 51.8 / 27.0 / 7.3
    at 10 / 20 / 40 ms
  - lead (collapse at peak, following cycle): CVT-5 16.8 / 1.7 / 0.0; CVT-15 51.0 / 26.7 / 7.0
  - The definitions differ and the results are consistent.
- **Per-window agreement** (`validate_ha2_step3cmp_chain_perwindow.csv`):
  - clean, AA, CT-mild and CVT: two-ended within 0.002 pp per window, tokens within 0.004
  - exception: in CT-mild a single tak0 value differs by 3.0, a guard-threshold flip at i0r of about 1e-6

**Cause of the CT-severe difference (verified).**
- The design writes i_e = sgn(l)(10/RP)(w|l|/(sqrt2 Vs))^S but never defines RP. I took RP = 1.
- The lead (`meas_chain.py`) uses RP = rms/peak of sin^S = 0.346. This lowers the knee flux by 4.7 %, so the 0.6 pu
  remanence is 0.63 of the effective knee.
- My earlier "1.6 %" estimate of this effect was wrong.
- With identical parameters, the lead's Newton and my bracketed Newton give the same i_e to within 6e-11 A. Both
  backward-Euler residuals are <= 2e-14 lambda_s.
- With the lead's parameterisation, my code reproduces the lead's saturation flags window for window (100 %
  agreement; 20.38 / 21.17 %).
- The i_s definition (actual i2 or ideal i1/N) changes the fraction by only 0.02 pp.
- The CT-severe token and H-A2 differences (<= 0.29 pp) follow from the RP choice.

H-A2 deltas (mine - lead):
- clean-trained cells: -0.07 to +0.29 pp
- matched FULL: +0.01 / +0.07 pp

All are inside the about 0.5 pp learner-noise band measured in part 5. The clean cells are the same 7.754 / 10.614 as
part 5: my clean recomputation equals my cached guarded tables bit for bit (max diff 0.0).

### Audit and leakage
- **`classical_oe.py`:**
  - oracle loop from labels (disclosed)
  - pre-fault cycle [nf - 10 - N, nf - 10), the same as mine
  - ZSA local and ZSB / Z0SB from the remote superimposed phasors (intended, disclosed oracle)
  - Z from the grid pickle
  - MV 8-14: local = bus 14, y = 1 - y, no remote, so E / MT are NaN
  - No label reaches an estimate except through the oracle loop.
  - **Defect:** the two unregistered E fallbacks above. Register them as an amendment, or use the 21a rule (NaN -> 0.5),
    and report both. The log line "T2 5.61" is from before 21a; the report CSV uses the fixed T2.
- **`classical_report.py`:**
  - H-A2 zero-shot predictions are re-fitted on `c1_grid_<src>` with `ALLOW` and `check_allow`
  - one-to-one join on (sim_idx, post_ms) with a length assert
  - NaN -> 0.5
  - post-fault bins (0, 17.5], (17.5, 32.5], (32.5, 52.5], > 52.5, equal to mine
  - Clean.
- **`meas_chain.py`, `chain_pass.py`, grid_pass hook:**
  - The chain is applied to each terminal record independently: `chain(XS, sid, 0)`, `chain(XR, sid, 1)`.
  - Tokens are computed from XS only, with the I0 guard applied inside `window_feats`.
  - XR enters only `td2_est`.
  - The clean-trained H-A2 is fitted on the clean `c1_grid_<src>` and tested on the distorted target. The matched model
    is fitted on FULL source only.
  - Alignment with the clean table is asserted.
  - The steady-state pad is 1,920 samples.
  - Leakage: **none found (CLEAN)**.
  - Defects:
    - RP is undefined in design 22; add its definition.
    - The chain's Takagi-I2 omits the 21a 3ph fix.

### Disclosure points
1. **Takeover by post-fault time.** H-A2 beats the best classical one-ended method on DL (0.74) and on adapt test
   (0.75 to 0.52, depending on E's undefined rule). It does not beat it on TG (1.10 - 1.20) or on MV. On DL and TG the
   advantage is confined to the first 15 ms: from 20 ms on, Eriksson with remote source impedances is as good or better.
2. Eriksson's score depends on how no-real-root and out-of-range cases are handled: up to 7 pp on adapt and MV. Register
   one rule. The classical methods use an oracle loop, pre-fault memory and remote source impedances, which favours them.
3. The CVT transient (tau 15 ms) is the dominant chain error for every method, including the two-ended locator
   (+4 - 6 pp). CT saturation (about 18 - 21 % of windows) costs H-A2 about 1.6 pp and the two-ended locator < 0.1 pp.
   Matched-chain training recovers most of the H-A2 loss.

Files:
- `results/validate_ha2_step3_{classical_perwindow.parquet, classical_summary.csv, classical_bolted.csv,
  classical_vs_ha2.csv, classical.log, eriksson_check.log, chain_tokens_{DL,TG}.parquet, chain_summary.csv,
  chain_saturation.csv, chain_cvt_residual.csv, chain_pred_*.csv, chain.log, frozen.sha256}`
- after the freeze: `results/validate_ha2_step3cmp_*`

---

## Conditioning + Fig 4
Spec: `claudedocs/validator_spec_cond.md` (design sections 21b and 23). CPU only. No retraining: I used my guarded
H-A2 predictions (part 5), my part-1 MLP / GRU npy predictions, my part-4 / part-6 two-ended estimates, and my frozen
part-6 classical table. The 21b rule applies to E: undefined estimates score as 0.5.

Code (`src/validate_ha2/`): `cond.py` (parts A and B), `freeze_cond.py`, and `compare_cond.py` (run after the freeze).

Independence: my numbers were frozen at **2026-10-04 22:09:18 +05:30**, before I opened `src/c1/conditioning.py`,
`fig4_data.py`, `equal_info_preds.py`, `results/c1_cond*`, `c1_fig4*` or `c1_equal_info_preds*`. The hashes are in
`results/validate_ha2_cond_frozen.sha256`.

### A. Conditioning
**Derivation (mine; it agrees with design 23).**
- Start from the loop equation V = d Z_L I + R_f I_F.
- A one-ended locator that assumes angle(I_F / I) = b^ multiplies V/I by e^{-j b^} and keeps the imaginary part. This gives
  d^ - d = rho |I_F/I| sin(b - b^) / sin(theta_L - b^), with rho = R_f / |Z_L| and b = angle(I_F / I).
- Reactance: b^ = 0.
- Takagi: dividing Im(V conj dI) / Im(Z I conj dI) by |I||dI| gives the same form with b^ = angle(dI / I).
- Synthetic check (20,000 random cases, any polarising current):
  - reactance matches the formula to within 2e-13
  - Takagi matches to within 5e-9 (worst case is a near-singular denominator)
  - `validate_ha2_cond_synthetic_check.csv`
- D_req(5 %) = arcsin(min(1, 0.05 / (rho kappa))). The design does not say which b^ goes into kappa. The lead uses
  Takagi's b^ (kappa_T). I computed both kappa_T and the linearised kappa at b^ = b. The share below 1 deg differs by
  at most 0.005 between the two.

**Data:** 1phg_shc windows, post-fault >= 25 ms, both ends measured. I_F is the sum of the faulted-phase currents at
both ends. I computed the unclipped estimates est_R and est_T again; they equal my part-6 table exactly (max diff 0.0).

| grid (n windows / episodes) | identity R, median / p90 (pp) | identity T, median / p90 (pp) | rho, median / p90 | kappa_T median | Takagi angle error, median / p90 (deg) | D_req(5 %) median (deg) | share D_req < 1 deg | share where Takagi angle error > D_req |
|---|---|---|---|---|---|---|---|---|
| DL (2,160 / 180) | 0.418 / 4.11 | 0.425 / 4.10 | 1.40 / 6.10 | 0.990 | 0.35 / 0.90 | 1.96 | **0.399** | 0.163 |
| TG (4,860 / 405) | 0.398 / 3.12 | 0.395 / 3.09 | 1.27 / 6.10 | 0.810 | 0.39 / 1.23 | 2.75 | **0.274** | 0.177 |
| adapt test (661 / 60) | 1.401 / 4.35 | 1.407 / 4.30 | 3.72 / 8.48 | 0.951 | 0.24 / 0.66 | 0.75 | **0.634** | 0.130 |
| CIGRE MV (7,560 / 630) | 1.519 / 17.83 | 1.580 / 19.02 | 9.58 / 81.74 | 1.373 | 3.95 / 14.74 | 0.158 | **0.762** | 0.966 |

**Lead's values.** Every cell agrees with the lead to 4 decimals except the MV Takagi-angle p90 (14.735 vs 14.738).
The window join matched 15,241 / 15,241 windows. The one apparent mismatch is a rounding artefact: adapt sim 1074 at
63.4375 ms rounds to 63.437 vs 63.438.

**Identity check.**
- The formula explains the observed errors on 110 kV: median deviation 0.4 pp, against an unclipped R_f = 40 ohm
  Takagi MAE of 10.6 % (DL) and 39.7 % (TG).
- The model residual |V - yZI - R_f I_F| / |V| has a median of 0.24 - 0.46 %.
- The pre-fault |I_S + I_R| / |I_S| is 1.4 - 2.8 %, which is the line charging current. This also confirms that both
  ends use an into-line current convention.
- On MV the p90 deviation is 18 - 19 pp. All of it comes from high rho: the R_f = 40 ohm median deviation is 12.8 pp,
  while the observed unclipped |e| is about 1,500 %. Model errors such as cable charging and the 24 % R1 mismatch
  (part 4) are multiplied by rho. Report the MV identity in relative terms.

**Stated expectation (design 23): confirmed.**
- On MV, 76 % of windows need D_req < 1 deg. On 110 kV the shares are 27 % (TG) and 40 % (DL).
- Takagi's actual polarising angle is off by a median of 3.95 deg on MV, against 0.35 - 0.39 deg on 110 kV. It misses
  D_req on 97 % of MV windows.
- By R_f (`validate_ha2_cond_by_rf.csv`): on DL / TG no window with R_f = 1 ohm needs < 1 deg, but 80 - 95 % of the
  windows with R_f = 40 ohm do. On MV the R_f = 10 and 40 ohm windows all need < 1 deg.
- Adapt test sits in between (63 %) because its R_f is continuous (rho median 3.7). A paper sentence of the form "most
  110 kV windows do not need sub-degree accuracy" holds for the DL / TG benchmark, not for adapt.

**(e) MAE (%) by rho bin.** All short-circuit windows with post-fault >= 25 ms. Mine; lead deltas are below.

| grid | rho bin | n | two-ended | R | T | E (21b) | H-A2 |
|---|---|---|---|---|---|---|---|
| DL | [0, 0.3) | 3,600 | 0.08 | 1.13 | 1.54 | 2.01 | 1.40 |
| DL | [0.3, 1) | 900 | 0.09 | 12.23 | 12.49 | 9.63 | 11.83 |
| DL | [1, 3) | 2,700 | 0.19 | 6.95 | 7.98 | 4.64 | 7.50 |
| DL | [3, 10) | 3,600 | 0.62 | 33.10 | 19.36 | 8.66 | 8.99 |
| TG | [0, 0.3) | 8,100 | 0.10 | 1.09 | 1.26 | 1.23 | 4.80 |
| TG | [0.3, 1) | 1,800 | 0.10 | 8.01 | 8.51 | 7.37 | 9.96 |
| TG | [1, 3) | 6,300 | 0.17 | 2.44 | 3.86 | 1.56 | 8.27 |
| TG | [3, 10) | 8,100 | 0.39 | 19.56 | 11.33 | 6.10 | 15.22 |
| MV | [0, 0.3) | 1,800 | 1.14 | 4.22 | 8.58 | 4.56 | 6.55 |
| MV | [0.3, 1) | 5,400 (two-ended 4,500) | 1.49 | 4.84 | 8.09 | 9.61 | 14.94 |
| MV | [1, 3) | 5,400 | 1.88 | 16.46 | 18.26 | 15.67 | 28.50 |
| MV | [3, 10) | 8,100 (7,200) | 1.48 | 23.30 | 26.23 | 23.12 | 31.78 |
| MV | [10, 30) | 9,900 (9,000) | 2.98 | 38.91 | 40.51 | 28.60 | 39.75 |
| MV | [30, inf) | 9,900 | 2.20 | 45.18 | 47.26 | 30.75 | 39.74 |

- Lead deltas:
  - two-ended, R, T and E: <= 0.002 pp in every cell, with identical n
  - H-A2: -0.30 to +0.61 pp (largest DL [0.3, 1), n = 900; TG [0, 0.3) +0.37). This is independent-refit noise, as
    in parts 1-6.
- The lead's right-closed bins (pd.cut) and my left-closed bins give identical counts, because no rho falls on an edge.
- Undefined E windows, scored 0.5: about 18,554 of 40,500 MV short-circuit windows (46 %), rising with rho (6,783 /
  9,900 in [30, inf)). So E's MV numbers at high rho are largely the 0.5 prior.
- The two-ended locator is flat in rho (0.08 - 0.62 % on 110 kV, 1.1 - 3.0 % on MV). Every one-ended method degrades
  with rho.

### B. Fig 4: zero-shot MAE (%) by post-fault time
DL = TG -> DL, TG = DL -> TG, all official windows. Neural cells are the mean of 3 single-seed MAEs. H-A2 is the 3-seed
mean prediction.

| grid | t (ms) | two-ended | R | T | E | H-A2 | MLP-raw | MLP+Z | GRU-raw | GRU+Z | best one-ended, mine (lead) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DL | 5 | 2.13 | 49.65 | 49.80 | 32.48 | 17.88 | 18.41 | 17.71 | 14.17 | 13.44 | GRU+Z (GRU+Z) |
| DL | 10 | 1.40 | 42.68 | 38.88 | 29.66 | 13.33 | 15.40 | 13.61 | 12.24 | 12.23 | GRU+Z (GRU+Z) |
| DL | 15 | 1.12 | 33.39 | 33.17 | 31.66 | 10.16 | 13.40 | 12.05 | 11.32 | 11.37 | H-A2 (H-A2) |
| DL | 20 | 0.91 | 17.01 | 12.86 | 8.71 | 7.37 | 12.82 | 11.32 | 11.18 | 11.00 | H-A2 (H-A2) |
| DL | 25 | 0.79 | 15.89 | 11.75 | 6.19 | 6.50 | 12.66 | 11.07 | 11.80 | 11.38 | E (E) |
| DL | 30 | 0.69 | 15.30 | 11.17 | 5.59 | 5.72 | 11.99 | 10.33 | 11.81 | 11.26 | E (E, margin 0.00) |
| DL | 35 | 0.63 | 15.25 | 11.10 | 5.73 | 5.73 | 12.04 | 10.59 | 11.26 | 10.74 | E by 0.003 (**H-A2** by 0.06) |
| DL | 40-80 | 0.07-0.58 | 14.0-15.1 | 9.9-11.0 | 5.2-5.5 | 5.9-6.8 | 12.1-13.3 | 10.3-12.3 | 10.7-11.9 | 10.0-11.1 | E (E) in every bin |
| TG | 5 | 1.40 | 46.60 | 49.03 | 32.53 | 19.29 | 21.22 | 20.30 | 20.40 | 21.78 | H-A2 (H-A2) |
| TG | 10 | 0.86 | 39.34 | 35.94 | 28.95 | 15.64 | 19.16 | 17.44 | 18.82 | 18.97 | H-A2 (H-A2) |
| TG | 15 | 0.69 | 29.13 | 30.29 | 31.22 | 11.77 | 18.13 | 16.15 | 18.27 | 18.00 | H-A2 (H-A2) |
| TG | 20 | 0.57 | 11.31 | 8.72 | 7.31 | 9.32 | 17.62 | 15.68 | 18.26 | 17.68 | E (E); 2nd T |
| TG | 25-80 | 0.10-0.50 | 7.9-10.1 | 5.7-7.4 | 3.2-4.2 | 8.8-10.4 | 16.9-18.0 | 15.0-16.5 | 17.2-18.5 | 16.3-18.0 | E (E); 2nd T in every bin |
| DL | all | 0.61 | 19.95 | 16.37 | 10.55 | 7.75 | 13.08 | 11.76 | 11.53 | 11.04 | H-A2 (H-A2) |
| TG | all | 0.41 | 14.45 | 12.61 | 8.83 | 10.61 | 17.78 | 15.95 | 18.07 | 17.48 | E (E) |

**Match with the lead's table:** 33 / 34 rows.
- Two-ended, R, T and E are identical to 2 decimals in every bin.
- H-A2 differs by -0.11 to +0.34 pp. The neural cells differ by -1.71 to +1.11 pp; these are independently trained
  nets, and the lead's re-run reproduces its own recorded MAEs to 2e-6.
- The single mismatch is DL 35 ms. It is a tie: E 5.728 vs H-A2 5.731 in my table, E 5.73 vs H-A2 5.67 in the lead's,
  so the margin is below the H-A2 refit noise. DL 25-35 ms is a tie band (margins 0.00 - 0.30 pp in both tables).

**What the figure supports:**
- DL: the GRU+Z equal-information net is the best one-ended method at 5 and 10 ms (10 ms by 1.1 pp over H-A2), not H-A2.
  H-A2 is best at 15-20 ms. Eriksson is best from 25 ms on.
- TG: H-A2 is best at 5-15 ms, and Eriksson from 20 ms on (T is second, H-A2 behind T).
- Eriksson uses remote source impedances as an oracle (design 21), so it is not strictly local information.
- The text must not claim that H-A2 is the best one-ended method "in the first cycle" on DL.

### Audit of the lead's code
- **`conditioning.py`:**
  - restricted to `flt_1phg_shc` with both terminals present, so MV 8-14 drops out
  - I_F = S[p] + Rr[p]; loop V / I / dI from `classical_oe.loop_quant` (k0 compensated), the same as mine
  - D_req uses |kappa_T|, which is a defensible reading of design 23 but should be stated in the paper
  - per-window agreement:
    - DL / TG / adapt: within 0.026 pp (e) and 3e-5 deg
    - MV: e differs by up to 69 pp on 720 windows, all on the two distributed overhead segments 12-13 and 13-14 with
      R_f >= 10 ohm. There |e| is 100 - 6,000 % of the line, so the relative difference is <= 0.3 %, consistent with
      a small Z0 / k0 parameter difference. Apart from the MV median |e_obs_R| (125.03 vs 125.01), no summary cell
      moves by more than 0.003.
  - **Defect:** the summary columns `mae_obs_R` / `mae_obs_T` hold the MEDIAN |e_obs| (`.median()`), not a mean.
    Rename them; my medians reproduce the lead's (e.g. MV R 125.03 vs 125.01). Their unclipped means are much larger:
    MV 534 / 574 %, DL 60.7 / 4.4 %.
- **`conditioning.rho_bins`:**
  - excludes incipient / hif and keeps post >= 25 ms
  - two-ended NaN (8-14) is dropped from the two-ended cell only
  - E / R / T NaN -> 0.5 (21b)
  - scored against the local y, which matches my y_local per bin exactly
  - Clean.
- **`fig4_data.py`:**
  - one-to-one joins on (sim_idx, post_ms rounded)
  - asserts no missing neural predictions; neural error = mean over seeds of |clipped pred - y| (`equal_info.fit_predict`
    clips to [0, 1])
  - H-A2 comes from `classical_report.zs_pred`, which re-fits with the allow-list (audited in part 6)
  - Clean. Note: the two-ended column also goes through fillna(0.5), which has no effect on DL / TG.
- **`equal_info_preds.py`:** the same recipe and seeds as `equal_info.py`, and it reproduces the recorded per-seed MAEs
  within 2e-6. Clean.
- **Leakage:** none found. Labels enter only as the oracle loop (R / T / E, disclosed), as y, and as R_f in the analysis
  quantities (rho, e_pred), which is intended for a diagnostic.

### Disclosure points
1. Say which b^ goes into kappa (Takagi's) and state that line charging is ignored in I_F (pre-fault KCL residual is
   1.4 - 2.8 %).
2. The MV identity holds only in relative terms: absolute p90 18 - 19 pp at rho up to 190.
3. 46 % of MV short-circuit E estimates are undefined (scored 0.5), so the high-rho E cells are mostly the prior.
4. In Fig 4, H-A2 is not the best one-ended method at 5-10 ms on DL (GRU+Z is). DL 25-35 ms is a tie between E and H-A2.

Files:
- `results/validate_ha2_cond_{perwindow_1phg.parquet, summary.csv, by_rf.csv, rho_bins.csv, fig4.csv,
  synthetic_check.csv, frozen.sha256}` and `validate_ha2_cond.log`
- after the freeze: `results/validate_ha2_condcmp_{perwindow, summary, rho_bins, fig4}.csv` and `validate_ha2_condcmp.log`

---

## Part 8: review-driven checks (design 24 / 24a) and cells never validated before
Spec: `claudedocs/validator_spec_part8.md`. Code (`src/validate_ha2/`): `part8.py` (A, B, D1), `part8_summary.py` (A-D1
tables, C), `part8_nn_mv.py` (D2, GPU), `part8_abl.py` (D3), `freeze_part8.py`, `compare_part8.py` (after the freeze).
Reused unchanged: `mv.two_ended`, `tokens.phasor`, `ingrid.rule_windows`, `neural.run`, `HGB_PARAMS`.

Independence: my numbers were frozen at **2026-10-06 07:56:05 +05:30**, before I opened `src/c1/review_checks.py`,
`sync_subsample.py`, `adapt_td.py`, `ablations.py`, `zs_equal_info.py` or any of `results/c1_review_*`,
`c1_adapt_td.csv`, `c1_ablations.csv`, `c1_zs_equal_info_MV.csv`. Hashes: `results/validate_ha2_part8_frozen.sha256`.
At most 6 worker processes; one GPU job (D2) only; sklearn HGB, LightGBM not used.

Choices the spec left open, with my option:
- Integer shift +k: remote record delayed, rem'[n] = rem[n - k] (np.roll), derivative shifted with it. Fractional delay:
  whole-record rfft phase shift exp(-j w tau), n = 4801, derivative recomputed. I ran both signs (spec: + only).
- A3: zr / zi / absz = Re, Im, |V/I| of the loop in ohm, **unclipped** (0 when |I| <= 1e-9, as in tokens.py).
- Post-fault bins: I first used p <= 15 / 30 / 50. That is wrong for adapt test (window ends are not on the 5 ms grid
  there); the comparison below uses the lead's edges (17.5, 32.5, 52.5), with which my DL / TG / MV bins are unchanged.

### Numbers: mine (frozen) against the lead. MAE, % of line length
| cell | mine | lead | delta |
|---|---|---|---|
| A phasor two-ended, DL / TG | 0.660 / 0.939 | 0.660 / 0.939 | 0.000 / 0.000 |
| A phasor two-ended, adapt test all / SC | 7.183 / 1.148 | 7.183 / 1.148 | 0.000 / 0.000 |
| A phasor two-ended, MV 14 seg all / SC | 4.580 / 2.933 | 4.589 / 2.933 | -0.009 / 0.000 |
| A by bin DL (<=15 / 20-30 / 35-50 / >=55) | 2.77 / 0.47 / 0.11 / 0.05 | same | 0.00 |
| A by bin TG | 3.89 / 0.69 / 0.16 / 0.08 | same | 0.00 |
| A by bin adapt (lead edges) | 8.99 / 4.91 / 5.23 / 8.51 | same | 0.00 |
| A by bin MV | 7.81 / 4.54 / 4.32 / 3.12 | 7.81 / 4.55 / 4.35 / 3.12 | <= 0.025 |
| B sync -20/-10/-5/-1/+1/+5/+10/+20 samples, DL | 34.94 / 28.60 / 21.52 / 7.08 / 7.26 / 21.51 / 28.53 / 35.09 | same | 0.00 |
| B sync, TG | 33.14 / 26.85 / 20.25 / 6.50 / 6.78 / 21.05 / 28.53 / 35.68 | same | 0.00 |
| B sub-sample +1 / +10 / +50 us, DL (TD) | 0.723 / 1.756 / 4.175 | 0.723 / 1.756 / 4.175 | 0.000 |
| B sub-sample +1 / +10 / +50 us, TG (TD) | 0.502 / 1.534 / 3.918 | same | 0.000 |
| B sub-sample -1 / -10 / -50 us, DL / TG (mine only) | 0.629 / 1.630 / 4.069; 0.439 / 1.300 / 3.666 | - | - |
| B R1, L1 x0.90 / 0.95 / 1.05 / 1.10, DL | 2.439 / 1.710 / 1.644 / 2.573 | same | 0.000 |
| B same, TG | 2.849 / 1.891 / 1.834 / 2.964 | same | 0.000 |
| B R1 only x0.8 / x1.2, DL / TG | 1.354 / 1.324; 1.176 / 1.109 | same | 0.000 |
| C oracle dtd, DL / TG / adapt / MV | 22.67 / 19.38 / 36.61 / 44.69 | 22.67 / 19.38 / 36.61 / 44.69 | 0.00 |
| C SC only, DL / TG / adapt / MV | 22.22 / 18.87 / 32.24 / 44.46 | same | 0.00 |
| D1 two-ended TD adapt test, all / SC (n 6,883 / 4,768) | 6.153 / 0.690 | 6.15 / 0.69 | 0.000 |
| 24(d) TD with pre-fault (<= 50 ms) / post-only (> 52.5), DL | 0.927 / 0.073 | same | 0.000 |
| 24(d) same, TG / adapt / MV | 0.589 / 0.105; 5.638 / 7.108; 4.759 / 2.712 | same | 0.000 |
| D2 MV zero-shot MLP -raw (3 seeds) | 41.29 +- 1.21 | 41.59 +- 0.25 | -0.30 |
| D2 MLP +Z | 48.16 +- 0.13 | 48.26 +- 0.10 | -0.10 |
| D2 GRU -raw | 43.87 +- 1.96 | 44.20 +- 1.43 | -0.33 |
| D2 GRU +Z | 46.78 +- 0.61 | 45.44 +- 2.08 | +1.34 |
| D3 TD tokens only, TG->DL / DL->TG | 11.483 / 11.825 | 11.364 / 11.836 | +0.12 / -0.01 |
| D3 H-A2 reference (same run) | 7.754 / 10.614 | 7.689 / 10.515 | +0.07 / +0.10 |
| D3 A3 un-normalised | **7.883 / 9.902** | **9.795 / 10.192** | **-1.91** / -0.29 |
| D3 A1 HGB raw window | 14.514 / 19.524 | 14.514 / 19.524 | 0.000 / 0.000 |
| D3 A1 raw window + Z | 14.062 / 19.254 | 14.062 / 19.267 | 0.000 / -0.013 |

Other part-8 numbers (mine): the TD locator by bin (lead edges) is DL 1.55 / 0.80 / 0.56 / 0.07, TG 0.99 / 0.50 /
0.36 / 0.10, adapt 6.51 / 5.27 / 5.01 / 7.11, MV 5.90 / 4.58 / 4.03 / 2.71, all equal to the lead. The C bins
(<= 17.5 / > 52.5 ms) equal the lead's le15 / ge55 in every grid.

### Explained differences
1. **MV phasor 4.580 vs 4.589.** It is not the denominator guard: no MV window has |Z1 (I1S + I1R)| <= 1e-9. I ran the
   lead's `review_checks.one` on the 180 MV high-impedance / incipient episodes (2,184 windows). The raw samples are
   bit-identical to my cache. The lead's Z1 comes from the grid pickle, rounded to float32 (MainLn12-13:
   2.31666 + 2.48184j against my 2.31688 + 2.48167j). On these ill-conditioned windows (MAE about 42 %, little through
   current), that 1e-4 relative difference moves the estimate by up to 0.22 on 760 windows. The SC rows are identical.
   The TD locator differs by up to 0.016 on the same windows, as in part 4.
2. **A3, TG->DL: 7.88 (mine) vs 9.79 (lead). This is a definition difference, not noise.** `ablations.unnormalise`
   multiplies the **clipped** per-unit tokens (zr, zi in [-5, 5], absz <= 10) by Z1. 81 % (TG) and 84 % (DL) of
   windows have at least one clipped z token (mostly healthy loops). For those windows the "ohm" token is
   5 x Z1-rotated, which is not V/I, and the clip level itself scales with Z1. I applied the lead's definition to my
   guarded tokens: **9.97 / 10.19** (`validate_ha2_part8cmp_diag.csv`), which reproduces the lead (9.79 / 10.19).
   With true unclipped ohm tokens, un-normalised is **equal or better** on the 110 kV pair:
   - TG->DL: 7.88 vs 7.75 for H-A2, +0.13, inside the noise band
   - DL->TG: 9.90 vs 10.61, -0.71
3. A1 + Z, DL->TG -0.013: my nameplate Z (spec, 5 digits) and the lead's (pickle, float32) differ by <= 4.8e-6 ohm,
   which moves a few HGB bin edges. A1 -raw is bit-identical, and the raw windows are identical (max diff 0).
4. D2: same windows, y_local and Z content as the lead's `c1_grid_MV_raw.npz` (checked: X max diff 0, y = y_local
   exactly). Every delta is within about one seed SD. GRU +Z (+1.34) has a lead SD of 2.08. Both versions are
   **far worse than the 0.5 mid-line constant** (30.29 % MAE on the MV y_local), and
   +Z makes the MLP worse in both.
5. TD-only +0.12 TG->DL: token float-level differences; same band as the H-A2 reference (+0.07 / +0.10, part 5).

### Audit of the lead's scripts
- **`review_checks.py`**
  - Phasor locator: formula = 24(a); last cycle [s1 - N, s1); positive sequence; nameplate Z1; NaN if |den| <= 1e-9.
    The guard never triggers.
  - TD locator: asserted equal to the stored `td2_est` (< 1e-9).
  - Sync: `np.roll` of the raw remote record, then derivative, which is equivalent to mine. The wrap never reaches a
    window (s1 - 500 >= 28).
  - Scaling: R1 and L1 together, or R1 only, as registered.
  - SC = not incipient / hif, the same as mine.
  - 24(d): the `td_with_prefault` cell uses post <= 50 and `td_post_only` uses > 52.5. OK.
  - Suonan: oracle-loop `dtd` from the stored tables; MV scored against the local y. OK.
  - Clean.
  - Lead extra, verified by me after the freeze: ADAPT integer shifts and parameter scaling, e.g. sync +-1 = 22.11 /
    22.48 and RL x0.90 = 13.85. These are equal to mine to 4 digits.
- **`sync_subsample.py`**
  - Positive delays only, as in 24a.
  - The phasor column has no denominator guard, which is harmless here.
  - Lead extra, verified: phasor under +1 / 10 / 50 us is DL 0.859 / 2.725 / 8.125 and TG 1.096 / 2.782 / 7.854,
    equal to mine.
  - Clean.
- **`adapt_td.py`**
  - Test split from `held_out/double_line/test.txt`; rule windows (floor nf, D = 289 or the incipient duration).
  - My per-window estimates match all 6,883 windows (max |diff| 0.0025 pp; y identical).
  - Clean.
  - Minor: its printout uses post <= 15 / >= 21 thresholds on a time grid that is not on 5 ms steps.
- **`ablations.py`**
  - A1 / A2 inputs and the `check_allow` allow-list are correct.
  - **Defect: A3 does not implement "in ohm"** (point 2 above). The design-19 sentence "on the 110 kV pair the Z1
    normalisation is mixed (9.91 vs 7.75 TG->DL, ...)" rests on this definition. Under true ohm tokens there is no
    normalisation benefit on 110 kV (-0.71 to +0.13 pp).
- **`zs_equal_info.py`**
  - Pooled sources with unique episode groups (sim_idx + 1e7 x grid) and the unchanged `equal_info.fit_predict`.
  - Standardisation on the training split; test = all 56,340 MV windows with y_local.
  - Clean.
- **Leakage:** none found in any part-8 script. The remote end enters only the two-ended locators. Labels enter only
  as y, as the oracle loop (C, disclosed) and as fault type for the SC subset.

### Defects and disclosure points
1. **A3 (design 19 / A3-MV): definition defect.** Either re-run A3 with true (unclipped) ohm tokens, or describe it as
   "clipped per-unit tokens rescaled by Z1". With true ohm tokens the 110 kV evidence does not favour Z1
   normalisation. The A3-MV rule ("claim only if A3 is clearly worse on CIGRE MV") should be re-checked with the
   corrected A3 (not part of this spec; not run by me).
2. **Design 24(b) wording:** "fit ... on the last 40 ms" is wrong. The dtd token (section 15, mine and the lead's)
   fits on [s0 + N, s1), which is the last **30 ms** of the 50 ms window; Delta i uses the cycle before.
3. **Synchronization:**
   - One whole sample (104 us) raises the two-ended TD MAE about 10x (0.61 -> 7.1 % DL; 0.41 -> 6.5-6.8 % TG; adapt
     6.15 -> 22.1 %).
   - At GPS-class 1 us the rise is about 0.1 pp (0.72 / 0.50). At 10 us the MAE is 1.5-1.8 %.
   - The phasor locator is about 1.5-2x more sensitive than TD (DL 10 us: 2.72 vs 1.76).
   - The sign of the offset matters a little: -1 us gives 0.63 / 0.44, +1 us gives 0.72 / 0.50.
4. **Line parameters:**
   - +-5 % on R1 and L1 together roughly triples the TD MAE (1.6-1.9 %); +-10 % gives 2.4-3.0 %.
   - R1 alone +-20 % gives 1.1-1.4 %. Most of that cost is in the windows with pre-fault samples: >= 55 ms is 0.06 -
     0.16 %.
5. **Phasor vs TD two-ended:** the phasor locator is better from 35 ms on (DL 0.11 vs 0.56, TG 0.16 vs 0.36), worse
   at <= 15 ms (2.8-3.9 vs 1.0-1.5), and worse on MV (4.58 vs 4.00) and adapt (7.18 vs 6.15). On adapt both are poor
   on incipient faults: the SC-only figures are 1.15 and 0.69.
6. The Suonan-type oracle dtd is a weak baseline everywhere: 19-23 % on 110 kV, 37 % adapt, and 45 % on MV (63 % of
   MV estimates saturate at the +2 clip). State that it uses nameplate R0 / L0 and an oracle loop.
7. The D2 MV neural baselines are reproduced. Report them as below the constant-0.5 predictor.

Files:
- `results/validate_ha2_part8_{perwindow.parquet, summary.csv, nn_mv_runs.csv, abl_summary.csv, abl_pred_*.csv,
  A3_ohm_{DL,TG}.parquet, mv_windows.npz, frozen.sha256}`, `validate_ha2_nnpred_*_DLTGtoMV_s*.npy` and the logs
  `validate_ha2_part8{,_nn_mv,_abl}.log`
- after the freeze: `results/validate_ha2_part8cmp_diag.csv`, `validate_ha2_part8cmp.log`

---

## Part 9: revamp statistics (design 25 a-d)
Spec: `claudedocs/validator_spec_part9.md`. Code (`src/validate_ha2/`): `part9_mvzid.py` (regenerates the per-window MV
two-ended TD with identified Z1, which part 4 kept only as a summary), `part9.py` (a-d), `freeze_part9.py`,
`compare_part9.py` (after the freeze). CPU only, 6 workers, no retraining.

Independence: frozen at **2026-10-06 14:01:40 +05:30** (`results/validate_ha2_part9_frozen.sha256`), before I opened
`src/c1/revamp_stats.py` or `results/c1_revamp_stats.csv`. No lead per-window file was read: every input is my own.
- two-ended TD: part-8 `td2e` (DL, TG, adapt test, MV 14 seg). MV identified Z: `part9_mvzid.py` (part-4 code), two
  variants: identified R1 only (my part-4 diagnostic) and identified R1 and X1.
- Eriksson: part-6 `E`, registered rule (NaN = no root in [-0.1, 1.1]).
- H-A2: part-5 guarded 3-seed predictions (TG->DL, DL->TG, MV zero-shot, adapt in-grid).
- Equal-information: part-1 npy (GRU+Z TG->DL, MLP+Z DL->TG), clipped, error averaged over the 3 seeds.
- Bootstrap: B = 2,000 episode resamples, my own RNG (`default_rng(20261006)`), the same resamples for every method on
  one window set (paired). MV one-ended rows are scored on all 56,340 windows against y_local.

### Mine (frozen) against the lead. MAE in %, CI = 95 % episode bootstrap
| set | method | mine MAE [CI] | lead MAE [CI] | delta | within tol. |
|---|---|---|---|---|---|
| DL | two-ended | 0.612 [0.544, 0.684] | 0.612 [0.543, 0.689] | 0.000 | yes |
| DL | Eriksson | 10.546 [9.896, 11.189] | 10.546 [9.919, 11.170] | 0.000 | yes |
| DL | H-A2 TG->DL | 7.754 [7.259, 8.291] | 7.689 [7.193, 8.188] | +0.065 | yes |
| DL | GRU+Z | 11.035 [10.465, 11.619] | 11.425 [10.832, 12.043] | -0.389 | yes (neural) |
| DL | 1 - H-A2/eq (%) | 29.7 [24.8, 34.2] | 32.7 [27.8, 37.2] | -3.0 | follows from eq |
| DL | switch | 5.757 [5.343, 6.193] | 5.754 [5.318, 6.208] | +0.004 | yes |
| TG | two-ended | 0.411 [0.383, 0.439] | 0.411 [0.384, 0.441] | 0.000 | yes |
| TG | Eriksson | 8.834 [8.540, 9.111] | 8.834 [8.580, 9.134] | 0.000 | yes |
| TG | H-A2 DL->TG | 10.614 [10.179, 11.068] | 10.515 [10.073, 10.990] | +0.099 | yes |
| TG | MLP+Z | 15.950 [15.223, 16.654] | 16.201 [15.487, 16.938] | -0.251 | yes (neural) |
| TG | 1 - H-A2/eq (%) | 33.5 [30.6, 36.2] | 35.1 [32.3, 37.8] | -1.6 | follows from eq |
| TG | switch | 5.540 [5.311, 5.785] | 5.492 [5.271, 5.744] | +0.048 | yes |
| MV | two-ended nameplate | 4.003 [3.746, 4.271] | 4.004 [3.747, 4.268] | -0.000 | yes |
| MV | two-ended identified R1+X1 | 1.804 [1.665, 1.954] | 1.804 [1.661, 1.958] | 0.000 | yes |
| MV | (identified R1 only, mine) | 1.810 [1.673, 1.958] | - | - | - |
| MV | Eriksson (56,340) | 23.746 [23.163, 24.309] | 23.746 [23.164, 24.313] | 0.000 | yes |
| MV | H-A2 zero-shot | 32.486 [31.701, 33.251] | 32.602 [31.787, 33.434] | -0.116 | yes |
| MV | switch | 27.165 [26.443, 27.886] | 27.220 [26.533, 27.913] | -0.055 | yes |
| ADAPT | two-ended | 6.153 [5.001, 7.359] | 6.153 [5.016, 7.412] | 0.000 | yes |
| ADAPT | Eriksson | 16.134 [14.859, 17.347] | 16.134 [14.970, 17.435] | 0.000 | yes (CI 0.11) |
| ADAPT | H-A2 in-grid | 12.048 [11.078, 13.022] | 11.977 [11.062, 12.948] | +0.071 | yes |
| ADAPT | switch | 12.534 [11.318, 13.640] | 12.671 [11.589, 13.879] | -0.137 | yes, but bin definition differs (below) |

Metres and P95 (mine; lead deltas):
- two-ended mean / P95: DL 158.7 m / 3.235 % (830 m); TG 121.4 m / 1.772 % (513 m); MV 68.7 m / 14.28 % (280 m);
  MV identified 32.7 m / 7.834 % (100 m); adapt 1,655 m / 36.37 % (9,655 m). Lead deltas <= 0.07 m and <= 0.002 pp.
- Eriksson: DL 3,108 m / 49.0 % (14,700 m); TG 2,772 m / 49.0 % (15,000 m); MV 396 m / 49.0 % (1,467 m); adapt
  4,465 m / 47.94 % (14,580 m). Identical to the lead (<= 0.01 m). The P95 of 49 % is the 0.5 prior on undefined windows.
- H-A2: DL 2,338 m / 27.35 % (8,839 m); TG 3,355 m / 34.91 % (11,603 m); MV 501 m / 83.81 % (2,066 m); adapt
  3,248 m / 36.91 % (9,727 m). Lead deltas: mean +20 to +35 m, P95 -0.32 to +0.38 pp (refit noise).

Eriksson on defined windows (MAE % / coverage %), all, <= 15 ms, >= 20 ms:
- DL 6.732 / 84.41, 32.196 / 48.38, 3.632 / 92.83; TG 5.986 / 87.46, 31.635 / 49.37, 2.917 / 96.35;
  MV 15.868 / 44.15, 36.099 / 8.28, 15.124 / 52.53. Identical to the lead (<= 0.0004).
- Adapt: all 11.284 / 64.07 (identical); <= 15 ms 33.366 / 35.52 (lead 32.577 / 37.10); >= 20 ms 7.912 / 71.31 (lead
  8.344 / 71.22). Cause below; with the lead's edges my values are 32.577 / 37.10 and 8.344 / 71.22, exact.

### Audit of `src/c1/revamp_stats.py`
- Scoring (`err`): NaN -> 0.5, then clip, |d^ - d| x 100. Equivalent to the registered rule. Clean.
- Bootstrap: `boot_idx` re-seeds `default_rng(0)` on every call, so all methods on one window set use the same resamples;
  the reduction CI is paired as required. Window-level MAE of the concatenated rows. MV two-ended uses its own 3,318-
  episode set. Clean.
- Inputs: lead per-window tables only (`c1_classical_*`, `c1_grid_*`, `zs_pred` refit, `c1_zs_MV_pred.csv`,
  `c1_adapt_td.csv`, `c1_adapt_ingrid_pred_adapt_test.csv`, `c1_equal_info_preds.parquet`). MV scored against one y
  (y_local); correct because the two-ended rows exclude MainLn8-14, the only flipped line.
- `two_zid` uses `c1_grid_MV_zid` = identified R1 AND X1 (reproduced exactly by my R1+X1 variant). The paper should say
  "identified Z1 (R1 and X1)"; with R1 only it is 1.810.
- **Defect 1 (bin edges, adapt only).** (c) uses `post_ms <= 17.5` for "le15" and `> 17.5` for "ge20", and (d) switches
  to Eriksson at `post_ms > 17.5`. Design 25 says <= 15 / >= 20 ms and "from 20 ms". On DL / TG / MV (5 ms grid) this
  is identical, but adapt-test windows are not on the grid (e.g. 18.23 ms): 398 adapt windows lie in (15, 20) ms. So
  the adapt le15 / ge20 cells and the adapt switch use Eriksson on 17.5-20 ms windows that the design assigns to H-A2.
  Effect: E_defined le15 32.58 -> 33.37 % (coverage 37.1 -> 35.5), ge20 8.34 -> 7.91 %, switch 12.67 -> 12.53
  (mine, spec edges) / 12.69 (mine, lead edges). Fix: use the design edges (<= 15, >= 20, switch at >= 20) or amend
  design 25 to the 17.5 ms edges and relabel the columns. No conclusion changes (the adapt switch is worse than H-A2
  alone in both versions: 12.5-12.7 against 12.0).
- **Minor.** `lengths()` concatenates DL, TG and MV and keeps the first row per line name; DL and TG share names with
  different lengths (MainLn1-2A is 20 km in DL, 25 km in TG). It is only used for adapt, where DL comes first, so the
  result is right (my adapt metres agree to 0.02 m), but it is order-dependent: map adapt lines from the DL table only.
- **Minor.** The classical / grid / prediction merges are inner joins without a row-count assert; the counts are right
  (14,640 / 32,940 / 56,340 / 6,883), so nothing was dropped.
- Leakage: none (report-only statistics on existing predictions).

### Disclosure points
1. The H-A2 gain over the best equal-information net is significant in both directions (paired CI excludes 0): mine
   29.7 % [24.8, 34.2] TG->DL and 33.5 % [30.6, 36.2] DL->TG; lead 32.7 / 35.1 %. The 1.6-3.0 pp spread comes from the
   independently trained nets (GRU+Z 11.04 vs 11.42). Quote the lower values or both.
2. MV Eriksson coverage (44.2 %) counts the 3,756 MainLn8-14 windows, which cannot have an Eriksson estimate (no remote
   end). On the 14 two-ended segments coverage is 47.3 % (MAE on defined windows 15.87 %, unchanged); MV Eriksson MAE
   on those 52,584 windows is 23.28 % against 23.75 % on all 56,340.
3. Eriksson's P95 of 49 % on DL / TG / MV is the 0.5 prior on undefined windows; P95 on defined windows would be
   informative if the paper quotes P95.
4. The switch (exploratory) helps on DL / TG (5.76 / 5.54 against H-A2 7.75 / 10.61) and on MV (27.2 against 32.5), and
   hurts on adapt test (12.5-12.7 against 12.0).

Files: `results/validate_ha2_part9_{summary.csv, perwindow.parquet, mv_zid_perwindow.parquet, frozen.sha256}`; after the
freeze `results/validate_ha2_part9cmp.csv`, `validate_ha2_part9cmp_diag.csv`.

---

## Part 10: the two CIGRE MV ablation cells (design 19 / 19a)
Spec: `claudedocs/validator_spec_part10.md`. Code (`src/validate_ha2/`): `part10.py`, `freeze_part10.py`. It reuses my
own `tokens.phasor`, `mv.line_z` / `mv.CACHE`, `part8_abl.ZTOK`, `HGB_PARAMS` and the part-8 DL / TG ohm tables, and
uses only CPU (6 workers for the MV ohm tokens, sklearn HGB).

Protocol: train on my guarded DL + TG official windows pooled (14,640 + 32,940, same row order as my part-5 MV cell).
Test zero-shot on all 56,340 CIGRE MV line-fault windows (local terminal; bus 14 for MainLn8-14; target y_local).
Seeds 0-2, 3-seed mean prediction, clipped to [0, 1].
- A3: zr / zi / absz = Re, Im, |V/I| in ohm (last-cycle phasors, k0-compensated ground loops), unclipped, 0 when
  |I| <= 1e-9; the other 59 tokens unchanged. Sanity check: ohm / Z1 equals my per-unit token to < 1e-6 on every
  unclipped window and loop. The share of windows with at least one clipped per-unit z token is DL 84.2 %, TG 81.3 %
  and MV 93.8 %.
- A1: raw 480 x 6 local window, 2,880 float32 samples. There is no per-window scaling, exactly as in my part-8 A1raw.
- Reference: H-A2 refit in the same run gives 32.486, identical to my part-5 / part-9 MV cell.

Independence: frozen at **2026-10-06 14:51:32 +05:30** (`results/validate_ha2_part10_frozen.sha256`). The freeze came
before I opened `src/c1/ablation_a3_fix.py`, `ablations.py`, `zs_eval.py`, `results/c1_ablation_a3_fixed.csv` or
`c1_ablations.csv`.

### Mine (frozen) against the lead. MAE, % of line length
| cell | mine | lead | delta | tolerance | verdict |
|---|---|---|---|---|---|
| A3 ohm tokens, DL+TG -> MV | 27.425 (seed SD 0.15) | 27.269 (`c1_ablation_a3_fixed.csv`) | +0.157 | 0.5 | within |
| A1 raw window, DL+TG -> MV | 38.848 (seed SD 0.24) | 38.848 (`c1_zs_MV.csv`, A1_raw_window) | 0.000 (bit-identical) | 1.0 | within |
| (H-A2 reference, MV) | 32.486 | 32.602 | -0.116 | 0.5 | within |
| (A3fix TG->DL / DL->TG, lead extra; mine from part 8) | 7.883 / 9.902 | 7.843 / 9.953 | +0.04 / -0.05 | 0.5 | within |

The MV A1 value is not in `c1_ablations.csv`, which holds only the 110 kV rows. It is in `c1_zs_MV.csv`, written by
`zs_eval.py`.

Other numbers (mine):
- A3 on MV: median 24.12, <= 15 ms 32.21 (lead 31.87), >= 21 ms 26.32 (lead 26.20), short circuits 27.32.
- A3 against H-A2 by fault class:
  - better on short circuits: 1phg 26.9 vs 30.2, 2ph 26.7 vs 35.8, 3ph 25.9 vs 30.1
  - worse on high-impedance faults (32.8 vs 27.2) and incipient faults (25.1 vs 15.7)
- A1 on MV is degenerate. Its predictions sit in a narrow band (5-95 % quantile 0.79-0.87, correlation with y_local
  0.13), because 20 kV samples lie outside the 110 kV training range. It is 8.6 pp worse than the constant 0.5 (30.29).

### Explained difference
A3 MV is +0.157. This is the same refit noise as the H-A2 MV cell (-0.116, opposite sign) and about 1 seed SD. Sources:
- The lead's Z1 / Z0 come from the grid pickle in float32; mine come from the spec in 5 digits (part 8, point 1). This
  moves the 59 per-Z1 tokens and k0 slightly, and so moves HGB bin edges.
- The raw samples are the same; the part-8 check found a max diff of 0, and A1 is bit-identical here.

### Audit of the lead's scripts
- **`ablation_a3_fix.py`** (new, design 19a)
  - Ohm tokens are taken from `local_features.window_feats` loop (V, I): last cycle, ground loops with
    I + k0 3 I0, k0 = (Z0 - Z1) / (3 Z1), V / I unclipped, 0 when |I| <= 1e-9. This matches 19a and my definition.
  - The flipped terminal for MainLn8-14 comes from `has_term` and uses the remote bus. This is correct.
  - Merge on (sim_idx, post_ms rounded to 3 decimals): `one_to_one` validation, row count and not-NaN are asserted.
  - The other tokens come from the post-guard `c1_grid_{DL,TG,MV}.parquet`.
  - `fit_eval` is shared with `ablations.py`: same PARAMS, seeds 0-2, mean then clip. `check_allow(ALLOW)` is applied.
  - MV target: `te.y` of `c1_grid_MV`, which is y_local (asserted equal to the raw npz y in `zs_eval.py`; checked in
    part 8).
  - Clean.
  - Minor: no per-window prediction file is written, so the cell cannot be checked window by window.
- **`zs_eval.py`, A1 rows**: raw `c1_grid_*_raw.npz` windows, pooled DL+TG, reshape to 2,880 features, with no
  scaling. y is asserted equal to the token table. Clean (bit-identical to mine).
- **`ablations.py`**: unchanged since part 8. The old clipped x Z1 `unnormalise` is still used by `zs_eval.py`.
- **Defect (reporting, not code): stale A3 row.** `c1_zs_MV.csv` still carries `A3_unnormalised` = 29.53, which uses
  the old, defective definition. The corrected value (27.27) is only in `c1_ablation_a3_fixed.csv`. The paper and the
  tables must quote 27.27 (validated: 27.43), not 29.53. Either drop the stale row or relabel it
  "clipped per-unit x Z1 (superseded, 19a)".
- Leakage: none. The remote end does not enter. Labels enter only as y.

### Disclosure points
1. **Normalisation.** Un-normalised (ohm) tokens are clearly **better** than Z1-normalised H-A2 zero-shot on CIGRE MV:
   27.3-27.4 against 32.5-32.6, about -5 pp, with seed SDs of 0.15-0.5. On 110 kV the result is also equal or better
   (7.84-7.88 vs 7.69-7.75; 9.90-9.95 vs 10.52-10.61). Under design 19 / A3-MV, no Z1-normalisation claim may be
   made. 19a already says the paper makes none. The text should not imply that per-unit tokens are what transfers.
2. On MV, H-A2 (32.5) is worse than the constant 0.5 (30.29). A3 (27.4) beats the constant by 2.9 pp and is the only
   one-ended learned cell below it.
3. A1 (raw-window boosting) on MV is a near-constant predictor (38.85, worse than 0.5). Report it as "fails to
   transfer across voltage level", not as a meaningful baseline value.

Files: `results/validate_ha2_part10_{summary.csv, A3_ohm_MV.parquet, pred_{HA2_ref,A3,A1raw}_MV.csv, frozen.sha256,
notes.md}` and the log `validate_ha2_part10.log`.
