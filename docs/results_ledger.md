# C1 gate: physics-structured fault location on EvEMTBench (young benchmark, 2026)
Source: research/B4_young_benchmarks.md (C1 and the C1 addendum). Created 2026-09-30 19:20, BEFORE any
data was inspected.

## Why this can give a LARGE margin (first principles)
Published learned baselines locate faults with 16-19 % MAE (in-grid, 50 ms windows) and 27-34 %
(zero-shot cross-grid). The majority floor is 24.4 %. No impedance or two-ended locator was benchmarked,
and the baselines never see line parameters. Two-ended synchronised-phasor location follows from the
line equations: V_S - Z_line d I_S = V_R - Z_line (1-d) I_R at the fault point (for the lumped model, or
the ABCD / distributed form). It is independent of fault resistance and remote infeed. The line
parameters can be identified PER EPISODE from pre-fault phasors at both ends (parameter-free, hence
grid-agnostic).

## Stage A (lever check, raw 1-s episodes, no windowing)
- Data: benchmark-DoubleLine (2.08 GB, FAU DataCloud, open).
- Method A0: closed-form two-ended locator on post-fault phasors (1 full cycle after inception, DFT),
  with line parameters identified from pre-fault phasors at both ends. No learning, and no use of
  settings.csv parameters (a parameter-aware variant is reported separately).
- PASS-A: median |d_hat - d| <= 3 % of line length AND MAE <= 5 % on DoubleLine line-view episodes.
- KILL-A: MAE > 8 %. That would mean the EMT data break the phasor assumptions.

## Stage B (benchmark protocol, 50 ms windows; only if Stage A passes)
Reconstruct the pinned window contract (480 samples, step 5 ms) and verify it against the published digests.
Bin windows by post-inception samples. Report A0 and a hybrid (physics tokens + small residual net).
- PASS-B: hybrid MAE <= 8 % overall on DoubleLine line view (published best 16.1 %), AND <= 5 % on
  windows with >= 1 cycle post-fault, AND zero-shot TestGrid 110 kV <= 10 % (published 27-34 % local,
  line view higher tier). If the windows with < 1 cycle dominate the error, the paper reports both the
  full benchmark and the >= 1-cycle subset, honestly.

## STAGE A RESULT (2026-09-30 19:45): PASS (by a large margin)
Script src/c1/stage_a.py; per-episode results in results/c1_stage_a_skip1p0.csv. Data: benchmark-DoubleLine
(2,083,367,135 B, sha256 prefix 31347006e0366825), 1,101 episodes, 924 line faults with location labels.
- Parameter-free two-ended positive-sequence locator (Z identified per episode from the pre-fault cycle;
  post-fault cycle starting 1 cycle after inception): **MAE 0.09 % of line length, median 0.04 %**.
  The learned published baselines are 5.1-18.6 % (held-out/in-grid).
- By type: 3ph 0.02, 2phg 0.06, 2ph 0.11, 1phg 0.12, 1phg+arc 0.11, incipient 0.15-0.17. By R: 1 ohm 0.16,
  10 ohm 0.05, 40 ohm 0.04. By location: 0.04 (50 %) to 0.12 (1 %/99 %).
- The reverse-orientation MAE is 61.6 %, so the convention (distance from bus X of MainLnX-Y) is unambiguous.
  The pre-fault |IS+IR|/|IS| is 0.031 mean, so the series-line model holds.
- CAVEAT: this is a TEXTBOOK locator on complete, noise-free episodes. The research contribution must come
  from the benchmark's real setting: short 50 ms windows (sub-cycle post-fault data), the local
  (one-ended) view, zero-shot grids, noise and CT/VT errors, and uncertainty.
- Lever noted for the method: for transposed lines Z1 = Z2, so for unbalanced faults (Z, d) is identifiable
  from FAULT data alone (2 complex equations, 3 real unknowns), with no pre-fault history needed.

## Protocol facts (from the repo's results/1.1.0/aggregated_results.csv and docs)
Official DoubleLine benchmark-family FL windows: n_test = 14,640 = 900 SHC episodes x 16 windows
(tau = 5..80 ms after inception, step 5 ms) + 24 incipient x 10 (tau = 5..50 ms). This is reproduced
EXACTLY by src/c1/td_locator.py, so the window count is verified (digests not checked: they need
producer memmaps). Published best line-view FL on this set: MLP 5.18 % (global GRU 5.10 %).

## Sub-cycle sweep (src/c1/sweep_tau.py; LS phasor with inception-aware start): MAE 53.8 / 41.8 / 13.8 /
4.1 / 1.4 / 1.0 % at tau = 3 / 5 / 7.5 / 10 / 15 / 20 ms. Per-line identified Z is constant across
episodes (CV <= 0.02 %).

## STAGE B (official window set, 14,640 windows)
| method | MAE % | median % | uses inception time? |
|---|---|---|---|
| published best learned (MLP, line) | 5.18 | - | n/a (learned) |
| phasor, window-end cycle / LS for tau < 21 ms | 3.121 | 0.067 | YES for tau < 21 ms (label info; NOT fair) |
| **time-domain aerial-mode two-ended R-L locator (Clarke alpha/beta, per-line R1, L1 from normal operation), all window samples** | **0.811** | **0.046** | **NO** |
| hybrid (phasor if tau >= 21 ms, else TD) | 0.511 | 0.053 | yes (diagnostic only) |
TD by tau: 2.25 (5 ms), 1.80 (10), 1.52 (15), 1.31 (20), ~1.2-0.82 (25-50), 0.02-0.03 (55-80 ms).
=> PASS-B criterion "hybrid MAE <= 8 %" is met by the deployable, inception-free TD locator at 0.81 %
   (6.4x below the best published). Still required: zero-shot TestGrid110kV (and CIGRE MV, IEEE39),
   noise / CT-VT robustness, the local (one-ended) view, and a literature novelty check (time-domain R-L
   location has classical roots).

## Noise robustness (official windows; src/c1/td_noise.py -> results/c1_td_noise.csv)
AWGN per channel at SNR relative to the channel's pre-fault RMS; R1, L1 from normal-operation calibration.
| SNR | central diff. MAE % | Savitzky-Golay (11, order 2, fixed a priori) MAE % | 5 ms windows (SG) |
|---|---|---|---|
| clean | 0.811 | **0.611** | 2.12 |
| 60 dB | 0.796 | 0.623 | 2.18 |
| 40 dB | 0.844 | 0.937 | 2.74 |
| 30 dB | 1.564 | 1.283 | 3.10 |
Even at 30 dB the locator is ~4x below the best published learned model (5.18 %, evaluated CLEAN).
Clean SG: 8.5x below.

## NOVELTY AUDIT (research/AUDIT_C1_novelty.md, 2026-09-30 ~20:05) - consequences
- The two-ended TD R-L locator, LS parameter identification, modal TD location, parameter-free two-ended
  location and the Z1 = Z2 lever are ALL classical (Kezunovic 1994/96; Djuric-Radojevic-Terzija 1999;
  Suonan 2005; Liao 2009; Preston/Radojevic/Terzija 2009-11). Being inception-free follows from the
  equation. The benchmark group (arXiv 2608.20181) already showed a phasor two-ended locator beating
  their learned models on PROTECT-90 and called it "expected / not equal-information".
- So the two-ended 0.61-0.81 % result is a SUPPORTING baseline, not the contribution.
- The novel core must be the ONE-ENDED (local view) physics-structured hybrid: keep the TD line equation
  at the local end, learn only the unmeasurable remote-end/fault-current term, solve d in closed form,
  and transfer zero-shot. Published local view: 18.6 % (adapt_grid) / 11.70 % (benchmark family,
  DoubleLine). Runner-up: MV laterals with candidate-path, set-valued output (CIGRE MV).
- Must-haves: an equal-information table; all grids and views; an inverse-crime check (dataset line
  model vs our model; distributed parameters for 345 kV); CT/CVT/sync/sampling errors; untransposed
  lines; uncertainty; runtime; ablations.
## Journal (verified live on IEEE Xplore 2026-09-30): TSG 10.1 (transmission protection OUT of scope;
distribution fault location in scope), TII 9.8, TPWRS 8.7, TPWRD 4.7. Decision pending with the user:
TII (hybrid, physics-structured data-driven) or TSG (if the core is the MV/distribution one-ended hybrid).

## G1: physics-only ONE-ENDED baselines (local view = terminal S; phase faults 2ph+3ph only; ORACLE loop;
## src/c1/one_ended.py -> results/c1_one_ended.csv; 5,760 official windows; tau < 21 ms -> 50 % prior)
| | reactance | Takagi (I2-polarised, pair rotation corrected) |
|---|---|---|
| all windows | 19.56 % | 19.35 % |
| 2ph, tau >= 21 ms | 11.77 % | 11.21 % |
| 3ph, tau >= 21 ms | 19.33 % | 19.33 % (= reactance) |
| by R_f (2ph): 1 / 10 / 40 ohm | 8.7 / 11.2 / 30.4 | 8.9 / 13.9 / 26.1 |
Bug found and fixed before recording: the I2 polariser was not rotated by the pair factor
(1-a^2, a^2-a, a-1). The pre-fix Takagi value of 29.9 % is discarded.
Reading: one-ended physics alone is about as poor as the learned local-view models (11.7-18.6 %). It
fails through remote infeed with fault resistance, which is exactly the gap the hybrid targets.

## Published FL leaderboard (results/1.1.0/aggregated_results.csv of the repo; best MAE % over baselines;
## benchmark family; n = official windows)
| grid (n) | line held-out | line zero-shot | local held-out | local zero-shot | global held-out |
|---|---|---|---|---|---|
| double_line (14,640) | 5.18 | 18.16 | 11.70 | 28.47 | 5.10 |
| testgrid_110kv (32,940) | 6.41 | 20.40 | 10.85 | 28.05 | 6.54 |
| cigre_mv (56,340) | 24.52 | 27.06 | 30.35 | 30.61 | 18.87 |
| ieee39 (128,004) | 9.40 | 16.35 | 21.40 | 27.41 | 20.39 |
Our training-free two-ended TD locator is zero-shot by construction: DoubleLine line view 0.61 % vs
18.16 % zero-shot / 5.18 % held-out.

## [V] INDEPENDENT VALIDATION (research/VALIDATION_C1.md, 2026-09-30): REPRODUCED
Spec-only re-implementation: MAE 0.6108 % (SG) / 0.8111 % (CD) on 14,640 windows; 5 ms windows 2.123 / 2.253 %.
Deltas vs lead -0.0003 / +0.0001. Calibration from the first cycle (no event_start) gives the identical Z1 and
MAE. No label enters d; event_target (located line) picks (R1, L1), which is the standard assumption and
must be disclosed. Breakdown (SG): 3ph 0.885, 2phg 0.773, 2ph 0.483, 1phg 0.448, 1phg-arc 0.126, incipient
5.64 / 3.70 %; SHC-only 0.543 %. NOTE: in results/c1_td_official.csv, err_td/err_hyb use the CD derivative.

## ZERO-SHOT TestGrid110kV (never used for any design choice; 2026-09-30 21:09)
Data: benchmark-TestGrid110kV.tar.gz 8,925,756,744 B (exact), 2,435 episodes, 9 lines x 231, 2,079 line faults.
Calibration: per-line Z1 from the grid's own pre-fault phasors (no labels). Scripts unchanged; env EVEMT_GRID.
- Full-cycle two-ended locator (stage_a): MAE 0.13 %, median 0.05 % (reverse orientation 61.63 %).
- OFFICIAL windows: 32,940 (= published n_test, exact). TD locator: CD 0.594 % (median 0.061 %), **SG 0.409 %**.
  Phasor variant 2.208 %. Published best: line held-out 6.41 %, line zero-shot 20.40 %
  => SG is 15.7x below the in-grid learned model and 50x below the zero-shot learned model.
- Noise (SG): 0.426 (60 dB), 0.732 (40 dB), 1.048 % (30 dB). 5 ms windows: 1.40 % clean SG.
Files: results/c1_stage_a_skip1p0_TestGrid110kV.csv, c1_window_eval_TestGrid110kV.csv,
c1_td_official_TestGrid110kV.csv, c1_td_noise_TestGrid110kV.csv.

## GATE H-A (physics-token hybrid, LOCAL view, ZERO-SHOT) - RESULT 2026-10-04: PASS
Design: claudedocs/c1_hybrid_design.md section 9 (pre-registered). Scripts: src/c1/local_features.py (tokens and
physics-only baselines), src/c1/hybrid_ha.py. Learner: sklearn HistGradientBoostingRegressor (absolute error,
800 iterations, lr 0.03, 31 leaves, min 50/leaf, fixed a priori; 3 seeds averaged). LightGBM is blocked
by a Windows Application Control policy on 2026-10-04 and was not bypassed. 63 dimensionless tokens; no
tau, fault type, resistance, grid id, line length or oracle-derived estimate is used as input.
| direction | n test | hybrid MAE | median | tau>=21 ms | tau=5 ms | best physics-only one-ended | published local zero-shot / held-out |
|---|---|---|---|---|---|---|---|
| TestGrid -> DoubleLine (GATING) | 14,640 | **8.94 %** | 4.85 % | 6.11 % | 23.9 % | 19.95 % (reactance, oracle loop) | 28.47 % / 11.70 % |
| DoubleLine -> TestGrid (reported) | 32,940 | 12.02 % | 7.39 % | 9.99 % | 23.0 % | 14.45 % | 28.05 % / 10.85 % |
PASS-Z: 8.94 <= 15 and 55 % below physics-only. Reverse: 17 % below physics-only (smaller training set:
924 vs 2,079 episodes). Caveats: same line type in both grids (easy transfer); sub-cycle windows
(5-10 ms) are still ~22-24 %; the learner outputs d directly (less physically constrained than design
section 3).
Physics-only one-ended baselines, all fault types (DoubleLine / TestGrid): reactance (oracle loop)
19.95 / 14.45; Takagi-I2 24.01 / 19.79; Takagi-superimposed 31.15 / 27.45; reactance on the min-|Z| loop
(no oracle) 23.32 / 19.10.

## GATE H-B (two-ended teacher -> one-ended closed form): as registered FAIL (teacher physics error, see
## design section 11); corrected H-B' PASSES its rule on complete windows
TG->DL deployable: 11.08 % overall, **3.89 % on tau >= 21 ms** (median 1.17 %) vs H-A 6.11 % on the same windows;
teacher bound 4.22 %; reactance 14.64 %. Loop classifier accuracy 0.988 (TG->DL), 0.62 (DL->TG).

## GATE H-C (completeness-gated: H-B' on complete windows, H-A otherwise; pre-registered design section 13)
| direction | H-A | H-B' | **H-C** | gate acc. | published local zero-shot / in-grid |
|---|---|---|---|---|---|
| TestGrid -> DoubleLine | 8.94 | 11.08 | **7.45 % (median 1.66)** | 0.967 | 28.47 / 11.70 |
| DoubleLine -> TestGrid | **12.02** | 21.56 | 16.53 | 0.945 | 28.05 / 10.85 |
TG->DL: H-C adopted (7.45 < 8.94), so the main zero-shot local result is 3.8x below published zero-shot and
36 % below published in-grid. DL->TG: H-C worse than H-A because the learned loop classifier transfers
poorly from the small DoubleLine set (acc 0.62). Next: a physics-based loop selector (classical phase
selection / consistency: d in [0,1], R_f >= 0).
Per-tau TG->DL (H-C): 23.9 (5 ms), 22.4 (10), 15.3 (15), 8.2 (20), 6.3 (25) ... 3.1-3.4 (55-80 ms).

## H-C-P (pre-registered physics loop selector, design section 14) - MAIN METHOD
| direction | H-A | H-C (learned loop) | **H-C-P** | H-B'-P (complete win.) | published zero-shot / in-grid |
|---|---|---|---|---|---|
| TestGrid -> DoubleLine | 8.94 | 7.45 | **10.03 %** (median 1.82) | tau>=21: 7.31 | 28.47 / 11.70 |
| DoubleLine -> TestGrid | 12.02 | 16.53 | **7.77 %** (median 1.57) | tau>=21: 3.66 | 28.05 / 10.85 |
| mean of directions | 10.48 | 12.0 | **8.90** | - | 28.3 / 11.3 |
H-C-P is robust in both directions (about 3x below published zero-shot and below published in-grid), uses
no learned loop classifier, and puts d through the closed-form physics equation on complete windows.
Remaining loss: physics loop misselection (accuracy 3ph 0.83, SLG 0.85-0.93) and sub-cycle windows
(5-15 ms: 15-24 %).

## TD tokens (design section 15) -> H-A2 adopted; NEW MAIN METHOD H-C-P2 (2026-10-04)
| direction | H-A | H-A2 | H-C-P | **H-C-P2** | tau<=15 H-A -> H-A2 | published zero-shot / in-grid |
|---|---|---|---|---|---|---|
| TG -> DL | 8.94 | 7.75 | 10.03 | **8.71 %** (median 1.67) | 20.40 -> 13.78 | 28.47 / 11.70 |
| DL -> TG | 12.02 | 10.58 | 7.77 | **6.62 %** (median 1.48) | 20.32 -> 15.58 | 28.05 / 10.85 |
Mean relative gain on tau <= 15 ms: 0.279 (bar 0.20), so ADOPTED. H-C-P2 mean over directions = 7.66 %
(3.3-4.2x below published zero-shot; 26-39 % below published in-grid). Files: results/c1_td_tokens*.parquet,
c1_hybrid_ha_*_td.csv, c1_hybrid_hcp2_*.csv. Scripts: src/c1/td_tokens.py, hybrid_ha2.py.

## !!! LEAKAGE CORRECTION (2026-10-04, found by the lead via a consolidated-pipeline reproduction check) !!!
hybrid_hb.py chose regressor inputs with an EXCLUSION list (feats()) after merging the teacher table, so the
theta regressors and loop classifier saw 29 non-local columns, INCLUDING THE TARGETS theta_star / sin_t /
cos_t computed from BOTH ends of the TEST grid. All H-B', H-C, H-C-P and H-C-P2 numbers above are therefore
INVALID: 3.89 %, 7.45 %, 10.03 / 7.77 %, 8.71 / 6.62 %, and the 0.02 deg theta error. Their result files are
renamed *_INVALID_LEAK.csv.
UNAFFECTED (verified: the inputs came only from c1_local_feats / c1_td_tokens, which hold no remote-end data):
two-ended locator results; G1 physics baselines; H-A (8.94 / 12.02); H-A2 (7.75 / 10.58); completeness gate.
LEAK-FREE re-test of the teacher-student idea (local tokens only): theta median error 1.2 / 0.7 deg (tau>=21)
vs theta* spread 15-16 deg. The closed-form d is too sensitive to angle errors: TG->DL 21.2 % (complete
windows 16.6 %), DL->TG 12.9 % (6.6 %). The consolidated leak-free H-C-P2 (src/c1/pipeline_hcp2.py)
gives TG->DL 17.38 %. VERDICT: the teacher-student closed form is NOT a working component as built.
==> VALID MAIN ONE-ENDED METHOD = H-A2 (physics tokens + TD tokens + HGB): TG->DL 7.75 % (median 3.98),
DL->TG 10.58 % (median 6.51).

## H-A2 (valid main one-ended method) under TEST-TIME noise (trained clean; results/c1_ha2_noise.csv)
| direction | clean | 40 dB | 30 dB | tau<=15 (clean/30dB) | tau>=21 (clean/30dB) |
|---|---|---|---|---|---|
| TG -> DL | 7.75 | 8.91 | 8.97 | 13.78 / 14.89 | 6.25 / 7.49 |
| DL -> TG | 10.58 | 11.69 | 11.70 | 15.58 / 16.46 | 9.42 / 10.59 |
Input columns: an explicit allow-list (63 local tokens + 18 TD tokens), asserted free of teacher/label columns.

## EQUAL-INFORMATION zero-shot baselines (benchmark MLP/GRU recipe re-implemented; 3 seeds; src/c1/equal_info.py;
## results/c1_equal_info.csv). '+Z' = plus the faulted line's nameplate R1, X1, R0, X0 (the same info our method uses)
| method | TG -> DL MAE | DL -> TG MAE |
|---|---|---|
| MLP -raw | 13.18 +- 0.24 | 17.83 +- 0.09 |
| MLP +Z | 11.86 +- 0.16 | 16.20 +- 0.12 |
| GRU -raw | 11.83 +- 0.99 | 18.40 +- 0.33 |
| GRU +Z | 11.42 +- 0.35 | 17.91 +- 0.47 |
| physics-only one-ended (oracle loop) | 19.95 | 14.45 |
| **H-A2 (ours)** | **7.75** | **10.58** |
H-A2 is 32 % (TG->DL) and 35 % (DL->TG) below the best equal-information learned baseline, so the gain is
physics STRUCTURE, not extra information. NOTE: the fair zero-shot comparator is this re-implementation on
the same TG<->DL transfer; the published 28.47 / 28.05 % zero-shot numbers use a different source grid.

## [V] INDEPENDENT VALIDATION of H-A, H-A2, physics baselines, equal-information baselines (2026-10-04, S3)
Validator agent (spec-only, claudedocs/validator_spec_HA2.md; Part A hashes frozen before reading lead code;
report research/VALIDATION_HA2.md, code src/validate_ha2/). Window counts exact (14,640 / 32,940); raw windows
bit-identical; tokens max abs diff 0.005 (99th pct <= 8e-6). ALL 18 cells REPRODUCED:
H-A 9.01 / 12.04 (lead 8.94 / 12.02); H-A2 7.81 / 10.52 (lead 7.75 / 10.58); physics baselines identical;
neural cells 0.04-0.43 pp LOWER than the lead's (MLP+Z 11.76 / 15.95, GRU+Z 11.04 / 17.48), so the gain of
H-A2 over the best equal-information baseline is 29 % (TG->DL) and 34 % (DL->TG) by the validator's numbers
(lead 32 / 35). REPORT THE CONSERVATIVE 29 / 34 %. HGB differences < ~0.25 pp are noise (max_features=0.8
samples columns by position, so column order alone moves the MAE by up to 0.2 pp).
Leakage verdict: H-A, H-A2, MLP/GRU -raw/+Z all CLEAN. Oracle-loop physics baselines use type/phase labels
(favours the baseline; disclose).
CORRECTIONS: H-A uses 59 tokens (not 63), H-A2 77. The headline scripts used an exclusion list (result clean);
src/c1/ha2_eval.py now uses the explicit ALLOW list of src/c1/tokens_core.py (same column order) and reproduces
H-A 8.937 / 12.018 and H-A2 7.747 / 10.584 and the noise table EXACTLY; it is the generating script of
results/c1_ha2_noise.csv (results/c1_ha2_eval.csv has all rows). hybrid_ha2.py's leaky H-C-P2 step is disabled.
Extra rows (c1_ha2_eval.csv): H-A2 short-circuit-only (no incipient) MAE 7.81 / 10.70 (incipient shortcut does not
drive the headline); H-A2 single-seed 7.88 +- 0.12 / 10.71 +- 0.17.
DISCLOSURES for the paper: HGB settings were fixed in code (hybrid_ha.py) before the H-A gate ran but are not
written in the design doc; H-A2-vs-H-A adoption was scored on the two test grids (no third grid -> CIGRE MV is
needed as a clean test); same 110 kV line type in both grids; discrete labels (locations {1,20,50,80,99} %,
R_f {1,10,40} ohm); H-A2 does not beat H-A on tau >= 21 ms TG->DL (6.25 vs 6.11); baselines use the fixed
published recipe (most runs hit the 60-epoch cap).

## GATE G2-A2 (in-grid, design sections 17/17a) - RESULT 2026-10-04: FAIL on the pre-registered 5.85 % bar
## [pending independent validation: validator part 2, claudedocs/validator_spec_ingrid.md]
PROTOCOL FINDING (labels only): adapt_grid fills the location label for every episode, incl. bus faults; the
published adapt test n_test 11,673 is reproduced (11,674) only if BUS faults are counted (line faults only:
6,883 windows). Window rule reconstructed: 5 ms grid from t = 1.0 s, active iff the window overlaps
[nf, nf + 289) samples (short circuits) or [nf, nf + floor(duration * 9600)) (incipient); the same rule gives
exactly 14,640 on the benchmark family. => the published in-distribution FL test numbers (local 18.60 %) mix
~41 % bus-fault windows with meaningless locations. Data: train 1,832 episodes / 30,260 windows, test 411 / 6,883
(continuous locations 1-99 %, R_f 0-50 ohm, random inception 1.1-1.3 s). Scripts src/c1/adapt_tokens.py,
adapt_ingrid.py; results/c1_adapt_ingrid.csv, c1_adapt_ingrid_pred_*.csv.
| method (trained on adapt train line faults) | adapt test MAE (median) | benchmark MAE (median) | bench post<=15 / >=21 |
|---|---|---|---|
| react oracle loop (physics) | 32.20 (23.49) | 19.95 (3.83) | 41.9 / 14.6 |
| H-A | 12.78 (8.14) | 10.57 (7.37) | 23.0 / 7.5 |
| **H-A2** | **11.98 (7.64)** | **9.14 (6.70)** | 17.4 / 7.1 |
| published best local (GRU) | 18.60 (incl. bus faults; not comparable) | 11.70 | - |
G2-A2: 9.14 > 5.85 bar -> FAIL (ratio to best physics 0.46 <= 0.8 met). H-A2 in-grid is WORSE on the benchmark
family than H-A2 zero-shot from TestGrid (7.75 %): the adapt->benchmark operating-point/label shift costs more
than the grid change. adapt test SC-only 8.18 %; incipient 20-21 % (random locations here, no 50 % shortcut).
Equal-information in-grid MLP/GRU: running (results/c1_adapt_equal_info.csv).
In-grid EQUAL-INFORMATION baselines (same adapt train line-fault windows, fixed recipe, 3 seeds; src/c1/adapt_equal_info.py,
results/c1_adapt_equal_info.csv): adapt test / benchmark MAE: MLP-raw 17.35 / 16.04; MLP+Z 16.68 / 14.73;
GRU-raw 14.46 / 11.97 (published GRU-raw on the benchmark family 11.70 -> re-implementation sanity check passes);
GRU+Z 14.21 / 11.95. H-A2 11.98 / 9.14 => 16 % (adapt test) and 24 % (benchmark) below the best equal-information
baseline: a MODERATE in-grid margin (zero-shot margins were 29 / 34 %).
TWO-ENDED TD locator (no learning, nameplate R1/L1, SG) on adapt_grid line faults (src/c1/adapt_td.py, results/c1_adapt_td.csv):
TEST split (6,883 windows): MAE 6.15 %, median 0.29 %, SHORT-CIRCUIT-only 0.69 % (1phg 0.57, arc 0.28, 2ph 0.66,
2phg 1.08, 3ph 0.82) with continuous locations and R_f 0-50 ohm; INCIPIENT 18.4-18.6 % (self-clearing faults of
2-80 ms: most windows extend past the clearing, so the fault model is wrong for part of the window -> disclose as a
limitation; a fault-interval-aware variant would be future work). All 2,243 episodes: 5.64 % (SC-only 0.73 %).
Published best line-view in-distribution test 16.10 % (includes bus faults, not comparable).
[V] G2-A2 VALIDATED (validator part 2, research/VALIDATION_HA2.md 'In-grid'): windows identical (37,143; 0 mismatches),
raw windows bit-identical, leakage CLEAN; validator H-A2 12.05 / 9.23 (lead 11.98 / 9.14), H-A 12.68 / 10.59,
physics identical. Validator counts: adapt test all faults 11,674 (published 11,673; 1-window gap unexplained),
line only 6,883; 2,030 bus faults in adapt carry a location value. Why physics collapses on adapt: R_f continuous
0-50 ohm + mixed grounding (solid/resistive/resonant); oracle-loop reactance on SC with post >= 25 ms: 0.96 %
(R_f <= 1 ohm), 2.7 % (5-10), 29.6 % (20-50). In-grid neural baselines: validator part 3 (running).
[V] In-grid neural baselines VALIDATED (validator part 3): 8/8 cells reproduced (deltas -0.07..+0.42 pp; the validator's
GRU early-stopped earlier). Conservative in-grid gain of H-A2 over the best equal-information baseline (validator's
numbers): adapt test 17.4 % (12.05 vs GRU+Z 14.59; lead 16 %), benchmark 22.4 % (9.23 vs GRU-raw 11.90; lead 24 %);
SC-only adapt test ~12 %. REPORT: in-grid gain 16-17 % (adapt test) / 22-24 % (benchmark), about half the zero-shot gain.
ha2_eval.py re-audited CLEAN.

## ABLATIONS (design section 19, report-only; src/c1/ablations.py, results/c1_ablations.csv) - zero-shot MAE TG->DL / DL->TG
| variant | TG->DL | DL->TG |
|---|---|---|
| H-A2 (77 tokens) | 7.75 | 10.58 |
| phasor tokens only (= H-A, 59) | 8.94 | 12.02 |
| TD tokens only (18) | 11.36 | 11.84 |
| A3 un-normalised impedance tokens (ohm) | 9.91 | 10.13 |
| A1 HGB on raw 480x6 window | 14.51 | 19.52 |
| A1 HGB on raw window + Z | 14.06 | 19.27 |
=> the gain is from the physics TOKENS, not the learner (same HGB on raw windows is 45-82 % worse, and worse than the
neural equal-info baselines). Both token groups contribute. Z1 normalisation: MIXED on the 110 kV pair (same line
type) -> do not claim it unless CIGRE MV supports it (A3-MV registered).
A4 classical one-ended by post-fault time (oracle-loop reactance, last-cycle phasor): at 50 ms DL 14.99 / TG 9.08 %;
>= 55 ms 14.0 / 7.9 %; sub-cycle (5-15 ms) 29-50 % (comparable to the 10.9 % single-ended reactance at 50 ms of
arXiv:2608.20181 on a different set). Takagi-I2 13.5-21 %.

## GATE G3-MV (CIGRE MV, design sections 18/18a) - RESULT 2026-10-04 (nameplate Z = registered primary)
Data: benchmark-CigreMVGrid, 3,555 line-fault episodes, 15 segments (0.24-4.89 km, X/R 1.1-1.4), windows 56,340 =
published n_test EXACTLY. Scripts: src/c1/grid_pass.py (MV), zs_eval.py MV DL TG, zs_equal_info.py MV DL TG.
Files: results/c1_grid_MV.parquet, c1_zs_MV.csv, c1_zs_MV_pred.csv, c1_zs_equal_info_MV.csv. [pending validation]
(a) TWO-ENDED TD locator (no learning; 14 segments measured at both ends, 52,584 windows): MAE 4.00 %, median 1.26 %,
    short circuits 2.05-2.56 % by type (all non-incipient incl. HIF: 3.30 %), HIF 36-37 %,
    incipient 47 %. Published best (held_out, IN-GRID trained): line 24.52 %, global 18.87 % -> 4.7-6.1x lower.
(b) ONE-ENDED ZERO-SHOT from pooled 110 kV (DL+TG): H-A2 32.98 %, H-A 32.03 %, A3 un-normalised 29.69 %,
    A1 HGB raw 38.85 / +Z 40.61 %, physics reactance (oracle loop) 32.98 % (median 20.0), Takagi-I2 34.56 %.
    Equal-information neural (3 seeds): MLP-raw 41.59, MLP+Z 48.26, GRU-raw 44.20, GRU+Z 45.44 +- 2.08 (all worse than the majority predictor). H-A2 32.98 is 21 % below the best (MLP-raw 41.59) but at majority level.
    Published local held_out 30.35 % (= majority), transfer_zeroshot 30.61 %.
    PRE-REGISTERED CLEAR-WIN RULE: H-A2 <= 0.8 x 30.35 = 24.3 %  -> FAIL (32.98). G3-MV(b) FAILS: the one-ended
    learner does NOT generalise from 110 kV to MV. Also: un-normalised tokens are not worse here (29.7), so the
    Z1-normalisation claim is NOT supported (do not claim it).
DIAGNOSIS (SC, post >= 25 ms): one-ended error grows with R_f/|Z_line|: ratio <= 1: H-A2 13.0, reactance 4.7,
two-ended 1.4 %; 1-3: 28.0 / 16.5 / 1.9; 10-30: 40.6 / 38.9 / 3.0; > 30: 40.9 / 45.2 / 2.2. By R_f: 1 ohm 19.0 /
5.7 / 1.6; 10 ohm 37.0 / 32.4 / 1.7; 40 ohm 41.2 / 46.7 / 2.9. On MV segments |Z1| = 0.2-4.0 ohm, so R_f/|Z1| reaches
~160 (110 kV: <= ~8): one-ended location is ill-conditioned (fault-resistance voltage swamps the line drop),
while the two-ended locator is insensitive to R_f. H-A2 is WORSE than plain physics at low R_f/|Z| (13.0 vs 4.7):
the learned 110 kV mapping does not transfer.
(a') SENSITIVITY (design 18a(1)(ii), disclosed; not used for any choice): two-ended TD locator with Z1 IDENTIFIED from
     pre-fault normal operation (label-free, median of 10 episodes per segment; results/c1_grid_MV_zid.parquet):
     MAE 1.80 %, median 0.27 %; SC by type 0.54-1.69 %; HIF 13.2-14.1 %; incipient 16.7-20.4 %. The nameplate R1 of
     the lumped segments is ~24 % low, which explains most of the 4.00 % (nameplate) error.
[V] G3-MV VALIDATED (validator part 4, research/VALIDATION_HA2.md 'CIGRE MV'; frozen 17:50:03): 6/6 reproduced
(two-ended 4.003, physics 32.975 / 34.556 / 38.606 identical; H-A 31.85 vs 32.03; H-A2 32.90 vs 32.98); 56,340
windows aligned, raw windows bit-identical, 8-14 flip confirmed (6.70 vs 62.84 %), leakage CLEAN. Identified Z1
reproduced to 4 digits (cable R1 x1.239-1.244, X x1.00; overhead lines 1.00); two-ended with identified Z 1.81 %
(SC 1.21 %) - TRANSDUCTIVE (uses the test episodes' own pre-fault data, no labels): disclose.
TOKEN BUG FOUND (validator): for 2ph/3ph faults I0 ~ 1e-14, so i0c/i0s and the three tak0 tokens are floating-point
noise (implementation-dependent). Not leakage, but must be GUARDED (e.g. set to 0 when i0r < 1e-6) and every H-A/H-A2
number re-run and re-validated (expected small change). hif and incipient episodes all sit at 50 % / 10 ohm on
CIGRE too (label shortcut). 'mae_shc_only' column = all NON-INCIPIENT (includes hif): rename in the paper.

## I0 GUARD (design section 20) - RE-RUN 2026-10-04 S4 [pending validator part 5, claudedocs/validator_spec_guard.md]
Rule: i0r < 1e-6 -> i0c = i0s = 0, ag/bg/cg_tak0 = 0.5. Guarded windows: DL 5,722 (all 2ph/3ph but 38 2ph), TG 12,960
(+25 incipient), MV 17,449 (+39 incipient), adapt 11,626 (incl. 1,300 incipient and 22 'dead' SC windows with all local
currents ~0); SNR30/40 tables 1-7 windows (noise lifts I0 above 1e-6). Patched tables == raw regeneration with the guarded
code (12 episodes per grid incl. 4 2ph + 4 3ph; max diff 0; results/c1_i0_guard_check.csv). Old tables *_preguard.parquet,
old results *_preguard.csv. Script src/c1/apply_i0_guard.py, re-runs src/c1/rerun_guard.sh (logs results/rerun_guard_*.log).
| cell | old | NEW | delta |
|---|---|---|---|
| H-A2 zero-shot TG->DL (clean / 40 dB / 30 dB) | 7.75 / 8.91 / 8.97 | **7.69** / 9.17 / 9.26 | -0.06 / +0.26 / +0.29 |
| H-A2 zero-shot DL->TG (clean / 40 dB / 30 dB) | 10.58 / 11.69 / 11.70 | **10.52** / 11.66 / 11.68 | -0.07 / -0.03 / -0.01 |
| H-A zero-shot TG->DL / DL->TG | 8.94 / 12.02 | 9.04 / 12.35 | +0.10 / +0.33 |
| H-A2 tau<=15 / >=21, TG->DL | 13.78 / 6.25 | 13.82 / 6.16 | |
| H-A2 tau<=15 / >=21, DL->TG | 15.58 / 9.42 | 15.35 / 9.40 | |
| G2-A2 in-grid H-A2 adapt test / benchmark | 11.98 / 9.14 | **11.98 / 9.31** | 0.00 / +0.17 |
| G2-A2 in-grid H-A adapt test / benchmark | 12.78 / 10.57 | 12.68 / 10.79 | |
| G3-MV(b) zero-shot H-A2 / H-A | 32.98 / 32.03 | **32.60** / 31.75 | -0.38 / -0.28 |
| ablation A3 un-normalised TG->DL / DL->TG / MV | 9.91 / 10.13 / 29.69 | 9.80 / 10.19 / 29.53 | |
| ablation A2 TD-only, A1 raw window, physics, two-ended | - | unchanged (no guarded token) | 0 |
All |delta| <= 0.38 pp (< the 0.5 pp investigate threshold). GATE VERDICTS UNCHANGED: G2-A2 FAIL (9.31 > 5.85; ratio to
physics 0.47), G3-MV(b) FAIL (32.60 > 24.3). H-A2 still beats H-A on every zero-shot cell.
Gain of H-A2 over the best equal-information baseline (equal-info numbers do not use tokens, unchanged):
zero-shot TG->DL 7.69 vs 11.42 lead / 11.04 validator -> 33 / 30 %; DL->TG 10.52 vs 16.20 / 15.95 -> 35 / 34 %.
in-grid adapt test 11.98 vs 14.21 / 14.59 -> 16 / 18 %; benchmark 9.31 vs 11.95 / 11.90 -> 22 / 22 %.
CIGRE MV 32.60 vs MLP-raw 41.59 -> 22 % (at majority level; the gate fails).
NOTE (disclose): under test-time noise the guard creates a small train/test mismatch for 2ph/3ph windows (trained on
guarded constants, tested on noise-dominated I0 angles): TG->DL noisy cells +0.26-0.29 pp. A noise-aware threshold would
be a design change; not done (pre-registered rule kept).
[V] I0 GUARD VALIDATED (validator part 5, research/VALIDATION_HA2.md 'I0 guard'; frozen 20:20:43): guarded masks identical
on all grids (0 mismatches); guard code and patch audited CORRECT; validator new H-A2 7.75 / 10.61 (zero-shot), 12.05 / 9.41
(in-grid), 32.49 (MV); all cells within 0.11 pp of the lead except H-A DL->TG (-0.33; column-order noise of HGB, which alone
moves that cell 0.48 pp -> differences < ~0.5 pp are not claimable). Gate verdicts unchanged. The 5 I0 tokens now agree
(<= 5e-6 DL/TG/adapt). Unguarded MV 2ph/3ph windows (4,190) are all on overhead segments 12-13, 13-14, 8-14: physical I0.
REPORT (conservative, validator's numbers): H-A2 gain over best equal-information baseline: zero-shot 30 % (TG->DL) /
34 % (DL->TG); in-grid 15 % (adapt test) / 21 % (benchmark); CIGRE MV 22 % but at majority level (gate fails).
Noise-table defect confirmed: TG->DL extra error under noise 1.17/1.22 -> 1.48/1.57 pp (40/30 dB); disclose.

## CLASSICAL ONE-ENDED BASELINES (design 21/21a, report-only) - 2026-10-04 S4 [pending validation]
Scripts src/c1/classical_oe.py (per grid), classical_report.py; results/c1_classical_<DL|TG|ADAPT|MV>.parquet,
c1_classical_report.csv, c1_classical.log. Oracle loop + PRE-FAULT MEMORY (favours baselines); MT and Eriksson use
the remote source impedance from the remote end's superimposed quantities (oracle settings knowledge). Bolted check
(DL, R_f <= 1 ohm, post >= 25 ms, bar 3 %): R 1.13, T 1.54, T2 1.52 (after 21a fix: 3ph -> reactance), MT 1.53, E 1.08 -> PASS.
MAE % (all windows; NaN -> 0.5) | post <= 15 ms | 20-30 | 35-50 | >= 55:
| setting | best classical | H-A2 | ratio | claim (<= 0.8) | E by time (<=15/20-30/35-50/>=55) | H-A2 by time |
|---|---|---|---|---|---|---|
| zero-shot TG->DL | E 10.46 | 7.69 | 0.74 | YES | 35.6 / 6.5 / 4.8 / 3.5 | 13.8 / 6.5 / 5.9 / 6.4 |
| zero-shot DL->TG | E 9.55 | 10.52 | 1.10 | no (E better) | 34.5 / 5.7 / 3.9 / 2.6 | 15.4 / 8.9 / 8.9 / 10.0 |
| in-grid benchmark DL | E 10.46 | 9.30 | 0.89 | no | as above | 17.6 / 7.6 / 7.1 / 7.4 |
| in-grid adapt test (R_f 0-50, mixed grounding) | E 22.88 | 11.98 | 0.52 | YES | 41.0 / 17.0 / 15.4 / 20.3 | 17.9 / 9.6 / 9.2 / 11.6 |
| zero-shot -> CIGRE MV | MT 28.83 | 32.60 | 1.13 | no | (E 30.87; 3,756 NaN on 8-14) | 36.7 / 31.7 / 31.7 / 31.5 |
Other classical (DL / TG / adapt / MV): R 19.95 / 14.45 / 32.20 / 32.98; T 16.37 / 12.61 / 26.79 / 34.56;
T2 20.50 / 15.34 / 32.35 / 31.31; MT 16.36 / 12.45 / 26.81 / 28.83.
READING (honest): with pre-fault memory and known source impedances, Eriksson beats H-A2 on every >= 20 ms window of
the 110 kV benchmark grids (discrete R_f {1, 10, 40} ohm). H-A2's advantage is (i) SUB-CYCLE windows (<= 15 ms:
13.8-17.6 vs 34.5-35.6 %), where phasor methods have no full post-fault cycle, and (ii) the adapt set with continuous
R_f 0-50 ohm and mixed grounding (all bins). The earlier "one-ended physics-only 19.95 / 14.45 %" was the plain
reactance method: a reviewer would rightly call it a weak baseline. Paper must compare with Eriksson.
On CIGRE MV every one-ended method is near majority level (28.8-34.6 %); the two-ended locator stays at 4.00 / 1.80 %.

## MEASUREMENT-CHAIN ROBUSTNESS (design 22, report-only) - 2026-10-04 S4 [pending validation]
Scripts src/c1/meas_chain.py (models + sanity), chain_pass.py; results/c1_chain_<scen>_<DL|TG>.parquet, c1_chain_eval.csv.
Model checks: CVT residual after a bolted terminal collapse (max over the next cycle, % of pre-fault peak) at 10/20/40 ms:
CVT-5 16.8/1.7/0.0 (peak), 11.0/2.4/0.0 (zero); CVT-15 51.0/26.7/7.0 (peak), 37.7/19.0/4.9 (zero) -> CVT-15 is a
stress case well beyond a well-damped CVT. CT (11 kA rms, full offset, X/R 12): CT-mild saturates after 70.7 ms,
CT-severe after 6.8 ms (deep saturation). Windows flagged saturated: CT-mild 0.0 % (DL, TG: it never saturates inside a
window, so CT-mild = ideal CT here), CT-severe 20.4 % (DL) / 21.2 % (TG).
MAE % TG->DL | DL->TG (H-A2 trained on the CLEAN source grid; two-ended and physics need no training):
| scenario | two-ended | react oracle | H-A2 | H-A2 post<=15 / >=21 |
|---|---|---|---|---|
| clean | 0.61 / 0.41 | 19.95 / 14.45 | 7.69 / 10.52 | 13.8 / 6.2 ; 15.4 / 9.4 |
| AA 2 kHz | 0.59 / 0.38 | 20.00 / 14.49 | 7.69 / 10.55 | |
| CT-mild | 0.61 / 0.41 | 19.95 / 14.45 | 7.69 / 10.52 | |
| CT-severe | 0.66 / 0.48 (sat windows 0.31 / 0.45) | 19.98 / 14.48 | 9.39 / 12.10 (sat windows 5.3 / 6.1) | 14.6 / 8.0 ; 16.3 / 11.1 |
| CVT-5 | 2.47 / 3.21 | 21.17 / 16.01 | 8.97 / 11.92 | 18.5 / 6.4 ; 20.0 / 9.7 |
| CVT-15 | 4.90 / 6.70 | 24.57 / 20.29 | 11.66 / 14.50 | 21.6 / 8.5 ; 23.3 / 11.8 |
| FULL (CT-severe + CVT-15 + AA) | 4.86 / 6.73 | 24.48 / 20.22 | 12.26 / 15.28; matched FULL training 8.17 / 11.78 | 20.7 / 9.7 ; 22.9 / 13.0 |
WORDING (registered rule: two-ended <= +1.0 pp, one-ended <= +2.0 pp): two-ended ROBUST to AA and CT saturation
(+0.05 / +0.07 pp even with 20 % saturated windows: the locator uses the whole window and the sum of both ends' currents);
NOT robust to CVTs (+1.9 / +2.8 pp CVT-5, +4.3 / +6.3 pp CVT-15): the voltage-based R-L model sees the CVT transient.
H-A2 robust to AA and CT-severe (+1.7 / +1.6 pp), NOT to CVTs (+1.3 / +1.4 CVT-5 within the rule; +4.0 / +4.0 CVT-15 not);
training with the chain (matched FULL) recovers most of it (8.17 / 11.78). Under FULL the two-ended locator (4.86 DL / 6.73 TG) is at the level of the
published CLEAN learned in-grid numbers (5.18 / 6.41; slightly below on DL, ABOVE on TG) - do not claim superiority
under CVT distortion; under CT saturation and AA the margin is intact. Limitation to state: CVT
compensation (standard in numerical relays) was not applied; it is the obvious fix and future work.

### [V] CLASSICAL + CHAIN VALIDATED (validator part 6, research/VALIDATION_HA2.md; frozen 21:37:34) WITH ONE CORRECTION
The validator found two unregistered Eriksson fallbacks (design 21b). Corrected, registered-rule ERIKSSON (undefined -> 0.5),
reproduced exactly by lead and validator: DL 10.55, TG 8.83, adapt test 16.13, CIGRE MV 23.75 % (undefined: 16 / 13 / 36 /
56 % of windows). R, T, T2 identical; MT within 0.9 pp (threshold details). By time (E | H-A2): TG->DL <=15 ms 31.3 | 13.8,
20-30 6.8 | 6.5, 35-50 5.4 | 5.9, >=55 5.3 | 6.4; DL->TG 30.9 | 15.4, 5.0 | 8.9, 3.4 | 8.9, 3.2 | 10.0; adapt 27.3 | 17.9,
14.2 | 9.6, 12.0 | 9.2, 13.5 | 11.6; MV 30.4 | 36.7, 22.4 | 31.7, 21.7 | 31.7, 22.4 | 31.5.
CLAIMS (H-A2 <= 0.8 x best classical): TG->DL YES (0.73); adapt test YES (0.74); DL->TG NO (1.19); in-grid bench NO (0.88);
MV NO (1.37: Eriksson with known source impedances reaches 23.75 %, below the 30.35 % majority level, where H-A2 does not).
THE TABLE ABOVE (first classical run) IS SUPERSEDED for E. Chain: all two-ended / physics cells reproduced (<= 0.015 pp);
H-A2 cells within -0.07..+0.29 pp; robustness verdicts identical; saturation flags identical given RP (22a). No leakage.

## ONE-ENDED CONDITIONING (design 23, report-only) - 2026-10-04 S4 [pending validation]
Script src/c1/conditioning.py; results/c1_cond_windows.parquet, c1_cond_summary.csv, c1_cond_rho_bins.csv.
Result: e = rho x kappa x sin(angle error), rho = R_f/|Z_L|. 1phg_shc, post >= 25 ms, both ends measured:
| | DL | TG | adapt test | CIGRE MV |
|---|---|---|---|---|
| windows (episodes) | 2,160 (180) | 4,860 (405) | 661 (60) | 7,560 (630) |
| identity |e_obs - e_pred| median / p90, reactance (pp) | 0.42 / 4.1 | 0.40 / 3.1 | 1.40 / 4.4 | 1.52 / 17.8 |
| median |e_obs| reactance, unclipped (pp) | 3.6 | 1.9 | 4.4 | 125 |
| rho median / p90 | 1.40 / 6.1 | 1.27 / 6.1 | 3.72 / 8.5 | 9.58 / 81.7 |
| kappa (Takagi) median | 0.99 | 0.81 | 0.95 | 1.37 |
| Takagi polarising-angle error median / p90 (deg) | 0.35 / 0.90 | 0.39 / 1.23 | 0.24 / 0.66 | 3.95 / 14.7 |
| required angle accuracy for 5 % error, median (deg) | 1.96 | 2.75 | 0.75 | 0.16 |
| share of windows needing < 1 deg | 40 % | 27 % | 63 % | 76 % |
The identity holds (median deviation 0.4 pp at 110 kV; ~1 % of the error on MV; residual = line charging and DFT).
On MV two factors compound: rho is ~7x larger (segments 0.2-4 ohm) AND the classical polarising angle is ~10x worse
(median 3.95 deg vs 0.35-0.39 deg; loads / non-homogeneous network), while the angle needed is ~12-17x tighter
(0.16 vs 1.96-2.75 deg). Pre-stated expectation CONFIRMED for MV vs the benchmark 110 kV grids (MV 76 % of windows need < 1 deg; DL / TG 60-73 %
do not); NOT for the adapt set (110 kV, continuous R_f up to 50 ohm: 63 % need < 1 deg) - state it that way.
MAE vs rho (SC, post >= 25 ms; c1_cond_rho_bins.csv): two-ended 0.1-0.6 % (110 kV) and 1.1-3.0 % (MV) in EVERY bin;
on MV all one-ended methods rise with rho: E 4.6 -> 30.7, R 4.2 -> 45.2, H-A2 6.3 -> 40.0 (rho <= 0.3 -> > 30).

## FIG 4 DATA: zero-shot MAE vs time since inception (src/c1/fig4_data.py, results/c1_fig4_data.csv) - 2026-10-04 S4
Equal-information per-window predictions re-generated (src/c1/equal_info_preds.py; all 24 runs reproduce the recorded
MAE to <= 4e-6 pp; results/c1_equal_info_preds.parquet). Best ONE-ENDED method per post-fault time (MAE %):
| t (ms) | TG->DL best | H-A2 | Eriksson | DL->TG best | H-A2 | Eriksson |
|---|---|---|---|---|---|---|
| 5 | GRU+Z 14.4 | 18.0 | 32.5 | H-A2 19.0 | 19.0 | 32.5 |
| 10 | GRU+Z 12.4 | 13.3 | 29.7 | H-A2 15.5 | 15.5 | 29.0 |
| 15 | H-A2 10.1 | 10.1 | 31.7 | H-A2 11.7 | 11.7 | 31.2 |
| 20 | H-A2 7.4 | 7.4 | 8.7 | Eriksson 7.3 | 9.2 | 7.3 |
| 25-50 | E ~ H-A2 (5.2-6.2 vs 5.6-6.4) | | | Eriksson 3.3-4.2 | 8.7-9.2 | 3.3-4.2 |
| 55-80 | Eriksson 5.2-5.4 | 6.2-6.7 | | Eriksson 3.2-3.3 | 9.7-10.4 | |
Two-ended: 0.07-2.13 % in every bin (both directions). Reading: NO one-ended method dominates. Learned window models
(GRU) lead at 5-10 ms in TG->DL; H-A2 leads at 5-15 ms in DL->TG and at 15-20 ms in TG->DL; Eriksson (pre-fault memory +
known source impedances) leads from 20-25 ms, and H-A2's error does not fall with longer windows on DL->TG (9.2 -> 10.4)
while Eriksson's does. Sub-cycle claim must be stated per direction. Overall: TG->DL H-A2 7.69 best one-ended;
DL->TG Eriksson 8.83 best one-ended.

[V] CONDITIONING + FIG 4 VALIDATED (validator part 7, research/VALIDATION_HA2.md 'Conditioning + Fig 4'; frozen 22:09:18):
summary cells identical to 4 decimals (2 cells differ in the 5th digit); rho-bin cells for two-ended / R / T / E within
0.002 pp, H-A2 -0.30..+0.61 pp (refit noise); Fig 4 best one-ended method per bin agrees in 33 / 34 rows (DL 35 ms is a
tie: E 5.73 vs H-A2 5.67-5.73). Disclose in the paper: D_req uses Takagi's kappa; line charging ignored in I_F (pre-fault
KCL residual 1.4-2.8 %); MV identity holds in relative terms only (p90 18-19 pp because model error is multiplied by
rho up to 190); 46 % of MV SC Eriksson estimates are undefined (scored 0.5), so its high-rho MV cells are mostly the prior.
Fixed: c1_cond_summary columns renamed med_abs_eobs_R/T (they are medians).

## REVIEW-DRIVEN CHECKS (design 24 / 24a, report-only) - 2026-10-06 S5 [pending validation, validator part 8]
Scripts src/c1/review_checks.py (two-ended TD reproduces the stored td2_est exactly, max dev < 1e-9), sync_subsample.py;
results/c1_review_checks.csv, c1_review_phasor_bytime.csv, c1_review_suonan.csv, c1_review_sync_us.csv.
(a) Two-ended PHASOR (C37.114, positive sequence, last cycle) vs TD, MAE %: DL 0.66 vs 0.61; TG 0.94 vs 0.41; adapt test
    7.18 (SC 1.15) vs 6.15 (SC 0.69); MV 4.59 vs 4.00. By time (TD | phasor): <=15 ms DL 1.55 | 2.77, TG 0.99 | 3.89,
    MV 5.90 | 7.81, adapt 6.51 | 8.99; 35-50 ms DL 0.56 | 0.11, TG 0.36 | 0.16; >=55 ms DL 0.07 | 0.05, TG 0.10 | 0.08.
    => the two-ended EQUATION, not the time-domain form, is what beats the learned models; the TD form wins in the first
    cycle (and is less sensitive to synchronization error), the phasor form is equal or better on later windows.
(b) Suonan-Qi-type one-ended TD (oracle-loop dtd): DL 22.67, TG 19.38, adapt 36.61, MV 44.69 % (weak).
(c) Two-ended TD sensitivity: registered whole-sample sync offsets (DL / TG): 1 sample (104 us) 7.1-7.3 / 6.5-6.8 %,
    5 samples 21 / 20-21 %, 20 samples 35 / 33-36 %. 24a sub-sample offsets (TD | phasor): 1 us 0.72 | 0.86 (DL),
    0.50 | 1.10 (TG); 10 us 1.76 | 2.72, 1.53 | 2.78; 50 us 4.18 | 8.13, 3.92 | 7.85. Line-parameter error R1 and L1
    x0.95/1.05: 1.71/1.64 (DL), 1.89/1.83 (TG); x0.90/1.10: 2.44/2.57, 2.85/2.96; R1 only x0.8/1.2: 1.35/1.32, 1.18/1.11.
    adapt: parameter +-5 % -> 11.1 / 9.8 (incipient-dominated).
(d) Inception-free cost: windows with pre-fault samples (tau <= 50) DL 0.93, TG 0.59 % vs fully post-fault 0.07 / 0.10 %.
[V] REVIEW CHECKS VALIDATED (validator part 8, research/VALIDATION_HA2.md 'Part 8'; frozen 07:56:05): phasor two-ended,
sync (whole-sample and sub-sample), parameter scaling, Suonan-type TD, adapt two-ended 6.15 / 0.69 all identical or within
0.01 pp; MV equal-information within ~1 seed SD (41.29 / 48.16 / 43.87 / 46.78 vs lead 41.59 / 48.26 / 44.20 / 45.44);
ablations TD-only and raw-window reproduced. DEFECT: A3 (ablations.py unnormalise) multiplied CLIPPED per-unit tokens by
Z1 (81-84 % of windows have a clipped impedance token) -> not V/I in ohm. With true ohm tokens (validator): 7.88 / 9.90 vs
H-A2 7.75 / 10.61 (equal or better). A3 numbers 9.79 / 10.19 / 29.53 are INVALID as 'ohm tokens' -> re-run (design 19a).
Wording fix: the TD-token fit uses the last 30 ms of the window (one 20 ms cycle of history needed), not 40 ms.
A3 CORRECTED (design 19a; src/c1/ablation_a3_fix.py, results/c1_ablation_a3_fixed.csv): true ohm, unclipped impedance
tokens: TG->DL 7.84, DL->TG 9.95, DL+TG->MV 27.27 % (H-A2 per-unit: 7.69 / 10.52 / 32.60). Matches validator part 8
(7.88 / 9.90). => per-unit normalisation does NOT help; on MV it HURTS (27.3 vs 32.6; ohm variant below the 30.35 %
majority level but above Eriksson 23.75). Paper: state this; no normalisation claim. Old A3 values (9.79/10.19/29.53) INVALID.
