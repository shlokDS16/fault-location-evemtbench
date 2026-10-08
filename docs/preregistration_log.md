# C1 core method (pre-registered design, 2026-09-30 20:10): one-ended physics-structured hybrid locator
Status: DESIGN ONLY. Nothing below has been run. Inputs: research/C1_GATE.md and research/AUDIT_C1_novelty.md.

## 1. Why the one-ended (local) view is the contribution
The two-ended TD locator (0.61-0.81 % on the official DoubleLine windows) is classical and is only a
supporting baseline. The local view (one terminal, 6 channels) is the benchmark's hardest fault-location
setting. Published best: 11.70 % (benchmark family, GRU) and 18.60 % (adapt_grid). Zero-shot cross-grid is
27-34 %. One-ended location is hard in principle: the fault current i_f = i_S + i_R contains the
UNMEASURED remote contribution, and remote infeed biases every classical one-ended method (reactance,
Takagi).

## 2. Physics (time domain, loop quantities with zero-sequence compensation)
For the fault loop selected by the fault type (a-g, b-c, ...):
  v_loop(t) = d [R1 i_loop(t) + L1 i_loop'(t)] + R_f i_f(t),   i_loop = i_ph + k0 * 3 i_0   (ground loops)
The unknowns are d, R_f and the waveform i_f(t). Classical Takagi (time-domain form) sets i_f ~ kappa *
delta_i_S(t) (superimposed local current), with kappa REAL, i.e. a homogeneous network. That fails when the
angle of kappa is non-zero (non-homogeneous sources, IBR infeed, double-circuit coupling).

## 3. Hybrid (the novel component)
Keep the loop equation exactly. Learn only what the local end cannot measure: a small network g_phi
predicts the fault-current shaping kappa (complex, 2 reals), or a short correction waveform for
i_f(t), from DIMENSIONLESS local features. Features: ratios of local sequence quantities, the angle
between delta_i and i_0 / i_2, and the pre-fault power-flow direction. All are normalised by the line
impedance, which makes them grid-agnostic. Then (d, R_f) is solved in CLOSED FORM by least squares over
the window samples. The network never outputs d directly, so the output stays physically constrained,
and d is differentiable through the LS solve (end-to-end training possible).
Uncertainty: the LS residual variance and a small ensemble of g_phi give an interval on d. It is
calibrated by split conformal on source-grid windows, only as a supporting element (not the headline;
lesson: conformal alone is not a contribution).
Fault loop: from the fault-type classifier. The published FC at 50 ms is 0.86-0.90 balanced accuracy;
also report the "all 6 loops + minimum residual" selection, which needs no classifier.

## 4. Information sets (fairness, required by the audit)
- Line positive/zero-sequence impedance per km and length: from commissioning or utility data. It is
  identified two-endedly from normal operation (as in Stage A) or read from the grid model.
- EQUAL-INFORMATION TABLE: rerun the benchmark's public MLP and GRU baselines with the same line-impedance
  scalars appended as inputs, on the same windows. Our gain must survive this.
- Also report the physics-only one-ended baselines (reactance, TD-Takagi, negative-sequence-polarised
  Takagi) with the same information.

## 5. Pre-registered gate (local view)
- G1 physics-only one-ended baselines on DoubleLine official windows: reported, no threshold.
- G2 hybrid in-grid (train adapt_grid-DoubleLine train split, test the benchmark family official windows):
  PASS if MAE <= 5.85 % (half of published 11.70 %) AND <= 0.8 x the best physics-only one-ended MAE.
- G3 zero-shot (train DoubleLine only, test TestGrid110kV official local windows): PASS if MAE <= 15 %
  (about half of the published 27-34 %).
- KILL if the hybrid is not at least 20 % better than the best physics-only one-ended locator (then the
  learning adds nothing and the paper collapses to a classical baseline).

## 6. Data needed (downloads are slow, ~1 MB/s from FAU; always verify the byte size)
- benchmark-TestGrid110kV.tar.gz 8,925,756,744 B (in progress 2026-09-30).
- adapt_grid-DoubleLine.tar.gz (~17 GB, training split for G2); later benchmark-CigreMVGrid (29 GB,
  laterals, TSG angle).
- Do not download two archives at the same time (they halve each other's speed).

## 7. Figures/tables (planned, 13-14 figures)
Pipeline; loop equation schematic; error vs tau (5-80 ms) for all methods; views (local/line/global) x
methods table; equal-information table; zero-shot grid matrix; noise/CT/CVT/sync robustness curves;
per-fault-type and per-location bars; fault-resistance sensitivity; kappa angle vs error (why Takagi
fails); interval calibration; runtime; failure cases.

## 8. Grid-model facts (restricted unpickler src/c1/read_graph.py; 2026-10-04)
All main lines in DoubleLine and TestGrid110kV are the SAME type (110kVDanubeAl/St435/55, transposed,
distributed constant-parameter model: i_dist=1, i_model=2, fd_model=0), 20-40 km.
Per km: r1 0.03339, x1 0.26002, r0 0.11814, x0 0.97862 ohm/km.
Check: label-free two-ended identified Z1 (Stage A) equals nameplate x length to 4 digits (e.g. 1-2A 20 km:
0.6679+j5.2005 nameplate vs 0.6689+j5.2005 identified). Signals behave as primary-referred.
Implications: (i) DoubleLine <-> TestGrid zero-shot is the EASY transfer (same line type, same voltage);
CIGRE MV and IEEE-39 are the real tests. (ii) Inverse crime: the data are distributed-parameter but NOT
frequency-dependent, so the paper must disclose this idealisation. (iii) Z0 for ground loops comes from
nameplate line data (utility knowledge), k0 = (Z0 - Z1)/(3 Z1).

## 9. PRE-REGISTERED GATE "H-A" (physics-token hybrid), fixed 2026-10-04 BEFORE any run
- Local view = smaller-bus terminal S (benchmark line_view convention, confirmed in code).
- Physics-only one-ended baselines on ALL fault types (oracle loop, k0-compensated ground loops):
  reactance and I2-/3I0-polarised Takagi, last-cycle phasors. Also a no-oracle variant (minimum
  |Z_app| loop).
- Hybrid H-A: per window, physics tokens (loop apparent impedances normalised by Z1; reactance and
  Takagi estimates for all 6 loops with I2, 3I0 and superimposed polarisers; sequence and onset ratios),
  all dimensionless. Then a LightGBM regressor (L1 loss) predicts the location fraction. Inputs never
  include grid identity or line length.
- Protocol Z (zero-shot): train on ALL TestGrid110kV official local windows, test on DoubleLine official
  local windows (14,640). Published DoubleLine local: zero-shot 28.47 %, held-out 11.70 %.
- PASS-Z: MAE <= 15 % AND >= 20 % lower than the best physics-only one-ended baseline on the same windows.
- KILL: hybrid not >= 20 % better than physics-only, meaning learning adds nothing.
- Reported but not gating: the reverse direction (DoubleLine -> TestGrid), and in-grid once adapt_grid
  is available (G2 bar 5.85 %).

## 10. PRE-REGISTERED GATE "H-B" (two-ended teacher -> one-ended student, closed-form d), 2026-10-04,
## fixed after H-A passed and BEFORE any H-B run
Physics: for the fault loop l, V_l = d Z1 I_l + R_f I_f (phasors, last cycle). With I_f = |k| e^{j theta} I_l,
d = Im(V_l conj(I_l) e^{-j theta}) / Im(Z1 |I_l|^2 e^{-j theta}). Here theta = 0 is the reactance method.
Teacher (training grid only): I_f = I_S,p + I_R,p from BOTH terminals (phase p of the loop; for phase loops
p is the first phase). theta* = arg(I_f / I_l), learned as (sin, cos) by HGB regressors on the same local
tokens as H-A. Loop: an HGB classifier on the tokens predicts the fault loop (6 classes), so there is no oracle.
Student (test grid, ONE end only): theta_hat, l_hat, then closed-form d. Same protocol Z (train TestGrid ->
test DoubleLine), same fixed learner settings as H-A.
Decision rule: H-B becomes the main method if its zero-shot MAE on tau >= 21 ms windows is <= 6.11 % (H-A).
Otherwise H-A stays the main method and H-B is an ablation. Also report: H-B with the oracle loop,
theta = 0 (reactance), and theta = theta* (teacher upper bound).

## 11. H-B RESULT AS PRE-REGISTERED (2026-10-04): FAIL, caused by a physics error in the teacher definition
TG->DL zero-shot: reactance (theta=0, oracle loop) 19.95 %; TEACHER bound (theta*, oracle loop) 22.31 %;
student oracle loop 22.37 %; DEPLOYABLE (predicted loop) 22.23 % (tau>=21: 17.05 % > 6.11 bar). The
teacher bound being WORSE than theta=0 proves the teacher is mis-specified. For a phase loop (p,q), the
fault-resistance voltage is R_f (I_f,p - I_f,q), the LOOP fault current, not I_f,p as registered (equal in
angle only for pure 2ph faults). Correction (physics, not tuning): phase loops I_f = I_f,p - I_f,q; ground
loops I_f = I_f,p. Suspected second issue: ground loops on double circuits miss the parallel line's
zero-sequence mutual coupling, which is invisible from one end. The corrected H-B' keeps the SAME decision
rule (deployable tau>=21 MAE <= 6.11 %). Both the failed and the corrected results are reported.

## 12. H-B' (corrected teacher) RESULT 2026-10-04
TG->DL (zero-shot): teacher bound 11.47 % (tau>=21: 4.22); student oracle loop 11.15 (3.96); DEPLOYABLE 11.08 %
(tau>=21 **3.89 %**, median 1.17 %; loop accuracy 0.988). Decision rule (tau>=21 <= 6.11 %): MET, so H-B' is the
main method on complete windows. Sub-cycle windows: 5 ms -> ~50 % (phasors undefined). Reverse DL->TG:
deployable 21.56 % (loop classifier accuracy 0.62 when trained on the small DoubleLine set); oracle loop
9.58 (2.58).

## 13. PRE-REGISTERED "H-C" (completeness-gated hybrid), fixed BEFORE running
A source-grid-trained HGB classifier on the same tokens predicts c = 1[tau >= 21 ms] (window holds a full
post-fault cycle). Inference: if p(c) >= 0.5 use the H-B' closed form, else the H-A prediction. All three
models are trained on the source grid only, with H-A/H-B' settings unchanged.
Adopt H-C as the main method if its TG->DL zero-shot overall MAE < 8.94 % (H-A); else keep H-A overall
and report H-B' for complete windows.

## 14. PRE-REGISTERED (before evaluating): physics loop selector replaces the learned one in H-B'/H-C
src/c1/phys_loop.py (classical sequence-component phase selection, thresholds fixed a priori, NO training):
accuracy 0.871 DL / 0.904 TG (tau>=21: 0.938 / 0.962). The learned selector gets 0.988 TG->DL but 0.62 DL->TG,
so the training-free selector is the principled choice for BOTH directions. H-C-P = completeness gate +
H-B' with the physics loop (else H-A). It is reported for both directions, with no selection on test results.

## 15. PRE-REGISTERED "TD tokens" for sub-cycle windows (before running)
Per window and loop (local terminal only, nameplate R1, L1, R0, L0), on samples [s0 + N, s1) with SG derivatives:
  ground loop p: v_p = d [R1 i_p + L1 i_p' + (R0-R1) i_0 + (L0-L1) i_0'] + g * Delta i_p
  phase loop pq: v_p - v_q = d [R1 (i_p - i_q) + L1 (i_p - i_q)'] + g * Delta(i_p - i_q)
  with Delta x(t) = x(t) - x(t - 20 ms). Tokens: d_td (2-column LS), d_rl (R_f = 0 fit), and the relative
  residual. That is 18 new dimensionless tokens.
H-A2 = H-A + TD tokens (same learner). Adopt H-A2 in H-C-P (for non-complete windows) if it lowers the MAE on
tau <= 15 ms windows by >= 20 % (relative), averaged over both zero-shot directions.

## 16. LEAKAGE CORRECTION (2026-10-04): sections 12-14 results are INVALID (teacher targets leaked into the
student's inputs through an exclusion-list feature selector). Leak-free teacher-student: theta predicted
to ~1 deg median, but the closed-form d is too angle-sensitive (TG->DL 21.2 %, DL->TG 12.9 %). Valid main
method = H-A2. Rule from now on: every learner takes an EXPLICIT allow-list of input columns, asserted
disjoint from any column derived from remote-end data or labels.

## 17. PRE-REGISTERED in-grid gate "G2-A2" (H-A2, benchmark held_out protocol), fixed 2026-10-04 BEFORE the
## adapt_grid archive was extracted or any of its labels were seen
Protocol = the benchmark's `held_out` protocol for DoubleLine (splits v1.0, committed in
external/evemtbench-benchmark/src/evemtbench/splits/v1.0/held_out/double_line, seed 20325215):
- Train: line-fault episodes of the adapt_grid TRAIN split (70 %). The val split is NOT used (H-A2 has fixed
  hyper-parameters and no early stopping). The adapt_grid TEST split and the benchmark family are never
  used for any choice.
- Windows: the benchmark contract (50 ms windows, 5 ms grid from the episode start; a window is scored iff
  the episode is a line fault (location not null) and the window overlaps the fault interval). For the
  benchmark family this is exactly the 14,640 windows already in use. Local terminal S, same tokens
  (59 local + 18 TD, explicit allow-list), same HGB settings, seeds 0-2 averaged, no re-tuning.
- Test sets (both reported, as in the benchmark): (a) adapt_grid TEST split line-fault windows
  (published best local MAE 18.60 %, GRU), (b) benchmark family 14,640 windows (published best local
  11.70 %, GRU).
- PASS (from section 5, G2): benchmark-family MAE <= 5.85 % AND <= 0.8 x the best physics-only one-ended
  MAE on the same windows. Reported regardless of the outcome; also reported: the adapt_grid test MAE,
  the zero-shot H-A2 numbers for contrast, and the equal-information MLP/GRU (+Z) trained on the same
  train split (run only after the validator has released the GPU).
- If the window count on the adapt_grid test split cannot be matched to the benchmark's n_test exactly,
  the discrepancy is reported and the benchmark-family number (exact count) is the gating one.
- Published n_test (results/1.1.0/aggregated_results.csv of the benchmark repo): adapt_grid test 11,673 windows,
  benchmark 14,640. Our window count on the adapt_grid test split must equal 11,673 (else disclosed, see above).

### 17a. PROTOCOL AMENDMENT (2026-10-04, from the adapt_grid LABELS only; no waveform read, no model run yet)
- adapt_grid labels hold `event_flt_target_line_location` for EVERY episode (also bus faults and switching);
  in the benchmark family it is NaN for bus faults. Window-count reconstruction on the test split: the
  published n_test 11,673 is reproduced (11,674) only if BUS faults are included: windows on the 5 ms grid from
  t = 1.0 s (ends 480 + 48k samples), fault-active iff the window overlaps [nf, nf + 289) samples for short
  circuits (about 30 ms; the same rule gives exactly 16 windows per SC and 10 per incipient fault on the
  benchmark family) and [nf, nf + floor(duration * 9600)) for incipient faults. Line faults only: 6,886 windows
  from 411 episodes. => the published in-distribution FL "test" MAE (local 18.60 %) mixes about 41 % bus-fault
  windows whose location label has no physical meaning (and the published models were trained on such
  windows). It is NOT a valid comparator for line-fault location.
- Amended evaluation (decision rule of section 17 unchanged; it gates on the benchmark family, 14,640 windows):
  train = adapt_grid TRAIN split, LINE faults only (1,832 episodes), windows by the rule above; tests =
  (a) adapt_grid TEST split line faults (411 episodes, about 6,886 windows; continuous locations 1-99 % and
  R_f 0-50 ohm), (b) benchmark family 14,640 windows. Line impedances from the adapt grid model
  (graphs/graph_grid0.pickle). Comparator for (a) and (b): the equal-information MLP/GRU (-raw, +Z) re-trained
  on the SAME line-fault train windows (fixed published recipe, 3 seeds), plus the physics-only baselines.
  The published 18.60 / 11.70 % are reported for context with this caveat.

## 18. PRE-REGISTERED CIGRE MV gate "G3-MV", fixed 2026-10-04 BEFORE the archive is complete (only the column
## names of result0..10 were seen: every MainLnX-Y segment is measured at both buses; loads/IBR on buses)
Data: benchmark-CigreMVGrid (benchmark family). Windows: the reconstructed benchmark rule (design 17a); the
line-fault window count must equal the published n_test 56,340 (else disclosed). Line data: the CIGRE grid model
(graph pickle, nameplate Z1/Z0 per faulted segment). Nothing from CIGRE is used for any choice below.
(a) TWO-ENDED TD locator (SG, unchanged code path; Z1 nameplate): reported vs published best line view 24.52 %
    and global 18.87 % (held_out, in-grid trained). No threshold (supporting result).
(b) ONE-ENDED H-A2 ZERO-SHOT from 110 kV to 20 kV: trained on ALL DoubleLine + TestGrid official windows (pooled;
    47,580 windows), unchanged tokens/settings/seeds, explicit allow-list. Comparators on the same windows:
    physics-only one-ended baselines; equal-information MLP/GRU (-raw, +Z) trained on the same pooled 110 kV windows
    (fixed recipe, 3 seeds; +Z inputs = faulted segment R1, X1, R0, X0 in ohm); published local held_out 30.35 %
    (in-grid trained, = majority predictor) and transfer_zeroshot 30.61 %.
    "CLEAR WIN" (checkpoint rule of RESUME_PROMPT step 5): H-A2 MAE <= 0.8 x the best equal-information baseline
    AND <= 0.8 x 30.35 % (published best local). Else the one-ended learner does not generalise to MV and the paper
    keeps H-A2 as a 110 kV result (protection-specialist framing).
(c) Journal rule: TSG only if (a) or (b) gives the paper's largest margin on CIGRE MV (MV-centric story); else TII.
Reported regardless: per fault type, per grounding type (if labelled), per segment length, post-fault time bins.

## 19. ABLATIONS requested by audit NC2 (research/AUDIT_C1_novelty_v2.md), REPORT-ONLY (no gate, no choice made on
## them), fixed 2026-10-04 before running. Zero-shot TG->DL and DL->TG, same HGB settings/seeds unless stated.
A1 learner without physics: HGB on the raw 480x6 local window (2,880 samples as features), -raw and +Z (adds R1, X1,
   R0, X0). Answers "is the gain from the tokens or from boosting?" (HGB on raw windows is a benchmark-group baseline).
A2 phasor tokens only (= H-A, 59) vs TD tokens only (18) vs both (= H-A2, 77).
A3 un-normalised tokens: the 6 x 3 apparent-impedance tokens (zr, zi, absz) in ohm instead of per Z1 (other tokens
   unchanged, they are already fractions or ratios) -> tests the grid-agnostic normalisation claim.
A4 classical one-ended baselines by post-fault time (5..80 ms), oracle loop: reactance, Takagi-I2, Takagi-superimposed,
   Takagi-3I0 (ground loops), to compare with the single-ended reactance 10.9 % at 50 ms reported in arXiv:2608.20181.
A3-MV (added 2026-10-04 after the 110 kV ablations, BEFORE any CIGRE data): on the 110 kV pair the Z1 normalisation
is mixed (un-normalised 9.91 vs 7.75 TG->DL, 10.13 vs 10.58 DL->TG; same line type in both grids). The CIGRE MV
zero-shot run therefore also reports A3 (un-normalised tokens) and A1 (HGB on raw windows +-Z), trained on pooled
DL+TG, report-only. The normalisation claim is made in the paper only if A3 is clearly worse on CIGRE MV.

### 18a. AMENDMENT (2026-10-04, from data checks only; no locator/learner run on CIGRE yet)
(1) Line impedance: src/c1/check_z_grid.py (two-ended identification from PRE-FAULT normal operation, no labels):
    distributed type (12-13, 13-14): identified = nameplate (|Z| ratio 1.000). Lumped type (12 other segments):
    X identical, R about 24 % higher than nameplate on every segment (|Z| ratio 1.086, angle 49.0 vs 55.0 deg;
    consistent with a conductor-temperature correction of R in the simulator). On DL/TG identified = nameplate.
    => Report BOTH variants on CIGRE: (i) nameplate Z1/Z0 = the REGISTERED primary (gate decisions use it);
    (ii) identified Z1 from normal operation (the two-ended locator's stated method; for one-ended tokens Z1
    identified and Z0 = nameplate with R0 scaled by the same identified R1 ratio). (ii) is a disclosed sensitivity
    variant, never used to pick between methods.
(2) Line MainLn8-14 is measured only at bus 14 (no cubicle at bus 8). Benchmark convention (line_view._termkey):
    local = smaller bus among MEASURED terminals -> bus 14 here; LINE view duplicates it. So: local terminal =
    bus 14; the two-ended locator is undefined on 8-14 (reported on the other 14 segments, n stated); the label
    orientation for 8-14 (is y measured from bus 8?) is fixed by a physics convention check on bolted complete
    windows (reactance from bus 14 vs y and 1 - y), as done for the 110 kV orientation in stage_a; if y is from
    bus 8, every method's prediction on 8-14 is mapped as y_hat = 1 - d_hat (same mapping for all methods).

## 20. AMENDMENT "I0 guard" (token bug found by validator part 4), fixed 2026-10-04 BEFORE any re-run
Seen before fixing: only the stored token tables' i0r / i2r histograms per fault type (no model run, no new
labels). Numerical-zero I0 (CSV-rounding level) for 2ph / 3ph windows: i0r <= 1.8e-7 on DL, TG, adapt, and
<= 1e-9 on MV. Physical I0: ground short circuits >= 1e-2; incipient ground faults 1e-7..1e-4 in early windows;
MV 2ph / 3ph windows have capacitive I0 of 1e-6..1e-2 (reproducible, kept). I2 is NOT at noise level on any
benchmark grid (2ph / 3ph min i2r >= 5e-6), so no I2 guard.
(1) Rule (single source: local_features.guard_i0, called at the end of window_feats): if i0r < 1e-6 then
    i0c = i0s = 0 and ag_tak0 = bg_tak0 = cg_tak0 = 0.5 (0.5 = est()'s existing value for an undefined
    estimate). No other token changes (TD tokens and loop quantities use I0 only additively, where 1e-14 is
    harmless).
(2) Tables: every stored token table is patched with the SAME function (it reads only i0r, which the guard
    does not change, so patching equals regeneration). Old tables are kept as *_preguard.parquet. Proof of
    equality: recompute >= 12 episodes per grid from the raw CSVs with the guarded code, max abs diff over the
    77 ALLOW tokens < 1e-9.
(3) Re-runs (unchanged code otherwise, same seeds): ha2_eval.py (H-A, H-A2 zero-shot + noise table),
    adapt_ingrid.py (G2-A2), zs_eval.py MV DL TG (G3-MV(b)), ablations.py (A2, A3; A1 / A4 do not use tokens).
    Old vs new recorded in research/C1_GATE.md. Gate rules unchanged; if any verdict flips it is reported.
    Expected change: within HGB noise (< ~0.25 pp). Any headline change > 0.5 pp is investigated before it is
    reported. Validator re-checks (claudedocs/validator_spec_guard.md).
(4) Also report: adapt_grid has 22 short-circuit windows (and some incipient) with all local currents ~0
    (every token saturated; i0r < 1e-9, i2r < 1e-7; the guard also applies to them). Kept as is (they are part
    of the registered window set).

## 21. CLASSICAL ONE-ENDED BASELINES (audit NC2 / TPWRD risk), REPORT-ONLY, fixed 2026-10-04 before any code ran
Grids / windows: DL (14,640), TG (32,940), adapt test line faults (6,883), CIGRE MV (56,340; Eriksson only where the
remote end is measured). Oracle loop (labels; favours the baseline, disclosed). Phasors: one-cycle DFT, last cycle of
the window. PRE-FAULT MEMORY ALLOWED for the classical methods: pre-fault phasors from the cycle ending 1 ms before
inception (a relay has it; H-A2 does not -> favours the baselines, disclosed). Superimposed dX = X_post - X_pre.
Methods (loop V, I with k0 compensation as in local_features):
  R  reactance:        d = Im(V / I) / Im(Z1)
  T  Takagi:           d = Im(V conj(dI_l)) / Im(Z1 I conj(dI_l))      (dI_l = loop superimposed current)
  T2 Takagi-I2:        polarised by I2 (as tak2)
  MT modified Takagi:  ground loops only, polarised by 3I0 e^{-jT}, T = angle of the zero-sequence current
                       distribution factor (Z0SR + (1-d) Z0L) / (Z0SL + Z0L + Z0SR), iterated 3x from d = 0.5;
                       Z0SL = -dV0/dI0 local (one-ended), Z0SR = -dV0R/dI0R from the REMOTE end (oracle settings
                       knowledge, as from short-circuit studies; disclosed). Phase loops: falls back to T.
  E  Eriksson (1985):  d^2 - K1 d + K2 - K3 Rf = 0 with K1 = 1 + ZSR/ZL + V/(ZL I), K2 = V/(ZL I) (1 + ZSR/ZL),
                       K3 = dI/(ZL I) (1 + (ZSL + ZSR)/ZL); eliminate Rf (Im/Re), root in [-0.1, 1.1] closest
                       to 0.5 if two; ZSL = -dV1/dI1 local, ZSR = -dV1R/dI1R remote (oracle settings, disclosed).
  Correctness check BEFORE reporting: on bolted (R_f <= 1 ohm) short circuits with post >= 25 ms on DL, every method
  must give MAE <= 3 %, else the implementation is fixed (only bugs, not method choices).
Reported: MAE / median overall and by post-fault time bins (5-15, 20-30, 35-50, 55-80 ms), short circuits only and
all, and by R_f bin. Comparison claim: H-A2 vs the BEST classical one-ended method per grid (lowest overall MAE);
"H-A2 beats classical one-ended" is claimed for a grid only if H-A2 MAE <= 0.8 x that best MAE.

## 22. MEASUREMENT-CHAIN ROBUSTNESS (audit), REPORT-ONLY, fixed 2026-10-04 before any code ran
Grids: DL and TG (110 kV; all official windows). Applied to the full record of every terminal channel (both ends,
independent per end) BEFORE windowing; currents mapped back to primary amperes (x ratio).
  AA   anti-aliasing: 2nd-order Butterworth low-pass, fc = 2 kHz, causal (lfilter), on all six channels.
  CT   IEEE PSRC 'CT SAT' model (Swift, C37.110 calculator theory): i_e = sgn(l) (10/RP) (w|l| / (sqrt2 Vs))^S,
       dl/dt = Rt i2 + Lb di2/dt, i2 = i1/N - i_e, backward Euler + Newton per sample; S = 22, ratio 1200:5,
       Rw = 0.6 ohm, Lb = 0 (numerical relay). CT-mild: C800 (Vs = 800 V), Rb = 1.0 ohm, rem = 0.
       CT-severe: C400 (Vs = 400 V), Rb = 2.0 ohm, rem = +0.6 pu of Vs flux (all phases, both ends).
       Saturation flag per window: max|i_e| > 0.05 max|i_s| in the window on any local phase (fraction reported).
  CVT  generic second-order CVT equivalent (series Ce, Lc tuned to 50 Hz, resistive burden incl. passive FSC):
       H(s) = s Ce R / (s^2 Lc Ce + s Ce R + 1), parameterised by its free-transient time constant tau = 2 Lc / R;
       CVT-5 (tau = 5 ms) and CVT-15 (tau = 15 ms). Residual voltage after a bolted terminal collapse at 10 / 20 /
       40 ms is reported for each variant (no claim of compliance with a specific IEC 61869-5 class).
  FULL = CT-severe + CVT-15 + AA.
Methods: two-ended TD locator; physics one-ended (oracle-loop reactance, Takagi-I2); H-A2 trained on the CLEAN source
grid, tested on the distorted target (TG->DL, DL->TG); and H-A2 trained AND tested with FULL (matched chain).
Wording rule: a method is called "robust to <scenario>" only if its MAE rises by <= 1.0 pp (two-ended) or <= 2.0 pp
(one-ended) over clean; otherwise the degradation is reported as is.
### 21a. AMENDMENT (2026-10-04, after the registered bolted-fault check failed for one method only)
Takagi-I2 failed the DL bolted check (5.61 %): all of it from 3ph faults (21.5 % DL, 18.2 % TG; other types 1.0-1.9 %),
where I2 ~ 0 and the I2 polarisation is noise. Fix (bug, same definition as G1 one_ended.py): for 3ph faults T2 = R.
Every other method passed (DL bolted: R 1.13, T 1.54, MT 1.53, E 1.08 %). NaN estimates (MV 8-14 has no remote end:
E and MT undefined; MT also where dI0 ~ 0) are scored as 0.5 (mid-line prior) and their counts reported.
### 21b. CORRECTION (2026-10-04, found by validator part 6): the first Eriksson code had two UNREGISTERED fallbacks
(negative discriminant -> p/2; no root in [-0.1, 1.1] -> out-of-range root nearest 0.5). Now as registered: both cases
are undefined (scored 0.5). The fallback variant is kept as results/c1_classical_*_E_unregistered.* and is NOT reported
as the method. Undefined share (registered rule): DL 2,282 / 14,640, TG 4,132 / 32,940, adapt 2,473 / 6,883, MV 31,466 /
56,340 (incl. 3,756 on 8-14).
### 22a. CLARIFICATION (validator part 6): RP in the CT model = rms/peak of |sin|^S over a cycle (PSRC definition;
0.346 for S = 22), as coded. The 'tak2_oracle' column of c1_chain_eval.csv is the token-based Takagi-I2 WITHOUT the 21a
3ph fallback (report-only; no verdict depends on it).

## 23. ONE-ENDED CONDITIONING ANALYSIS (paper claim C4), REPORT-ONLY, fixed 2026-10-04 before any computation
Derivation (lumped line, loop quantities at terminal S, fault current I_F through R_f):
  V / I = d Z_L + R_f (I_F / I).  Any one-ended locator that assumes I_F / I has angle b^ solves
  d^ = Im((V/I) e^{-j b^}) / Im(Z_L e^{-j b^}), so its error is
  e = d^ - d = rho * |I_F/I| * sin(b - b^) / sin(theta_L - b^),   rho = R_f / |Z_L|, b = angle(I_F / I).
  Reactance: b^ = 0 -> e_R = R_f Im(I_F / I) / X_L. Takagi: b^ = angle(dI_loop / I).
  => error = rho x kappa x sin(angle error), kappa = |I_F/I| / sin(theta_L - b^). For a target error eps the
  polarising angle must be right to within D_req = arcsin(min(1, eps / (rho kappa))).
Data: 1phg_shc (R_f to ground, well defined) windows with post >= 25 ms on lines measured at both ends (DL, TG, adapt
test, CIGRE MV); I_F = I_Sp + I_Rp (faulted-phase currents of both ends, last-cycle phasors; line charging ignored);
R_f = label; y = label. Other fault types reported descriptively only.
Report: (a) identity check: observed reactance and Takagi errors (unclipped) vs predicted e from the formula: median
and 90th percentile of |e_obs - e_pred| (pp), per grid; (b) rho, kappa per grid (median, 90th pct); (c) D_req for
eps = 5 % per grid and the share of windows with D_req < 1 deg; (d) Takagi's actual polarising-angle error
|b - angle(dI_loop/I)| per grid; (e) MAE vs rho bins (all SC windows, post >= 25 ms) for two-ended, R, T, E, H-A2
(figure data). Stated expectation (before running): MV rho >> 110 kV rho, so on MV most windows need D_req < 1 deg
(below the ~1 deg angle accuracy the leak-free learned angle model reached, design 16), while on 110 kV most do not.

## 24. REVIEW-DRIVEN CHECKS (audits AUDIT_DRAFT1_claims / _review), REPORT-ONLY, fixed 2026-10-06 before any run
(a) Two-ended PHASOR locator (IEEE C37.114 synchronized form), positive-sequence last-cycle DFT phasors of both ends:
    d = Re[(V1S - V1R + Z1 I1R) / (Z1 (I1S + I1R))]; same windows as the TD locator (DL, TG, adapt test, CIGRE MV with
    both ends). Reported overall and by post-fault time; no threshold.
(b) Suonan-Qi-type one-ended time-domain R-L identification = the 'dtd' token of the ORACLE loop (fit of
    v = d y + g Delta i on the last 40 ms), physics only; DL, TG, adapt test, MV. Reported as a classical baseline.
(c) Two-ended TD locator sensitivity (DL, TG; nameplate Z): (i) synchronization error: remote record shifted by
    +/- {1, 5, 10, 20} samples (0.10 to 2.08 ms); (ii) line-parameter error: R1 and L1 both scaled by
    {0.90, 0.95, 1.05, 1.10}; (iii) R1 only x{0.8, 1.2}. Reported as MAE; no threshold.
(d) Inception-free cost (no new run): two-ended MAE for windows that contain pre-fault samples (tau <= 50 ms) vs
    fully post-fault windows (tau >= 55 ms), from the existing tables; stated in the paper as observed.
Wording rule: none of (a)-(d) changes a registered result; they are reported as context and limitations.
### 24a. AMENDMENT (2026-10-06, after the DL result of 24(c)(i) only): the registered synchronization offsets are whole
samples (>= 104 us at 9.6 kHz), far coarser than GPS/PTP time synchronization (~1 us; IEC 61850-9-3). Added, report-only:
sub-sample offsets {1, 10, 50} us applied as a fractional delay to the remote record (FFT phase shift of the whole
record), DL and TG. The registered whole-sample results are reported unchanged.
### 19a. CORRECTION (2026-10-06, validator part 8): A3 'un-normalised tokens' was implemented as clipped per-unit tokens x Z1,
which is not V/I in ohm. Corrected definition (before re-running): for each loop, zr, zi, absz = Re, Im, |V/I| in ohm from
the last-cycle phasors, NOT clipped (0 when |I| < 1e-9); all other tokens unchanged; same HGB settings and seeds;
TG->DL, DL->TG and DL+TG->MV. Report-only (no claim depends on it; the paper makes no normalisation claim).
Also: section 15 / 24(b) TD tokens use the last 30 ms of the window (s1 - 480 + 192 .. s1), not 40 ms.

## 25. REVAMP ANALYSES (research/REVIEW_DRAFT2_PDF.md s.7 "analyses a referee will ask for"), REPORT-ONLY, fixed
## 2026-10-06 S6 before any of them was computed. No gate; no registered result changes; existing per-window predictions
## only (no retraining except the deterministic re-fit of the H-A2 zero-shot models by classical_report.zs_pred).
Scoring as in the paper: estimates clipped to [0, 1], undefined (NaN) -> 0.5; error e = |d^ - d| x 100 (% of length).
Test sets: DL and TG benchmark families (line faults), CIGRE MV (two-ended rows: the 52,584 windows on sections measured
at both ends; one-ended rows: all 56,340), adapt test split (6,883 line-fault windows).
(a) Episode-level bootstrap 95 % CI: B = 2,000 resamples of episodes (sim_idx) with replacement, numpy
    default_rng(0), percentile interval of the window-level MAE, for: two-ended TD (DL, TG, MV nameplate, MV identified
    Z, adapt test); Eriksson (DL, TG, adapt test, MV); H-A2 zero-shot (TG->DL, DL->TG, DL+TG->MV) and in-grid adapt test;
    best equal-information baseline zero-shot (TG->DL GRU+Z, DL->TG MLP+Z; error of each seed's prediction, averaged
    over the 3 seeds per window). Paired CI (same resamples) of the relative reduction 1 - MAE(H-A2) / MAE(best eq) in
    both zero-shot directions. If the paired CI includes 0 the gain is reported as not significant.
(b) Physical units: per window error in metres = e/100 x length_km x 1000 (line length of the faulted line); report
    mean (m) and 95th percentile (% and m) for two-ended TD, Eriksson and H-A2 on the sets of (a).
(c) Eriksson scored only where its quadratic has a root in [-0.1, 1.1]: MAE on defined windows and coverage (share
    defined), overall and for windows ending <= 15 ms and >= 20 ms after inception; DL, TG, adapt test, MV.
(d) EXPLORATORY (post hoc: the per-time results of Fig. 4 had been seen): one-ended switch rule = H-A2 for windows ending
    < 20 ms after inception; from 20 ms, Eriksson where defined, else H-A2. MAE for zero-shot TG->DL, DL->TG,
    DL+TG->MV and in-grid adapt test. Must be labelled exploratory in the paper.
(e) Indicative computational cost (one CPU core, wall clock, this laptop; not validated, stated as indicative): median
    time per window for the two-ended TD estimate (Clarke + Savitzky-Golay + LS), the local feature extraction (59 phasor
    + 18 TD features) and H-A2 inference (3 boosting models), measured on in-memory records of 50 DL episodes (CSV
    reading excluded); and the H-A2 training time on the TG source grid (3 seeds).
Validation: validator part 9 re-implements (a)-(d) from claudedocs/validator_spec_part9.md; tolerance: point values
<= 0.01 pp (H-A2 cells <= 0.5 pp: boosting refit noise), CI bounds <= 0.15 pp (Monte-Carlo noise of B = 2,000).
## 26. PROFESSOR REVISION ANALYSES (research/PROF_COMMENTS_2026-10-08.md M1, M5, M6), REPORT-ONLY, fixed 2026-10-08
## BEFORE any of them was computed or any new output was seen. No gate; no registered result changes; wording rules below.
Scoring as in s.25: estimates clipped to [0, 1], undefined (NaN) -> 0.5; e = |d^ - d| x 100.
(a) M1 TWO-ENDED LEARNERS AT EQUAL INFORMATION. Inputs per official window (both terminals, same 480-sample windows as
    src/c1/raw_windows.py): (P) "physical inputs": the four sequences u_alpha, u_beta, w_alpha, w_beta of Eq. (3)
    (Clarke, Savitzky-Golay 11/2 derivatives, nameplate R1, L1 of the faulted line, exactly as the TD locator), each
    window divided by its own RMS of (w_alpha, w_beta) jointly (scale-free; d = u/w is invariant to it);
    (Z) "two-ended +Z": raw 12 channels (S and R terminals, Ia Ib Ic Va Vb Vc) plus R1, X1, R0, X0 of the faulted line.
    Models: MLP and GRU of src/c1/equal_info.py, identical recipe (AdamW 1e-3, wd 1e-4, cosine, 60 epochs, batch 256,
    patience 12, episode-grouped 10 % validation, channel/target standardisation, MSE), input width adapted only.
    Seeds 0, 1, 2. Directions: TG->DL, DL->TG, DL+TG->MV (MV test = the 52,584 two-ended windows; train = all line-fault
    windows of both 110 kV grids). Report MAE mean and SD over seeds per (input, model, direction) and the best cell.
    Wording rule: if the best two-ended learner's MAE is >= 2x the TD locator's MAE on a test set, the paper may state that
    the physics advantage holds against learners given the same two-ended physical inputs on that set; if it is within
    2x, the claim is softened to "comparable"; the title/abstract say "equal terminals" unless the rule holds on all
    three sets.
(b) M5 TD NOISE. Report the existing sweep (src/c1/td_noise.py, Sep 30, SG derivative, official windows, AWGN per
    channel at 60/40/30 dB of the channel's pre-fault RMS, seeded per episode) for DL and TG: results/c1_td_noise.csv,
    c1_td_noise_TestGrid110kV.csv. Rows added to Table VI for 40 and 30 dB (two-ended). Validator re-implements the 30 dB
    and 40 dB SG cells. Wording rule of s.22 applies (robust if rise <= 1.0 pp).
(c) M6 d = 50 % ARTEFACT. Existing per-window predictions only (no retraining; learners were trained with these
    episodes). Exclusion set: every window whose true location y = 0.5 (|y - 0.5| < 1e-9), all fault types. Recompute the
    MAE of: TD and phasor two-ended (DL, TG, DL-adapt test, MV two-ended set); reactance, Takagi, Eriksson (undefined as
    0.5), R-L identification, best equal-information MLP/GRU, PFB (columns of Table III where per-window predictions exist).
    Also report: share of windows with y = 0.5 per set; share of all windows where Eriksson is undefined AND y = 0.5
    (zero error by construction); PFB MAE on incipient + HIF windows vs short circuits. Published models: not recomputable
    (no per-window predictions); stated as such. Wording rule: a ranking stated in the paper is kept only if it holds in
    the excluded-set table; otherwise the text is changed.
Validation: validator part 10b (claudedocs/validator_spec_part10b.md) re-implements (a) evaluation of saved predictions,
(b) 30/40 dB cells and (c) from the spec only; tolerance 0.01 pp (network cells: re-scoring of saved predictions only).
### 26a. AMENDMENT (2026-10-08, after the s.26(a) MAE table and the s.26(c) output were seen; no excluded-set number of
### the two-ended learners seen yet): the d = 0.5 exclusion of 26(c) is also applied to the 26(a) learners (same rule,
### per seed then averaged) and to the TD locator on the same test sets, because the new rows enter Table II. Also
### reported for them: MAE on incipient + HIF windows vs short circuits. Report-only; no wording rule changes.
