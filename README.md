# Code and results: where learning helps in impedance-based fault location

For the manuscript *Where Learning Helps in Impedance-Based Fault Location: An Equal-Information Evaluation on an Open
EMT Benchmark* by Shlok Goenka and Ganesh Khekare (School of Computer Science and Engineering, Vellore Institute of
Technology), submitted to IEEE Transactions on Power Delivery.

The repository holds the code that produces every number, table and figure of the paper from the public EvEMTBench
records, the aggregate result tables, the pre-registration log, the results ledger and the report of the independent
re-implementation.

## Source data (not redistributed)

| Set | File on the EvEMTBench FAU DataCloud share | Size (bytes) |
|---|---|---|
| DoubleLine, benchmark family | `benchmark-DoubleLine.tar.gz` | 2,083,367,135 |
| TestGrid 110 kV, benchmark family | `benchmark-TestGrid110kV.tar.gz` | 8,925,756,744 |
| CIGRE MV, benchmark family | `benchmark-CigreMVGrid.tar.gz` | 31,216,663,114 |
| DoubleLine adapt_grid (train/test splits) | `adapt_grid-DoubleLine.tar.gz` | 17,733,535,772 |

Share: https://data.fau.de/share/0e8d60feb7e65616c60aab78b93db77053275da53fd894bf5b75fc5e9ee7dfbf/
Benchmark code, splits and published results: https://github.com/EvEMTBench/evemtbench-benchmark (clone into
`external/evemtbench-benchmark`). Cite the EvEMTBench papers (arXiv:2609.28149, arXiv:2608.19777) when you use the data.

The licence of the data is stated as CC BY 4.0 in the benchmark repository and CC BY-NC-ND 4.0 on the dataset paper.
Until this is clarified, no per-window table derived from the records is redistributed here; the scripts regenerate
all of them.

`src/c1/dl_segmented.sh <file> <bytes> [K]` downloads one archive in K byte ranges with resume and a final size check.
Extract each archive to `data/raw/evemt/<GridName>/` (`DoubleLine`, `TestGrid110kV`, `CigreMVGrid`,
`adapt/DoubleLine`).

## Layout

| Folder | Contents | Licence |
|---|---|---|
| `src/c1/` | locators, tokens, learners, baselines, sensitivity studies | MIT |
| `src/paper/` | figure scripts, number macros (`make_macros.py` writes every number used in the paper), release builder | MIT |
| `src/validate_ha2/` | independent re-implementation used to check every headline number | MIT |
| `results/` | aggregate result tables quoted in the paper | CC BY 4.0 |
| `docs/preregistration_log.md` | analysis plans written before each run, with amendments and corrections | CC BY 4.0 |
| `docs/results_ledger.md` | every number with its provenance, including superseded and invalidated results | CC BY 4.0 |
| `docs/independent_validation.md` | report of the independent re-implementation (parts 1 to 8) | CC BY 4.0 |

## Environment

Python 3.14; versions in `requirements.txt`. Windows 11, 24 CPU threads and one NVIDIA RTX 5070 Ti laptop GPU (12 GB).
The MLP/GRU baselines need CUDA; everything else runs on CPU. LightGBM is not used (scikit-learn's histogram gradient
boosting is). Run every command from the repository root.

## Reproducing the paper

```bash
# 1. local tokens, time-domain tokens and raw windows, 110 kV benchmark families (clean and test-time noise)
for G in DoubleLine TestGrid110kV; do
  EVEMT_GRID=$G python src/c1/local_features.py
  EVEMT_GRID=$G python src/c1/td_tokens.py
  EVEMT_GRID=$G python src/c1/raw_windows.py
  for S in 30 40; do EVEMT_GRID=$G EVEMT_SNR=$S python src/c1/local_features.py; EVEMT_GRID=$G EVEMT_SNR=$S python src/c1/td_tokens.py; done
done
# 2. zero-shot one-ended learners and equal-information baselines (GPU)
python src/c1/ha2_eval.py
python src/c1/equal_info.py
python src/c1/equal_info_preds.py
# 3. one pass per grid: windows, tokens, two-ended locator (MV twice: nameplate and identified impedance)
python src/c1/grid_pass.py DoubleLine DL
python src/c1/grid_pass.py TestGrid110kV TG
python src/c1/grid_pass.py CigreMVGrid MV
python src/c1/grid_pass.py CigreMVGrid MV_zid zid
# 4. in-grid protocol on adapt_grid
python src/c1/adapt_tokens.py
python src/c1/adapt_ingrid.py
python src/c1/adapt_equal_info.py
python src/c1/adapt_td.py
# 5. CIGRE MV zero-shot and ablations
python src/c1/zs_eval.py MV DL TG
python src/c1/zs_equal_info.py MV DL TG
python src/c1/ablations.py
python src/c1/ablation_a3_fix.py
# 6. classical one-ended baselines, measurement chain, conditioning, time-resolved data, sensitivity checks
for G in DL TG ADAPT MV; do python src/c1/classical_oe.py $G; done
python src/c1/classical_report.py
python src/c1/chain_pass.py all
python src/c1/conditioning.py
python src/c1/fig4_data.py
python src/c1/review_checks.py
python src/c1/sync_subsample.py
# 7. confidence intervals, errors in metres, Eriksson on defined windows, switch rule; indicative runtime
python src/c1/revamp_stats.py
python src/c1/runtime.py
# 8. numbers, figures and the Word review copy of the paper
python src/paper/grid_topology.py
python src/paper/make_macros.py
python src/paper/figs_v3.py
python src/paper/build_docx.py
```
`make_macros.py` also reads `results/validate_ha2_part9_summary.csv` from the independent re-implementation: the paper
quotes the lower of the two implementations' confidence bounds for the gain of the physics-feature learner.

Each script states its inputs and outputs in its docstring. Every learner takes an explicit allow-list of 77 input
columns (`src/c1/tokens_core.py`); no label, fault type or remote-end quantity reaches it. The results ledger lists
three results that were invalidated during the project (teacher-target leakage, an unregistered estimator fallback,
a wrong ablation definition) and the corrected numbers; only corrected numbers appear in the paper.

## Citation

See `CITATION.cff`. The archived version has the DOI given on its Zenodo page.
