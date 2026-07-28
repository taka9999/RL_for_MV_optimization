# Phase 1A Report Revision Map

Companion to `REPORT_SOURCE_INVENTORY.md`. Read-only mapping — no edits made
to `ICA_report.tex` yet. All line numbers refer to the baseline commit
(`7876d66`) of `ICA report/ICA_report.tex`.

**Scope reminder:** Phase 1A = common-covariance track only (5 training
seeds, 0–4). The regime-dependent-covariance track (`tab:regcov_metrics`,
`tab:regcov_parameters`, `fig:stage1_regcov`, and every prose passage that
discusses it) is **out of scope** — no aggregate data exists for it yet.

## 5. Locations reporting a single-seed number

| Line(s) | What | Value(s) | In scope? |
|---|---|---|---|
| 1268, 1277–1297 (`tab:configs`) | Stage 1 iteration count etc. | `9000` iters — **matches** Phase 1A config, not a result number | config, not a result |
| 1270, 1312, 1479 | Stage 2 legacy eval path count | **"$500$ … evaluation paths"** | ⚠️ Phase 1A used **1000** paths — discrepancy, not yet reconciled |
| 1397–1411 (`tab:common_metrics`) | Stage 1 common-cov single-seed summary | EW `1.070/0.272/0.074/0.309/0.187`; MinVar `1.071/0.261/0.068/0.311/0.182`; RL `1.197/0.266/0.071/0.611/0.115` | **in scope** |
| 1413 | Prose interpreting the above | "raises mean terminal wealth by approximately $0.126$"; "nearly doubles the target-hit probability" | **in scope** |
| 1438–1452 (`tab:regcov_metrics`) | Regime-dep-cov single-seed summary | EW/MinVar/RL rows | out of scope |
| 1454 | Prose interpreting regcov table | — | out of scope |
| 1476–1491 (`tab:stage2_means`) | Stage 2 legacy single-seed summary | Direct Boundary RL `1.248`; CenterOnly daily `1.136`; CenterOnly monthly `1.147`; EW monthly `1.052` | **in scope** |
| 1495 | Prose interpreting the above | "$0.101$ above monthly … $0.112$ above daily" | **in scope** |
| 1549 | Regcov approximate-HJB claim | "higher mean terminal wealth and lower dispersion" (qualitative, regcov) | out of scope |

## 6. Locations using "representative run" / "single-seed" / "legacy" framing

| Line(s) | Section | Note |
|---|---|---|
| 112 (Abstract) | "Representative simulations show…" | mixes Stage 1 (in-scope) + Stage 2 (in-scope) qualitative claims |
| 121 | Introduction | "present representative experiments do not isolate each benefit" — general caveat, likely keep |
| 141 | Introduction | about the regcov surrogate specifically — out of scope |
| 1302, 1314–1318 (`tab:stage2_config` caption + rows) | Numerical Study | documents legacy run's **non**-traceability — Phase 1A run **is** fully traceable (checkpoint hashes, manifest); table needs a decision: replace vs. keep-as-history + add companion table |
| 1366–1370 (`\subsection{Draft robustness protocol…}`) | Numerical Study | **directly anticipates this exact revision** ("five-seed replication…in progress but was not completed by the draft cutoff") — prime rewrite target for the common-cov/Stage 2 portions; the regcov portion of this caveat still applies |
| 1374, 1393 | Stage 1 fig/caption | "representative run" / "Representative Stage 1 results under common covariance" |
| 1400 | `tab:common_metrics` caption | "Representative single-seed common-covariance…" |
| 1434, 1441 | Stage 1 regcov fig/table captions | out of scope |
| 1458 | Stage 2 results intro | **"Turnover, cumulative transaction costs, and trade counts are not yet reported as separate summary statistics"** — now **factually outdated**: Phase 1A's Table 2 / Fig. 4 report exactly these |
| 1472, 1479, 1493 | Stage 2 fig/table captions + prose | "Representative Stage 2 evaluation…"; "Legacy representative single-seed illustration…"; non-archival note |
| 1559 (Limitations) | Discussion | "based on representative single-seed runs…currently in progress" — resolve for common-cov/Stage 2, keep for regcov |
| 1579 (Conclusion) | Conclusion | same mixed in-scope/out-of-scope pattern as line 112 |

## 8. Existing LaTeX table ↔ new aggregate table correspondence

| Existing table | Replace with / draw from | Notes |
|---|---|---|
| `tab:common_metrics` (Stage 1, EW/MinVar/RL) | `aggregate/tables/table1_stage1_robustness.csv` (+ `.tex`) | Direct column match: mean/std/target-hit-prob/target-MSD/shortfall, now 5-seed mean + 95% CI instead of one number |
| `tab:stage2_means` (Stage 2, 4 methods) | `aggregate/tables/table2_stage2_cost_aware_performance.csv` (+ `.tex`) | Old table had only mean net wealth; new table adds gross wealth, cumulative tc, turnover, trades, band width, target-hit-prob, all with 95% CI |
| *(no existing equivalent)* | `aggregate/tables/table3_paired_comparisons.csv` (+ `.tex`) | New: Stage2-vs-Stage1, gross-vs-net, and method-vs-method paired differences with CIs — supports/replaces the prose deltas at line 1413 and 1495 with defensible statistics |
| `tab:stage2_config` (reproducibility status) | `aggregate/run_manifest.csv` + `aggregation_config.json` | Old table documents what's *missing*; new run has it all (checkpoint hashes, seeds, git commit, eval config) — needs a rewritten table or companion table, not a mechanical CSV swap |
| `tab:configs`, `tab:common_parameters` | *(no change expected)* | Hyperparameters/estimated-params match Phase 1A exactly (verified programmatically in `validate()`) — these are config tables, not per-seed results |

## 9. Figure replacement candidates

| Existing figure | Candidate replacement | Status |
|---|---|---|
| `fig:stage1_common`, panel (c) terminal hist | `aggregate/figures/fig2_terminal_wealth_distribution.png/.pdf` (Stage 1 panel: RL/EW/MinVar violin, pooled 5×1000) | ready |
| *(no existing equivalent)* | `aggregate/figures/fig1_mean_vs_std.png/.pdf` | new — shows the 5 individual seeds + mean±CI, directly visualizes the "not a single seed" point |
| *(no existing equivalent)* | `aggregate/figures/fig5_training_seed_variability.png/.pdf` | new — bar chart of mean terminal wealth by seed, Stage 1 vs Stage 2 side by side |
| `fig:stage2_eval`, panel (b) terminal hist | `aggregate/figures/fig2_terminal_wealth_distribution.png/.pdf` (Stage 2 panel: 4-method violin, pooled 5×1000) | ready |
| `fig:stage2_eval`, panel (a) avg wealth **path over time** | **no replacement available** | blocked — see `aggregate/figures/fig3_average_wealth_path_NOTE.txt`: no per-timestep trajectory data was ever persisted by the eval scripts; would need new evaluation code + a re-run, out of scope for a read-only aggregation |
| *(no existing equivalent)* | `aggregate/figures/fig4_cost_turnover.png/.pdf` | new — cumulative tc / turnover / trades / band width, 4-method bar comparison with CI, directly fills the "not yet reported as separate summary statistics" gap at line 1458 |
| `fig:stage1_common` panel (a), `fig:stage2_training` (learning/loss curves) | **no direct replacement** | `aggregate_phase1a.py` only pulls the *final* training iteration into `training_diagnostics.csv`; a multi-seed overlay of full training curves would need a new plotting pass over `stage1/seed_*/metrics.csv` and `stage2/seed_*/metrics.csv` (data exists, script does not — not built in this inventory pass) |

## Candidate files to change (when the revision itself is authorized)

- `ICA report/ICA_report.tex` — the only source file; all edits land here.
- New figure assets need to be placed somewhere `\imgorplaceholder`/
  `\includegraphics` can find them (e.g. copy the relevant
  `aggregate/figures/*.png` into a new `ICA report/file/phase1a/` folder, or
  point the macro calls at the aggregate path directly) — not done yet.
- **Not** `aggregate_phase1a.py` or any file under `results/…` — those are
  the frozen, already-validated data source for the revision, not something
  this revision should modify.

## Summary for this pass

- Baseline branch `ica2026-report-phase1a-results` created, baseline
  (unmodified) 30-page PDF + source committed at `7876d66`.
- Nine in-scope single-seed/representative-run locations identified across
  the Numerical Study and Discussion/Conclusion sections; two out-of-scope
  (regcov) tables/figures identified and excluded.
- Three ICA-report-ready tables and four ICA-report-ready figures are
  available now; two figure types (time-series wealth path, multi-seed
  training-curve overlay) are **not** available without new code/data and
  are flagged as gaps rather than silently skipped.
- One factual discrepancy flagged for the revision spec to resolve: the
  legacy Stage 2 text says 500 evaluation paths; Phase 1A used 1000.

Awaiting the academic revision specification before any edits to
`ICA_report.tex`.
