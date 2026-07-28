# ICA Report — Source Inventory

Read-only inventory only. No content, numbers, tables, or figures in
`ICA_report.tex` were changed while producing this document. Branch:
`ica2026-report-phase1a-results`, baseline commit `7876d66`.

## 1. Main file and included files

- Single self-contained file: `ICA report/ICA_report.tex` (1787 lines).
- **No** `\input{...}` or `\include{...}` anywhere — everything (title,
  abstract, all sections, appendices, bibliography) lives in this one file.
- No separate `.cls` file: `\documentclass[a4paper,10pt]{article}` (standard
  LaTeX `article` class).

## 2. Bibliography, class, and style files

- No `.bib` file. Bibliography is inline via `\begin{thebibliography}{99}`
  … `\end{thebibliography}` at lines 1688–1786 (manual `\bibitem` entries,
  not BibTeX/biblatex).
- No custom `.cls`/`.sty` file. Packages loaded (lines 3–17):
  `geometry, amsmath, amssymb, amsthm, mathtools, bm, graphicx, booktabs,
  array, enumitem, algorithm, algpseudocode, caption, subcaption, float,
  microtype, hyperref, xcolor, tikz`.
- **Local compile-environment finding (not a document-content issue):** this
  machine's TeX Live install (`texlive/2025basic`) is missing `enumitem`,
  `algorithm`, `algorithmic`, `algorithmicx`/`algpseudocode` by default, and
  `tlmgr` cannot install them here (no write permission to the system tree,
  and user-mode install is blocked by a TeX Live 2025-vs-2026 CTAN mismatch).
  Worked around for this inventory only by fetching the four `.sty` files
  from CTAN into a scratch `TEXINPUTS` directory — no change to the repo or
  to any system state.

## 3. Current compiled PDF

- `ICA report/Draft-Structure-Guided Reinforcement Learning for Partially
  Observed Multi-Period Portfolio Optimization with Transaction Costs.pdf`
- **30 pages** (via `pypdf`).
- Custom figure-inclusion macro (lines 56–59):
  ```
  \newcommand{\imgorplaceholder}[3]{%
    \IfFileExists{#3}{\includegraphics[...]{#3}}{%
      \fbox{\parbox[c][#2][c]{#1}{\centering Missing figure\\[0.3em]\texttt{#3}}}}%
  }
  ```
  All 10 figure files it references (see §4) are **absent** from this
  checkout (`ICA report/` has no `file/` subdirectory and no loose PNGs
  next to the `.tex`). The existing 30-page PDF must therefore have been
  compiled elsewhere (e.g. Overleaf, or a local copy with the images
  present) — not reproducible from what's in this repo checkout as-is.

### Baseline compile result (this session)

- Direct `pdflatex` on the untouched source: **fails** (`enumitem.sty` not
  found — environment issue above, not a document bug).
- After supplying the 4 missing packages via `TEXINPUTS`: **fails again**,
  now with `! Missing $ inserted.` at `\imgorplaceholder{...}{file/learning_
  curves_comcov.png}` (line 1380). Root cause: since the image is missing,
  `\IfFileExists` takes the FALSE branch and typesets the raw filename via
  `\texttt{#3}` — the filename contains `_` (e.g.
  `learning_curves_comcov.png`), and plain LaTeX treats an unescaped `_` in
  text mode as an error (needs `\_` or the `underscore` package). This is a
  **latent, pre-existing bug** in the placeholder-fallback branch of
  `\imgorplaceholder` — it only ever triggers when a referenced figure file
  is genuinely absent, which is exactly the state of this checkout.
- Diagnostic-only workaround (scratch directory, not committed): supplied
  ten 1×1-pixel dummy PNGs at the exact referenced paths so `\IfFileExists`
  takes the TRUE branch and the placeholder bug is never reached. Result:
  **compiles cleanly, 2-pass, 0 warnings, 31 pages.**
- Interpretation: the 1-page difference from the checked-in PDF (30 pages)
  is expected/benign — real figures have different aspect ratios than 1×1
  dummies and shift page breaks slightly. The document's LaTeX mechanics are
  otherwise sound; the only blockers are (a) the missing image files
  themselves and (b) the `\texttt{#3}` underscore bug that only surfaces
  because of (a).
- **Baseline PDF committed to this branch is the existing, already-compiled
  30-page PDF, unmodified** (see commit `7876d66`) — not a fresh local
  recompile, per the reasons above. The diagnostic 31-page recompile (with
  dummy images) exists only in the session scratchpad; it is not part of
  the repo since it does not contain real figures.

## 4. Existing tables/figures used in "Numerical Study" (`\section{Numerical
Study}`, lines 1264–1513)

| Label | Kind | Content | Line |
|---|---|---|---|
| `tab:configs` | table | Stage 1 config (common-cov vs regime-dep-cov) | 1277 |
| `tab:stage2_config` | table | Stage 2 config + reproducibility status of the **legacy representative run** | 1299 |
| `tab:common_parameters` | table | True vs estimated params, common-covariance | 1325 |
| `tab:regcov_parameters` | table | True vs SJM-estimated params, regime-dep-cov | 1345 |
| `fig:stage1_common` | figure | 3 subfigs: learning curve / avg wealth / terminal hist, **common-cov, single seed** | 1376 |
| `tab:common_metrics` | table | EW/MinVar/RL summary stats, **common-cov, single seed** | 1397 |
| `fig:stage1_regcov` | figure | Same 3 subfigs, **regime-dep-cov, single seed** | 1417 |
| `tab:regcov_metrics` | table | EW/MinVar/RL summary stats, **regime-dep-cov, single seed** | 1438 |
| `fig:stage2_eval` | figure | avg wealth path + terminal hist, **Stage 2, single seed** | 1460 |
| `tab:stage2_means` | table | 4-method net terminal wealth, **Stage 2 legacy single seed** | 1476 |
| `fig:stage2_training` | figure | Stage 2 training diagnostics (learning/loss curve), **single seed** | 1499 |

**Scope note:** Phase 1A (the 5-seed aggregate this revision is based on)
only covers the **common-covariance** track. Its `run_config.json`
`policy_params`/`true_params` match `tab:common_parameters` exactly (checked
programmatically). The **regime-dependent-covariance** track
(`tab:regcov_metrics`, `tab:regcov_parameters`, `fig:stage1_regcov`) is a
**separate, not-yet-aggregated experiment** (matches the in-progress,
untracked `poemv_rs/*_qvi_commoncov.py` files) — **out of scope** for this
revision.

## 7. Phase 1A aggregate deliverables (input for the revision)

All under `results/ica2026_additional_runs/phase1a_endtoend_20260725/aggregate/`,
produced by `aggregate_phase1a.py` (commit `aaed20a`), read-only, deterministic:

| File | Rows | Notes |
|---|---|---|
| `seed_level_results.csv` | 35 (5 seeds × 7 methods) | per-seed, per-method stats |
| `aggregate_results.csv` | 462 (stage×method×metric) | cross-seed mean/std/SE/95%CI |
| `paired_comparisons.csv` | 11 | episode-level paired diffs, seed-aggregated |
| `episode_level_combined.parquet` | 35,000 | full per-path data, all seeds/methods |
| `training_diagnostics.csv` | 10 | avg_abs_action etc., training-time only |
| `run_manifest.csv` | 12 | seed/stage/checkpoint hash/status/exclusion |
| `aggregation_config.json` | — | exact config, thresholds, validation info |
| `DATA_QUALITY_REPORT.md` | — | validation results, gaps, exclusions |
| `PHASE1A_RESULTS_SUMMARY.md` | — | headline 5-seed numbers, CIs, paired comps |
| `tables/table{1,2,3}_*.csv` + `.tex` | — | ICA-report-ready tables, CSV+LaTeX |
| `figures/fig{1,2,4,5}_*.{pdf,png}` + `_data.csv` | — | ICA-report-ready figures + source data |
| `figures/fig3_average_wealth_path_NOTE.txt` | — | documents why Fig. 3 (time-series wealth path) could not be produced |

## 10. Current page count

**30 pages** (checked-in PDF, `pypdf`). See §3 for the compile-diagnostic
31-page figure (dummy images, not representative of final page count once
real figures are inserted).
