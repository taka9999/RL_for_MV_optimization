# Phase 1A LaTeX Revision Report

Branch: `ica2026-report-phase1a-results`. Base commit before this revision:
`1655e81` (inventory). This report documents the revision of `ICA_report.tex`
to incorporate the Phase 1A five-training-seed common-covariance results, per
the academic revision specification (sections A–O of the request).

## Changed sections

| Section (spec ref) | LaTeX location | Nature of change |
|---|---|---|
| Title page (C) | titlepage | Removed "Repeated-seed robustness analysis is in progress"; updated draft date and status line |
| Abstract (D) | `\begin{abstract}...\end{abstract}` | Fully replaced with the supplied 5-seed abstract text |
| Introduction (E1–E4) | Intro paragraph, contributions (ii)/(iii), new empirical-validation paragraph | Ablation-vs-robustness distinction; regcov single-seed framing; Stage 2 5-seed summary; new paragraph before "Actuarial relevance" |
| Experimental design (F1–F4) | `\subsection{Experimental design}` | 500→1000 paths (legacy 500 remains only in the superseded-result sentence); leverage-cap vs. Stage-1 action-bound distinction; `tab:stage2_config` fully rewritten from config files; robustness subsection renamed and rewritten |
| Stage 1 results (G1–G3) | `\subsection{Stage 1 results}`, `\label{sec:stage1_results}` | `tab:common_metrics` replaced (label kept) with 5-seed table; `fig:stage1_common` replaced (label kept) with Fig. 1 + Fig. 5; prose rewritten with 5-seed numbers, old single-seed numbers removed |
| Regime-dependent covariance (H) | new `\subsubsection{Exploratory regime-dependent-covariance experiment}`, `\label{sec:regcov_results}` | Scope disclaimer added; `tab:regcov_metrics` values unchanged; `fig:stage1_regcov` removed (no retained image) rather than shown as placeholder |
| Stage 2 results (I1–I5) | `\subsection{Stage 2 results}` | Legacy `tab:stage2_means` numbers removed from the main comparison and mentioned once (superseded); new 5-seed `tab:stage2_means` (label kept); new `tab:paired_comparisons`; `fig:stage2_eval` (label kept) repointed to the terminal-distribution figure; new `fig:stage2_cost_turnover`; old training-diagnostic figure (`fig:stage2_training`) removed, no retained image |
| Discussion (J1–J4) | Practical implications, Lessons, Sources of performance, Limitations | Trade-count-reduction point added; Stage 1/Stage 2 stability language with explicit non-causal framing and the ~70x std ratio; turnover/cost claim updated to reflect Table 2/Fig. 4; Limitations paragraph replaced with the 11-point list |
| Conclusion (K) | `\section{Conclusion}` | Numerical-results paragraph replaced with the supplied 5-seed text; regcov/QVI/ALM framing retained |
| New Appendix (I3) | `\section{Complete Phase~1A Paired Comparisons}`, `\label{app:paired_full}` | Full 11-row paired-comparison table |
| Macro fix (L) | `\imgorplaceholder` (line 58) | `\texttt{#3}` → `\texttt{\detokenize{#3}}` to prevent the underscore-escaping compile failure on any future missing figure |

Also updated for terminology consistency (Section B): "Center+DNNBand, the
implemented Direct Boundary RL policy" defined at its first occurrence
(Introduction, contribution (iii) lead-in); used thereafter in all
numerical-results-facing prose (Numerical Study, Discussion, Conclusion,
positioning table). "Direct Boundary RL" was intentionally left as-is only in
the Stage 2 *methodology/derivation* sections (Sections 7.x on QVI design,
the algorithm pseudocode, and the appendix QVI critic derivation) where the
text is naming the algorithm class rather than reporting the evaluated
policy's results.

## Inserted tables and figures — source mapping

| LaTeX label | Source file | Rows/content used |
|---|---|---|
| `tab:common_metrics` | `aggregate/tables/table1_stage1_robustness.csv` | All 3 rows (RL/EW/MinVar), all columns |
| `tab:stage2_config` | `stage2/seed_*/run_config.json`, `eval_stage2/seed_*/eval_config.json`, `aggregation_config.json`, `run_manifest.csv` (read directly, not via a single aggregate CSV) | Config values only, no per-seed results |
| `tab:stage2_means` | `aggregate/tables/table2_stage2_cost_aware_performance.csv` | All 4 rows, all columns |
| `tab:paired_comparisons` | `aggregate/tables/table3_paired_comparisons.csv` | 4 of 11 rows (rows 0, 5, 6, 7) |
| `tab:paired_full` (Appendix) | `aggregate/tables/table3_paired_comparisons.csv` | All 11 rows |
| `fig:stage1_common` | `aggregate/figures/fig1_mean_vs_std.png`, `fig5_training_seed_variability.png` (copied to `ICA report/file/phase1a/`) | full images, unmodified |
| `fig:stage2_eval` | `aggregate/figures/fig2_terminal_wealth_distribution.png` | full image (both stage panels), unmodified |
| `fig:stage2_cost_turnover` | `aggregate/figures/fig4_cost_turnover.png` | full image, unmodified |

## All numerical values and their source rows

Cross-checked programmatically against the CSVs immediately before finalizing
(see session transcript): every number appearing in `tab:common_metrics`,
`tab:stage2_means`, `tab:paired_comparisons`/`tab:paired_full`, and the
prose sentences quoting them (mean terminal wealth, intervals, target MSD,
trade counts, the ~70x cross-seed std ratio) matches
`aggregate/tables/table{1,2,3}_*.csv` and `aggregate/aggregate_results.csv`
to the rounding precision displayed (4 decimal places, or as specified in
the revision spec). No value was hand-derived outside these files except:
(a) simple arithmetic already shown in-line (e.g. $1.1811-1.0530=0.1281$),
and (b) the trade-count reduction percentage ($1-175.8/252\approx30\%$).

## Removed legacy results

- `tab:common_metrics`: EW $1.070/0.272/0.074/0.309/0.187$, MinVar
  $1.071/0.261/0.068/0.311/0.182$, "HJB-guided RL" $1.197/0.266/0.071/0.611/0.115$
  — removed, replaced by the 5-seed table.
- Prose: "$+0.126$", "nearly doubles the target-hit probability" — removed.
- `tab:stage2_means`: Direct Boundary RL $1.248$, CenterOnly daily $1.136$,
  CenterOnly monthly $1.147$, EW monthly $1.052$, "$500$ independently
  generated evaluation paths" — removed from the primary comparison. Referenced
  exactly once, in a single sentence explaining supersession
  (Stage 2 results, opening paragraph), per the revision spec's allowance for
  a one-sentence footnote/provenance mention.
- Prose: "$0.101$ above monthly … $0.112$ above daily" — removed, replaced by
  the 5-seed paired-comparison numbers ($+0.0780$/$+0.0910$).
- `fig:stage2_training` (learning/loss curve, single-seed) — removed entirely;
  no retained image file existed to relabel as illustrative.
- `fig:stage1_common` panels (a) training diagnostics and (b) average wealth,
  and `fig:stage1_regcov` (entire figure) — removed entirely for the same
  reason (no retained image files).

## Retained exploratory regime-dependent-covariance results

`tab:regcov_metrics` (EW $1.078/0.271/\dots$, MinVar $1.078/0.259/\dots$,
"Approximate HJB RL" $1.132/0.172/\dots$) and its interpretive prose are
**unchanged in value**, now isolated in
`\subsubsection{Exploratory regime-dependent-covariance experiment}`
(`\label{sec:regcov_results}`) with an explicit scope disclaimer stating it is
not part of the five-seed Phase 1A replication. `tab:regcov_parameters` is
also unchanged. The corresponding figure (`fig:stage1_regcov`) was removed
(no retained image) rather than left as a placeholder.

## Compile command

```bash
cd "ICA report"
export TEXINPUTS=".:<scratch>/texlocal:"   # see note below
pdflatex -interaction=nonstopmode -halt-on-error ICA_report.tex   # pass 1
pdflatex -interaction=nonstopmode -halt-on-error ICA_report.tex   # pass 2
```

`TEXINPUTS` addendum: this machine's TeX Live install (`texlive/2025basic`) is
missing `enumitem.sty`, `algorithm.sty`, `algorithmic.sty`, and
`algorithmicx`/`algpseudocode.sty`; `tlmgr` cannot install them here (no
write permission, and a TeX Live 2025-vs-2026 CTAN version mismatch blocks
user-mode install). Worked around by fetching the four packages from CTAN
into a scratch directory and adding it to `TEXINPUTS` for this compile only —
no change to the repository or to system TeX state. A TeX install with these
four standard packages present will compile `ICA_report.tex` directly with no
workaround.

## Compile warnings/errors

- Pass 1 and pass 2: **exit code 0**, no errors.
- 1 trivial `Overfull \hbox (1.35977pt too wide)` at lines 1395–1404 (Table 1
  region) — sub-millimeter, not visually detectable, not fixed.
- The two larger overfull boxes found during editing (Table 2 at ~27pt, a
  Limitations bullet containing a long `\texttt{...}` identifier at ~37pt)
  were fixed (Table 2 wrapped in `\resizebox{\textwidth}{!}{...}`; the
  `\texttt{share_paths_avg_leverage_near_cap}` identifier given `\allowbreak`
  points) and no longer appear.
- 1 `undefined references` warning found on the first post-edit compile
  (`fig:stage1_regcov`, from a leftover lead-in sentence not caught by the
  original edit) — fixed by removing the now-orphaned sentence; 0 undefined
  references on the final compile.
- 0 "multiply defined labels" warnings.

## Final page count

**32 pages** (via `pypdf`), up from the 30-page baseline. Within the
requested 32–36 page target and the ICA 40-page limit.

## Unresolved missing figures

**None.** Every `\includegraphics`/`\imgorplaceholder` call in the final
document points to a file that exists (`file/phase1a/fig{1,2,4,5}_*.png`).
Figures without a retained source image (`fig:stage1_regcov`,
`fig:stage2_training`, and two of `fig:stage1_common`'s three original
panels) were removed rather than left as placeholders, per the revision
spec.

## Old-value grep results (final state)

Checked all terms listed in the revision spec (Section N):

| Term | Occurrences in final `.tex` |
|---|---|
| `representative single-seed` | 0 |
| `replication is in progress` | 0 |
| `1.248`, `1.136`, `1.147`, `1.052` | 1 each, all in the single sentence explaining supersession of the legacy Stage 2 result (Stage 2 results, opening paragraph) — intentional, not leftover |
| `1.197` | 0 |
| `0.126` | 0 |
| `nearly doubles` | 0 |
| `not yet reported as separate summary statistics` | 0 |
| `do not yet provide a full turnover` | 0 |
| `Repeated-seed robustness analysis is in progress` | 0 |
| `500` (bare, evaluation-path count) | 0 outside the same one legacy sentence |
| `cryptographic` | 0 (replaced with "byte-identical … consistency check" per spec) |

Remaining occurrences of the word "representative"/"legacy" (checked
individually) are all either (a) inside the isolated regime-dependent-
covariance subsubsection, explicitly marked single-seed/exploratory and out
of Phase 1A scope, or (b) unrelated uses of "representative" in the SJM
math derivation (e.g. "regime representatives $\vartheta_1,\vartheta_2$")
and the Stage 1 actor-signal paragraph, which do not refer to single-seed
result framing.

## Git diff summary

```
ICA report/ICA_report.tex | 270 ++++++++++++++++++++++++++--------------------
1 file changed, 152 insertions(+), 118 deletions(-)
```
(relative to baseline commit `7876d66`). Additional new files in this
commit: `ICA_report.pdf` (recompiled), `file/phase1a/fig{1,2,4,5}_*.png`
(4 new figure assets copied read-only from the Phase 1A aggregate output),
and this report.

## Commit hash

See the commit immediately following this file in the branch history
(reported in the final chat summary after this file was committed).
