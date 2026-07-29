# Final Technical QA Report

Branch: `ica2026-report-phase1a-results`. Base commit before this QA pass:
`d50af92` (Phase 1A 5-seed revision). This report documents a technical QA
pass that checked the Phase 1A revision against the actual implementation
code, run configs, and manifests (`poemv_rs/stage2_direct_boundary.py`,
`poemv_rs/stage2_eval.py`, `poemv_rs/eval_compare.py`,
`stage2/seed_*/run_config.json`, `eval_stage2/seed_*/eval_config.json`,
`aggregation_config.json`, `run_manifest.csv`), and corrected the LaTeX
source and figure-generation script wherever the reported specification
disagreed with what was actually run. **No new training or evaluation was
executed; all corrections use already-archived checkpoints, configs, and
episode-level outputs.**

## 1. Stage 2 leverage cap

**Fact confirmed by reading `poemv_rs/stage2_direct_boundary.py` in full**:
no `--lev_cap` argument exists in the training script at all, and no
gross-leverage-norm projection (`Π_W`, hard `‖w‖₁ ≤ L_max`) appears anywhere
in its code. The only leverage control active during training is the soft
L1 regularizer `gross_lev_coef * gross_lev_pen` (`gross_lev_coef=1e-5`),
computed on the post-band-projection target `w_tgt_t` with no further cap
applied (`stage2_direct_boundary.py:344,362,375`). The wealth recursion used
for backprop (`stage2_direct_boundary.py:328-341`) sets
`w_tgt_t = min(max(w_cur_t, lower_t), upper_t)` and uses it directly as
`w_k^+` -- no leverage-norm step follows.

The hard cap **is** applied, identically, to every Stage 2 comparator during
**evaluation only**, via `apply_leverage_cap_to_weights()`
(`poemv_rs/eval_compare.py:220-227`, imported into `stage2_eval.py`), called
for Center+DNNBand, CenterOnly_Daily, CenterOnly_Monthly, and EW_Monthly
alike (`stage2_eval.py:474-475,807-820` and the static-comparator calls at
`stage2_eval.py:761-764`). The function is a proportional (radial) rescaling
`w -> w * (L_max/‖w‖₁)` when `‖w‖₁ > L_max`, not a generic unspecified
projection. Empirically confirmed against `eval_stage2/seed_0/eval_terminal_by_path.csv`:
`CenterOnly_Daily`'s per-step gross leverage reaches exactly `1.500000`
(the cap binding), `Center+DNNBand`'s stays at or below `1.487` (cap not
binding for this policy), consistent with a real, uniformly-applied cap.

**Changes made**: Section "From a frictionless center to an inaction
region" (`eq:projection`) now defines only the componentwise band
projection, states explicitly that `w_k^+ = w_k^band` is what training
differentiates through, and introduces a new labeled equation
(`eq:eval_projection`) for the hard cap, explicitly marked evaluation-only
and identical across all four comparators. Algorithm 2's generic "apply the
projection rule" step now correctly refers only to the band step (nothing
in it was claiming the leverage cap; the ambiguity was resolved by fixing
`eq:projection` itself). `tab:stage2_config` (Table 6) already stated the
evaluation-only status correctly before this QA pass and needed no change
to that specific row.

## 2. Stage 2 objective, exactly as implemented

All facts below were read directly from
`poemv_rs/stage2_direct_boundary.py` (`_utility_torch`,
`simulate_episode_direct_boundary`, `train_direct_boundary`,
`_run_validation`) and cross-checked against `stage2/seed_0/run_config.json`.

| Item | Fact |
|---|---|
| Exact utility $U(x)$ | `utility_kind="log"` for all 5 seeds: $U(x)=\log(\max(x,\epsilon))$, $\epsilon=10^{-10}$ |
| `utility_gamma` (=2.0) | Passed as the `gamma` argument to `_utility_torch`; used only by the unused "power"/CRRA branch. The "log" branch ignores it entirely. |
| `gamma_risk` (=5.0) | Used only inside `_qvi_base_width()`, itself only called when `gap_mode="qvi_exp"`. Phase 1A's `gap_mode="direct"` never reaches that branch, so `gamma_risk` had **zero** effect on any Phase 1A training run. |
| "Risk-aversion parameter" naming | **Not accurate** for this configuration -- neither `utility_gamma` nor `gamma_risk` multiplies or exponentiates the utility actually used. Corrected in the text (was previously drafted, in an earlier session, as "$\gamma_{\mathrm{risk}}=5$" attached to "Log utility" in Table 6 -- this was itself a QA-session error caught and fixed before finalizing, see the git diff). |
| $\lambda_U$ | Code field `utility_scale` (=1.0 for Phase 1A); the objective also has an additive shift `utility_shift` (=0.0 for Phase 1A), not previously shown in `eq:band_loss`. |
| Positivity floor | $\epsilon=10^{-10}$, applied via `torch.clamp(w_t, min=eps)` directly inside the utility evaluation. |
| Bankruptcy penalty | **None** -- there is no separate term; the floor inside $U(x)$ is the only nonpositive-wealth handling. |
| Turnover regularizer normalization | Per-step $\|w_k^+-w_k^-\|_1$ (`simulate_episode_direct_boundary` line 343), **time-averaged** over $N=252$ steps per episode (`turnover_pen = mean over k`), then averaged over the `episodes_per_iter=128` simulated episodes per training iteration (`loss = mean over episodes`). |
| Gap-L2 regularizer normalization | Per step, `lower_gap.pow(2).mean() + upper_gap.pow(2).mean()` -- an **asset-average** (mean over the 2 risky assets, i.e. divided by 2), not a sum; then time-averaged over $N$ steps, then episode-averaged. The paper's previous formula ($\|\cdot\|_2^2$, i.e. an un-normalized sum over assets) was corrected to show the explicit $\tfrac12$ factor. |
| Gross-leverage regularizer normalization | Per step, $\|w_k^+\|_1$ (true sum over both assets, matching the paper's original formula, no correction needed there), time-averaged over $N$ steps, episode-averaged. |
| Validation objective / checkpoint selection | `_run_validation()` computes `val_utility = mean(utility)` over `val_n_paths=64` held-out paths every `val_every=25` iterations; `best_checkpoint.pt` is saved whenever `val_utility > best_val_utility` (`train_direct_boundary`, line 563) -- i.e. **highest mean validation log-utility**, not lowest validation loss. |

**Changes made**: `eq:band_loss` rewritten with explicit per-step/per-episode
normalization for all three regularizers, the $\tfrac12$ asset-average
factor added to $\mathcal L_g$, $\lambda_U$/$s_U$ mapped to their code
names, and a new paragraph gives the exact utility, floor, checkpoint
criterion, and the corrected (non-)role of `utility_gamma`/`gamma_risk`.
Table 6 (`tab:stage2_config`) updated to match.

## 3. Evaluation metric definitions

New metric-definition paragraph added to the Numerical Study section
(before the mean-shortfall equation), stating exactly (all read from
`poemv_rs/stage2_eval.py`):
- **Turnover** ($0.0397$ for Center+DNNBand): per-step time average over the
  $252$ steps of a path (`stage2_eval.py:480,852`: `mean(sim["turnover"])`
  per path, then `.mean()` over paths in `eval_summary.csv`), further
  averaged over the 5 seeds by this session's aggregation. **Not** a
  cumulative sum, **not** trade-conditioned.
- **Number of trades**: `sum(turnover > 1e-9)` (`stage2_eval.py:487`) -- the
  exact trade threshold is $10^{-9}$ in weight-space turnover units.
- **Mean band width** ($1.740$ for Center+DNNBand): per-step, the two-asset
  *average* (not sum) of `upper-lower` (`stage2_eval.py:330`:
  `np.mean(upper-lower)`), then time-averaged over the $252$ steps
  (`stage2_eval.py:836-837`: `np.mean(sim["band_width"])`), then
  path-averaged, then seed-averaged.
- **Target achievement probability**: not a native `eval_summary.csv` column
  for Stage 2 -- computed by this session's aggregation as the empirical
  fraction of paths with net (Stage 2) / plain (Stage 1) terminal wealth
  $\ge z$.

## 4. Gross terminal wealth

**Fact confirmed by reading `stage2_eval.py:483-485,832-846`**:
`wealth_gross[k+1] = wealth_gross[k] + dot(current_u, disc_ret)` -- a
parallel ledger sharing the *identical* realized post-trade dollar
positions `current_u` (and hence identical realized per-step P&L) as net
wealth, but never debiting the transaction cost `tc`. `cumulative_tc` is
then *defined* as `gross_terminal_wealth - net_terminal_wealth`
(`stage2_eval.py:846`) -- an accounting identity by construction, not an
independently-tracked series compared post hoc. This is **not** a
counterfactual no-cost re-simulation (trade sizes are not rescaled for a
hypothetical larger fee-free capital base) and does **not** credit the
foregone investment return the paid transaction-cost dollars would have
earned. The near-$10^{-16}$ "invariant" reported in Table 6/the
DATA_QUALITY_REPORT is therefore correctly understood as a check on
aggregation arithmetic, not an independent physical validation -- the text
was revised to state this precisely.

**Changes made**: metric-definition paragraph, `tab:stage2_means` caption,
and a new paragraph immediately following that table all state this
explicitly, per the exact "not a counterfactual re-simulation" / "does not
include foregone investment returns" language requested.

## 5. Common evaluation path wording

"if and only if the same underlying evaluation paths were used" removed.
Confirmed (via `poemv_rs/eval_compare.py:generate_test_path` and
`stage2_eval.py`) that **no scenario hash or per-path seed identifier is
persisted** anywhere in the archived outputs -- only the integer path index
$0$--$999$ -- so there is no stronger verification artifact available than
the byte-identical-comparator check already used. Replaced with the exact
requested sentence: "Byte-identical deterministic benchmark outputs across
all seeds provide a strong consistency check that the same evaluation
scenarios were used," with additional context on why no stronger artifact
exists.

## 6. Future-tense wording

Found and fixed one exact match: "Those items must be recorded in the
frozen final manifest. ... the final implementation record will specify
..." (SJM estimation-sample reproducibility paragraph, Section "Statistical
jump model"). Rewritten in past/present tense as an explicit reproducibility
**limitation** of the single-seed exploratory regime-dependent-covariance
experiment specifically (contrasted with the fully-archived common-covariance
pipeline), per the instruction, rather than a future promise. The other two
target phrases ("The frozen implementation used for the final tables will
follow ...", "The final manifest will therefore record ...") were part of
the *same* paragraph (Stage 2 wealth-recursion timing convention) and were
already corrected in the prior revision session; re-verified clean by grep
in this pass (0 occurrences).

## 7. Utility-consistent Stage 2 evaluation

Computed directly from already-archived episode-level terminal wealth
(`eval_stage2/seed_*/eval_terminal_by_path.csv` net_terminal_wealth column
and `eval_stage1(_postfix)/seed_*/eval_terminal_by_path.csv` terminal_wealth
for Stage 1 RL), no re-simulation:
`mean_terminal_utility = mean(log(max(x_T, 1e-10)))`,
`log_utility_certainty_equivalent = exp(mean_terminal_utility)`, using the
exact positivity floor from `_utility_torch`. Computed and cross-seed
aggregated (mean, sample std ddof=1, SE, 95% Student-$t$ interval df=4) for
Stage 1 RL and all four Stage 2 comparators. New script function
`utility_consistent_metrics()` in `aggregate_phase1a.py`; output
`aggregate/utility_consistent_metrics.csv`; folded into
`seed_level_results.csv`/`aggregate_results.csv` via the existing
cross-seed-stats machinery (added to `METRIC_COLS`). Not computed for
Stage 1 EW/MinVar (out of the requested scope: those are not evaluated
under the terminal-utility objective) -- reported as "--" in the appendix
table rather than fabricated.

## 8. Full metric intervals

New `make_table4_full_intervals()` function and two new appendix tables
(`tab:full_intervals`, `tab:full_intervals_cost`) report 95% Student-$t$
training-seed intervals for target achievement probability, target MSD,
mean shortfall, cumulative transaction cost, turnover, trade count, band
width, and the new utility-consistent metric, for all 3 Stage 1 and 4
Stage 2 methods. Table 1 (`tab:common_metrics`) and Table 2
(`tab:stage2_means`) captions both now state explicitly that every entry is
a 5-seed mean of an already within-seed-averaged per-seed statistic.

## 9. Figure revision (script-only, no re-run)

`aggregate_phase1a.py`'s `fig2_terminal_wealth_distribution()` and
`fig4_cost_turnover()` modified: short method labels
(`STAGE2_METHOD_SHORT`/`STAGE1_METHOD_SHORT`, e.g. "Center-Daily" instead of
"CenterOnly_Daily"), a `z=1.2` target reference line added to both violin
panels of Fig. 2, an explicit "descriptive visualization ... not treated as
5000 independent observations for inference" line added to Fig. 2's title,
and Fig. 1/Fig. 4's error-bar legend/suptitle text changed from generic
"95% CI" to "95% Student-$t$ training-seed interval". Regenerated by
re-running `aggregate_phase1a.py` against the same already-validated
`--base-dir` (no new training or evaluation); new PNGs/PDFs copied into
`ICA report/file/phase1a/`, overwriting only the 4 files already introduced
in the prior revision.

## 10. Wording refinement

- The Discussion's Stage 1-vs-Stage 2 standard-deviation comparison
  (~$0.0163$ vs ~$0.000232$, "$70\times$") now explicitly states this is
  the between-seed dispersion of the **end-to-end Stage 1-plus-Stage 2
  pipeline** (each seed's Stage 2 policy trains against that same seed's
  own Stage 1 checkpoint, so the two stages' initializations are not varied
  independently), not an isolated measurement of "the Stage 2 optimizer
  alone" with a fixed Stage 1 center.
- All four legacy numeric values ($1.248$/$1.136$/$1.147$/$1.052$) and the
  "$500$ evaluation paths" figure were removed from the Stage 2 results
  opening paragraph entirely (a prior QA round had left them in one
  sentence explaining supersession; this round removes them per the new,
  stricter instruction) and replaced with exactly: "An earlier
  incompletely archived single-seed illustration has been superseded by
  the fully traceable five-seed Phase~1A replication and is not used in
  the present inference."

## 11. Title-page cleanup

**Confirmed by extracting text from the compiled PDF** (not just reading
the source): the conditional `\ICAAuthorName\ifx\ICAAffiliation\empty\else,
\ICAAffiliation\fi` in the copyright block was rendering "Takahiro
Kobayashi," with a stray trailing comma even though `\ICAAffiliation` is
empty -- a well-known LaTeX pitfall (`\ifx` comparison against
`\empty` is not reliable for `\newcommand`-defined empty macros). No
affiliation or email was supplied to add, so per the "delete if
unnecessary" branch of the instruction, the conditional was removed
entirely; the line now reads simply `\ICAAuthorName`. Re-verified via a
fresh PDF text extraction: no trailing comma.

## 12. Validation

- **Two-pass compile**: clean, exit code 0 both passes.
- **Undefined references**: 0.
- **Missing figures**: 0 (all 4 `\includegraphics` calls point to files
  present in `file/phase1a/`).
- **Major overfull boxes**: 0. (One cosmetic, sub-millimeter overfull
  hbox, $1.36$pt, remains in the Table 1 region -- not visually
  detectable, not addressed, same as in the prior revision.) A genuine
  $41$pt overfull box introduced by the new `eq:eval_projection` piecewise
  definition was found and fixed by restructuring it as a two-line
  `align` block.
- **Final page count**: **35 pages** (up from 32 after the prior revision,
  35 after this QA pass's additions: metric-definition paragraph, expanded
  `eq:band_loss` discussion, two new appendix tables). Within the 40-page
  ICA limit.
- **Numerical values**: every new/changed number cross-checked
  programmatically against `aggregate/tables/table4_full_metric_intervals.csv`
  and `aggregate/utility_consistent_metrics.csv` (script-verified match,
  see session transcript); the `gamma_risk`/`utility_gamma` mix-up
  described in Section 2 above was caught during this same verification
  pass and corrected before finalizing.
- **No training/evaluation process started**: confirmed (`ps`/`tmux ls`
  checked immediately before writing this report -- no matching processes,
  no tmux server).
- **Phase 2 not started**: confirmed -- no transaction-cost-sensitivity,
  parameter-comparison, or actor-signal-ablation work was performed.

## Unresolved items

- None of the twelve QA items were left unaddressed. The one residual
  cosmetic overfull box ($1.36$pt) is below the threshold of visual concern
  and was not fixed, consistent with the prior revision's treatment of the
  same warning.
- Section 2's finding that neither `utility_gamma` nor `gamma_risk`
  functions as a risk-aversion parameter for the Phase 1A configuration is
  a substantive finding, not just a wording fix -- it means the paper's
  Stage 2 objective, as actually run, has no explicit risk-aversion control
  beyond the shape of $\log(x)$ itself. This is now stated plainly in the
  text; no further action was taken (out of scope for this QA pass, which
  covers only accurately describing the implemented experiment).
