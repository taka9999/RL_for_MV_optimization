# Final Submission Revision Report

Branch: `ica2026-final-submission-revision`, created from `ed9bf7c` (tip of
`ica2026-report-phase1a-results` at the end of the technical-QA phase). This
document is a repository work-record only; it is never cited from
`ICA_report.tex` and describes the process by which the submission-candidate
revision was produced, not the paper's scientific content itself.

Constraints observed throughout: no new training, evaluation, simulation, or
parameter estimation was run. The only executable actions taken were (a)
read-only inspection of `poemv_rs/*.py` source code and archived
`run_config.json`/manifest files, and (b) editing and rerunning
`aggregate_phase1a.py` (a pure re-aggregation/plotting pass over already
-archived per-seed evaluation CSVs, producing new PNGs from unchanged
underlying numbers) followed by LaTeX edit + recompile.

## 1. Code/config facts newly verified in this pass

| Claim | Verified from | Result |
|---|---|---|
| Stage~1 action bound mechanics | `poemv_rs/agent.py` (`_squash_action_np`, `_squash_log_prob`, `POEMVAgent.act`), `poemv_rs/env.py` (`apply_action_projection` branch and its inline comment) | Dollar-scale, componentwise `tanh` squash: `scale = |x|*a_max`, `u = scale*tanh(z/scale)`, applied identically to the sampled action (training) and the distribution mean (deterministic/eval use in `eval_compare.py`); a Jacobian log-det correction is added to the log-prob/entropy. `apply_action_projection=false` for the five-seed runs, confirmed against `run_config.json`; the alternate hard L1 projection in `env.py` is coded but inert (source comment: "the extra hard projection is disabled by default and only kept as an optional guard"). |
| Filter discretization | `poemv_rs/filtering.py` (`wonham_filter_q_update`) | First-order Euler CTMC prediction (`p_pred = p*(1-lam1*dt) + (1-p)*lam2*dt`) followed by an exact Gaussian-likelihood Bayes correction in log-space. This is the discrete daily prediction-correction implementation; the paper already distinguished it from the continuous-time Wonham filter theorem, and that distinction is retained/tightened. |
| Parameter-estimation provenance | `poemv_rs/train.py` (`make_estimated_params_via_heuristic`, hardcoded `true_params` block) | Fully synthetic: "true" parameters are hardcoded constants in the training script (labeled in-code as "the paper's simulation 'true' parameters"), not fitted to real market data. "Estimated" parameters used by the filter/actor are produced once by simulating 30 years of synthetic prices under those true parameters, labeling the first 20 years by an equal-weight-index drawdown heuristic, and fitting regime-conditional moments/transition rates to the labeled synthetic returns; the resulting fixed numbers were then passed as CLI arguments, identically, to all five training seeds (confirmed identical `est_mu1_json` etc. across all five archived `run_config.json` files). |
| `share_paths_avg_leverage_near_cap` / threshold `1.980` | `aggregate_phase1a.py` (`LEV_NEAR_CAP_THRESHOLD = 1.980`, `per_seed_stage1`/`per_seed_stage2`), `aggregate/seed_level_results.csv` | Computed identically for Stage~1 (uncapped `avg_gross_leverage`) and Stage~2 (`gross_lev`, post evaluation-time 1.5 cap). Archived values: Stage~1 RL ≈0.014–0.015 across seeds; **every** Stage~2 method/seed = exactly 0.000, because Stage~2 gross leverage is bounded above by the 1.5 cap and can never reach the 1.980 threshold. The 1.980 constant has no derivation recorded in code/config linking it to `a_max=1.0` or the 1.5 cap — its specific numeric provenance is not established and is stated as such in the paper rather than guessed at. |
| Baseline comparator mechanics | `poemv_rs/eval_compare.py` (`ew_weights`, `gmv_weights`, `mv_target_weights`, `rebalance_steps`, `simulate_method_on_path`) | Equal weight: fixed `(1/d,...,1/d)`. Minimum variance: fixed global-min-variance weights from the pseudo-inverse of the unconditional (stationary-mixture) covariance of the true simulator, computed once. CenterOnly daily/monthly: mechanical rebalancing (no band) to the frozen Stage~1 center weight, every 1 or 21 steps respectively. All comparators pass through the same evaluation-time leverage cap as Center+DNNBand. |
| Algorithm~2 (unified two-stage procedure) | `poemv_rs/train.py` estimation branch | Previously stated "fit a two-state statistical jump model" as a forced first step; corrected to state that the common-covariance track (reported five-seed results) actually uses drawdown-threshold heuristic labeling, and SJM is used only for the regime-dependent-covariance extension. |
| Stage~2 checkpoint-selection vs. reported-evaluation policy mismatch | `poemv_rs/stage2_direct_boundary.py` (`_run_validation`, `simulate_episode_direct_boundary`) | Validation utility used for checkpoint selection is computed by the same band-only simulator used in training (no evaluation-time leverage cap); the final reported evaluation metrics additionally apply that cap. This mismatch is now stated explicitly in the paper. |

## 2. Section-level changes (old -> new)

| Area | Nature of change |
|---|---|
| Title page | Removed the "Draft version... incorporated" process-status line; now a plain date line. |
| Abstract | Fully rewritten (~230 words) to lead with the belief-state HJB reduction (incl. regime-dependent-covariance extension) as the central contribution, then Stage~1, then Stage~2/Direct Boundary policy optimization, then the five-seed empirical results, ending with the ALM-scope disclaimer. All internal/process language removed. |
| Introduction / contributions | Restructured from 4 flat, un-ordered contributions into a tiered list: (i) belief-state HJB reduction incl. regime-dependent covariance (central/theoretical), (ii) Stage~1 structure-guided actor--critic, (iii) Stage~2 transaction-cost architecture with Direct Boundary policy optimization, (iv) TD-inspired actor-signal refinement (minor). Five-seed robustness moved into its own paragraph after the list. |
| Algorithm~2 | Fixed to not force SJM fitting for the common-covariance track (see table above). |
| Terminology | "Direct Boundary RL" replaced by "Direct Boundary policy optimization" throughout (with an explicit critic-free explanatory clause at first use in the Introduction and in Section~7.4); Stage~1 log-utility numbers relabeled "post-hoc distributional summaries" (not "utility-consistent"), reserving "objective-consistent" for Stage~2. |
| Leverage-cap composition | Trade-count-reduction claims (Discussion, Conclusion, Stage~2-results paragraph) rewritten to attribute the reduction to "the evaluated band-plus-cap policy," not "the learned no-trade band alone"; checkpoint-selection/final-evaluation-policy mismatch now stated explicitly. |
| `share_paths_avg_leverage_near_cap` | Limitations bullet rewritten to state the metric is uninformative (identically zero) for Stage~2 and meaningful only for uncapped Stage~1, and that the 1.980 threshold's numeric origin is not established in the archived code/config. |
| Baseline definitions | New "Baseline comparator definitions" paragraph added to the Numerical Study (Equal Weight, Minimum Variance, CenterOnly daily/monthly) giving exact mechanics from `eval_compare.py`. |
| Stage~1 configuration table (`tab:configs`) | Expanded from 9 to 26 rows with verified optimizer/actor/critic hyperparameters (action bound, learning rates, schedule, mixing weight, exploration scale, polynomial degree, gradient clipping, checkpoint frequency, compute device), sourced from `run_config.json` and cross-checked against `train.py`/`agent.py` defaults; reformatted to wrapped paragraph columns to fix a resulting table overflow. |
| Code/artifact availability | New Appendix D ("Code and Data Availability") added; commit-hash and manifest-hash detail moved out of the main-text `tab:stage2_config` row into this appendix, replaced in the main table by a one-line pointer. |
| Figures | `aggregate_phase1a.py`: `fig4_cost_turnover` suptitle shortened and given a two-line, correctly-margined title (`tight_layout(rect=...)`) to fix the title-clipping/overflow bug; `fig1_mean_vs_std` and `fig5_training_seed_variability` enlarged (11x4.5 -> 13x5.5) with a reserved top margin. LaTeX layout for the combined Stage~1 figure (`fig:stage1_common`) changed from two 0.49-textwidth side-by-side subfigures to two full-width stacked subfigures, since each source PNG already contains two internal panels. Figure~2 caption (`fig:stage2_eval`) gained a sentence stating the pooled violin plots are descriptive visualizations of non-independent (repeated-scenario) observations, not 5000 independent draws. All four figures regenerated by rerunning `aggregate_phase1a.py` against the same archived per-seed CSVs (no new simulation) and re-copied into `ICA report/file/phase1a/`. |
| Process-narrative language | Removed/reworded sentences describing the paper's own revision history ("previously reported," "superseded," "this revision," "regenerated... for this revision," "earlier drafts," a path-quoting bug-fix narrative). All 16 occurrences of the internal label "Phase~1A" replaced with neutral phrasing ("the five-seed replication," "the reported runs," etc.). |
| Conclusion | Shortened from five paragraphs to three; removed a duplicated ALM-scope limitation sentence (previously stated in two consecutive paragraphs); re-ordered to match the new tiered contributions; added a `\label{sec:actuarial}` cross-reference (was previously an unreferenceable starred subsection). |
| Cross-references | Added `\label{sec:experimental_design}` and repointed two mis-targeted `\ref{sec:stage1_results}` citations (leverage-cap discussion; metric-definition paragraph) that actually belong to the Experimental Design subsection; removed the one remaining vague "Section~X onward" reference. |

## 3. Numerical-value source mapping

No numerical value was newly computed or hand-derived in this pass beyond
simple arithmetic already shown inline in the surviving text (unchanged from
the prior technical-QA phase). All Stage~1/Stage~2 five-seed table and figure
values are unchanged from the QA-phase baseline and continue to trace to
`aggregate/tables/table{1,2,3,4}_*.csv` and `aggregate/seed_level_results.csv`
under `results/ica2026_additional_runs/phase1a_endtoend_20260725/aggregate/`,
produced by `aggregate_phase1a.py`. The only newly added numeric content is:
(a) the ~17 new Stage~1 configuration-table rows, each read directly from an
archived `run_config.json` field or a `train.py`/`agent.py` default (see
Section 1 above); (b) the `share_paths_avg_leverage_near_cap` per-method
values quoted in the Limitations bullet, read directly from
`aggregate/seed_level_results.csv`.

## 4. Legacy-value and meta-language grep sweep (final state)

Final case-insensitive sweep of `ICA_report.tex`, all zero hits except where
noted:

| Term | Hits |
|---|---|
| `Phase~1A` / `Phase 1A` | 0 |
| `previously reported` | 0 |
| `superseded` | 0 |
| `legacy run` | 0 (`legacy threshold labeling` / `legacy configuration` retained twice: legitimate description of an alternative estimation method, not the paper's own draft history) |
| `technical QA` | 0 |
| `Claude`, `chat`, `conversation` | 0 |
| Absolute paths (`/Users/taka`, `results/ica2026...`, `poemv_rs copy`) | 0 |
| `1.248`, `1.136`, `1.147`, `1.052`, `1.197`, `0.126` (legacy single-seed values) | 0 |
| `$500$` / "500 independently"/"500 evaluation" (legacy path count) | 0 |
| `this revision`, `current manuscript`, `revision discovered`, `bug fixed`, `draft cutoff` | 0 |
| `independent validation` | 0 |
| vague `Section~\ref{...} onward` | 0 |
| `cryptographic` | 0 |

## 5. Compile / validation results

- Two-pass `pdflatex` compile (via the same scratch-`TEXINPUTS` workaround as
  the prior phase, for `enumitem`/`algorithm`/`algorithmic`/`algpseudocode`,
  which remain absent from this machine's system TeX install): **exit code 0**
  on both passes.
- 0 undefined references, 0 multiply-defined labels.
- 2 overfull `\hbox` warnings, both sub-3pt (2.47pt and 1.36pt; not visually
  detectable). The one large (512pt) overfull box produced by the newly
  expanded Stage~1 configuration table was found and fixed by switching that
  table to wrapped paragraph columns before this final compile.
- Final page count: **37 pages** (via `pypdf`), within the 40-page ICA limit.
- 0 occurrences of "Missing figure" (the `\imgorplaceholder` fallback text) in
  the extracted PDF text; all four `\includegraphics` calls resolve to actual
  files in `file/phase1a/`.
- No `tmux` server and no matching `poemv_rs`/training/supervisor processes
  running at the time of this revision; no Phase~2 experiment was started.

## 6. Unresolved issues (stated explicitly, not guessed)

- The exact numeric provenance of the `1.980` leverage-near-cap threshold in
  `aggregate_phase1a.py` could not be traced to any derivation in the code or
  archived configuration; the paper now states this rather than inventing a
  rationale.
- The regime-dependent-covariance track's archived training configuration
  (iteration count, episodes per iteration) could not be matched with
  certainty to a single specific run directory under `runs/`: one candidate
  run (`stage1_regime_cov_jm_like_commoncfg_sigmas_long_bs64_rev`) was
  inspected and found to use different iteration/episode counts and different
  estimated parameters than the values already displayed in
  `tab:sjm_results`/`tab:regcov_parameters`, indicating it is a *different,
  later* experiment and not the source of the currently displayed numbers.
  The true source run for the regcov table was not re-identified in this
  pass (it was already validated in an earlier phase); no value in
  `tab:regcov_parameters`, `tab:sjm_results`, or `tab:regcov_metrics` was
  changed, and the newly expanded `tab:configs` therefore states the regcov
  optimizer-hyperparameter column as "same" only for fields directly
  confirmed identical across the one archived regcov config file found, with
  the iteration-count/episode-count row explicitly flagged as not
  independently re-verified for that exploratory track.
- The repository is not yet public; the Code and Data Availability appendix
  states this and gives a fallback ("available upon request... subject to
  institutional approval") rather than asserting a URL that does not exist.

## 7. Commits

Two commits on `ica2026-final-submission-revision`, per the requested
separation:
1. Scientific content and reproducibility corrections (LaTeX text: abstract,
   contributions, algorithm fix, terminology renames, leverage-cap wording,
   baseline definitions, configuration table expansion, code-availability
   appendix, conclusion, cross-reference fixes, meta-language purge).
2. Figures/layout/language cleanup (`aggregate_phase1a.py` figure-generation
   edits, regenerated PNGs, LaTeX figure-layout/caption changes, recompiled
   PDF).

Commit hashes are recorded in the branch history immediately following this
report's own commit.

## 8. Editorial-cleanup pass (post-review)

A follow-up editorial-cleanup pass, prior to pushing the branch, removed the
remaining implementation-history wording flagged in review:

- Renamed "Legacy threshold labeling" -> "Threshold-based labeling" and "In
  the legacy configuration" -> "In the threshold-based configuration"; `legacy`
  no longer appears anywhere in the paper.
- Verified, read-only, the SJM rolling-feature construction code
  (`poemv_rs/train_stage1_cov_modes_with_estimation.py`: `_roll_mean`,
  `_roll_std`, `_rolling_downside_dev`, `_rolling_drawdown_from_returns`,
  `_rolling_cumret`, all sliced as `x[max(0, t-win+1):t+1]`). Every raw
  rolling feature is confirmed causal (backward-looking only). However, the
  subsequent `_zscore_cols` standardization step normalizes using the full
  estimation sample's mean/standard deviation, which is **not** a causal,
  online transformation. The paper's sentence was rewritten to state the
  verified causality of the raw features precisely, while disclosing the
  non-causal batch normalization step, rather than the previous blanket "is
  intended to" claim.
- Removed the unsupported "its noisier learning curve cannot be attributed to
  a single source" sentence (Discussion, sources-of-performance section) and
  the unsupported "the training curve is noisier" clause (Stage~1 regcov
  results paragraph): no regcov training-curve figure or archived multi-seed
  training-curve data is presented anywhere in the manuscript, so these
  claims were not supportable as written. Replaced with a source-agnostic
  statement that the reported performance difference cannot be attributed to
  any single error layer (regime identification, parameter estimation,
  filtering, control approximation, optimization), which is the scientifically
  supported claim.
- Shortened the SJM reproducibility-limitation paragraph in the main text to
  a concise statement, moving the itemized detail (what is and is not
  retained: estimation horizon, jump penalty, initializations, iteration
  limit vs. rolling windows, normalization statistics, estimation seed) to
  the Code and Data Availability appendix; removed workflow/session language
  from the main exposition.
- Deleted, without replacement, the standalone paragraph following
  Figure~4 ("Per-iteration Stage~2 training diagnostics ... for each of the
  five training seeds."): it described artifact-availability/figure-
  production history rather than a scientific result and was unnecessary for
  interpreting the figure. Post-deletion, `learning curve`/`training curve`
  no longer appear anywhere in the manuscript (case-insensitive sweep: 0
  hits for both terms).
- Final case-insensitive sweep for `legacy`, `current revision`, `previous
  version`, `superseded`, `earlier illustration`, `could not be regenerated`,
  `missing file`, `retained image`, `technical QA`, `inventory`, `revision
  map`, `bug`, `failed invocation`, `conversation`, `chat`, `prompt`,
  `Claude`, `ChatGPT`, `Phase 1A`/`Phase~1A`, `as shown in the figure`,
  `visual inspection`: **0 hits** for every term. `learning curve` has one
  remaining hit, which explicitly states that no such figure is shown (used
  to disclaim absence, not to claim unsupported visual evidence), and was
  judged compliant with the review's intent rather than removed.
- Recompiled twice: exit code 0 both passes; 0 undefined references; 0
  multiply-defined labels; 2 pre-existing overfull `\hbox` warnings, both
  sub-3pt (not major); 0 "Missing figure" placeholder text in the extracted
  PDF; final page count unchanged at 37 pages. No tmux session and no
  matching training/evaluation/supervisor process was running at the time of
  this pass.
- This cleanup was committed separately from the two commits described above,
  per the requested three-commit history for this branch.

## 9. Second editorial-cleanup pass (manuscript-only, pre-push)

A further manuscript-only cleanup pass, still prior to pushing the branch,
removed remaining artifact-management/workflow language while preserving
scientific reproducibility content:

- Replaced the "Code and Data Availability" appendix with a short "Code and
  Reproducibility Materials" section (author-request availability statement;
  synthetic-data statement; five-seed reproducibility statement; regcov
  exploratory-status statement), removing all mention of commit identifiers,
  checkpoint/manifest hashes, `run_manifest.csv`, and an as-yet-unpublished
  repository URL.
- Rewrote the repeated-seed-protocol paragraph to remove the history of two
  initially empty Stage~1 evaluation outputs and their invocation-error
  cause, replacing it with a direct description of the deterministic
  evaluation-scenario design and the byte-identical-benchmark consistency
  check.
- Deleted, without replacement, the sentence about an "earlier, incompletely
  archived exploratory run" and its unrecorded utility/coefficient settings.
- Replaced the regime-dependent-covariance results' opening paragraph to
  drop the mention of a non-regenerable figure/retained image, stating only
  the experiment's scientific purpose and single-seed scope.
- Simplified both Stage~1 and Stage~2 configuration-table captions to remove
  references to `run_config.json`/`eval_config.json`/`aggregation_config.json`/
  `run_manifest.csv`, source-script filenames, and "not transcribed from
  memory" wording.
- Removed the "Training/evaluation code provenance" row (commit identifier,
  checkpoint/manifest hashes) from the main Stage~2 configuration table,
  keeping the scientifically relevant checkpoint-selection-criterion row.
- Replaced the long `utility_gamma`/`gamma_risk` walk-through with a short
  statement that both parameters are inert under the reported log-utility,
  direct-gap configuration; removed the archived-manifest cross-reference
  and the literal `best_checkpoint.pt` filename (the checkpoint-selection
  criterion itself, and the checkpoint-selection/evaluation-policy mismatch
  noted in an earlier pass, are retained as scientifically relevant).
- Replaced "In trial runs... The final experiment therefore uses..." with a
  direct statement that the tested QVI-residual/intervention optimization
  scheme did not converge reliably, motivating Direct Boundary policy
  optimization.
- Removed the unsupported claim that "preliminary generic deep-RL and
  Transformer-style formulations also failed" from the Stage~2 discussion
  paragraph, retaining only the documented QVI critic/intervention
  instability. (The similarly worded, but separately scoped, Introduction
  sentence about preliminary Stage~1 architecture trials was left
  unchanged, as it was outside this item's specified scope.)
- Removed all mentions of `share_paths_avg_leverage_near_cap` and the
  `1.980` threshold from the manuscript (table, captions, and the
  Limitations list), replacing the corresponding limitation with a single
  sentence stating that a time-resolved leverage-cap saturation measure is
  not available.
- Replaced two data-retention Limitations bullets (per-timestep wealth
  trajectories; separate lower/upper boundary summaries) with
  scientific-scope wording that states what the evaluation output contains
  rather than what the evaluation code did or did not persist.
- Replaced "remain future work unless completed before submission" with
  "remain directions for future work".
- Replaced the Appendix~B sentence describing the QVI critic/intervention
  residual objective as "the intended... direction" that was found unstable
  "in practice" with a direct statement that it is the QVI-based alternative
  considered in the study, unstable under the tested optimization scheme.
- Replaced both occurrences of "reconstructed from archived episode-level
  terminal wealth" / "computed directly from the archived episode-level
  terminal wealth" with "computed from episode-level terminal wealth".
- Replaced the SJM section's retained-configuration paragraph with a two-
  sentence statement of what is not documented and the resulting exploratory
  interpretation, removing the sentence pointing to the (now-restructured)
  Code and Data Availability appendix.
- Verified the Generative-AI acknowledgment paragraph was not modified.
- Final case-insensitive sweep across `archived`, `manifest`, `commit`,
  `hash`, `run_config`, `eval_config`, `aggregation_config`, `run_manifest`,
  `not transcribed`, `evaluation-invocation`, `empty output`, `no policy was
  retrained`, `earlier run`, `retained image`, `missing figure`, `could not
  be regenerated`, `trial runs`, `final experiment`, `available
  experiments`, `before submission`, `training-script`, `CLI`, `legacy`,
  `revision`, `technical QA`, `conversation`, `chat`, `prompt`, `Claude`,
  `ChatGPT`, `Phase 1A`. Two categories of retained hits, both judged
  compliant: (a) all 10 hits of `commit` are the financial term
  "precommitted" (defined in the paper as the terminal criterion being fixed
  at time 0, not reoptimized) or "committees" (an unrelated actuarial-
  governance word), none referring to a source-control commit; (b) the
  single `missing figure` hit is the literal fallback text embedded in the
  `\imgorplaceholder` LaTeX macro definition (infrastructure code that would
  only render if a figure file were absent, which none are), not narrative
  text about the manuscript's own production history. Two initial residual
  hits for `archived`/`manifest` and two for `training-script`/`CLI` were
  found and fixed during this pass (an "archived manifest" phrase in the
  Stage~2 wealth-recursion paragraph, an "archived run" phrase in the
  expanded Stage~1 configuration table, and "training-script default"/"CLI-
  configurable" phrasing in two configuration-table cells); after these
  fixes, the sweep returns zero further hits.
- Recompiled twice: exit code 0 both passes; 0 undefined references; 0
  multiply-defined labels; the same 2 pre-existing sub-3pt overfull `\hbox`
  warnings (not major); 0 "Missing figure" placeholder text in the extracted
  PDF; page count unchanged at 37 pages. No tmux session and no matching
  training/evaluation/supervisor process was running at the time of this
  pass.
- This pass amended the same editorial-cleanup commit (still unpushed)
  rather than adding a new commit.
