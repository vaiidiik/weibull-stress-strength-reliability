# Guide to the statistical code

The report explains the equations and the notebook shows their outputs. This file maps the main functions to those calculations. Function descriptions are kept here rather than in Python comments or docstrings.

| Function | Calculation | Report location or equation label |
|---|---|---|
| `reliability` | Shared-stress s-out-of-K probability, with explicit system validation | `eq:reliability` |
| `removal_plan`, `censor_block` | Planned withdrawals and actual Type-I or adaptive Type-II observations | Observation schemes; `eq:block-likelihood` |
| `make_data`, `exposures` | Retained failure counts, records and weighted exposures | `eq:exposure-a`, `eq:exposure-b` |
| `loglik` | Joint Weibull censoring/record log-likelihood | `eq:loglik` |
| `fit_mle`, `conditional_rates` | Shape profiling, unrestricted rates and order-boundary pooling | `eq:mle-rates`, `eq:ordered-rates` |
| `observed_information`, `asymptotic_intervals` | Covariance and positive log-scale intervals | `eq:information`, `eq:parameter-aci` |
| `posterior_grid` | Analytically remove the total rate, then integrate shape and rate share | `eq:collapsed` |
| `hpd_interval` | Shortest 95 percent interval under the interpolated reliability density | Credible intervals section |
| `posterior_draws` | Quadrature state selection followed by a conditional Gamma total-rate draw | Bayesian computation |
| `simulate_data`, `run_simulation` | Generate the 48 main scenarios and summarize four estimators | Algorithms 1 and 2 |
| `bootstrap_reliability` | Original complete-experiment percentile-bootstrap draws | Confidence intervals section |
| `goodness_of_fit` | Weibull K-S and common-shape likelihood-ratio bootstrap calibration | Real-data analysis |
| `extract_records` | Strict upper k-record extraction from a finite acquisition sequence | `eq:record-likelihood` |
| `mcmc`, `chain_diagnostics` | Independent collapsed-posterior sampling and Monte Carlo checks | Checking the calculations |

## Additional checks

`review_extensions.py` contains the added bootstrap, design and prior comparisons. `batch_generate` uses exponential spacings on the transformed Weibull strength scale. With independent noninformative withdrawals, the next spacing has exponential rate equal to the number of active specimens. The procedure preserves each block's own target, time limit and withdrawal plan. It does not combine heterogeneous blocks into one sample.

`batch_fit` finds the same profile-score root as the scalar optimizer, for many independent datasets at once. Its implementation is checked against 200 separate scalar fits; the saved largest parameter discrepancy is about 5.3e-8. The generator is also compared with the original complete-lifetime generator. Validation results are stored in `data/results/batch_validation.json` and `generator_validation.csv`.

The new studentized interval uses the log-rate ratio, not an unrestricted normal interval clipped to [0,1]. For each unrestricted bootstrap fit it computes the observed-information standard error of log(b/a), forms a studentized pivot, inverts its quantiles and transforms the endpoints through reliability. An ordinary studentized interval is not assigned to a pooled ordered MLE. Its percentile interval is assessed separately.

`review_applications.py` applies those calculations to the jute example, the original carbon random-split boundary check and the new primary gauge groups. It also reports a boundary-bootstrap order diagnostic and the unrestricted posterior probability of the order. These are evidence about an assumption, not an automatic pretest-and-select rule with guaranteed coverage.

`nonparametric_reliability` uses all complete observations and averages a hypergeometric U-statistic over stress observations. Its resampling intervals use both complete groups. It therefore has different information from the censored/record fit and is reported as a robustness comparison, not as its replacement.

`review_checks.py` independently checks the general reliability formula, the fast generator and quadrature against MCMC in three additional scenarios for a 2-out-of-4 system.

## Data and report preparation

`run_notebook.py` executes the notebook. `prepare_graphs.py` rebuilds its original enlarged panels. `prepare_report.py` creates the main manuscript tables. `prepare_review_report.py` creates the added comparison tables and graphs. `build_notebook.py` rebuilds notebook source and replaces saved outputs, so it is a maintenance tool rather than the normal execution command.

The primary gauge data are public CSV files with specimen identifiers, original positions and measured failure stress. `data/raw/primary_carbon_provenance.json` records the exact author files, checksums, units and row-selection rule. The row rule was all valid specimen identifiers, with no filtering based on fit or on the reported reliability.
