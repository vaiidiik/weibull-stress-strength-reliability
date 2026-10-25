# Data files

## Raw published observations

`raw/jute_fibre.csv` contains 30 measurements from each of the 15 mm and 20 mm jute groups in Table 10 of Sultana et al., *Inference for block Type-I hybrid censored Weibull populations*, DOI 10.1080/03610926.2025.2579720. That article attributes them to Xia et al. (2009). The original experimental file was not supplied. The source values are already divided by 1000 and their printed order is preserved. The 10 mm group is not used because its reproduced table contains an unverified value, 70.32383.

`raw/carbon_fibre.csv` contains 100 carbon-fibre strengths in GPa from Section 5.1.1 of Newer and Alanazi, *Empirical Bayes and Bootstrap Inference for Multicomponent Stress-Strength Reliability Under Ordered Ranked Set Sampling*, DOI 10.1155/jom/5561353. The article attributes them to Nichols and Padgett (2006). The source is the supplied reproduction, not a directly obtained original experimental file. Analysis divides both allocated groups by a fixed 3 GPa unit.

The source positions identify rows in reproduced tables. They are not verified historical acquisition times.

## Retrospective sampling

`processed/` records the seeded block allocations, observed strength failures, actual withdrawals, terminal survivors and retained upper k-records. These files are derived from the raw observations; they are not additional experiments.

The jute strength group is split into three blocks of ten, with target seven failures, balanced withdrawals and time limit 0.5. Its comparison group supplies six upper 2-records. The carbon sample is randomly divided into disjoint groups of fifty using seed 250150115. Strength block sizes are 17, 17 and 16; their targets are the integer parts of 70 percent of block size and their time limit is one in scaled units. The comparison half supplies five upper 4-records.

For strength tables, `terminal_survivors` is a block count repeated beside its failure rows. Count it once per block when checking specimen conservation. `withdrawn_at_failure` is the actual number removed at the displayed failure. Acquisition positions and original source positions are different quantities for the randomized carbon example.

## Computed and simulated results

`results/` contains outputs from the stated calculations:

- `simulation_design.csv`: all 48 settings and fixed truths.
- `simulation_raw.csv`: 57,600 fitted rows from 14,400 datasets and four estimators.
- `simulation_summary.csv`: bias, absolute bias, MSE, RMSE and interval performance by scenario and target.
- `interval_comparison.csv`: regime averages for ACI, equal-tail and numerical HPD intervals.
- `real_estimates.csv`, `carbon_estimates.csv`: fitted application estimates and reliability intervals.
- `jute_hpd_intervals.csv`, `carbon_hpd_intervals.csv`: numerical HPD alternatives.
- `goodness_of_fit.csv`, `carbon_goodness_of_fit.csv` and common-shape JSON files: calibrated fit checks.
- `posterior_diagnostics.csv`, `posterior_chains.npz`: the independent sampling check.
- `sensitivity.csv`, `acquisition_order_sensitivity.csv`, `predictive_checks.csv`: sensitivity and predictive results.
- `report_graph_index.json`, `report_graph_coverage.json`: the mapping from notebook figures to report panels.

Other files hold bootstrap draws, parameter intervals and numerical refinement checks. Missing regular-normal interval entries for ordered MLE are intentional; its boundary interval is assessed by the separate bootstrap study.

The public data and results contain no measured operating loads. Reliability in the applications is a same-unit laboratory comparison of breaking-strength populations.
