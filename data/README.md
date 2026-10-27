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
- `simulation_raw.csv`: 192,000 fitted rows from 48,000 datasets and four estimators.
- `simulation_summary.csv`: bias, absolute bias, MSE, RMSE and interval performance by scenario and target.
- `interval_comparison.csv`: regime averages for ACI, equal-tail and numerical HPD intervals.
- `real_estimates.csv`, `carbon_estimates.csv`: fitted application estimates and reliability intervals.
- `jute_hpd_intervals.csv`, `carbon_hpd_intervals.csv`: numerical HPD alternatives.
- `goodness_of_fit.csv`, `carbon_goodness_of_fit.csv` and common-shape JSON files: calibrated fit checks.
- `posterior_diagnostics.csv`, `posterior_chains.npz`: the independent sampling check.
- `sensitivity.csv`, `acquisition_order_sensitivity.csv`, `predictive_checks.csv`: sensitivity and predictive results.
- `report_graph_index.json`, `report_graph_coverage.json`: the mapping of all saved figure panels and the selected report subset.

Other files hold bootstrap draws, parameter intervals and numerical refinement checks. `application_quadrature_refinement.csv` compares the standard and finer quadrature grids for all three examples and both posterior supports. Missing regular-normal interval entries for ordered MLE are intentional; its boundary interval is assessed by the separate bootstrap study.

The public data and results contain no measured operating loads. Reliability in the applications is a same-unit laboratory comparison of breaking-strength populations.

## Primary carbon gauge groups

`raw/primary_carbon_gauges.csv` contains maximum failure stress from the authors' 20 mm and 80 mm Results workbooks in [Zenodo DOI 10.5281/zenodo.4033353](https://doi.org/10.5281/zenodo.4033353). The related article is Kumar, Mikkelsen, Lilholt and Madsen (2024), *Weibull parameters determined from a comprehensive dataset of tensile testing of single carbon fibres*, Data in Brief 55, 110717, DOI 10.1016/j.dib.2024.110717.

The endpoint gauges were chosen before fitting. Each group has 153 positive identified specimen rows. Specimen IDs, source positions and original MPa values are preserved; GPa is MPa divided by 1000. Analysis divides GPa by a fixed 5 GPa unit for both groups. `raw/primary_carbon_provenance.json` records exact author filenames, published MD5 checksums and the row rule. These are all identified positive workbook rows, not a reconstruction of experimental-error exclusions or the paper's plotted subset. Preparation failures absent from the published measurements cannot be recovered here.

The 20 mm group supplies three strength blocks of 51 with target 35 and threshold one. The 80 mm group supplies six upper 2-records. The workbook order did not reach that target. One fixed seeded permutation, using 250150246, defines the imposed acquisition order; no repeated search for preferred estimates is performed. `primary_allocation.csv`, `primary_censored_strength.csv` and `primary_records.csv` preserve the allocations, actual withdrawals and acquisition/source positions.

## Added studies and checks

- `bootstrap_extended_raw.csv`, `bootstrap_extended_summary.csv`: 12 settings, 500 outer datasets and 399 resamples per fitted model; unrestricted ACI, Boot-p, log-rate-ratio Boot-t and ordered Boot-p.
- `design_extensions_raw.csv`, `design_extensions_summary.csv`: fourteen 2-out-of-4 designs with 500 replications each; crossed strength/record sizes, unequal blocks and early/late withdrawals.
- `prior_extensions_raw.csv`, `prior_extensions_summary.csv`: three regimes with 500 paired datasets each and two proper priors.
- `primary_goodness_of_fit.csv`, `primary_common_shape_test.json`, `primary_estimates.csv`: complete-group fit checks and retained-data inference for the primary gauge groups.
- `order_evidence.csv`: boundary-bootstrap likelihood diagnostics and unrestricted posterior order probabilities for all three examples.
- `application_bootstrap_intervals.csv`: additional application intervals with 3999 bootstrap resamples.
- `nonparametric_reliability.csv`: complete-data shared-stress U-statistic estimates with 2000 two-sample bootstrap resamples and matching complete-data Weibull estimates. These use more observations than retained-data fits.
- `general_system_checks.csv`, `batch_validation.json`, `generator_validation.csv`, `mcmc_scenario_checks.csv`: independent formula, optimizer, generator and posterior checks.
- `source_access_checks.json`: dated access scope for the assigned SAGE reference.

The original random carbon split is a boundary check from one source population. Its fitted reliability below 0.5 is not by itself a significance test; the new boundary-bootstrap tail area is 0.101. The primary gauge groups are distinct laboratory populations, still not operating-load measurements.
