# Weibull stress-strength reliability

**Vaidik Maheshwari (250150045) and Anant Gupta (250150004)**

DA2141, Mehta Family School of Data Science and AI, IIT Guwahati

This project estimates the probability that at least three of five component strengths exceed the same random stress. Strength and stress have Weibull distributions with a common shape. Strength observations are progressively censored in blocks, and stress observations are upper k-records.

Start with the [40-page report](Project_Report.pdf). The [executed notebook](code/Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb) contains calculations, saved tables and graphs. The [LaTeX source](latex/main.tex) follows the supplied article structure and margins.

## What we did

1. Derived shared-stress reliability and the likelihood for the actual observation schemes.
2. Obtained unrestricted and ordered maximum likelihood estimates and Bayesian estimates under a dependent gamma-Dirichlet prior.
3. Compared four estimators in 48 settings, with 1000 replications per setting: 48,000 datasets and 192,000 fitted rows.
4. Compared asymptotic, equal-tail, numerical HPD and bootstrap intervals, reporting coverage, width and Monte Carlo uncertainty.
5. Added a 12-setting bootstrap study, a 14-design 2-out-of-4 study, and a paired prior comparison, each with 500 datasets per setting.
6. Analyzed jute measurements, a carbon random-split boundary check, and separate primary carbon gauge populations, with fit checks, order evidence and a complete-data nonparametric comparison.
7. Checked quadrature against independent posterior sampling and checked optimization, derivatives, specimen conservation and general system formulas.

The report explains the derivations, simulation pseudocode, data selection, important graphs, results and limitations. Full Python code is kept in the code folder. Function explanations are in [code/GUIDE.md](code/GUIDE.md); the source contains no comments or docstrings.

## Important files

| Item | Purpose |
|---|---|
| [Project_Report.pdf](Project_Report.pdf) | Main report for reading and submission |
| [latex/](latex) | Editable manuscript, bibliography, sections and result tables |
| [graphs/](graphs) | All 18 figure groups and 61 enlarged panels; 23 selected panels appear in the report |
| [data/raw/](data/raw) | Published measurements and primary-data provenance |
| [data/processed/](data/processed) | Retrospective allocations, censoring and retained records |
| [data/results/](data/results) | Simulation summaries, raw fits, intervals and diagnostic results |
| [Notebook](code/Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb) | Executed analysis with saved outputs |
| [analysis.py](code/analysis.py) | Main statistical functions |
| [Additional studies](code/review_extensions.py) | Bootstrap, observation-design and prior comparisons |
| [Application checks](code/review_applications.py) | Primary gauge analysis, order evidence and nonparametric comparisons |
| [Verification note](code/notes/shortcomings_verification.md) | Decisions on all 22 review findings and remaining limits |

All graphs are in one folder. Composite plots are saved as PNG and PDF. The report uses 23 selected panels; additional fit, trace and distribution plots remain available here and in the notebook. Full raw-data and scenario listings are kept in data rather than printed in the report. [data/README.md](data/README.md) describes data sources, units and result files.

## Main findings and limits

The order restriction helps in some correctly ordered cases but causes large bias when false. For the main 3-out-of-5 system it excludes reliability below 0.5; the general bound is `1 - s/(K+1)`. Ordered equal-tail intervals cannot cover the equality boundary. Numerical HPD intervals can include it, but their finite-sample coverage is still approximate.

The added log-rate-ratio Boot-t interval improves average unrestricted coverage in the tested bootstrap settings, usually with greater width. It is not guaranteed to improve every setting. The broader-prior comparison shows that small-sample Bayesian performance depends on prior concentration.

One jute group fails the calibrated Weibull fit check. The original carbon split comes from one population and is a boundary check, not a distinct-population validation. The added 20 mm and 80 mm carbon groups come directly from the authors' public workbooks. They are compatible with the working Weibull and common-shape assumptions; their complete-data nonparametric and Weibull reliability estimates are both near 0.81.

The applications compare laboratory breaking strengths in common units. They do not use measured operating loads. Censoring and acquisition orders are imposed retrospectively. The primary data include all positive identified workbook rows, not a reconstruction of the article's experimental-error exclusions. Only six stress records enter the primary retained-data fit, so it has different information from the complete-data comparison.

## Running the notebook

Run from the repository root:

```bash
python3 -m pip install -r code/requirements.txt
python3 code/run_notebook.py
```

Python 3.14 was used for the completed run. The notebook is already executed. Its implementation cell sets `RUN_SIMULATION=False`: this loads the saved 1000-replication main results and repeats the other notebook calculations. To regenerate all 48,000 main datasets, change that switch to `True` and run all cells. Seeds, priors and designs are fixed in the source.

The notebook's final section displays saved additional-study tables and figures. To regenerate those studies as well:

```bash
python3 code/review_extensions.py check
python3 code/review_extensions.py bootstrap
python3 code/review_extensions.py design
python3 code/review_extensions.py prior
python3 code/review_applications.py
python3 code/review_checks.py
```

These studies are separate scripts to keep a normal notebook run manageable. The application script uses the saved raw CSV files; downloading the source Excel workbooks again is not required for analysis. Their URLs, specimen-row rule and checksums remain in the provenance JSON.

## Preparing the report

After calculations, refresh graphs and tables in this order:

```bash
python3 code/prepare_graphs.py
python3 code/prepare_report.py
python3 code/prepare_review_report.py
```

With a local TeX installation:

```bash
latexmk -pdf -cd latex/main.tex
cp latex/main.pdf Project_Report.pdf
```

For Overleaf, upload the `latex` and `graphs` folders with their relative structure preserved and select `latex/main.tex` as the main document. The data and code folders are not needed just to compile it. The document uses A4, 11-point article text, 0.9-inch side margins and 1-inch top and bottom margins. Its five main sections follow the professor's template. The supplied `generic.sty` had no active formatting rules, so the included file is a minimal package with that name.

`code/build_notebook.py` rebuilds notebook source and replaces saved outputs. It is a maintenance script; execute the rebuilt notebook afterwards. [code/GUIDE.md](code/GUIDE.md) maps functions to report equations.

## Source notes

Detailed derivations and source-use notes are in [code/notes/](code/notes). The primary carbon data are from [Kumar et al.'s Zenodo deposit](https://doi.org/10.5281/zenodo.4033353), linked by their [2024 Data in Brief article](https://doi.org/10.1016/j.dib.2024.110717). Jute and the original carbon table are transcriptions from the supplied papers, with original experimental attributions recorded separately.

For the assigned SAGE reference, this work used the publisher abstract and citation details. Full text was not accessed in this session; another institution may have different access. A dated record is saved in `data/results/source_access_checks.json`.

Original supplied papers are kept locally in the ignored `sources` folder. Earlier versions and the external review are in the ignored `local_archive` folder. Neither folder is included by the repository's normal Git commands.
