# Weibull stress-strength reliability

**Vaidik Maheshwari (250150045) and Anant Gupta (250150004)**

DA2141, Mehta Family School of Data Science and AI, IIT Guwahati

This project estimates the probability that at least three of five component strengths exceed one common random stress. Strength and stress have Weibull distributions with a common shape. Strength observations are progressively censored in blocks, and stress observations are upper k-records.

The comparison includes maximum likelihood and Bayesian estimates, both with and without the order restriction that strength is stochastically larger than stress. The Bayesian model uses a dependent gamma-Dirichlet prior.

## Main files

| File or folder | Purpose |
|---|---|
| [Project report](reports/Project_Report.pdf) | The paper to read first: theory, simulation, data analysis and conclusions |
| [Jupyter notebook](Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb) | Executed calculations, tables and graphs |
| [LaTeX manuscript](manuscript/main.tex) | Editable report following the shared article outline |
| [Bibliography](manuscript/reference.bib) | References cited in the manuscript |
| [analysis.py](analysis.py) | Statistical functions used in the notebook |
| [data](data) | Transcribed measurements, block allocations and retained observations |
| [results](results) | Simulation results, estimates, intervals and numerical checks |
| [figures](figures) | Figures saved by the notebook as PNG and PDF |
| [Derivations](reports/derivations.md) | Additional mathematical details |
| [Literature review](reports/literature_review.md) | How the supplied articles were used |

The `scripts` folder contains two maintenance scripts: `build_notebook.py` rebuilds the notebook source, and `prepare_report.py` creates the manuscript tables and figures from computed results. Neither is needed just to read the project.

## Simulation

There are 48 scenarios and 300 replications per scenario, giving 14,400 simulated datasets. The factors are rate order, common shape, block size, censoring protocol and record order. Every dataset is used for all four estimators.

The saved results report absolute bias and mean squared error for the shape, both rates and reliability. Reliability intervals are compared by coverage and average width. These include the unrestricted asymptotic interval, Bayesian equal-tail intervals and numerical HPD intervals. Ordered MLE bootstrap coverage is checked separately in six scenarios.

HPD intervals are obtained from an interpolated posterior density on the reliability scale. Their probability mass is checked against 0.95 and application endpoints are compared on a finer grid. They are numerical approximations for unimodal marginal densities. They are not interchangeable with equal-tail intervals.

## Data and interpretation

The jute measurements are transcribed from Sultana et al., Table 10, which attributes them to Xia et al. The carbon-fibre measurements are transcribed from Newer and Alanazi, Section 5.1.1, which attributes them to Nichols and Padgett. The original experimental files were not supplied. The data folder records the source and the retrospective sampling design.

One jute group fails the bootstrap-calibrated Weibull goodness-of-fit test at 5 percent, so its estimates have a stated model-fit limitation. The additional carbon groups are compatible with the fitted Weibull model. Both examples compare laboratory breaking-strength populations in the same units; neither uses measured operating loads.

An incorrect order restriction can increase bias. In the simulation's violated-order case, true reliability is below one half but the ordered model cannot estimate below one half. Boundary coverage also depends on whether an equal-tail or HPD interval is used.

The source-use review records an access limitation for the assigned SAGE reference: its abstract was available, but its full text was not supplied. No full-text derivation from that source is claimed.

## Running the notebook

The saved notebook includes its outputs and can be read without rerunning it. Python 3.14 was used for the completed run. From the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 run_notebook.py
```

The implementation cell sets `RUN_SIMULATION=False`. This loads the saved Monte Carlo tables and repeats the remaining calculations, including the data analyses and numerical checks. To regenerate all simulation results, change it to `RUN_SIMULATION=True` in the notebook and run all cells. Repeating the full study takes longer. Seeds and scenario settings are saved.

The mathematical explanations are written in Markdown cells.

## Building the report

After running the notebook, update the manuscript tables and figures with:

```bash
python3 scripts/prepare_report.py
```

Upload the contents of `manuscript` to Overleaf and select `main.tex`. This includes `reference.bib`, `generic.sty`, `tables` and `figures`. Compile with the normal LaTeX and BibTeX workflow, then save the PDF as `reports/Project_Report.pdf`. With an existing local TeX installation, `latexmk -pdf main.tex` can be run from the manuscript folder.

The report follows the shared A4, 11-point article layout, margins and section order. The supplied `generic.sty` contains only commented settings, so it adds no active formatting rules. The included file is a minimal comment-free package with the same name.

Rebuilding the notebook with `python3 scripts/build_notebook.py` replaces its saved outputs. Run it only when changing notebook structure, then execute the notebook again.

## Local material

Original supplied papers, extracted text and the assignment list are kept in the ignored `sources` folder. Earlier report versions and obsolete build files are kept in the ignored `local_archive` folder. They are not part of the GitHub submission.
