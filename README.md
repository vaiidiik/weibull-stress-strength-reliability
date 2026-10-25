# Weibull stress-strength reliability

**Vaidik Maheshwari (250150045) and Anant Gupta (250150004)**

DA2141, Mehta Family School of Data Science and AI, IIT Guwahati

This project studies a system that works when at least three of five component strengths exceed one common stress. Strength and stress follow Weibull distributions with a common shape. Strength observations are progressively censored in blocks, and stress observations are upper k-records.

Start with the [report](Project_Report.pdf). The [executed notebook](code/Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb) contains the calculations and saved outputs. The [LaTeX source](latex/main.tex) follows the supplied article layout.

## What we did

1. Derived the system reliability and the likelihood for the two observation schemes.
2. Calculated unrestricted and ordered maximum likelihood estimates.
3. Used a dependent gamma-Dirichlet prior to calculate both Bayesian cases.
4. Compared point errors and asymptotic, equal-tail and numerical HPD intervals in 48 simulation settings, with 300 replications each.
5. Analyzed published jute-fibre and carbon-fibre measurements, with fit checks and a reproducible retrospective sampling design.
6. Examined prior, protocol and acquisition-order sensitivity, and checked posterior and numerical calculations.

The report includes pseudocode for data generation, the Monte Carlo comparison and posterior calculation. It also contains the important graphs, parameter and reliability results, retained observations, data insights and limitations. The code itself is kept in the notebook and Python files.

## Repository layout

```text
Project_Report.pdf
README.md
latex/
    main.tex
    generic.sty
    reference.bib
    sections/
    tables/
graphs/
data/
    raw/
    processed/
    results/
code/
    Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb
    analysis.py
    run_notebook.py
    prepare_graphs.py
    prepare_report.py
    build_notebook.py
    requirements.txt
    notes/
```

| Item | Purpose |
|---|---|
| [Report](Project_Report.pdf) | Main document for reading and submission |
| [LaTeX folder](latex) | Complete editable report source, bibliography and tables |
| [Graphs](graphs) | All 13 notebook figure groups, plus the 49 individual panels used in the report |
| [Data](data) | Raw measurements, retained observations and all computed results |
| [Code](code) | Notebook, statistical functions, execution and report preparation scripts |

All graphs are stored in one folder. The report uses enlarged panels from the same figures so that axis labels and legends remain readable. The complete composite plots are also retained as PNG and PDF.

## Main findings

The order assumption is useful in some correctly ordered simulation settings, but it increases bias when false. For this 3-out-of-5 system, the ordered model cannot estimate reliability below one half. This explains its poor results when the true reliability lies below that bound.

Equal-tail and HPD intervals behave differently at the equality boundary. The ordered equal-tail intervals start above one half, while numerical HPD intervals may include the boundary. Both coverage and width are reported; a narrow interval alone is not evidence of a better method.

One jute group fails the calibrated Weibull goodness-of-fit check at 5 percent. Its estimates are therefore reported with a model-fit limitation. The additional carbon groups are compatible with the fitted Weibull model. Both applications compare laboratory breaking-strength populations in common units; they do not use measured operating loads.

## Data and results

See [data/README.md](data/README.md) for sources, scaling and the distinction between original observations, retrospective selection and simulation.

The study has 14,400 simulated datasets and four fitted estimators per dataset. `data/results/simulation_summary.csv` reports point errors for the shape, both rates and reliability. It also reports reliability interval coverage, average width and Monte Carlo uncertainty. Ordered MLE bootstrap coverage is assessed separately in six settings.

Numerical HPD intervals are shortest intervals under an interpolated reliability density, used as HPD approximations for unimodal marginals. Probability mass and grid-refinement checks are saved. They are not labelled as exact intervals.

## Running the calculations

The notebook already contains its outputs. To execute it again, run these commands from the repository root:

```bash
python3 -m pip install -r code/requirements.txt
python3 code/run_notebook.py
```

Python 3.14 was used for the completed run. The implementation cell sets `RUN_SIMULATION=False`, which loads the saved Monte Carlo tables and repeats the other calculations. To regenerate the full simulation, set it to `True` in the notebook and run all cells. The prior, master seed and scenario settings are fixed in the code.

`code/analysis.py` contains the same statistical functions as the notebook implementation cell. Raw records are not treated as independent Weibull observations. The censoring likelihood uses actual withdrawals and terminal survivors.

## Preparing the report

After executing the notebook, refresh the report graphs and tables from the saved results:

```bash
python3 code/prepare_graphs.py
python3 code/prepare_report.py
```

For Overleaf, upload the `latex` and `graphs` folders with their relative structure preserved, then select `latex/main.tex` as the main document. The data and code folders are not needed just to compile the report.

With an existing local TeX installation, run from the repository root:

```bash
latexmk -pdf -cd latex/main.tex
cp latex/main.pdf Project_Report.pdf
```

The document uses the supplied A4, 11-point article format and margins. Its main sections remain Introduction, Maximum Likelihood Estimation, Bayesian Estimation, Numerical Illustration and Conclusion. The supplied `generic.sty` had no active formatting rules; the included file is a minimal package with the same name.

`code/build_notebook.py` is a maintenance script that rebuilds notebook source and replaces saved outputs. Use it only when changing notebook structure, then execute the notebook again.

## Supporting source notes

Detailed derivations and source-use notes are in [code/notes](code/notes). Original supplied papers and the assignment list remain in the ignored local `sources` folder. Earlier versions are preserved in the ignored `local_archive` folder. Neither folder is uploaded by the Git commands used for this repository.

The assigned SAGE reference was available only at abstract level. Its full text was not supplied, and no inaccessible full-text derivation is claimed.
