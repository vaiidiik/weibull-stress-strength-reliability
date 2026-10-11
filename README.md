# DA2141 project 3

Vaidik Maheshwari (250150045) and Anant Gupta (250150004)

Open `Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb` in Jupyter and run all cells from this folder. Saved outputs include the generated graphs and computed tables. The primary implementation is self-contained in the notebook; `analysis.py` contains the same computational functions for reuse.

The project covers Weibull distribution properties, shared-stress system reliability, block progressive hybrid censoring, upper k-records, unrestricted and restricted MLEs, a dependent joint prior, both Bayesian cases, interval estimation, 48 simulation scenarios with 300 replications each, published-data validation, sensitivity and posterior diagnostics.

## Deliverables

- `reports/Project_Report.pdf`: formal project report with derivations, results and figures.
- `reports/derivations.md`: complete mathematical development and assumptions.
- `reports/literature_review.md`: source-use report for all supplied articles and assigned reference links.
- `reports/results_discussion.md`: discussion based on executed calculations.
- `reports/Notebook_Report.html`: readable notebook export with retained outputs.
- `data/`: published measurement transcriptions, disjoint allocations, censored strength data and retained records.
- `results/`: raw simulations, summaries, intervals, fit checks, numerical audits and posterior chains.
- `figures/`: every plot as PNG and PDF.
- `sources/text/`: page-numbered extracted text from the five original PDFs.

## Reproduction

Use Python 3.11 or later with the packages in `requirements.txt`. The completed run used Python 3.14. The original PDFs and CSV are retained. No additional datasets are downloaded during execution.

Run the notebook through Jupyter, or execute `python3 run_notebook.py`. `build_notebook.py` rebuilds the unexecuted notebook source and therefore replaces its saved outputs; use it only when changing the notebook structure. After notebook execution, run `python3 render_readable_figures.py` followed by `python3 create_report.py` to build the formal PDF with enlarged individual graph panels and typeset equations. The report renderer uses the system Arial fonts on macOS. The complete run includes 119,400 inner restricted-bootstrap resamples and may take several minutes depending on hardware.

All Python code is free of comments. Explanations are in Markdown cells and reports. Fixed seeds reproduce the measurements retained by the retrospective design and the numerical study.

## Interpretation

The applications are same-unit laboratory breaking-strength comparisons, not measured in-service stress experiments. The calibrated 15 mm jute goodness-of-fit test detects a Weibull limitation; additional published carbon-fibre data provide a compatible validation. Rate-order restrictions are assumptions, and the simulation explicitly shows their consequences when false. The SAGE reference was available only at abstract level because its full text was not supplied and the publisher full-text request was inaccessible; this limitation is recorded in the literature review.
