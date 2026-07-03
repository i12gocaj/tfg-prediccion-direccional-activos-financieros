# Final Degree Project - Directional Prediction of Financial Assets with Machine Learning and Alternative Data

This repository contains the code and artifacts for my Final Degree Project, focused on predicting market direction across several financial assets by combining:

- Technical indicators
- Baseline and machine learning models
- Deep learning models
- Alternative sentiment data (GDELT)

## Author

- Javier Gonzalez Casares

## Objective

The goal is to compare different modelling approaches for predicting the evolution of financial assets (`BTC-USD`, `TSLA`, `NVDA`) across several time horizons, and to evaluate the impact of news sentiment as an explanatory variable.

## Tech Stack

- Python 3.x
- pandas, numpy, scipy
- scikit-learn, xgboost, statsmodels
- tensorflow
- yfinance, requests
- matplotlib, seaborn
- ta

## Project Structure

```text
.
├── main.py                      # Full pipeline: download, preprocessing, experiments, metrics, charts
├── requirements.txt
├── data/
│   ├── raw/                     # Downloaded data
│   └── processed/               # Processed data for modelling
├── src/
│   ├── data/                    # Download and preprocessing
│   ├── models/                  # Baselines, ML, and DL
│   ├── evaluation/              # Temporal validation and metrics
│   └── utils/                   # Reproducibility
├── scripts/                     # Experiment execution and output generation
├── results/                     # Result CSVs and charts
└── memoria/                     # Written thesis (LaTeX), figures, and appendices
```

## Installation

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Run the full workflow:

```bash
python main.py
```

The pipeline performs:

1. Download of financial and sentiment data
2. Preprocessing and feature construction
3. Execution of experiments A-D
4. Calculation of additional metrics
5. Chart generation
6. Export of LaTeX-ready tables

## Results

Main artifacts are generated in:

- `results/`: CSV tables and metrics
- `results/plots/`: visualizations
- `memoria/Imagenes/`: synchronized figures for the written thesis

## Reproducibility

- Global seed fixed at `42` in the experiment scripts.
- The default download date range is defined in `src/data/downloader.py`.

## Thesis Document

The academic documentation and appendices are in the `memoria/` folder.

## Note

Academic project.
