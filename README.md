# DamDataPredictionAndTypeEvaluation

Reproducible baselines for dam measurement data preprocessing (Wavelet + IsolationForest) and prediction (PatchTST, LSTM, XGBoost, Decision Tree).  
Implements short/long gap handling (≤12 h linear interpolation; >12 h model-based completion).

## Quickstart

```bash
# 1) Create & activate environment
pip install -r requirements.txt
pre-commit install

# 2) (Optional) conda
# conda env create -f environment.yml
# conda activate dam-monitoring

# 3) Run your existing experiment scripts (kept under ./scripts)
python scripts/PatchTST.py
python scripts/LSTM.py
python scripts/XGBoost.py
python scripts/Decision_Tree.py
python scripts/wavelet_and_isoloation_forest.py

# 4) Run tests
pytest
```

> Tip: Enable GitHub Actions (CI) by pushing to `main`. Dependency caching is enabled via `actions/setup-python`.

## Project layout
```
DamDataPredictionAndTypeEvaluation/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml
├─ requirements.txt
├─ environment.yml
├─ .pre-commit-config.yaml
├─ .github/workflows/ci.yml
├─ data/{raw,interim,processed}/
├─ notebooks/
├─ scripts/                 # your original scripts (kept intact)
├─ src/dam_monitoring/
│   ├─ __init__.py
│   ├─ utils/{io.py,metrics.py}
│   └─ preprocess/wavelet_iso.py
├─ tests/
└─ paper/                   # manuscript & docs
```

## How to publish this on GitHub (first time)

```bash
# inside DamDataPredictionAndTypeEvaluation/
git init
git add -A
git commit -m "Initial commit: reproducible scaffold + scripts"
git branch -M main
git remote add origin https://github.com/ryulkr/DamDataPredictionAndTypeEvaluation.git
git push -u origin main
```

Then open your repo on GitHub:
- **Citations**: GitHub will read `CITATION.cff` and show a “Cite this repository” button.
- **DOI**: Link GitHub ↔ Zenodo, then make a Release to mint a DOI.
- **CI**: See the status under the “Actions” tab.

## Data policy
- Do **not** commit heavy or private data. Place files under `data/` and consider Git LFS/DVC if needed.
- Document schema/units/timezone/NA markers in this README (FAIR).

## License
MIT (see `LICENSE`).

