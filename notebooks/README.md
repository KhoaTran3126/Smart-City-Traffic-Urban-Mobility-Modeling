# Notebooks

This folder contains the exploratory analysis and forecasting experiments for the Smart City Traffic & Urban Mobility project.

| Notebook | Purpose |
|---|---|
| `01_smart_city_traffic_EDA_and_clustering.ipynb` | Data-quality assessment, traffic patterns, spatial and environmental analysis, hotspot detection, and intersection clustering. |
| `02_smart_city_traffic_24h_forecasting_GBDT_vs_PatchTST.ipynb` | Leakage-safe 1–24 hour forecasting comparison between GBDT models, seasonal baselines, and PatchTST. |
| `03_tuned_PatchTST_LightGBM_XGBoost_CatBoost_ensembles.ipynb` | Optuna tuning, nested time validation, ensemble evaluation, diagnostics, and final forecasts for LightGBM, XGBoost, CatBoost, and PatchTST. |

## Recommended order

Run the notebooks sequentially from `01` to `03`. Update `DATA_CANDIDATES` near the beginning of each notebook if the CSV is stored elsewhere.

The notebooks require Python 3.10+ and common data-science packages including pandas, NumPy, Polars, Matplotlib, Seaborn, scikit-learn, Optuna, LightGBM, XGBoost, CatBoost, PyTorch, and NeuralForecast.

The tuned evaluation identified **LightGBM** as the strongest model, achieving an outer-test MAE of approximately **514 vehicles** for citywide 1–24 hour forecasts.
