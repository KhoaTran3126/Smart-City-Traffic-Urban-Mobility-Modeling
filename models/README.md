# Forecasting Models

These joblib bundles forecast **citywide hourly `vehicle_count` 1–24 hours ahead**. Each bundle includes the fitted model, Optuna-selected parameters, feature/training metadata, library versions, and validation results.

| File | Model | Outer-test MAE |
|---|---|---:|
| `lightgbm_vehicle_count_24h_model.joblib` | LightGBM | **514.1** |
| `catboost_vehicle_count_24h_model.joblib` | CatBoost | 537.5 |
| `xgboost_vehicle_count_24h_model.joblib` | XGBoost | 626.2 |
| `patchtst_vehicle_count_24h_model.joblib` | PatchTST | 7,161.4 |

LightGBM is the recommended model based on four untouched 24-hour evaluation windows. Results are specific to this simulated, short-duration dataset.

## Loading a bundle

```python
import joblib

bundle = joblib.load("models/lightgbm_vehicle_count_24h_model.joblib")
model = bundle["model"]
print(bundle["best_params"])
print(bundle["validation"])
```

Tree models require the stored 34-column feature contract. The PatchTST bundle contains a fitted `NeuralForecast` object. See the project forecasting notebooks and production script for complete preprocessing and inference examples.

Install the corresponding libraries before loading (`lightgbm`, `xgboost`, `catboost`, or `neuralforecast`/`torch`). On macOS, LightGBM and XGBoost also require an OpenMP runtime such as `libomp`.

> Only load joblib files from trusted sources; deserialization can execute arbitrary code.
