# Smart City Traffic and Urban Mobility Modeling

## Project overview

This project studies traffic behavior across a simulated smart city and builds hourly forecasts for future vehicle demand. The work has two connected goals. The first goal is to explain when, where, and under which conditions congestion becomes most severe. The second goal is to forecast citywide vehicle count from 1 to 24 hours ahead using reproducible time series methods.

The project includes exploratory analysis, intersection clustering, baseline forecasting, gradient boosted tree models, a PatchTST transformer, Optuna tuning, ensemble evaluation, saved model bundles, and a production style forecasting script.

## Dataset

The dataset contains 204,000 hourly records, 100 intersections, 2,040 timestamps, and 47 fields. It covers 1 January 2026 through 26 March 2026. The features describe location, roadway design, traffic flow, speed, queues, delay, weather, incidents, public activity, signal timing, emissions, fuel waste, and congestion.

The supplied dataset description credits Mobeen Fatima as the author. The city and sensor network are simulated. Results therefore demonstrate an analytical workflow and should not be treated as evidence about a real city without external validation.

Initial validation found a complete panel. Every timestamp contains all 100 intersections. Record identifiers and intersection timestamp pairs are unique. The 47 fields contain no missing values, and the main physical range checks passed. This level of completeness is useful for experimentation, but it is cleaner than a typical live sensor system.

## Project files

| Location | Contents |
|---|---|
| `notebooks/01_smart_city_traffic_EDA_and_clustering.ipynb` | Data validation, descriptive analysis, spatial patterns, environmental analysis, hotspot screening, and intersection clustering. |
| `notebooks/02_smart_city_traffic_24h_forecasting_GBDT_vs_PatchTST.ipynb` | Initial 1 to 24 hour comparison among seasonal baselines, gradient boosted trees, and PatchTST. |
| `notebooks/03_tuned_PatchTST_LightGBM_XGBoost_CatBoost_ensembles.ipynb` | Optuna tuning, nested time validation, ensemble assessment, diagnostics, and final forecasts. |
| `models/` | Versioned joblib bundles for LightGBM, XGBoost, CatBoost, and PatchTST. |
| `04_lightgbm_production_forecaster.py` | Batch inference program for the recommended LightGBM model. |

## Exploratory methodology

The exploratory notebook begins with data structure and integrity checks. It verifies timestamp continuity, panel completeness, duplicate records, missing values, and plausible numeric ranges. It then examines distributions through means, medians, standard deviations, upper quantiles, and visual summaries.

Temporal analysis compares hour of day, day of week, weekday status, weekend status, and named peak periods. Spatial analysis compares city zones, road types, and individual intersections. Weather, accidents, construction, public events, emergency vehicles, and signal settings are examined through descriptive group differences. These comparisons describe association. They do not establish that a condition caused the observed outcome.

Pearson correlation measures linear relationships among numeric variables. Phi K measures associations across numeric and categorical variables and can identify relationships that are not strictly linear. Traffic density, speed, queue length, delay, congestion, emissions, and fuel waste are also examined together to show how mobility and environmental performance interact.

Intersection clustering uses long run operating summaries. Features are standardized so that measurement units do not control the result. K means models with two through six clusters are compared using silhouette score. Six clusters produced the strongest score. Principal component analysis provides a two dimensional view for interpretation. A transparent priority score combines congestion, upper queue length, delay, severe congestion share, emissions, and fuel waste to identify locations for further review.

## Main exploratory findings

The highest recurring congestion cell occurred on Monday at 08:00. Downtown Core had the highest zone average congestion score, approximately 52.5. Heavy Rain had the highest average congestion among the observed weather categories. Records with a reported accident had a congestion score approximately 25.5 percent higher than records without an accident.

The six intersection clusters separate distinct operating profiles rather than predicting future outcomes. The first locations identified for operational review were INT_013, INT_012, INT_042, INT_091, and INT_047. These rankings are screening results. Field evidence would be required before changing signals, road design, staffing, or enforcement.

The analysis also supports tracking delay and queue measures together with emissions and fuel waste. An intervention that improves vehicle throughput may still have an environmental cost, so mobility and sustainability measures should be reviewed together.

![Heat map of average congestion by weekday and hour](images/congestion_by_weekday_hour.png)

The hourly pattern is highly structured. Weekday congestion rises sharply during the morning and remains elevated through the afternoon. Weekend demand rises later and changes more gradually. This regularity explains why hourly and weekly lag features are useful for forecasting.

![Average congestion under different weather and accident conditions](images/congestion_context.png)

Heavy rain had the highest weather group average. Accident records also had a visibly higher average congestion score. These comparisons are descriptive and may include differences in time, place, demand, or other conditions.

## Forecasting target and features

The forecasting target is total citywide vehicle count per hour. Counts from the 100 intersections are summed to create one operational demand series. The same code can be adapted to mean network speed, but all reported model results use vehicle count.

The tree models use a direct multi horizon design. One training row represents one forecast origin and one future horizon. Inputs include the previous 24 hourly values, plus lags at 48, 72, 168, and 336 hours. They also include hour, day of week, weekend status, and forecast horizon. Calendar fields describe the future timestamp and are known before the forecast is made.

Realized future weather, accidents, congestion, and speed are excluded. Using them would reveal information that is unavailable at prediction time and would make the evaluation misleading.

PatchTST uses an autoregressive input window of 168 hours. It learns representations from patches of the ordered history and predicts all 24 horizons directly.

## Validation and tuning methodology

Time order is preserved throughout the study. Random train and test splitting is not used.

Three earlier 24 hour windows form the inner validation layer. Optuna uses these windows to select model settings and to estimate ensemble weights. Four later 24 hour windows form the outer evaluation layer. Hyperparameters and ensemble rules are frozen before these outer windows are evaluated.

Each fold uses an expanding history. A model can use only observations that occurred before its forecast origin. This design provides a practical estimate of how the system would have behaved if it had been operated on those historical days.

The tree model search uses eight Optuna trials for each of LightGBM, XGBoost, and CatBoost. PatchTST uses five Optuna trials, 60 training steps during tuning, and 200 steps during final evaluation. These budgets make the notebook practical on a laptop. They are not exhaustive searches.

Four ensemble rules are evaluated. Mean4 uses an equal average of the four model families. Median4 uses their median and is less sensitive to one poor forecast. InverseMAE4 gives more weight to models with lower inner validation error. NNLS_All uses nonnegative least squares across the four models and both seasonal baselines. All ensemble weights are learned before outer evaluation.

## Evaluation metrics

MAE is the primary metric. It is the average absolute difference between actual and predicted vehicle counts, so it can be read directly in vehicles.

RMSE gives more weight to large errors and helps reveal occasional severe misses.

sMAPE expresses error as a symmetric percentage of actual and predicted values.

WAPE divides total absolute error by total actual volume. It is useful for comparing aggregate forecast error with total demand.

MASE compares model error with a historical seasonal reference. A value below one indicates improvement over that reference scale.

## Forecasting results

| Method | MAE | RMSE | sMAPE percent | WAPE percent |
|---|---:|---:|---:|---:|
| LightGBM | 514.1 | 844.2 | 0.664 | 0.613 |
| Median4 ensemble | 520.6 | 868.5 | 0.711 | 0.621 |
| NNLS_All ensemble | 523.7 | 874.2 | 0.668 | 0.625 |
| CatBoost | 537.5 | 896.9 | 0.683 | 0.641 |
| InverseMAE4 ensemble | 544.5 | 855.9 | 0.852 | 0.649 |
| XGBoost | 626.2 | 1,012.4 | 0.976 | 0.747 |
| Same hour last week | 704.0 | 1,126.2 | 0.937 | 0.840 |
| PatchTST | 7,161.4 | 8,846.5 | 19.214 | 8.540 |
| Same hour yesterday | 8,370.2 | 22,416.8 | 11.510 | 9.981 |

![Mean absolute error for models, ensembles, and seasonal baselines](images/forecast_model_comparison.png)

The chart separates methods below and above 1,000 MAE so that differences among the competitive models remain visible. Both panels use the same metric. Lower values indicate more accurate forecasts.

LightGBM produced the lowest outer evaluation MAE. It improved MAE by approximately 27.0 percent compared with the same hour last week baseline. The Median4 ensemble was the strongest ensemble, but it remained slightly less accurate than LightGBM. The main lesson is simple. More complexity did not automatically produce a better forecast.

PatchTST performed much worse in this experiment. The likely reasons are the short single series, limited neural tuning, abrupt rush hour transitions, and the lack of explicit future calendar inputs. The tree models received a stronger feature representation for this problem. PatchTST may be more competitive when trained globally across the 100 intersection series, with more history, equal future information, longer tuning, early stopping, and several random seeds.

The PatchTST result is specific to this dataset and experimental budget. It is not evidence that transformers are generally inferior for time series forecasting.

## Saved models

The `models` folder contains joblib bundles for LightGBM, XGBoost, CatBoost, and PatchTST. Each bundle includes the fitted model, Optuna settings, target and aggregation rules, training dates, data hash, library versions, validation metrics, and feature contract.

LightGBM is the recommended point forecast model because it achieved the lowest untouched outer evaluation MAE. The other bundles are retained for comparison, monitoring, ensemble experiments, and future research.

Joblib files must be loaded only from trusted sources. Deserialization can execute code. Matching or compatible package versions should be used when loading a saved model.

## Production inference

The production script validates the input schema, duplicate intersection records, recent hourly continuity, and intersection completeness. It reproduces the 34 stored LightGBM features, checks simple indicators of input drift, clips vehicle count predictions at zero, emits structured logs, and writes CSV or JSON output through an atomic replacement step.

Example command:

```bash
python 04_lightgbm_production_forecaster.py \
  --input data/smart_city_traffic_mobility.csv \
  --output forecast.csv
```

The input must provide at least 336 consecutive recent hours with `timestamp`, `intersection_id`, and `vehicle_count`. The default validation expects all 100 intersections in every recent hour. Partial panels are rejected because missing intersections would bias the citywide total.

On macOS, LightGBM and XGBoost require an OpenMP runtime such as `libomp`. The production script can use a compatible copy bundled with PyTorch or scikit learn when one is available.

## Reproduction

Use Python 3.10 or newer. Install NumPy, pandas, Polars, Matplotlib, Seaborn, SciPy, scikit learn, statsmodels, Phi K, Optuna, LightGBM, XGBoost, CatBoost, PyTorch, NeuralForecast, joblib, Jupyter, and notebook execution tools.

Place `smart_city_traffic_mobility.csv` in the project data location, or update `DATA_CANDIDATES` near the beginning of each notebook. Run the notebooks in numeric order. Notebook 01 performs exploratory analysis. Notebook 02 introduces the forecasting design. Notebook 03 performs final tuning, ensemble assessment, model diagnostics, and next day forecasting.

All delivered notebooks were executed from beginning to end. Notebook 01 contains 19 figures. Notebook 02 contains 8 figures. Notebook 03 contains 16 figures. The final runs contained no failed cells, no unexecuted code cells, and no embedded tracebacks.

The saved LightGBM production bundle was retrained on all 2,040 hourly observations. This created 40,620 supervised rows across 24 horizons and 34 features. Its production forecasts matched the final notebook values within 0.001 vehicles. The XGBoost, CatBoost, and PatchTST bundles were also reloaded in fresh processes and reproduced their notebook forecasts within 0.001.

## Limitations

The data are simulated and unusually complete. Real traffic sensors experience outages, duplication, delayed messages, calibration changes, and location changes.

The outer evaluation contains four days, or 96 hourly forecasts per method. This is enough for a controlled comparison, but not enough to establish performance across seasons, holidays, rare incidents, or policy changes.

The citywide aggregate can hide local problems. A good total forecast may still miss an important intersection or zone.

The Optuna budgets are modest. PatchTST in particular would benefit from broader tuning, longer training, several random seeds, and a larger collection of related series.

The model disagreement range is a warning signal, not a calibrated prediction interval. It should not be interpreted as a guaranteed probability range.

## Recommended next steps

The next study should train a global model across all 100 intersection series and reconcile local forecasts with the citywide total. Evaluation should cover several months and include daily forecast origins across weekdays, weekends, holidays, weather events, and incidents.

Future models may include scheduled public events, planned construction, and weather forecasts when those inputs are genuinely available before prediction time. Rolling conformal methods can provide calibrated uncertainty intervals. Production monitoring should track error by horizon, traffic regime, day type, location, and model disagreement.

Before operational use, the project should also test timezone handling, daylight saving changes, missing hours, partial sensor panels, duplicated messages, schema changes, and delayed data.

## Conclusion

The project shows that rigorous validation matters more than model novelty. A carefully tuned LightGBM model with transparent lags and calendar features outperformed the transformer, the seasonal baselines, and every tested ensemble. At the same time, the exploratory analysis identified clear temporal, spatial, weather, and incident patterns that can guide further investigation.

The results support LightGBM as the current forecasting choice. They also define a clear path for improving spatial detail, uncertainty measurement, and real world robustness.
