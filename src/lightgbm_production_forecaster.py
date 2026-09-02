#!/usr/bin/env python3
"""Production-style batch inference for the tuned 1–24 hour traffic model.

The serialized joblib artifact is a versioned bundle containing the fitted
LGBMRegressor, feature order, lags, aggregation rule, training range, package
versions, and validation metrics.

Example
-------
python 04_lightgbm_production_forecaster.py \
    --input ../work/smart_city_traffic_mobility.csv \
    --output forecast.csv

Required runtime packages
-------------------------
pandas, numpy, joblib, lightgbm, scikit-learn
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import sysconfig
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_MODEL_PATH = SCRIPT_PATH.with_name("lightgbm_vehicle_count_24h_model.joblib")
LOGGER = logging.getLogger("traffic_forecaster")


def ensure_macos_openmp() -> None:
    """Re-launch once with a locally available libomp search path on macOS."""
    if sys.platform != "darwin" or os.environ.get("TRAFFIC_FORECAST_OPENMP_READY") == "1":
        return

    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    standard_locations = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path("/opt/local/lib/libomp/libomp.dylib"),
    ]
    if any(path.exists() for path in standard_locations):
        return

    site_packages = Path(sysconfig.get_paths()["purelib"])
    bundled_candidates = [
        site_packages / "torch" / "lib" / "libomp.dylib",
        site_packages / "sklearn" / ".dylibs" / "libomp.dylib",
    ]
    available = next((path for path in bundled_candidates if path.exists()), None)
    if available is None:
        return

    new_environment = os.environ.copy()
    library_dir = str(available.parent)
    new_environment["DYLD_LIBRARY_PATH"] = (
        library_dir if not existing else f"{library_dir}:{existing}"
    )
    new_environment["TRAFFIC_FORECAST_OPENMP_READY"] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
        new_environment,
    )


ensure_macos_openmp()

import joblib  
import numpy as np  
import pandas as pd  


def log_event(event: str, **fields: Any) -> None:
    """Emit a compact JSON log record suitable for collection by a scheduler."""
    payload = {
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    LOGGER.info(json.dumps(payload, default=str, sort_keys=True))


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    try:
        bundle = joblib.load(path)
    except OSError as exc:
        if "libomp" in str(exc).lower():
            raise RuntimeError(
                "LightGBM's OpenMP runtime is unavailable. On macOS run "
                "`brew install libomp`, or launch Python with DYLD_LIBRARY_PATH "
                "pointing to a compatible libomp directory."
            ) from exc
        raise

    required = {
        "bundle_version", "model", "target", "aggregation", "horizon", "lags",
        "feature_columns", "training_profile", "training_end",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    if not str(bundle["bundle_version"]).startswith("1."):
        raise ValueError(f"Unsupported bundle version: {bundle['bundle_version']}")
    return bundle


def load_and_validate_history(
    csv_path: Path,
    bundle: dict[str, Any],
    allow_partial_panel: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw intersection records and return a validated hourly target series."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    target = bundle["target"]
    required_columns = {"timestamp", "intersection_id", target}
    header = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = required_columns.difference(header)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    raw = pd.read_csv(
        csv_path,
        usecols=["timestamp", "intersection_id", target],
        parse_dates=["timestamp"],
    )
    if raw.empty:
        raise ValueError("Input CSV contains no rows.")
    if raw[["timestamp", "intersection_id", target]].isna().any().any():
        raise ValueError("Input contains missing timestamp, intersection_id, or target values.")
    if raw.duplicated(["timestamp", "intersection_id"]).any():
        duplicate_count = int(raw.duplicated(["timestamp", "intersection_id"]).sum())
        raise ValueError(f"Input contains {duplicate_count} duplicate timestamp/intersection pairs.")

    raw[target] = pd.to_numeric(raw[target], errors="raise")
    if not np.isfinite(raw[target]).all():
        raise ValueError("Target contains non-finite values.")
    if bundle.get("clip_lower_bound") == 0.0 and (raw[target] < 0).any():
        raise ValueError("Target contains negative values, which violate the model contract.")

    aggregation = bundle["aggregation"]
    if aggregation not in {"sum", "mean"}:
        raise ValueError(f"Unsupported aggregation rule in bundle: {aggregation}")
    hourly = (
        raw.groupby("timestamp", as_index=False)
        .agg(
            y=(target, aggregation),
            intersections=("intersection_id", "nunique"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    max_lag = max(bundle["lags"])
    if len(hourly) < max_lag:
        raise ValueError(f"At least {max_lag} hourly observations are required; received {len(hourly)}.")

    recent = hourly.tail(max_lag).copy()
    expected_clock = pd.date_range(recent.timestamp.iloc[0], recent.timestamp.iloc[-1], freq="h")
    if not pd.DatetimeIndex(recent.timestamp).equals(expected_clock):
        missing_hours = expected_clock.difference(pd.DatetimeIndex(recent.timestamp))
        raise ValueError(
            f"The most recent {max_lag}-hour window is not continuous; "
            f"missing {len(missing_hours)} hour(s)."
        )

    expected_intersections = bundle.get("expected_intersections")
    incomplete_hours = 0
    if expected_intersections is not None:
        incomplete_hours = int(recent.intersections.ne(expected_intersections).sum())
        if incomplete_hours and not allow_partial_panel:
            raise ValueError(
                f"{incomplete_hours} recent hour(s) do not contain the expected "
                f"{expected_intersections} intersections. Use --allow-partial-panel only "
                "when biased citywide totals are acceptable."
            )

    diagnostics = {
        "raw_rows": int(len(raw)),
        "hourly_rows": int(len(hourly)),
        "history_start": hourly.timestamp.iloc[0].isoformat(),
        "history_end": hourly.timestamp.iloc[-1].isoformat(),
        "latest_intersections": int(hourly.intersections.iloc[-1]),
        "incomplete_recent_hours": incomplete_hours,
    }
    return hourly, diagnostics


def calendar_features(timestamp: pd.Timestamp) -> dict[str, float | int]:
    hour, day_of_week = timestamp.hour, timestamp.dayofweek
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend": int(day_of_week >= 5),
    }


def make_forecast_features(
    hourly: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Reproduce the exact training feature contract for horizons 1 through 24."""
    values = hourly.y.to_numpy(dtype=float)
    origin = len(values)
    horizon = int(bundle["horizon"])
    # Generate one extra point and drop the observed endpoint. This avoids
    # ambiguous datetime arithmetic across pandas/NumPy version combinations.
    future_dates = pd.date_range(
        start=pd.Timestamp(hourly.timestamp.iloc[-1]),
        periods=horizon + 1,
        freq="h",
    )[1:]

    rows: list[dict[str, float | int]] = []
    for step, forecast_time in enumerate(future_dates, start=1):
        row = {
            f"lag_{lag}": float(values[origin - int(lag)])
            for lag in bundle["lags"]
        }
        row.update(calendar_features(forecast_time))
        row["horizon"] = step
        rows.append(row)

    features = pd.DataFrame(rows)
    expected_columns = list(bundle["feature_columns"])
    missing = set(expected_columns).difference(features.columns)
    extra = set(features.columns).difference(expected_columns)
    if missing or extra:
        raise ValueError(f"Feature-schema mismatch. Missing={sorted(missing)}; extra={sorted(extra)}")
    features = features[expected_columns]
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Forecast features contain missing or non-finite values.")
    return features, future_dates


def drift_diagnostics(hourly: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    """Return simple input-shift indicators for monitoring, not automatic retraining."""
    profile = bundle["training_profile"]
    recent = hourly.y.tail(min(168, len(hourly))).astype(float)
    standard_deviation = max(float(profile["hourly_std"]), 1e-9)
    recent_mean_z = (float(recent.mean()) - float(profile["hourly_mean"])) / standard_deviation
    outside_reference_share = float(
        ((recent < profile["hourly_p01"]) | (recent > profile["hourly_p99"])).mean()
    )
    warnings_found: list[str] = []
    if abs(recent_mean_z) > 1.0:
        warnings_found.append("Recent 168-hour mean is more than one training standard deviation from the training mean.")
    if outside_reference_share > 0.10:
        warnings_found.append("More than 10% of recent values fall outside the training 1st–99th percentile range.")
    return {
        "recent_168h_mean": float(recent.mean()),
        "recent_mean_z_score": float(recent_mean_z),
        "outside_training_p01_p99_share": outside_reference_share,
        "warnings": warnings_found,
    }


def write_forecast_atomically(forecast: pd.DataFrame, output_path: Path) -> None:
    """Write CSV or JSON through a temporary file, then replace the final path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise ValueError("Output extension must be .csv or .json")
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if suffix == ".csv":
        forecast.to_csv(temporary, index=False)
    else:
        forecast.to_json(temporary, orient="records", date_format="iso", indent=2)
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a leakage-safe 1–24 hour citywide traffic forecast."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw intersection-level CSV history.")
    parser.add_argument("--output", required=True, type=Path, help="Destination .csv or .json file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Versioned joblib model bundle.")
    parser.add_argument(
        "--allow-partial-panel",
        action="store_true",
        help="Allow recent hours with fewer intersections than training (may bias totals).",
    )
    parser.add_argument(
        "--max-data-age-hours",
        type=float,
        default=None,
        help="Optional maximum age of the latest input timestamp relative to UTC now.",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    started = datetime.now(timezone.utc)

    bundle = load_model_bundle(args.model.resolve())
    log_event(
        "model_loaded",
        model=bundle.get("model_name"),
        bundle_version=bundle["bundle_version"],
        trained_through=bundle["training_end"],
        validation=bundle.get("validation"),
    )

    hourly, input_diagnostics = load_and_validate_history(
        args.input.resolve(), bundle, args.allow_partial_panel
    )
    if args.max_data_age_hours is not None:
        latest = pd.Timestamp(hourly.timestamp.iloc[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600
        if age_hours > args.max_data_age_hours:
            raise ValueError(
                f"Latest input is {age_hours:,.1f} hours old; limit is {args.max_data_age_hours:,.1f}."
            )
        input_diagnostics["data_age_hours"] = age_hours

    drift = drift_diagnostics(hourly, bundle)
    log_event("input_validated", **input_diagnostics, drift=drift)

    features, future_dates = make_forecast_features(hourly, bundle)
    predictions = np.asarray(bundle["model"].predict(features), dtype=float)
    lower_bound = bundle.get("clip_lower_bound")
    if lower_bound is not None:
        predictions = np.clip(predictions, float(lower_bound), None)

    forecast = pd.DataFrame(
        {
            "forecast_timestamp": future_dates,
            "horizon_hours": np.arange(1, len(future_dates) + 1),
            "prediction": predictions,
            "model_name": bundle.get("model_name", "LGBMRegressor"),
            "bundle_version": bundle["bundle_version"],
            "history_through": hourly.timestamp.iloc[-1],
        }
    )
    if bundle["target"] == "vehicle_count":
        forecast["prediction_rounded"] = np.rint(predictions).astype(int)

    write_forecast_atomically(forecast, args.output.resolve())
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log_event(
        "forecast_written",
        output=str(args.output.resolve()),
        forecast_rows=int(len(forecast)),
        forecast_start=future_dates[0].isoformat(),
        forecast_end=future_dates[-1].isoformat(),
        prediction_min=float(predictions.min()),
        prediction_max=float(predictions.max()),
        elapsed_seconds=elapsed,
    )
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
        log_event("forecast_failed", error_type=type(exc).__name__, error=str(exc))
        raise
