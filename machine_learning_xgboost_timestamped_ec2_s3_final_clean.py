
"""
machine_learning_xgboost_timestamped_ec2_s3_with_metrics_s3_only.py

Updated XGBoost version that:
  1) Adapts to the newer cleaned/preprocessed dataset schema.
  2) Uses historical analog sampling instead of last-row persistence.
  3) Uses 70% training / 30% testing by default for all models.
  4) Stores outputs in S3 only by default (temporary local files are deleted).
  5) Uploads to: s3://ibrahim1995-dust-datasets/datasets/cleaned/
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "xgboost is required for this script. Install it with: pip install xgboost"
    ) from e


# ---------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------
def _s3_parse_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError(f"Expected an S3 URI starting with s3://, got: {s3_uri}")
    no_scheme = s3_uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0].strip()
    key = parts[1].lstrip("/") if len(parts) == 2 else ""
    return bucket, key


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _s3_find_latest_object(
    s3_client,
    bucket: str,
    prefix: str,
    prefer_suffix: Tuple[str, ...] = (".parquet", ".csv"),
    exclude_names: Tuple[str, ...] = ("latest.parquet", "latest.csv", "manifest.json"),
) -> str:
    prefix = prefix.lstrip("/")
    paginator = s3_client.get_paginator("list_objects_v2")
    newest_key, newest_time = None, None

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            name = k.split("/")[-1].lower()
            if name in exclude_names:
                continue
            if prefer_suffix and not any(name.endswith(suf) for suf in prefer_suffix):
                continue
            lm = obj["LastModified"]
            if newest_time is None or lm > newest_time:
                newest_time, newest_key = lm, k

    if not newest_key:
        raise RuntimeError(f"No suitable input object found under s3://{bucket}/{prefix}")
    return newest_key


def _s3_download_to_temp(s3_client, bucket: str, key: str) -> str:
    ext = ".parquet" if key.lower().endswith(".parquet") else ".csv" if key.lower().endswith(".csv") else (Path(key).suffix or ".data")
    fd, tmp_path = tempfile.mkstemp(prefix="ml_input_", suffix=ext)
    os.close(fd)
    s3_client.download_file(bucket, key, tmp_path)
    return tmp_path


def _s3_upload_file(s3_client, local_path: str, bucket: str, key: str) -> None:
    s3_client.upload_file(local_path, bucket, key)


# ---------------------------------------------------------------------
# Output naming helpers
# ---------------------------------------------------------------------
def make_timestamp_str(use_utc: bool = True) -> str:
    dt = datetime.now(timezone.utc) if use_utc else datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def add_timestamp_to_path(output_path: str, stamp: str) -> str:
    if "." in Path(output_path).name:
        suffix = Path(output_path).suffix
        return output_path[: -len(suffix)] + f"_{stamp}{suffix}"
    return output_path + f"_{stamp}"


def build_metrics_output_path(prediction_output_path: str) -> str:
    p = Path(prediction_output_path)
    name = p.name
    if "predictions" in name:
        metrics_name = name.replace("predictions", "metrics", 1)
    else:
        metrics_name = p.stem + "_metrics.csv"
    return str(p.with_name(metrics_name))


def write_df_to_temp_csv(df: pd.DataFrame, prefix: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    return tmp_path


# ---------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Parquet file '{path}'. Install pyarrow or fastparquet. Error: {e}"
            )
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' column.")
    if "city" not in df.columns:
        raise ValueError("Dataset must contain 'city' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["city", "timestamp"]).copy()
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    return df




# ---------------------------------------------------------------------
# Numeric feature sanitation helpers
# ---------------------------------------------------------------------
def sanitize_feature_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert model feature columns to a strict float frame safe for XGBoost.
    Handles pandas nullable dtypes (Int64 / Float64), pd.NA, strings that
    should be numeric, and inf/-inf values.
    """
    work = df.loc[:, feature_cols].copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)

    ref = reference_df if reference_df is not None else df
    ref_work = ref.loc[:, feature_cols].copy()
    for col in feature_cols:
        ref_work[col] = pd.to_numeric(ref_work[col], errors="coerce")
    ref_work = ref_work.replace([np.inf, -np.inf], np.nan)

    medians = ref_work.median(numeric_only=True).reindex(feature_cols)

    for col in feature_cols:
        fill_value = medians[col]
        if pd.isna(fill_value):
            fill_value = 0.0
        work[col] = work[col].fillna(fill_value)

    return work.astype(float)


def sanitize_feature_array(
    df: pd.DataFrame,
    feature_cols: List[str],
    reference_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Return a strict float NumPy array safe for XGBoost / scikit-learn."""
    return sanitize_feature_frame(df, feature_cols, reference_df=reference_df).to_numpy(dtype=float)

# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def safe_json_dumps(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def safe_float(val) -> Optional[float]:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def evaluate_classifier_metrics(
    *,
    model_name: str,
    target_name: str,
    y_true,
    y_pred,
    y_proba=None,
    labels=None,
    average: str = "binary",
    city: str = "ALL",
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_true = np.unique(y_true)

    row = {
        "model_name": model_name,
        "model_type": "classifier",
        "target_name": target_name,
        "city": city,
        "n_samples": int(len(y_true)),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": safe_float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": safe_float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "loss": None,
        "auc_roc": None,
        "r2_score": None,
        "mse": None,
        "rmse": None,
        "mae": None,
        "confusion_matrix": safe_json_dumps(confusion_matrix(y_true, y_pred, labels=labels).tolist()),
        "classification_report": safe_json_dumps(classification_report(y_true, y_pred, zero_division=0, output_dict=True)),
    }

    if y_proba is not None:
        try:
            row["loss"] = safe_float(log_loss(y_true, y_proba, labels=labels))
        except Exception:
            row["loss"] = None

        try:
            proba_arr = np.asarray(y_proba)
            if len(unique_true) == 2:
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                    row["auc_roc"] = safe_float(roc_auc_score(y_true, proba_arr[:, 1]))
                elif proba_arr.ndim == 1:
                    row["auc_roc"] = safe_float(roc_auc_score(y_true, proba_arr))
            elif proba_arr.ndim == 2:
                row["auc_roc"] = safe_float(roc_auc_score(y_true, proba_arr, multi_class="ovr", average="weighted"))
        except Exception:
            row["auc_roc"] = None

    return row


def evaluate_regressor_metrics(
    *,
    model_name: str,
    target_name: str,
    y_true,
    y_pred,
    city: str = "ALL",
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model_name": model_name,
        "model_type": "regressor",
        "target_name": target_name,
        "city": city,
        "n_samples": int(len(y_true)),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "loss": safe_float(mse),
        "auc_roc": None,
        "r2_score": safe_float(r2_score(y_true, y_pred)),
        "mse": safe_float(mse),
        "rmse": safe_float(np.sqrt(mse)),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": None,
        "classification_report": None,
    }


# ---------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------
def make_xgb_classifier(num_classes: int) -> XGBClassifier:
    common_kwargs = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if num_classes <= 2:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_kwargs,
        )
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        **common_kwargs,
    )


def make_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


# ---------------------------------------------------------------------
# Calendar / historical analog helpers
# ---------------------------------------------------------------------
def month_to_season_code(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _maybe_set_calendar_fields(row: pd.Series, future_date: pd.Timestamp) -> pd.Series:
    row["timestamp"] = future_date
    if "date" in row.index:
        row["date"] = future_date.normalize().date().isoformat()

    month = future_date.month
    dow = future_date.weekday()
    week = int(future_date.isocalendar().week)
    doy = future_date.dayofyear

    updates = {
        "year": future_date.year,
        "month": month,
        "day_of_year": doy,
        "day_of_week": dow,
        "week_of_year": week,
        "month_drought": month,
        "day_of_week_drought": dow,
        "month_dust": month,
        "day_of_week_dust": dow,
    }
    for col, val in updates.items():
        if col in row.index:
            row[col] = val

    if "season_drought" in row.index:
        if isinstance(row["season_drought"], str):
            row["season_drought"] = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}[month_to_season_code(month)]
        else:
            row["season_drought"] = month_to_season_code(month)

    if "season_drought_code" in row.index:
        row["season_drought_code"] = month_to_season_code(month)

    if "season_dust_code" in row.index:
        row["season_dust_code"] = ((month % 12 + 3) // 3)

    return row


def build_future_by_historical_sampling(
    df: pd.DataFrame,
    days_ahead: int,
    target_cols_to_clear: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["city", "timestamp"]).sort_values(["city", "timestamp"]).reset_index(drop=True)

    future_rows = []
    target_cols_to_clear = target_cols_to_clear or []

    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        last_date = pd.to_datetime(group["timestamp"].max())
        historical_pool = group[group["timestamp"] < last_date].copy()
        if historical_pool.empty:
            historical_pool = group.copy()

        print(f"[INFO] (Historical analog) Building {days_ahead}-day horizon for city={city}, starting from {last_date.date()}")

        for step in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=step)
            m = future_date.month
            d = future_date.day

            candidates = historical_pool[
                (historical_pool["timestamp"].dt.month == m) &
                (historical_pool["timestamp"].dt.day == d)
            ]
            if candidates.empty:
                candidates = historical_pool[historical_pool["timestamp"].dt.month == m]
            if candidates.empty:
                candidates = historical_pool

            sampled = candidates.loc[rng.choice(candidates.index.to_numpy(), size=1)[0]].copy()
            sampled = _maybe_set_calendar_fields(sampled, future_date)

            for col in target_cols_to_clear:
                if col in sampled.index:
                    sampled[col] = np.nan

            future_rows.append(sampled)

    if not future_rows:
        raise RuntimeError("No future records were generated by historical analog sampling.")

    return pd.DataFrame(future_rows).reset_index(drop=True)


# ---------------------------------------------------------------------
# Drought helpers
# ---------------------------------------------------------------------
def select_feature_columns_for_drought(df: pd.DataFrame) -> List[str]:
    exclude_cols = {
        "city",
        "timestamp",
        "date",
        "drought_flag",
        "drought_severity",
        "drought_severity_code",
        "sim_source_drought",
        "data_source_drought",
        "sim_source_dust",
        "data_source_dust",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for drought models.")
    return feature_cols


def select_feature_columns_for_precip(df: pd.DataFrame) -> List[str]:
    if "precipitation_sum" not in df.columns:
        raise ValueError("Expected 'precipitation_sum' column for precipitation model.")
    exclude_cols = {
        "city",
        "timestamp",
        "date",
        "drought_flag",
        "drought_severity",
        "drought_severity_code",
        "precipitation_sum",
        "sim_source_drought",
        "data_source_drought",
        "sim_source_dust",
        "data_source_dust",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for precipitation model.")
    return feature_cols


def compute_climatology_normal_precip(df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    if "precipitation_sum" not in df.columns:
        raise ValueError("Dataset must contain 'precipitation_sum' for climatology.")
    work = df.copy()
    work["doy"] = work["timestamp"].dt.dayofyear
    clim = work.groupby(["city", "doy"])["precipitation_sum"].mean().reset_index()
    return {(row["city"], int(row["doy"])): float(row["precipitation_sum"]) for _, row in clim.iterrows()}


@dataclass
class DroughtModels:
    model_flag: XGBClassifier
    model_severity: XGBClassifier
    model_precip: XGBRegressor
    severity_encoding: Dict[str, int]
    severity_decoding: Dict[int, str]
    feature_cols_drought: List[str]
    feature_cols_precip: List[str]
    climatology_precip: Dict[Tuple[str, int], float]
    metrics_rows: List[Dict[str, object]]


def train_drought_models(df: pd.DataFrame, train_ratio: float = 0.70) -> DroughtModels:
    feature_cols_drought = select_feature_columns_for_drought(df)
    feature_cols_precip = select_feature_columns_for_precip(df)
    metrics_rows: List[Dict[str, object]] = []
    test_size = 1.0 - train_ratio

    target_df = df.dropna(subset=["drought_flag", "drought_severity"]).copy()
    X_drought = sanitize_feature_array(target_df, feature_cols_drought)
    y_flag = target_df["drought_flag"].astype(int).values

    unique_sev = sorted(target_df["drought_severity"].dropna().astype(str).unique().tolist())
    ordered_labels = [lab for lab in ["none", "moderate", "severe", "extreme"] if lab in unique_sev]
    severity_encoding = {lab: i for i, lab in enumerate(ordered_labels)}
    severity_decoding = {i: lab for lab, i in severity_encoding.items()}
    y_sev = target_df["drought_severity"].astype(str).map(severity_encoding).astype(int).values

    X_train, X_val, y_flag_train, y_flag_val, y_sev_train, y_sev_val = train_test_split(
        X_drought,
        y_flag,
        y_sev,
        test_size=test_size,
        random_state=42,
        stratify=y_flag if len(np.unique(y_flag)) > 1 else None,
    )

    flag_sample_weight = compute_sample_weight(class_weight="balanced", y=y_flag_train)
    model_flag = make_xgb_classifier(num_classes=len(np.unique(y_flag_train)))
    model_flag.fit(X_train, y_flag_train, sample_weight=flag_sample_weight)
    y_flag_pred_val = model_flag.predict(X_val)
    y_flag_proba_val = model_flag.predict_proba(X_val)

    flag_metrics = evaluate_classifier_metrics(
        model_name="drought_flag_xgboost_classifier",
        target_name="drought_flag",
        y_true=y_flag_val,
        y_pred=y_flag_pred_val,
        y_proba=y_flag_proba_val,
        labels=sorted(np.unique(y_flag)),
        average="binary" if len(np.unique(y_flag)) == 2 else "weighted",
        city="ALL",
    )
    metrics_rows.append(flag_metrics)

    sev_sample_weight = compute_sample_weight(class_weight="balanced", y=y_sev_train)
    model_severity = make_xgb_classifier(num_classes=len(np.unique(y_sev_train)))
    model_severity.fit(X_train, y_sev_train, sample_weight=sev_sample_weight)
    y_sev_pred_val = model_severity.predict(X_val)
    y_sev_proba_val = model_severity.predict_proba(X_val)

    sev_metrics = evaluate_classifier_metrics(
        model_name="drought_severity_xgboost_classifier",
        target_name="drought_severity",
        y_true=y_sev_val,
        y_pred=y_sev_pred_val,
        y_proba=y_sev_proba_val,
        labels=sorted(np.unique(y_sev)),
        average="weighted",
        city="ALL",
    )
    metrics_rows.append(sev_metrics)

    precip_df = df.dropna(subset=["precipitation_sum"]).copy()
    X_precip = sanitize_feature_array(precip_df, feature_cols_precip)
    y_precip = precip_df["precipitation_sum"].values
    Xp_train, Xp_val, yp_train, yp_val = train_test_split(
        X_precip, y_precip, test_size=test_size, random_state=42
    )

    model_precip = make_xgb_regressor()
    model_precip.fit(Xp_train, yp_train)
    yp_val_pred = model_precip.predict(Xp_val)

    precip_metrics = evaluate_regressor_metrics(
        model_name="precipitation_sum_xgboost_regressor",
        target_name="precipitation_sum",
        y_true=yp_val,
        y_pred=yp_val_pred,
        city="ALL",
    )
    metrics_rows.append(precip_metrics)

    print("\n[INFO] Drought_flag 70/30 test classification report:")
    print(classification_report(y_flag_val, y_flag_pred_val, digits=3, zero_division=0))
    print("\n[INFO] Drought_severity 70/30 test classification report:")
    print(classification_report(y_sev_val, y_sev_pred_val, digits=3, zero_division=0))
    print("\n[INFO] Precipitation 70/30 test metrics:", {
        "r2_score": precip_metrics["r2_score"],
        "mse": precip_metrics["mse"],
        "rmse": precip_metrics["rmse"],
        "mae": precip_metrics["mae"],
    })

    climatology_precip = compute_climatology_normal_precip(df)
    return DroughtModels(
        model_flag=model_flag,
        model_severity=model_severity,
        model_precip=model_precip,
        severity_encoding=severity_encoding,
        severity_decoding=severity_decoding,
        feature_cols_drought=feature_cols_drought,
        feature_cols_precip=feature_cols_precip,
        climatology_precip=climatology_precip,
        metrics_rows=metrics_rows,
    )


def build_future_horizon_for_drought(df: pd.DataFrame, days_ahead: int, random_state: int = 42) -> pd.DataFrame:
    return build_future_by_historical_sampling(
        df=df,
        days_ahead=days_ahead,
        target_cols_to_clear=["drought_flag", "drought_severity", "drought_severity_code"],
        random_state=random_state,
    )


def predict_drought_on_horizon(horizon_df: pd.DataFrame, models: DroughtModels) -> pd.DataFrame:
    X_future_drought = sanitize_feature_array(horizon_df, models.feature_cols_drought, reference_df=horizon_df)
    drought_flag_pred = models.model_flag.predict(X_future_drought)
    drought_severity_code = models.model_severity.predict(X_future_drought)
    drought_severity_pred = [models.severity_decoding.get(int(code), "unknown") for code in drought_severity_code]

    X_future_precip = sanitize_feature_array(horizon_df, models.feature_cols_precip, reference_df=horizon_df)
    precip_pred = models.model_precip.predict(X_future_precip)

    cities = horizon_df["city"].values
    timestamps = horizon_df["timestamp"].values
    normal_precip_list = []
    precip_deficit_list = []

    for c, ts, p_pred in zip(cities, timestamps, precip_pred):
        dt = pd.to_datetime(ts)
        key = (c, int(dt.dayofyear))
        normal = models.climatology_precip.get(key, np.nan)
        normal_precip_list.append(normal)
        precip_deficit_list.append(np.nan if pd.isna(normal) else normal - p_pred)

    out = pd.DataFrame(
        {
            "city": cities,
            "timestamp": timestamps,
            "drought_flag_pred": np.asarray(drought_flag_pred).astype(int),
            "drought_severity_pred": drought_severity_pred,
            "precipitation_sum_pred": precip_pred,
            "precipitation_sum_normal": normal_precip_list,
            "precip_deficit_pred": precip_deficit_list,
        }
    ).sort_values(["city", "timestamp"]).reset_index(drop=True)

    out["drought_duration_pred"] = 0
    for city, idxs in out.groupby("city").groups.items():
        idx_list = list(idxs)
        flags = out.loc[idx_list, "drought_flag_pred"].to_numpy(dtype=int)
        durations = np.zeros_like(flags, dtype=int)
        for i in range(len(flags)):
            if flags[i] == 0:
                durations[i] = 0
            else:
                j = i
                count = 0
                while j < len(flags) and flags[j] == 1:
                    count += 1
                    j += 1
                durations[i] = count
        out.loc[idx_list, "drought_duration_pred"] = durations
    return out


# ---------------------------------------------------------------------
# Dust helpers
# ---------------------------------------------------------------------
def auto_detect_dust_target(df: pd.DataFrame) -> str:
    if "dust_event" in df.columns:
        return "dust_event"
    if "dust_event_x" in df.columns:
        return "dust_event_x"
    raise ValueError(f"Could not find dust target. Available columns: {df.columns.tolist()}")


def safe_get_numeric(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    val = row[col]
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def build_and_predict_dust_future(
    df: pd.DataFrame,
    horizon_days: int,
    train_ratio: float = 0.70,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["city", "timestamp"]).sort_values(["city", "timestamp"]).reset_index(drop=True)

    target_col = auto_detect_dust_target(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        target_col,
        "dust_intensity_code",
        "dust_intensity_level",
        "duration_hours",
        "is_simulated",
        "drought_flag",
        "drought_severity_code",
    }
    feature_cols = [c for c in numeric_cols if c not in exclude]
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found for dust model.")

    metrics_rows: List[Dict[str, object]] = []
    future_records: List[Dict[str, object]] = []
    test_size = 1.0 - train_ratio

    context_cols = [
        "dust_intensity_level",
        "dust_intensity_code",
        "pm10",
        "pm25",
        "aod",
        "temp_mean",
        "humidity_mean",
        "wind_speed_mean",
    ]

    for city in sorted(df["city"].astype(str).unique()):
        city_df = df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)
        city_df = city_df.dropna(subset=[target_col]).copy()
        if len(city_df) < 10:
            print(f"[WARN] City '{city}' has less than 10 labeled records; skipping dust model for this city.")
            continue

        X_all = sanitize_feature_frame(city_df, feature_cols, reference_df=city_df)
        y_all_raw = city_df[target_col].astype(int)
        raw_classes = sorted(y_all_raw.unique().tolist())
        class_to_idx = {label: idx for idx, label in enumerate(raw_classes)}
        idx_to_class = {idx: label for label, idx in class_to_idx.items()}
        y_all = y_all_raw.map(class_to_idx).astype(int)

        stratify_target = y_all if len(np.unique(y_all)) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=42, stratify=stratify_target
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=42, stratify=None
            )

        model = make_xgb_classifier(num_classes=len(raw_classes))
        dust_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=dust_sample_weight)

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        city_metrics = evaluate_classifier_metrics(
            model_name=f"dust_xgboost_classifier_{city.lower().replace(' ', '_')}",
            target_name=target_col,
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            labels=sorted(np.unique(y_all)),
            average="binary" if len(np.unique(y_all)) == 2 else "weighted",
            city=city,
        )
        metrics_rows.append(city_metrics)

        print(f"\n[INFO] Dust 70/30 test classification report for city={city}:")
        print(classification_report(y_test, y_test_pred, digits=3, zero_division=0))

        full_sample_weight = compute_sample_weight(class_weight="balanced", y=y_all)
        model.fit(X_all, y_all, sample_weight=full_sample_weight)

        analog_future_df = build_future_by_historical_sampling(
            df=city_df,
            days_ahead=horizon_days,
            target_cols_to_clear=[target_col],
            random_state=random_state,
        ).sort_values("timestamp").reset_index(drop=True)

        X_future = sanitize_feature_frame(analog_future_df, feature_cols, reference_df=city_df)
        y_pred_future = model.predict(X_future).astype(int)
        y_proba_future = model.predict_proba(X_future)
        positive_proba = y_proba_future[:, 1] if y_proba_future.ndim == 2 and y_proba_future.shape[1] >= 2 else (y_proba_future[:, 0] if y_proba_future.ndim == 2 else None)
        y_pred_future_raw = [int(idx_to_class[int(code)]) for code in y_pred_future]

        for idx, row in analog_future_df.iterrows():
            rec = {
                "city": city,
                "timestamp": row["timestamp"],
                f"{target_col}_pred": int(y_pred_future_raw[idx]),
                f"{target_col}_prob": float(positive_proba[idx]) if positive_proba is not None else np.nan,
            }
            for c in context_cols:
                if c in row.index:
                    rec[c] = safe_get_numeric(row, c)
            future_records.append(rec)

    if not future_records:
        raise RuntimeError("No future dust prediction records generated.")

    out_df = pd.DataFrame(future_records).sort_values(["city", "timestamp"]).reset_index(drop=True)
    return out_df, metrics_rows


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified XGBoost future prediction script for dust and drought."
    )
    parser.add_argument("--input", type=str, default="DUST_RF_cleaned_preprocessed.parquet")
    parser.add_argument("--s3_input_uri", type=str, default="")
    parser.add_argument("--s3_region", type=str, default="eu-north-1")
    parser.add_argument(
        "--s3_output_uri",
        type=str,
        default="s3://ibrahim1995-dust-datasets/datasets/cleaned/",
        help="S3 URI prefix for prediction and metrics outputs.",
    )
    parser.add_argument(
        "--upload_to_s3",
        action="store_true",
        default=True,
        help="Upload predictions and metrics to S3 (default: enabled).",
    )
    parser.add_argument(
        "--no_upload_to_s3",
        action="store_false",
        dest="upload_to_s3",
        help="Disable S3 upload.",
    )
    parser.add_argument(
        "--write_s3_latest",
        action="store_true",
        default=True,
        help="Also overwrite stable LATEST files in S3 for the dashboard (default: enabled).",
    )
    parser.add_argument(
        "--no_write_s3_latest",
        action="store_false",
        dest="write_s3_latest",
        help="Disable writing stable LATEST files in S3.",
    )
    parser.add_argument("--s3_latest_name", type=str, default="unified_next30_predictions_LATEST.csv")
    parser.add_argument("--s3_metrics_latest_name", type=str, default="unified_next30_metrics_LATEST.csv")
    parser.add_argument("--s3_manifest", action="store_true")
    parser.add_argument("--horizon_days", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", type=str, default="unified_next30_predictions.csv")
    parser.add_argument("--metrics_output", type=str, default="")
    parser.add_argument("--no_timestamped_output", action="store_false", dest="timestamped_output")
    parser.set_defaults(timestamped_output=True)
    parser.add_argument("--timestamp_tz", type=str, choices=["utc", "local"], default="utc")
    parser.add_argument(
        "--save_local_outputs",
        action="store_true",
        help="Also keep local CSV files. By default, files are stored in S3 only.",
    )
    parser.add_argument("--write_latest", action="store_true")
    parser.add_argument("--latest_output", type=str, default="unified_next30_predictions_LATEST.csv")
    parser.add_argument("--latest_metrics_output", type=str, default="unified_next30_metrics_LATEST.csv")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be between 0 and 1.")

    local_input_path = args.input
    used_input_s3_uri = ""

    if args.s3_input_uri:
        s3_client = boto3.client("s3", region_name=args.s3_region)
        in_bucket, in_key_or_prefix = _s3_parse_uri(args.s3_input_uri)
        is_prefix = args.s3_input_uri.endswith("/") or in_key_or_prefix.endswith("/") or in_key_or_prefix == ""
        if is_prefix:
            prefix = in_key_or_prefix
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            latest_key = f"{prefix}LATEST.parquet"
            in_key = latest_key if _s3_object_exists(s3_client, in_bucket, latest_key) else _s3_find_latest_object(s3_client, in_bucket, prefix)
        else:
            in_key = in_key_or_prefix
        used_input_s3_uri = f"s3://{in_bucket}/{in_key}"
        print("[INFO] Downloading cleaned input from:", used_input_s3_uri)
        local_input_path = _s3_download_to_temp(s3_client, in_bucket, in_key)
        print("[INFO] Downloaded to local temp file:", local_input_path)

    print("[INFO] Loading unified dataset:", local_input_path)
    df = load_dataset(local_input_path)

    print("\n[INFO] Training drought models with XGBoost...")
    drought_models = train_drought_models(df, train_ratio=args.train_ratio)

    print(f"[INFO] Building drought future horizon for {args.horizon_days} days...")
    drought_horizon = build_future_horizon_for_drought(df, days_ahead=args.horizon_days, random_state=args.random_state)

    print("[INFO] Predicting drought variables on future horizon...")
    drought_predictions = predict_drought_on_horizon(drought_horizon, drought_models)

    print("\n[INFO] Building dust future predictions with XGBoost...")
    dust_predictions, dust_metrics_rows = build_and_predict_dust_future(
        df,
        horizon_days=args.horizon_days,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
    )

    print("\n[INFO] Merging drought and dust predictions into ONE DataFrame...")
    merged = pd.merge(drought_predictions, dust_predictions, on=["city", "timestamp"], how="outer", sort=True)
    merged = merged.sort_values(["city", "timestamp"]).reset_index(drop=True)

    metrics_rows = []
    metrics_rows.extend(drought_models.metrics_rows)
    metrics_rows.extend(dust_metrics_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_column_order = [
        "model_name", "model_type", "target_name", "city", "n_samples",
        "accuracy", "loss", "precision", "recall", "f1_score", "auc_roc",
        "r2_score", "mse", "rmse", "mae", "confusion_matrix", "classification_report",
    ]
    for col in metrics_column_order:
        if col not in metrics_df.columns:
            metrics_df[col] = None
    metrics_df = metrics_df[metrics_column_order]

    stamp = make_timestamp_str(use_utc=(args.timestamp_tz == "utc"))
    if not args.timestamped_output:
        output_path = args.output
        metrics_output_path = args.metrics_output.strip() or build_metrics_output_path(args.output)
    else:
        output_path = add_timestamp_to_path(args.output, stamp)
        metrics_output_path = add_timestamp_to_path(args.metrics_output.strip(), stamp) if args.metrics_output.strip() else build_metrics_output_path(output_path)

    local_artifacts_to_delete = []

    if args.save_local_outputs:
        print("[INFO] Saving merged predictions locally to:", output_path)
        merged.to_csv(output_path, index=False)
        print("[INFO] Saving model metrics locally to:", metrics_output_path)
        metrics_df.to_csv(metrics_output_path, index=False)

        if args.write_latest:
            print("[INFO] Writing stable latest local prediction copy to:", args.latest_output)
            merged.to_csv(args.latest_output, index=False)
            print("[INFO] Writing stable latest local metrics copy to:", args.latest_metrics_output)
            metrics_df.to_csv(args.latest_metrics_output, index=False)

        local_pred_for_upload = output_path
        local_metrics_for_upload = metrics_output_path
        local_latest_pred_for_upload = args.latest_output if args.write_latest else output_path
        local_latest_metrics_for_upload = args.latest_metrics_output if args.write_latest else metrics_output_path
    else:
        print("[INFO] Local output saving is disabled. Files will be stored in S3 only.")
        local_pred_for_upload = write_df_to_temp_csv(merged, "xgb_predictions_")
        local_metrics_for_upload = write_df_to_temp_csv(metrics_df, "xgb_metrics_")
        local_artifacts_to_delete.extend([local_pred_for_upload, local_metrics_for_upload])

        if args.write_s3_latest:
            local_latest_pred_for_upload = write_df_to_temp_csv(merged, "xgb_predictions_latest_")
            local_latest_metrics_for_upload = write_df_to_temp_csv(metrics_df, "xgb_metrics_latest_")
            local_artifacts_to_delete.extend([local_latest_pred_for_upload, local_latest_metrics_for_upload])
        else:
            local_latest_pred_for_upload = local_pred_for_upload
            local_latest_metrics_for_upload = local_metrics_for_upload

    if args.upload_to_s3:
        s3_client = boto3.client("s3", region_name=args.s3_region)
        out_bucket, out_prefix = _s3_parse_uri(args.s3_output_uri)
        if out_prefix and not out_prefix.endswith("/"):
            out_prefix += "/"

        pred_name = Path(output_path).name
        metrics_name = Path(metrics_output_path).name
        s3_pred_key = f"{out_prefix}{pred_name}"
        s3_metrics_key = f"{out_prefix}{metrics_name}"

        print(f"[INFO] Uploading timestamped predictions to s3://{out_bucket}/{s3_pred_key}")
        _s3_upload_file(s3_client, local_pred_for_upload, out_bucket, s3_pred_key)
        print(f"[INFO] Uploading timestamped metrics to s3://{out_bucket}/{s3_metrics_key}")
        _s3_upload_file(s3_client, local_metrics_for_upload, out_bucket, s3_metrics_key)

        if args.write_s3_latest:
            s3_latest_pred_key = f"{out_prefix}{args.s3_latest_name.strip() or 'unified_next30_predictions_LATEST.csv'}"
            s3_latest_metrics_key = f"{out_prefix}{args.s3_metrics_latest_name.strip() or 'unified_next30_metrics_LATEST.csv'}"
            print(f"[INFO] Uploading stable latest predictions to s3://{out_bucket}/{s3_latest_pred_key}")
            _s3_upload_file(s3_client, local_latest_pred_for_upload, out_bucket, s3_latest_pred_key)
            print(f"[INFO] Uploading stable latest metrics to s3://{out_bucket}/{s3_latest_metrics_key}")
            _s3_upload_file(s3_client, local_latest_metrics_for_upload, out_bucket, s3_latest_metrics_key)

        if args.s3_manifest:
            manifest = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_s3_uri": used_input_s3_uri,
                "output_timestamped_s3_uri": f"s3://{out_bucket}/{s3_pred_key}",
                "metrics_timestamped_s3_uri": f"s3://{out_bucket}/{s3_metrics_key}",
                "train_ratio": args.train_ratio,
                "test_ratio": 1.0 - args.train_ratio,
                "future_builder": "historical_analog_same_month_day_random_past_year",
            }
            if args.write_s3_latest:
                manifest["output_latest_s3_uri"] = f"s3://{out_bucket}/{out_prefix}{args.s3_latest_name.strip() or 'unified_next30_predictions_LATEST.csv'}"
                manifest["metrics_latest_s3_uri"] = f"s3://{out_bucket}/{out_prefix}{args.s3_metrics_latest_name.strip() or 'unified_next30_metrics_LATEST.csv'}"
            fd, mpath = tempfile.mkstemp(prefix="xgb_manifest_", suffix=".json")
            os.close(fd)
            Path(mpath).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            mkey = f"{out_prefix}manifest.json"
            print(f"[INFO] Uploading manifest to s3://{out_bucket}/{mkey}")
            _s3_upload_file(s3_client, mpath, out_bucket, mkey)
            local_artifacts_to_delete.append(mpath)

    for tmp_file in local_artifacts_to_delete:
        try:
            if tmp_file and Path(tmp_file).exists():
                Path(tmp_file).unlink()
        except Exception as cleanup_error:
            print(f"[WARN] Could not delete temporary file {tmp_file}: {cleanup_error}")

    print("\n[INFO] Done.")
    print("[INFO] Example merged predictions:")
    print(merged.head())
    print("\n[INFO] Example metrics rows:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
