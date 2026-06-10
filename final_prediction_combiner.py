from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------
def _s3_parse_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError(f"Expected S3 URI starting with s3://, got: {s3_uri}")
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


def _s3_list_objects(s3_client, bucket: str, prefix: str) -> List[dict]:
    out: List[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def _s3_find_latest_matching_object(
    s3_client,
    bucket: str,
    prefix: str,
    include_substrings: Iterable[str],
    prefer_latest_name: Optional[str] = None,
) -> str:
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    if prefer_latest_name:
        candidate = f"{prefix}{prefer_latest_name}"
        if _s3_object_exists(s3_client, bucket, candidate):
            return candidate

    include_substrings = tuple(s.lower() for s in include_substrings)
    objects = _s3_list_objects(s3_client, bucket, prefix)

    filtered = []
    for obj in objects:
        key = obj["Key"]
        name = key.split("/")[-1].lower()
        if not name.endswith(".csv"):
            continue
        if all(sub in name for sub in include_substrings):
            filtered.append(obj)

    if not filtered:
        raise RuntimeError(
            f"No matching CSV found under s3://{bucket}/{prefix} with parts {include_substrings}"
        )

    filtered.sort(key=lambda x: x["LastModified"], reverse=True)
    return filtered[0]["Key"]


def _s3_download_to_temp(s3_client, bucket: str, key: str) -> str:
    suffix = Path(key).suffix or ".csv"
    fd, tmp_path = tempfile.mkstemp(prefix="combine_", suffix=suffix)
    os.close(fd)
    s3_client.download_file(bucket, key, tmp_path)
    return tmp_path


def _s3_upload_file(s3_client, local_path: str, bucket: str, key: str) -> None:
    s3_client.upload_file(local_path, bucket, key)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------
def make_timestamp_str(use_utc: bool = True) -> str:
    dt = datetime.now(timezone.utc) if use_utc else datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def add_timestamp_to_path(output_path: str, stamp: str) -> str:
    p = Path(output_path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_{stamp}{p.suffix}"))
    return output_path + f"_{stamp}"


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_df_to_temp_csv(df: pd.DataFrame, prefix: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    return tmp_path


# ---------------------------------------------------------------------
# Model sources
# ---------------------------------------------------------------------
@dataclass
class ModelSource:
    alias: str
    metrics_s3_uri: str
    predictions_s3_uri: str
    metrics_latest_name: str
    predictions_latest_name: str


def default_sources(bucket: str) -> List[ModelSource]:
    base = f"s3://{bucket}/datasets/predictions"
    return [
        ModelSource(
            alias="random_forest",
            metrics_s3_uri=f"{base}/",
            predictions_s3_uri=f"{base}/",
            metrics_latest_name="unified_next30_metrics_LATEST.csv",
            predictions_latest_name="unified_next30_predictions_LATEST.csv",
        ),
        ModelSource(
            alias="xgboost",
            metrics_s3_uri=f"{base}/xgboost/",
            predictions_s3_uri=f"{base}/xgboost/",
            metrics_latest_name="xgboost_next30_metrics_LATEST.csv",
            predictions_latest_name="xgboost_next30_predictions_LATEST.csv",
        ),
        ModelSource(
            alias="lstm",
            metrics_s3_uri=f"{base}/lstm/",
            predictions_s3_uri=f"{base}/lstm/",
            metrics_latest_name="lstm_next30_metrics_LATEST.csv",
            predictions_latest_name="lstm_next30_predictions_LATEST.csv",
        ),
    ]


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_csv_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_metrics_from_s3(s3_client, source: ModelSource) -> Tuple[pd.DataFrame, str]:
    bucket, prefix = _s3_parse_uri(source.metrics_s3_uri)
    key = _s3_find_latest_matching_object(
        s3_client,
        bucket,
        prefix,
        include_substrings=("metrics",),
        prefer_latest_name=source.metrics_latest_name,
    )
    local_path = _s3_download_to_temp(s3_client, bucket, key)
    df = pd.read_csv(local_path)
    df["source_model"] = source.alias
    df["source_metrics_s3_uri"] = f"s3://{bucket}/{key}"
    return df, local_path


def load_predictions_from_s3(s3_client, source: ModelSource) -> Tuple[pd.DataFrame, str]:
    bucket, prefix = _s3_parse_uri(source.predictions_s3_uri)
    key = _s3_find_latest_matching_object(
        s3_client,
        bucket,
        prefix,
        include_substrings=("predictions",),
        prefer_latest_name=source.predictions_latest_name,
    )
    local_path = _s3_download_to_temp(s3_client, bucket, key)
    df = load_csv_any(local_path)
    df["source_predictions_s3_uri"] = f"s3://{bucket}/{key}"
    return df, local_path


# ---------------------------------------------------------------------
# Target mapping
# ---------------------------------------------------------------------
TARGET_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "drought_flag": ["drought_flag_pred"],
    "drought_severity_code": ["drought_severity_code_pred", "drought_severity_pred"],
    "precipitation_sum": ["precipitation_sum_pred"],
    "dust_event": ["dust_event_pred", "dust_event_x_pred"],
}

COMPANION_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "drought_flag": [],
    "drought_severity_code": [],
    "precipitation_sum": ["precipitation_sum_normal", "precip_deficit_pred"],
    "dust_event": ["dust_event_prob", "dust_event_x_prob"],
}

FINAL_STANDARD_COLUMNS: Dict[str, str] = {
    "drought_flag": "drought_flag_pred",
    "drought_severity_code": "drought_severity_pred",
    "precipitation_sum": "precipitation_sum_pred",
    "dust_event": "dust_event_pred",
}


def choose_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------
# Metrics selection
# ---------------------------------------------------------------------
def normalize_model_type(val: object) -> str:
    s = str(val).strip().lower()
    if s in {"classifier", "classification"}:
        return "classification"
    if s in {"regressor", "regression"}:
        return "regression"
    return s


def metric_sort_values(row: pd.Series) -> Tuple:
    target = str(row["target_name"])
    model_type = normalize_model_type(row.get("model_type"))

    if target == "precipitation_sum" or model_type == "regression":
        rmse = row.get("rmse")
        r2 = row.get("r2_score")
        mae = row.get("mae")
        return (
            np.inf if pd.isna(rmse) else float(rmse),
            -(float(r2)) if pd.notna(r2) else np.inf,
            np.inf if pd.isna(mae) else float(mae),
        )

    f1 = row.get("f1_score")
    acc = row.get("accuracy")
    auc = row.get("auc_roc")
    loss = row.get("loss")
    return (
        -(float(f1)) if pd.notna(f1) else np.inf,
        -(float(acc)) if pd.notna(acc) else np.inf,
        -(float(auc)) if pd.notna(auc) else np.inf,
        float(loss) if pd.notna(loss) else np.inf,
    )


def choose_best_metrics_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    work = metrics_df.copy()
    work["target_name"] = work["target_name"].astype(str)
    work["city"] = work["city"].fillna("ALL").astype(str)

    numeric_cols = [
        "accuracy", "loss", "precision", "recall", "f1_score",
        "auc_roc", "r2_score", "mse", "rmse", "mae",
    ]
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    winners = []
    global_targets = {"drought_flag", "drought_severity_code", "precipitation_sum"}

    for target, group in work.groupby("target_name"):
        if target in global_targets:
            grouped_rows = [("ALL", group)]
        else:
            grouped_rows = [(city, city_group) for city, city_group in group.groupby("city")]

        for city, g in grouped_rows:
            valid = g.copy()
            valid["sort_key"] = valid.apply(metric_sort_values, axis=1)
            valid = valid.sort_values("sort_key")
            winners.append(valid.iloc[0].drop(labels=["sort_key"]))

    return pd.DataFrame(winners).reset_index(drop=True)


# ---------------------------------------------------------------------
# Prediction preparation and combination
# ---------------------------------------------------------------------
def prepare_prediction_source(df: pd.DataFrame, source_alias: str) -> pd.DataFrame:
    work = df.copy()
    if "timestamp" not in work.columns or "city" not in work.columns:
        raise ValueError(f"Prediction dataframe for {source_alias} must contain timestamp and city columns.")
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", "city"]).copy()
    work["city"] = work["city"].astype(str)

    rename_map = {}
    for target, candidates in TARGET_COLUMN_CANDIDATES.items():
        existing = choose_existing_column(work, candidates)
        if existing:
            rename_map[existing] = f"{FINAL_STANDARD_COLUMNS[target]}__{source_alias}"

    for companions in COMPANION_COLUMN_CANDIDATES.values():
        for col in companions:
            if col in work.columns:
                rename_map[col] = f"{col}__{source_alias}"

    keep = ["city", "timestamp"] + list(rename_map.keys())
    out = work[keep].rename(columns=rename_map).copy()
    return out


def merge_prediction_sources(prepared_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not prepared_dfs:
        raise ValueError("No prediction dataframes provided.")
    merged = prepared_dfs[0].copy()
    for df in prepared_dfs[1:]:
        merged = merged.merge(df, on=["city", "timestamp"], how="outer")
    return merged.sort_values(["city", "timestamp"]).reset_index(drop=True)


def apply_global_selection(result_df: pd.DataFrame, merged_predictions: pd.DataFrame, winners_df: pd.DataFrame, target: str) -> None:
    row = winners_df[winners_df["target_name"] == target].iloc[0]
    source = row["source_model"]
    selected_col = f"{FINAL_STANDARD_COLUMNS[target]}__{source}"
    if selected_col not in merged_predictions.columns:
        raise KeyError(f"Selected column '{selected_col}' was not found in merged predictions.")

    result_df[FINAL_STANDARD_COLUMNS[target]] = merged_predictions[selected_col]
    result_df[f"{target}_source_model"] = source

    for companion in COMPANION_COLUMN_CANDIDATES.get(target, []):
        source_col = f"{companion}__{source}"
        if source_col in merged_predictions.columns:
            result_df[companion] = merged_predictions[source_col]


def apply_city_selection(result_df: pd.DataFrame, merged_predictions: pd.DataFrame, winners_df: pd.DataFrame, target: str) -> None:
    final_col = FINAL_STANDARD_COLUMNS[target]
    result_df[final_col] = np.nan
    result_df[f"{target}_source_model"] = None

    for _, row in winners_df[winners_df["target_name"] == target].iterrows():
        city = row["city"]
        source = row["source_model"]
        source_col = f"{final_col}__{source}"
        if source_col not in merged_predictions.columns:
            continue

        mask = result_df["city"].astype(str) == str(city)
        result_df.loc[mask, final_col] = merged_predictions.loc[mask, source_col]
        result_df.loc[mask, f"{target}_source_model"] = source

        for companion in COMPANION_COLUMN_CANDIDATES.get(target, []):
            source_companion_col = f"{companion}__{source}"
            if source_companion_col in merged_predictions.columns:
                if companion not in result_df.columns:
                    result_df[companion] = np.nan
                result_df.loc[mask, companion] = merged_predictions.loc[mask, source_companion_col]


def compute_drought_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "drought_flag_pred" not in df.columns:
        return df

    out = df.sort_values(["city", "timestamp"]).copy()
    out["drought_duration_pred"] = 0

    for city, idx in out.groupby("city").groups.items():
        idx_list = list(idx)
        flags = pd.to_numeric(out.loc[idx_list, "drought_flag_pred"], errors="coerce").fillna(0).astype(int).to_numpy()
        durations = np.zeros_like(flags, dtype=int)
        n = len(flags)

        for i in range(n):
            if flags[i] == 0:
                durations[i] = 0
            else:
                count = 0
                j = i
                while j < n and flags[j] == 1:
                    count += 1
                    j += 1
                durations[i] = count

        out.loc[idx_list, "drought_duration_pred"] = durations

    return out


def build_combined_predictions(merged_predictions: pd.DataFrame, winners_df: pd.DataFrame) -> pd.DataFrame:
    result = merged_predictions.loc[:, ["city", "timestamp"]].copy()

    apply_global_selection(result, merged_predictions, winners_df, "drought_flag")
    apply_global_selection(result, merged_predictions, winners_df, "drought_severity_code")
    apply_global_selection(result, merged_predictions, winners_df, "precipitation_sum")
    apply_city_selection(result, merged_predictions, winners_df, "dust_event")

    if "precipitation_sum_pred" in result.columns and "precipitation_sum_normal" in result.columns:
        pred = pd.to_numeric(result["precipitation_sum_pred"], errors="coerce")
        normal = pd.to_numeric(result["precipitation_sum_normal"], errors="coerce")
        result["precip_deficit_pred"] = normal - pred

    result = compute_drought_duration(result)
    return result.sort_values(["city", "timestamp"]).reset_index(drop=True)


def build_winner_summary(winners_df: pd.DataFrame) -> pd.DataFrame:
    summary = winners_df.copy()
    keep_cols = [
        "target_name", "city", "source_model", "model_name", "model_type",
        "f1_score", "accuracy", "auc_roc", "loss",
        "rmse", "r2_score", "mae", "source_metrics_s3_uri",
    ]
    for c in keep_cols:
        if c not in summary.columns:
            summary[c] = None
    return summary[keep_cols].sort_values(["target_name", "city"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combine RF, XGBoost, and LSTM prediction datasets using the best metrics per target."
    )
    parser.add_argument("--s3_region", type=str, default="eu-north-1")
    parser.add_argument("--bucket", type=str, default="ibrahim1995-dust-datasets")
    parser.add_argument(
        "--combined_output_s3_uri",
        type=str,
        default="s3://ibrahim1995-dust-datasets/datasets/predictions/combined/",
    )
    parser.add_argument("--upload_to_s3", action="store_true", default=True)
    parser.add_argument("--no_upload_to_s3", action="store_false", dest="upload_to_s3")
    parser.add_argument("--write_s3_latest", action="store_true", default=True)
    parser.add_argument("--no_write_s3_latest", action="store_false", dest="write_s3_latest")
    parser.add_argument("--save_local_outputs", action="store_true")
    parser.add_argument("--timestamp_tz", type=str, choices=["utc", "local"], default="utc")
    parser.add_argument("--output", type=str, default="combined_best_predictions.csv")
    parser.add_argument("--winners_output", type=str, default="combined_best_model_selection.csv")
    parser.add_argument("--latest_output", type=str, default="combined_best_predictions_LATEST.csv")
    parser.add_argument("--latest_winners_output", type=str, default="combined_best_model_selection_LATEST.csv")
    parser.add_argument("--s3_latest_name", type=str, default="combined_best_predictions_LATEST.csv")
    parser.add_argument("--s3_latest_winners_name", type=str, default="combined_best_model_selection_LATEST.csv")
    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.s3_region)
    sources = default_sources(args.bucket)

    temp_files: List[str] = []
    cleanup_paths: List[str] = []
    metrics_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []

    print("[INFO] Loading latest metrics and predictions...")
    for source in sources:
        print(f"[INFO] Source: {source.alias}")
        metrics_df, metrics_tmp = load_metrics_from_s3(s3_client, source)
        pred_df, pred_tmp = load_predictions_from_s3(s3_client, source)
        temp_files.extend([metrics_tmp, pred_tmp])
        metrics_frames.append(metrics_df)
        prediction_frames.append(prepare_prediction_source(pred_df, source.alias))

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    winners = choose_best_metrics_rows(all_metrics)
    winner_summary = build_winner_summary(winners)

    print("[INFO] Winner summary:")
    print(winner_summary)

    merged_predictions = merge_prediction_sources(prediction_frames)
    combined_predictions = build_combined_predictions(merged_predictions, winners)

    stamp = make_timestamp_str(use_utc=args.timestamp_tz == "utc")
    timestamped_output = add_timestamp_to_path(args.output, stamp)
    timestamped_winners_output = add_timestamp_to_path(args.winners_output, stamp)

    if args.save_local_outputs:
        ensure_parent_dir(timestamped_output)
        ensure_parent_dir(timestamped_winners_output)
        combined_predictions.to_csv(timestamped_output, index=False)
        winner_summary.to_csv(timestamped_winners_output, index=False)

        ensure_parent_dir(args.latest_output)
        ensure_parent_dir(args.latest_winners_output)
        combined_predictions.to_csv(args.latest_output, index=False)
        winner_summary.to_csv(args.latest_winners_output, index=False)

        local_pred = timestamped_output
        local_winners = timestamped_winners_output
        local_latest_pred = args.latest_output
        local_latest_winners = args.latest_winners_output
    else:
        local_pred = write_df_to_temp_csv(combined_predictions, "combined_predictions_")
        local_winners = write_df_to_temp_csv(winner_summary, "combined_winners_")
        cleanup_paths.extend([local_pred, local_winners])

        if args.write_s3_latest:
            local_latest_pred = write_df_to_temp_csv(combined_predictions, "combined_predictions_latest_")
            local_latest_winners = write_df_to_temp_csv(winner_summary, "combined_winners_latest_")
            cleanup_paths.extend([local_latest_pred, local_latest_winners])
        else:
            local_latest_pred = local_pred
            local_latest_winners = local_winners

    if args.upload_to_s3:
        out_bucket, out_prefix = _s3_parse_uri(args.combined_output_s3_uri)
        if out_prefix and not out_prefix.endswith("/"):
            out_prefix += "/"

        pred_key = f"{out_prefix}{Path(timestamped_output).name}"
        winners_key = f"{out_prefix}{Path(timestamped_winners_output).name}"

        print(f"[INFO] Uploading combined predictions to s3://{out_bucket}/{pred_key}")
        _s3_upload_file(s3_client, local_pred, out_bucket, pred_key)

        print(f"[INFO] Uploading winner summary to s3://{out_bucket}/{winners_key}")
        _s3_upload_file(s3_client, local_winners, out_bucket, winners_key)

        if args.write_s3_latest:
            latest_pred_key = f"{out_prefix}{args.s3_latest_name}"
            latest_winners_key = f"{out_prefix}{args.s3_latest_winners_name}"

            print(f"[INFO] Uploading stable latest combined predictions to s3://{out_bucket}/{latest_pred_key}")
            _s3_upload_file(s3_client, local_latest_pred, out_bucket, latest_pred_key)

            print(f"[INFO] Uploading stable latest winner summary to s3://{out_bucket}/{latest_winners_key}")
            _s3_upload_file(s3_client, local_latest_winners, out_bucket, latest_winners_key)

            manifest = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "combined_predictions_s3_uri": f"s3://{out_bucket}/{pred_key}",
                "combined_winners_s3_uri": f"s3://{out_bucket}/{winners_key}",
                "combined_predictions_latest_s3_uri": f"s3://{out_bucket}/{latest_pred_key}",
                "combined_winners_latest_s3_uri": f"s3://{out_bucket}/{latest_winners_key}",
                "sources": [
                    {
                        "alias": s.alias,
                        "metrics_s3_uri": s.metrics_s3_uri,
                        "predictions_s3_uri": s.predictions_s3_uri,
                    }
                    for s in sources
                ],
            }
            fd, manifest_path = tempfile.mkstemp(prefix="combined_manifest_", suffix=".json")
            os.close(fd)
            Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            cleanup_paths.append(manifest_path)
            manifest_key = f"{out_prefix}manifest.json"
            print(f"[INFO] Uploading manifest to s3://{out_bucket}/{manifest_key}")
            _s3_upload_file(s3_client, manifest_path, out_bucket, manifest_key)

    for p in cleanup_paths + temp_files:
        try:
            if p and Path(p).exists():
                Path(p).unlink()
        except Exception as e:
            print(f"[WARN] Failed to remove temporary file {p}: {e}")

    print("[INFO] Done.")
    print(combined_predictions.head())


if __name__ == "__main__":
    main()
