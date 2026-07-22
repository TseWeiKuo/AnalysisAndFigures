"""Configuration helpers shared by tracking-QC-aware plotting workflows."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackingQCConfig:
    """Tracking-QC options without coupling them to a plotting workflow."""

    enabled: bool = False
    error_thresholds: dict | None = None
    error_max: float = 50
    score_min: float = 0.8
    min_cameras: int = 2
    max_interp_gap_frames: int = 5
    min_valid_fraction: float = 0.7
    require_score: bool = False

    @property
    def max_invalid_fraction(self):
        return 1.0 - float(self.min_valid_fraction)

    def output_metadata(self):
        """Return the common QC fields written to result dataframes."""
        return {
            "Apply_Tracking_QC": bool(self.enabled),
            "Min_Cameras": self.min_cameras if self.enabled else np.nan,
            "Error_Max": self.error_max if self.enabled else np.nan,
            "Score_Min": self.score_min if self.enabled else np.nan,
            "Max_Interp_Gap_Frames": (
                self.max_interp_gap_frames if self.enabled else np.nan
            ),
            "Min_Valid_Fraction": (
                self.min_valid_fraction if self.enabled else np.nan
            ),
            "Max_Invalid_Fraction": (
                self.max_invalid_fraction if self.enabled else np.nan
            ),
            "Require_Score": bool(self.require_score) if self.enabled else False,
        }


def from_legacy_arguments(
        apply_tracking_qc=False,
        tracking_error_thresholds=None,
        min_cameras=2,
        max_interp_gap_frames=5,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Build a config while public plotting methods retain existing arguments."""
    return TrackingQCConfig(
        enabled=apply_tracking_qc,
        error_thresholds=tracking_error_thresholds,
        error_max=error_max,
        score_min=score_min,
        min_cameras=min_cameras,
        max_interp_gap_frames=max_interp_gap_frames,
        min_valid_fraction=min_valid_fraction,
        require_score=require_score,
    )


def _resolve_error_threshold(error_thresholds, keypoint, default_error_max=50):
    if error_thresholds is None:
        return default_error_max
    if isinstance(error_thresholds, dict):
        value = error_thresholds.get(keypoint, default_error_max)
    else:
        value = error_thresholds
    if value is None or pd.isna(value):
        return default_error_max
    return float(value)


def _optional_score_array(point):
    for attr in ("score", "likelihood", "confidence", "probability"):
        if hasattr(point, attr):
            values = getattr(point, attr)
            if values is not None:
                return np.asarray(values, dtype=float), getattr(point, "score_column", attr)
    return None, None


def point_invalid_components(
        point,
        keypoint,
        min_cameras=2,
        error_thresholds=None,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Return frame-wise invalid components for one 3D keypoint trace."""
    x = np.asarray(point.x_coord, dtype=float)
    y = np.asarray(point.y_coord, dtype=float)
    z = np.asarray(point.z_coord, dtype=float)
    camera_count = np.asarray(point.camera_count, dtype=float)
    error = np.asarray(point.error, dtype=float)
    n_frames = min(len(x), len(y), len(z), len(camera_count), len(error))

    x = x[:n_frames]
    y = y[:n_frames]
    z = z[:n_frames]
    camera_count = camera_count[:n_frames]
    error = error[:n_frames]

    threshold = _resolve_error_threshold(error_thresholds, keypoint, error_max)
    xyz_missing = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
    camera_missing = ~np.isfinite(camera_count)
    low_camera = np.isfinite(camera_count) & (camera_count < min_cameras)
    error_missing = ~np.isfinite(error)
    error_high = np.isfinite(error) & (error > threshold)

    score, score_column = _optional_score_array(point)
    score_column_missing = score is None
    if score is not None:
        score = score[:n_frames]
        score_missing = ~np.isfinite(score)
        score_low = np.isfinite(score) & (score < score_min)
    elif require_score:
        score_missing = np.ones(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)
    else:
        score_missing = np.zeros(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)

    invalid = (
        xyz_missing
        | camera_missing
        | low_camera
        | error_missing
        | error_high
        | score_missing
        | score_low
    )
    xyz = np.column_stack([x, y, z])
    components = {
        "xyz_missing": xyz_missing,
        "camera_missing": camera_missing,
        "low_camera": low_camera,
        "error_missing": error_missing,
        "error_high": error_high,
        "score_missing": score_missing,
        "score_low": score_low,
        "invalid": invalid,
    }
    metadata = {
        "Error_Threshold": threshold,
        "Score_Column": score_column,
        "Score_Column_Missing": bool(score_column_missing),
        "Min_Cameras": min_cameras,
        "Score_Min": score_min,
        "Require_Score": bool(require_score),
    }
    return xyz, components, metadata


def summarize_invalid_mask(
        invalid_mask,
        components=None,
        start_frame=None,
        end_frame=None,
        max_interp_gap_frames=5,
        min_valid_fraction=0.7,
        require_start_end_valid=False,
):
    """Summarize one frame-wise invalid mask using the current QC rule."""
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    n_frames = len(invalid_mask)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames - 1
    start_frame = max(int(start_frame), 0)
    end_frame = min(int(end_frame), n_frames - 1)
    window_invalid = (
        invalid_mask[start_frame:end_frame + 1]
        if end_frame >= start_frame and n_frames
        else np.array([], dtype=bool)
    )
    window_valid = ~window_invalid
    gap_lengths = _true_run_lengths(window_invalid)
    total_frames = len(window_invalid)
    invalid_frames = _count_true(window_invalid)
    invalid_fraction = _fraction(invalid_frames, total_frames)
    valid_fraction = _fraction(total_frames - invalid_frames, total_frames)
    max_invalid_fraction = 1.0 - float(min_valid_fraction)
    max_gap = int(max(gap_lengths)) if gap_lengths else 0
    long_gap_count = int(sum(gap > max_interp_gap_frames for gap in gap_lengths))
    interpolatable_count = int(sum(
        gap for gap in gap_lengths if gap <= max_interp_gap_frames
    ))
    start_valid = (
        bool(not invalid_mask[start_frame])
        if n_frames and 0 <= start_frame < n_frames
        else False
    )
    end_valid = (
        bool(not invalid_mask[end_frame])
        if n_frames and 0 <= end_frame < n_frames
        else False
    )

    exclusion_reasons = []
    if total_frames == 0:
        exclusion_reasons.append("empty_qc_window")
    if pd.isna(invalid_fraction) or invalid_fraction > max_invalid_fraction:
        exclusion_reasons.append("invalid_fraction_above_threshold")
    if max_gap > max_interp_gap_frames:
        exclusion_reasons.append("long_invalid_gap")
    if require_start_end_valid and not start_valid:
        exclusion_reasons.append("start_frame_invalid")
    if require_start_end_valid and not end_valid:
        exclusion_reasons.append("end_frame_invalid")

    summary = {
        "Valid_Frame_Fraction": valid_fraction,
        "Invalid_Frame_Fraction": invalid_fraction,
        "Invalid_Frame_Count": invalid_frames,
        "Max_Invalid_Gap_Frames": max_gap,
        "Long_Gap_Count": long_gap_count,
        "Interpolated_Frame_Count": interpolatable_count,
        "Interpolatable_Invalid_Frame_Count": interpolatable_count,
        "Interpolatable_Invalid_Fraction": _fraction(interpolatable_count, total_frames),
        "Start_Frame_Valid": start_valid,
        "End_Frame_Valid": end_valid,
        "Max_Interp_Gap_Frames": max_interp_gap_frames,
        "Min_Valid_Fraction": min_valid_fraction,
        "Max_Invalid_Fraction": max_invalid_fraction,
        "QC_Passed": len(exclusion_reasons) == 0,
        "QC_Exclusion_Reason": ";".join(exclusion_reasons),
    }

    if components is not None:
        for name, mask in components.items():
            if name == "invalid":
                continue
            mask = np.asarray(mask, dtype=bool)
            window = (
                mask[start_frame:end_frame + 1]
                if end_frame >= start_frame and len(mask)
                else np.array([], dtype=bool)
            )
            column = "".join(part.capitalize() for part in name.split("_"))
            summary[f"{column}_Frame_Count"] = _count_true(window)
            summary[f"{column}_Fraction"] = _fraction(_count_true(window), total_frames)

    return summary


def interpolate_invalid_xyz_gaps(xyz, invalid_mask, max_gap_frames=5):
    """Set invalid xyz frames to NaN and linearly interpolate short invalid runs."""
    xyz = np.asarray(xyz, dtype=float).copy()
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    xyz[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(xyz)
    x_index = np.arange(n_frames)
    runs = []
    i = 0
    while i < n_frames:
        if not invalid_mask[i]:
            i += 1
            continue
        start = i
        while i < n_frames and invalid_mask[i]:
            i += 1
        runs.append((start, i))

    for start, stop in runs:
        gap_len = stop - start
        left = start - 1
        right = stop
        if gap_len > max_gap_frames or left < 0 or right >= n_frames:
            continue
        if not np.all(np.isfinite(xyz[left])) or not np.all(np.isfinite(xyz[right])):
            continue
        for dim in range(xyz.shape[1]):
            xyz[start:stop, dim] = np.interp(
                x_index[start:stop],
                [left, right],
                [xyz[left, dim], xyz[right, dim]],
            )
        interpolated_total += gap_len
    return xyz, interpolated_total


def interpolate_invalid_trace_gaps(values, invalid_mask, max_gap_frames=5):
    """Set invalid scalar frames to NaN and linearly interpolate short invalid runs."""
    values = np.asarray(values, dtype=float).copy()
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    values[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(values)
    x_index = np.arange(n_frames)
    i = 0
    while i < n_frames:
        if not invalid_mask[i]:
            i += 1
            continue
        start = i
        while i < n_frames and invalid_mask[i]:
            i += 1
        stop = i
        gap_len = stop - start
        left = start - 1
        right = stop
        if (
                gap_len <= max_gap_frames
                and left >= 0
                and right < n_frames
                and np.isfinite(values[left])
                and np.isfinite(values[right])
        ):
            values[start:stop] = np.interp(
                x_index[start:stop],
                [left, right],
                [values[left], values[right]],
            )
            interpolated_total += gap_len
    return values, interpolated_total


def _resolve_group(group_name):
    from group_config_new import GROUP_INFO, build_one_group

    if group_name in GROUP_INFO:
        return build_one_group(group_name)

    matches = [
        key for key, info in GROUP_INFO.items()
        if info.get("group_name") == group_name
    ]
    if len(matches) == 1:
        return build_one_group(matches[0])
    if len(matches) > 1:
        raise ValueError(
            f"Group name '{group_name}' matches multiple config keys: {matches}"
        )
    raise KeyError(f"Could not find group '{group_name}' in group_config_new.GROUP_INFO.")


def _read_trial_timing(group_info, fly, trial):
    fly_idx = int(fly) - 1
    trial_idx = int(trial) - 1

    if group_info.moc_data is None:
        raise ValueError(f"MOC metadata is not available for {group_info.group_name}.")
    if group_info.mol_data is None:
        raise ValueError(f"MOL metadata is not available for {group_info.group_name}.")
    if fly_idx < 0 or fly_idx >= group_info.total_fly_number:
        raise IndexError(f"Fly {fly} is outside the configured fly range.")
    if trial_idx < 0 or trial_idx >= group_info.trial_num:
        raise IndexError(f"Trial {trial} is outside the configured trial range.")

    moc = group_info.moc_data.iloc[fly_idx, trial_idx]
    mol = group_info.mol_data.iloc[fly_idx, trial_idx]
    fps = group_info.fps[fly_idx]

    if pd.isna(moc) or pd.isna(mol):
        raise ValueError(
            f"Missing MOC/MOL metadata for {group_info.group_name} F{fly}T{trial}."
        )
    if mol < moc:
        raise ValueError(
            f"MOL ({mol}) is earlier than MOC ({moc}) for "
            f"{group_info.group_name} F{fly}T{trial}."
        )

    return int(round(moc)), int(round(mol)), float(fps)


def _read_2d_projection_h5(h5_path):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"2D projection H5 file does not exist: {h5_path}")

    with pd.HDFStore(h5_path, mode="r") as store:
        keys = store.keys()
        if "/df_with_missing" in keys:
            df = store["/df_with_missing"]
        elif len(keys) == 1:
            df = store[keys[0]]
        else:
            raise ValueError(
                f"Could not choose an H5 dataset from {h5_path}. Available keys: {keys}"
            )

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DeepLabCut-style MultiIndex columns in the H5 file.")
    if "bodyparts" not in df.columns.names or "coords" not in df.columns.names:
        raise ValueError(
            "Expected H5 columns with MultiIndex levels named 'bodyparts' and 'coords'."
        )

    return df


def _find_projection_h5_files(projection_path, fly, trial):
    projection_path = Path(projection_path)
    if not projection_path.exists():
        raise FileNotFoundError(f"2D projection path does not exist: {projection_path}")

    fly_tag = f"F{int(fly):03d}"
    trial_tag = f"T{int(trial):03d}"
    search_roots = [
        projection_path / fly_tag / "pose-2d-proj",
        projection_path / fly_tag,
        projection_path,
    ]

    camera_paths = {}
    for root in search_roots:
        if not root.exists():
            continue
        for h5_path in root.rglob(f"{fly_tag}_{trial_tag}_*Cam*.h5"):
            stem = h5_path.stem
            cam_text = stem.rsplit("Cam", 1)[-1]
            if not cam_text.isdigit():
                continue
            camera_paths[int(cam_text)] = h5_path
        if camera_paths:
            break

    expected = set(range(1, 7))
    missing = sorted(expected - set(camera_paths))
    if missing:
        raise FileNotFoundError(
            f"Missing projection H5 files for cameras {missing} under {projection_path}."
        )
    return {camera: camera_paths[camera] for camera in sorted(expected)}


def _read_3d_trial_dataframe(group_info, fly, trial):
    key = group_info._trial_key(int(fly), int(trial))
    data_paths = getattr(group_info, "fly_kinematic_data_path", {})
    if key not in data_paths:
        raise FileNotFoundError(
            f"No 3D kinematic CSV path configured for {group_info.group_name} F{fly}T{trial}."
        )

    csv_path = Path(data_paths[key])
    if not csv_path.exists():
        raise FileNotFoundError(f"3D kinematic CSV file does not exist: {csv_path}")
    return pd.read_csv(csv_path), csv_path


def _slice_series(series, start_frame, stop_frame, value_mode):
    values = series.astype(float)
    if value_mode == "change":
        values = values.diff()
    return values.iloc[start_frame:stop_frame + 1]


def _extract_2d_coord(df, keypoint, coord):
    try:
        return df.xs((keypoint, coord), level=("bodyparts", "coords"), axis=1).iloc[:, 0]
    except KeyError as exc:
        raise KeyError(
            f"Could not find coordinate '{coord}' for keypoint '{keypoint}' in projection H5."
        ) from exc


def _extract_2d_coord_optional(df, keypoint, coord, start_frame, stop_frame):
    try:
        values = _extract_2d_coord(df, keypoint, coord)
    except KeyError:
        return pd.Series(
            np.nan,
            index=np.arange(start_frame, stop_frame + 1),
            dtype=float,
        )
    return values.astype(float).iloc[start_frame:stop_frame + 1]


def _resolve_axis_ylim(axis_ylim, values=None, symmetric=False):
    if axis_ylim is None:
        if values is None or not symmetric:
            return None
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) == 0:
            return None
        max_abs = float(np.nanmax(np.abs(finite_values)))
        if max_abs == 0:
            max_abs = 1.0
        return -max_abs, max_abs
    if np.isscalar(axis_ylim):
        limit = abs(float(axis_ylim))
        return -limit, limit
    if len(axis_ylim) != 2:
        raise ValueError("Axis limits must be None, a number, or a 2-value tuple.")
    return tuple(axis_ylim)


def _initialize_standard_metadata(group_info):
    if getattr(group_info, "trial_metadata", None):
        return
    group_info.initialize_manual_data(require_kinematics=False)


def _get_analysis_window(group_info, meta, n_frames, margin_s, window_mode):
    if window_mode not in {"moc_mol", "full"}:
        raise ValueError("window_mode must be 'moc_mol' or 'full'.")
    if n_frames <= 0:
        raise ValueError("Trial dataframe has no frames.")

    fly = int(meta["Fly#"])
    trial = int(meta["Trial#"])
    fps = float(meta["fps"])

    if window_mode == "full":
        return 0, n_frames - 1, np.nan, np.nan, fps

    moc, mol, fps = _read_trial_timing(group_info, fly, trial)
    start_frame = max(0, int(round(moc - margin_s * fps)))
    stop_frame = min(n_frames - 1, int(round(mol + margin_s * fps)))
    if stop_frame < start_frame:
        raise ValueError(
            f"Invalid analysis window for {group_info.group_name} F{fly}T{trial}: "
            f"start={start_frame}, stop={stop_frame}."
        )
    return start_frame, stop_frame, moc, mol, fps


def _find_score_column(df, keypoint):
    candidates = [
        f"{keypoint}_score",
        f"{keypoint}_likelihood",
        f"{keypoint}_confidence",
        f"{keypoint}_probability",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _count_true(values):
    values = np.asarray(values, dtype=bool)
    return int(np.count_nonzero(values))


def _fraction(count, total):
    if total == 0:
        return np.nan
    return float(count) / float(total)


def _true_run_lengths(values):
    values = np.asarray(values, dtype=bool)
    lengths = []
    run_length = 0
    for value in values:
        if value:
            run_length += 1
        elif run_length:
            lengths.append(run_length)
            run_length = 0
    if run_length:
        lengths.append(run_length)
    return lengths


def _summarize_one_trial_keypoint(
        group_info,
        meta,
        kine_df,
        keypoint,
        margin_s,
        window_mode,
        error_max,
        score_min,
        min_cameras,
        max_interp_gap_frames,
        max_invalid_fraction,
        require_score,
):
    fly = int(meta["Fly#"])
    trial = int(meta["Trial#"])
    trial_key = group_info._trial_key(fly, trial)
    start_frame, stop_frame, moc, mol, fps = _get_analysis_window(
        group_info,
        meta,
        len(kine_df),
        margin_s,
        window_mode,
    )
    window = kine_df.iloc[start_frame:stop_frame + 1]
    total_frames = len(window)

    coord_columns = [f"{keypoint}_{axis}" for axis in ("x", "y", "z")]
    missing_coord_columns = [column for column in coord_columns if column not in kine_df.columns]
    if missing_coord_columns:
        raise KeyError(
            f"3D coordinate columns not found for {group_info.group_name} {trial_key} "
            f"{keypoint}: {missing_coord_columns}"
        )

    xyz = window[coord_columns].astype(float)
    xyz_missing = xyz.isna().any(axis=1).to_numpy()

    error_column = f"{keypoint}_error"
    if error_column in kine_df.columns:
        error_values = window[error_column].astype(float)
        error_missing = error_values.isna().to_numpy()
        error_high = (error_values > error_max).fillna(False).to_numpy()
        finite_error_values = error_values[np.isfinite(error_values)]
        median_error = float(finite_error_values.median()) if len(finite_error_values) else np.nan
        p95_error = float(finite_error_values.quantile(0.95)) if len(finite_error_values) else np.nan
    else:
        error_missing = np.ones(total_frames, dtype=bool)
        error_high = np.zeros(total_frames, dtype=bool)
        median_error = np.nan
        p95_error = np.nan

    ncams_column = f"{keypoint}_ncams"
    if ncams_column in kine_df.columns:
        ncams = window[ncams_column].astype(float)
        camera_missing = ncams.isna().to_numpy()
        low_camera = (ncams < min_cameras).fillna(False).to_numpy()
        finite_ncams = ncams[np.isfinite(ncams)]
        median_ncams = float(finite_ncams.median()) if len(finite_ncams) else np.nan
        min_ncams_value = float(finite_ncams.min()) if len(finite_ncams) else np.nan
    else:
        camera_missing = np.ones(total_frames, dtype=bool)
        low_camera = np.ones(total_frames, dtype=bool)
        median_ncams = np.nan
        min_ncams_value = np.nan

    score_column = _find_score_column(kine_df, keypoint)
    score_column_missing = score_column is None
    if score_column is not None:
        score_values = window[score_column].astype(float)
        score_missing = score_values.isna().to_numpy()
        score_low = (score_values < score_min).fillna(False).to_numpy()
        finite_score_values = score_values[np.isfinite(score_values)]
        median_score = float(finite_score_values.median()) if len(finite_score_values) else np.nan
        p05_score = float(finite_score_values.quantile(0.05)) if len(finite_score_values) else np.nan
    elif require_score:
        score_missing = np.ones(total_frames, dtype=bool)
        score_low = np.zeros(total_frames, dtype=bool)
        median_score = np.nan
        p05_score = np.nan
    else:
        score_missing = np.zeros(total_frames, dtype=bool)
        score_low = np.zeros(total_frames, dtype=bool)
        median_score = np.nan
        p05_score = np.nan

    error_invalid = error_missing | error_high
    camera_invalid = camera_missing | low_camera
    score_invalid = score_missing | score_low
    invalid = xyz_missing | error_invalid | camera_invalid | score_invalid

    invalid_lengths = _true_run_lengths(invalid)
    interpolatable_frames = sum(
        length for length in invalid_lengths
        if length <= max_interp_gap_frames
    )
    long_gap_count = sum(
        1 for length in invalid_lengths
        if length > max_interp_gap_frames
    )
    longest_invalid_gap = max(invalid_lengths, default=0)

    invalid_frames = _count_true(invalid)
    invalid_fraction = _fraction(invalid_frames, total_frames)
    pass_invalid_fraction = invalid_fraction <= max_invalid_fraction
    pass_long_gap = longest_invalid_gap <= max_interp_gap_frames

    reason_counts = {
        "XYZ_Missing": _count_true(xyz_missing),
        "Error_Missing": _count_true(error_missing),
        "Error_High": _count_true(error_high),
        "Camera_Missing": _count_true(camera_missing),
        "Low_Camera_Count": _count_true(low_camera),
        "Score_Missing": _count_true(score_missing),
        "Score_Low": _count_true(score_low),
    }
    primary_failure_reason = "None"
    if invalid_frames:
        primary_failure_reason = max(reason_counts, key=reason_counts.get)

    return {
        "Group": group_info.group_name,
        "Fly": fly,
        "Trial": trial,
        "Trial_Key": trial_key,
        "Trial_Type": meta.get("TrialType"),
        "Keypoint": keypoint,
        "Window_Mode": window_mode,
        "Start_Frame": start_frame,
        "Stop_Frame": stop_frame,
        "MOC_Frame": moc,
        "MOL_Frame": mol,
        "FPS": fps,
        "Total_Frames": total_frames,
        "QC_Error_Max": error_max,
        "QC_Score_Min": score_min,
        "QC_Min_Cameras": min_cameras,
        "QC_Max_Interp_Gap_Frames": max_interp_gap_frames,
        "QC_Max_Invalid_Fraction": max_invalid_fraction,
        "QC_Require_Score": bool(require_score),
        "Score_Column": score_column,
        "Score_Column_Missing": bool(score_column_missing),
        "Error_Median": median_error,
        "Error_P95": p95_error,
        "Score_Median": median_score,
        "Score_P05": p05_score,
        "Ncams_Median": median_ncams,
        "Ncams_Min": min_ncams_value,
        "XYZ_Missing_Frames": reason_counts["XYZ_Missing"],
        "Error_Missing_Frames": reason_counts["Error_Missing"],
        "Error_High_Frames": reason_counts["Error_High"],
        "Camera_Missing_Frames": reason_counts["Camera_Missing"],
        "Low_Camera_Count_Frames": reason_counts["Low_Camera_Count"],
        "Score_Missing_Frames": reason_counts["Score_Missing"],
        "Score_Low_Frames": reason_counts["Score_Low"],
        "Error_Invalid_Frames": _count_true(error_invalid),
        "Camera_Invalid_Frames": _count_true(camera_invalid),
        "Score_Invalid_Frames": _count_true(score_invalid),
        "Invalid_Frames": invalid_frames,
        "XYZ_Missing_Fraction": _fraction(reason_counts["XYZ_Missing"], total_frames),
        "Error_Missing_Fraction": _fraction(reason_counts["Error_Missing"], total_frames),
        "Error_High_Fraction": _fraction(reason_counts["Error_High"], total_frames),
        "Error_Invalid_Fraction": _fraction(_count_true(error_invalid), total_frames),
        "Camera_Missing_Fraction": _fraction(reason_counts["Camera_Missing"], total_frames),
        "Low_Camera_Count_Fraction": _fraction(reason_counts["Low_Camera_Count"], total_frames),
        "Camera_Invalid_Fraction": _fraction(_count_true(camera_invalid), total_frames),
        "Score_Missing_Fraction": _fraction(reason_counts["Score_Missing"], total_frames),
        "Score_Low_Fraction": _fraction(reason_counts["Score_Low"], total_frames),
        "Score_Invalid_Fraction": _fraction(_count_true(score_invalid), total_frames),
        "Invalid_Fraction": invalid_fraction,
        "Interpolatable_Invalid_Frames": interpolatable_frames,
        "Interpolatable_Invalid_Fraction": _fraction(interpolatable_frames, total_frames),
        "Longest_Invalid_Gap_Frames": longest_invalid_gap,
        "Long_Invalid_Gap_Count": long_gap_count,
        "Pass_Invalid_Fraction": bool(pass_invalid_fraction),
        "Pass_Long_Gap": bool(pass_long_gap),
        "QC_Passed": bool(pass_invalid_fraction and pass_long_gap),
        "Primary_Failure_Reason": primary_failure_reason,
    }


def summarize_tracking_qc_by_trial_keypoint(
        group_name,
        keypoints=None,
        trial_types=("Landing",),
        margin_s=0.2,
        window_mode="moc_mol",
        error_max=50,
        score_min=0.8,
        min_cameras=2,
        max_interp_gap_frames=5,
        max_invalid_fraction=0.3,
        require_score=False,
        include_good_fly_only=True,
):
    """
    Build a trial-keypoint-level tracking QC summary table for one group.

    Each output row is one fly x trial x keypoint trajectory within the selected
    analysis window. Missing reprojection error, high reprojection error, low
    camera count, missing coordinates, and low/missing score are tracked as
    separate fractions, then combined into the overall invalid-frame fraction.
    """
    group_info = _resolve_group(group_name)
    _initialize_standard_metadata(group_info)

    if keypoints is None:
        keypoints = list(group_info.joints)
    if isinstance(trial_types, str):
        trial_types = [trial_types]
    trial_types = set(trial_types)

    rows = []
    skipped = []
    for trial_key, meta in group_info.trial_metadata.items():
        if trial_types and meta.get("TrialType") not in trial_types:
            continue
        if include_good_fly_only and int(meta["Fly#"]) not in group_info.good_fly_index:
            continue
        try:
            kine_df, csv_path = _read_3d_trial_dataframe(
                group_info,
                int(meta["Fly#"]),
                int(meta["Trial#"]),
            )
            for keypoint in keypoints:
                row = _summarize_one_trial_keypoint(
                    group_info=group_info,
                    meta=meta,
                    kine_df=kine_df,
                    keypoint=keypoint,
                    margin_s=margin_s,
                    window_mode=window_mode,
                    error_max=error_max,
                    score_min=score_min,
                    min_cameras=min_cameras,
                    max_interp_gap_frames=max_interp_gap_frames,
                    max_invalid_fraction=max_invalid_fraction,
                    require_score=require_score,
                )
                row["Kinematic_CSV"] = str(csv_path)
                rows.append(row)
        except Exception as exc:
            skipped.append({
                "Group": group_info.group_name,
                "Trial_Key": trial_key,
                "Fly": meta.get("Fly#"),
                "Trial": meta.get("Trial#"),
                "Trial_Type": meta.get("TrialType"),
                "Skip_Reason": f"{type(exc).__name__}: {exc}",
            })

    summary_df = pd.DataFrame(rows)
    if skipped:
        skipped_df = pd.DataFrame(skipped)
        summary_df.attrs["skipped_trials"] = skipped_df
    return summary_df


def _stripplot_by_keypoint(
        ax,
        df,
        y_column,
        keypoint_order,
        ylabel,
        threshold=None,
        color="0.25",
):
    positions = np.arange(len(keypoint_order))
    rng = np.random.default_rng(0)
    for pos, keypoint in zip(positions, keypoint_order):
        values = df.loc[df["Keypoint"] == keypoint, y_column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=22,
            alpha=0.7,
            color=color,
            edgecolors="none",
        )
        median = float(np.nanmedian(values))
        ax.plot([pos - 0.25, pos + 0.25], [median, median], color="black", linewidth=1.5)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(keypoint_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _save_figure(fig, output_dir, filename):
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")


def plot_tracking_qc_summary(
        qc_summary_df,
        output_dir=None,
        show=True,
        keypoint_order=None,
        max_invalid_fraction=None,
        max_interp_gap_frames=None,
        threshold_values=None,
        figsize=(11, 5),
):
    """
    Plot trial-keypoint-level QC distributions from `summarize_tracking_qc_by_trial_keypoint`.

    Returns a dictionary of matplotlib figures. Every dot in the distribution
    plots is one fly x trial x keypoint row from the summary table.
    """
    if qc_summary_df.empty:
        raise ValueError("qc_summary_df is empty; no QC summary plots can be made.")
    if keypoint_order is None:
        keypoint_order = list(dict.fromkeys(qc_summary_df["Keypoint"]))
    if max_invalid_fraction is None:
        max_invalid_fraction = float(qc_summary_df["QC_Max_Invalid_Fraction"].dropna().iloc[0])
    if max_interp_gap_frames is None:
        max_interp_gap_frames = int(qc_summary_df["QC_Max_Interp_Gap_Frames"].dropna().iloc[0])
    if threshold_values is None:
        threshold_values = np.arange(0.05, 0.51, 0.05)

    figures = {}
    group_name = str(qc_summary_df["Group"].dropna().iloc[0])

    distribution_specs = [
        (
            "invalid_fraction",
            "Invalid_Fraction",
            "Invalid frame fraction",
            max_invalid_fraction,
            "#4c78a8",
        ),
        (
            "missing_reprojection_error_fraction",
            "Error_Missing_Fraction",
            "Missing reprojection error fraction",
            None,
            "#8c564b",
        ),
        (
            "high_reprojection_error_fraction",
            "Error_High_Fraction",
            "High reprojection error fraction",
            None,
            "#ff7f0e",
        ),
        (
            "low_camera_count_fraction",
            "Low_Camera_Count_Fraction",
            "Low camera-count fraction",
            None,
            "#7f7f7f",
        ),
        (
            "score_invalid_fraction",
            "Score_Invalid_Fraction",
            "Score invalid fraction",
            None,
            "#2ca02c",
        ),
        (
            "interpolatable_invalid_fraction",
            "Interpolatable_Invalid_Fraction",
            "Interpolatable invalid fraction",
            None,
            "#9467bd",
        ),
        (
            "longest_invalid_gap",
            "Longest_Invalid_Gap_Frames",
            "Longest invalid gap (frames)",
            max_interp_gap_frames,
            "#d62728",
        ),
        (
            "reprojection_error_p95",
            "Error_P95",
            "95th percentile reprojection error",
            None,
            "#ff7f0e",
        ),
    ]

    for fig_key, column, ylabel, threshold, color in distribution_specs:
        if column not in qc_summary_df.columns:
            continue
        if qc_summary_df[column].dropna().empty:
            continue
        if (
                fig_key == "score_invalid_fraction"
                and "Score_Column_Missing" in qc_summary_df.columns
                and qc_summary_df["Score_Column_Missing"].all()
                and qc_summary_df[column].fillna(0).eq(0).all()
        ):
            continue
        fig, ax = plt.subplots(figsize=figsize)
        _stripplot_by_keypoint(
            ax,
            qc_summary_df,
            column,
            keypoint_order,
            ylabel,
            threshold=threshold,
            color=color,
        )
        ax.set_title(f"{group_name}: {ylabel}")
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{group_name}_{fig_key}.png")
        if not show:
            plt.close(fig)
        figures[fig_key] = fig

    if "Score_Column_Missing" in qc_summary_df.columns and not qc_summary_df["Score_Column_Missing"].all():
        fig, ax = plt.subplots(figsize=figsize)
        _stripplot_by_keypoint(
            ax,
            qc_summary_df,
            "Score_P05",
            keypoint_order,
            "5th percentile score",
            threshold=float(qc_summary_df["QC_Score_Min"].dropna().iloc[0]),
            color="#2ca02c",
        )
        ax.set_title(f"{group_name}: Score distribution")
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{group_name}_score_p05.png")
        if not show:
            plt.close(fig)
        figures["score_p05"] = fig

    pivot = qc_summary_df.pivot_table(
        index="Trial_Key",
        columns="Keypoint",
        values="Invalid_Fraction",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=keypoint_order)
    fig_height = max(4, 0.25 * len(pivot) + 2)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(keypoint_order) + 4), fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{group_name}: Trial-keypoint invalid fraction")
    ax.set_xlabel("Keypoint")
    ax.set_ylabel("Trial")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Invalid frame fraction")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{group_name}_invalid_fraction_heatmap.png")
    if not show:
        plt.close(fig)
    figures["invalid_fraction_heatmap"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    for keypoint in keypoint_order:
        values = qc_summary_df.loc[
            qc_summary_df["Keypoint"] == keypoint,
            "Invalid_Fraction",
        ].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        retained = [np.mean(values <= threshold) for threshold in threshold_values]
        ax.plot(threshold_values, retained, marker="o", linewidth=1.5, label=keypoint)
    ax.axvline(max_invalid_fraction, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Allowed invalid-frame fraction")
    ax.set_ylabel("Retained trial-keypoint fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{group_name}: QC threshold sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{group_name}_threshold_sensitivity.png")
    if not show:
        plt.close(fig)
    figures["threshold_sensitivity"] = fig

    return figures


def plot_2d_keypoint_xy_traces(
        group_name,
        fly,
        trial,
        projection_path=None,
        h5_path=None,
        keypoints=None,
        margin_s=0.2,
        window_mode="moc_mol",
        value_mode="change",
        projection_ylim=None,
        xyz_ylim=None,
        change_ylim=None,
        output_dir=None,
        show=True,
        figsize=(12, 12),
):
    """
    Plot 2D projection traces from all cameras with matching 3D trial data.

    Parameters
    ----------
    group_name : str
        Config key or display group name from group_config_new.
    fly, trial : int
        Metadata fly/trial numbers, using the same 1-based indexing as Excel.
    projection_path : str or Path, optional
        Root folder containing per-fly 2D projection H5 files.
    h5_path : str or Path, optional
        Backward-compatible single-camera path. If projection_path is omitted,
        its parent folder is searched for all six camera files.
    keypoints : sequence[str], optional
        Bodyparts to plot. Defaults to all bodyparts present in the H5 file.
    margin_s : float
        Seconds added before MOC and after MOL when window_mode="moc_mol".
    window_mode : {"moc_mol", "full"}
        Plot either the MOC-to-MOL window with margin, or the whole video.
    value_mode : {"change", "absolute"}
        Plot frame-to-frame coordinate changes or absolute coordinates.
    projection_ylim : None, number, or tuple
        Shared y-axis limit for the 2D x/y panels.
    xyz_ylim : None, number, or tuple
        Shared y-axis limit for the 3D xyz panel.
    change_ylim : None, number, or tuple
        Deprecated alias for projection_ylim.
    output_dir : str or Path, optional
        If provided, save one PNG per keypoint in this directory.
    show : bool
        If True, display figures. If False, close them after optional saving.
    figsize : tuple
        Matplotlib figure size for each keypoint.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One 6x1 figure per plotted keypoint.
    """
    if window_mode not in {"moc_mol", "full"}:
        raise ValueError("window_mode must be 'moc_mol' or 'full'.")
    if value_mode not in {"change", "absolute"}:
        raise ValueError("value_mode must be 'change' or 'absolute'.")
    if projection_ylim is None and change_ylim is not None:
        projection_ylim = change_ylim
    if projection_path is None:
        if h5_path is None:
            raise ValueError("Provide projection_path, or h5_path for backward-compatible lookup.")
        projection_path = Path(h5_path).parent

    group_info = _resolve_group(group_name)
    moc, mol, fps = _read_trial_timing(group_info, fly, trial)
    camera_paths = _find_projection_h5_files(projection_path, fly, trial)
    camera_dfs = {
        camera: _read_2d_projection_h5(path)
        for camera, path in camera_paths.items()
    }
    kine_df, _ = _read_3d_trial_dataframe(group_info, fly, trial)

    first_df = next(iter(camera_dfs.values()))
    available_keypoints = list(dict.fromkeys(first_df.columns.get_level_values("bodyparts")))
    if keypoints is None:
        keypoints = available_keypoints
    else:
        missing = [kp for kp in keypoints if kp not in available_keypoints]
        if missing:
            raise KeyError(f"Keypoints not found in H5 file: {missing}")

    missing_3d = []
    for keypoint in keypoints:
        for suffix in ("x", "y", "z", "error", "ncams"):
            column = f"{keypoint}_{suffix}"
            if column not in kine_df.columns:
                missing_3d.append(column)
    if missing_3d:
        raise KeyError(f"3D kinematic columns not found: {missing_3d}")

    max_projection_len = min(len(df) for df in camera_dfs.values())
    max_len = min(max_projection_len, len(kine_df))
    if window_mode == "moc_mol":
        start_frame = max(0, int(round(moc - margin_s * fps)))
        stop_frame = min(max_len - 1, int(round(mol + margin_s * fps)))
    else:
        start_frame = 0
        stop_frame = max_len - 1

    if stop_frame < start_frame:
        raise ValueError(
            f"Invalid plotting window: start={start_frame}, stop={stop_frame}."
        )

    frame_index = np.arange(start_frame, stop_frame + 1)
    red_colors = plt.cm.Reds(np.linspace(0.9, 0.35, 6))
    blue_colors = plt.cm.Blues(np.linspace(0.9, 0.35, 6))
    green_colors = plt.cm.Greens(np.linspace(0.9, 0.35, 6))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for keypoint in keypoints:
        x_traces = {}
        y_traces = {}
        confidence_traces = {}
        for camera, df in camera_dfs.items():
            x = _extract_2d_coord(df, keypoint, "x")
            y = _extract_2d_coord(df, keypoint, "y")
            x_traces[camera] = _slice_series(x, start_frame, stop_frame, value_mode)
            y_traces[camera] = _slice_series(y, start_frame, stop_frame, value_mode)
            confidence_traces[camera] = _extract_2d_coord_optional(
                df,
                keypoint,
                "likelihood",
                start_frame,
                stop_frame,
            )

        x3d = _slice_series(kine_df[f"{keypoint}_x"], start_frame, stop_frame, value_mode)
        y3d = _slice_series(kine_df[f"{keypoint}_y"], start_frame, stop_frame, value_mode)
        z3d = _slice_series(kine_df[f"{keypoint}_z"], start_frame, stop_frame, value_mode)
        error = kine_df[f"{keypoint}_error"].astype(float).iloc[start_frame:stop_frame + 1]
        ncams = kine_df[f"{keypoint}_ncams"].astype(float).iloc[start_frame:stop_frame + 1]

        projection_values = np.concatenate([
            trace.to_numpy(dtype=float)
            for trace in list(x_traces.values()) + list(y_traces.values())
        ])
        xyz_values = np.concatenate([
            x3d.to_numpy(dtype=float),
            y3d.to_numpy(dtype=float),
            z3d.to_numpy(dtype=float),
        ])
        resolved_projection_ylim = _resolve_axis_ylim(
            projection_ylim,
            projection_values,
            symmetric=value_mode == "change",
        )
        resolved_xyz_ylim = _resolve_axis_ylim(
            xyz_ylim,
            xyz_values,
            symmetric=value_mode == "change",
        )

        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
        coordinate_label = "change" if value_mode == "change" else "coordinate"
        fig.suptitle(
            f"{group_info.group_name} F{fly}T{trial} {keypoint} 2D projections and 3D data"
        )

        for color_idx, camera in enumerate(sorted(camera_dfs)):
            axes[0].plot(
                frame_index,
                x_traces[camera].to_numpy(dtype=float),
                color=red_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )
            axes[1].plot(
                frame_index,
                y_traces[camera].to_numpy(dtype=float),
                color=blue_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )
            axes[2].plot(
                frame_index,
                confidence_traces[camera].to_numpy(dtype=float),
                color=green_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )

        axes[0].set_ylabel("2D x" if value_mode == "absolute" else "2D dx")
        axes[1].set_ylabel("2D y" if value_mode == "absolute" else "2D dy")
        axes[2].set_ylabel("confidence")
        axes[2].set_ylim(-0.05, 1.05)

        axes[3].plot(frame_index, x3d.to_numpy(dtype=float), color="#d62728", linewidth=1, label="x")
        axes[3].plot(frame_index, y3d.to_numpy(dtype=float), color="#1f77b4", linewidth=1, label="y")
        axes[3].plot(frame_index, z3d.to_numpy(dtype=float), color="black", linewidth=1, label="z")
        axes[3].set_ylabel("3D xyz" if value_mode == "absolute" else "3D dxyz")

        axes[4].plot(frame_index, error.to_numpy(dtype=float), color="#ff7f0e", linewidth=1)
        axes[4].set_ylabel("reproj. error")

        axes[5].plot(
            frame_index,
            ncams.to_numpy(dtype=float),
            color="0.25",
            marker=".",
            markersize=2,
            linewidth=0.8,
        )
        axes[5].set_ylabel("3D ncams")
        axes[5].set_xlabel("Frame")

        for ax in axes:
            ax.axvline(moc, color="black", linestyle="--", linewidth=1)
            ax.axvline(mol, color="black", linestyle=":", linewidth=1)
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(True, which="major", axis="y", alpha=0.25)
            ax.grid(True, which="both", axis="x", alpha=0.18)
        if resolved_projection_ylim is not None:
            axes[0].set_ylim(resolved_projection_ylim)
            axes[1].set_ylim(resolved_projection_ylim)
        if resolved_xyz_ylim is not None:
            axes[3].set_ylim(resolved_xyz_ylim)
        axes[0].legend(loc="upper right", ncol=6, fontsize="small")
        axes[3].legend(loc="upper right", ncol=3, fontsize="small")

        fig.tight_layout()

        if output_dir is not None:
            safe_keypoint = str(keypoint).replace("/", "_").replace("\\", "_")
            fig.savefig(
                output_dir / (
                    f"{group_info.group_name}_F{fly}T{trial}_{safe_keypoint}_2D3D_{coordinate_label}.png"
                ),
                dpi=200,
            )
        if not show:
            plt.close(fig)
        figures.append(fig)

    return figures
