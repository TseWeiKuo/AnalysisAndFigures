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

    # Master switch: when False, callers can bypass all QC filtering.
    enabled: bool = False
    # Reprojection-error cutoff used to mark a frame invalid.
    error_max: float = 50
    # Minimum accepted score/likelihood when a score column is available.
    score_min: float = 0.8
    # Minimum number of cameras contributing to the reconstructed 3D point.
    min_cameras: int = 2
    # Maximum invalid gap that can be linearly interpolated, expressed in seconds.
    max_interp_gap_s: float = 0.02
    # Minimum fraction of valid frames required within the analysis window.
    min_valid_fraction: float = 0.7
    # If True, a missing score/likelihood column makes every frame score-invalid.
    require_score: bool = False

    @property
    def max_invalid_fraction(self):
        # Convert the valid-frame threshold into the equivalent invalid-frame limit.
        return 1.0 - float(self.min_valid_fraction)

    def output_metadata(self):
        """Return the common QC fields written to result dataframes."""
        # Store active thresholds beside downstream analysis results for provenance.
        return {
            "Apply_Tracking_QC": bool(self.enabled),
            "Min_Cameras": self.min_cameras if self.enabled else np.nan,
            "Error_Max": self.error_max if self.enabled else np.nan,
            "Score_Min": self.score_min if self.enabled else np.nan,
            "Max_Interp_Gap_s": self.max_interp_gap_s if self.enabled else np.nan,
            "Min_Valid_Fraction": (
                self.min_valid_fraction if self.enabled else np.nan
            ),
            "Max_Invalid_Fraction": (
                self.max_invalid_fraction if self.enabled else np.nan
            ),
            "Require_Score": bool(self.require_score) if self.enabled else False,
        }


def build_config(
        apply_tracking_qc=False,
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Build a QC config from the current fixed-threshold, time-gap rule."""
    # Keep the public plotting arguments separate from the dataclass constructor.
    return TrackingQCConfig(
        enabled=apply_tracking_qc,
        error_max=error_max,
        score_min=score_min,
        min_cameras=min_cameras,
        max_interp_gap_s=max_interp_gap_s,
        min_valid_fraction=min_valid_fraction,
        require_score=require_score,
    )


def interp_gap_frames_from_fps(max_interp_gap_s, fps):
    """Convert the time-based interpolation threshold into trial-local frames."""
    # A finite positive FPS is required because the interpolation rule is time-based.
    if fps is None or pd.isna(fps) or float(fps) <= 0:
        raise ValueError("fps must be finite and > 0 to resolve time-based QC interpolation.")
    # Round to the nearest whole-frame gap and keep at least one frame interpolable.
    return max(1, int(round(float(max_interp_gap_s) * float(fps))))


def _optional_score_array(point):
    # Accept several common confidence-column names used by tracking pipelines.
    for attr in ("score", "likelihood", "confidence", "probability"):
        # Skip names that are not present on the Point object.
        if hasattr(point, attr):
            values = getattr(point, attr)
            # A present-but-empty score attribute is treated as unavailable.
            if values is not None:
                return np.asarray(values, dtype=float), getattr(point, "score_column", attr)
    # Returning None lets the caller decide whether missing scores should fail QC.
    return None, None


def point_invalid_components(
        point,
        keypoint,
        min_cameras=2,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Return frame-wise invalid components for one 3D keypoint trace."""
    # Convert point attributes to numeric arrays so NaN/finite checks are consistent.
    x = np.asarray(point.x_coord, dtype=float)
    y = np.asarray(point.y_coord, dtype=float)
    z = np.asarray(point.z_coord, dtype=float)
    camera_count = np.asarray(point.camera_count, dtype=float)
    error = np.asarray(point.error, dtype=float)
    # Required 3D channels must be frame-aligned; missing values should be NaN,
    # not shorter arrays that silently change the analyzed window.
    required_lengths = {
        "x": len(x),
        "y": len(y),
        "z": len(z),
        "camera_count": len(camera_count),
        "error": len(error),
    }
    if len(set(required_lengths.values())) != 1:
        raise ValueError(
            f"Frame-length mismatch for keypoint '{keypoint}': {required_lengths}"
        )
    # Use the validated shared length for all downstream frame-wise masks.
    n_frames = len(x)

    # The current QC rule uses one fixed reprojection-error cutoff for every keypoint.
    threshold = float(error_max)
    # Missing xyz coordinates invalidate a frame because downstream geometry needs all axes.
    xyz_missing = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
    # Missing camera count invalidates a frame because camera support cannot be verified.
    camera_missing = ~np.isfinite(camera_count)
    # Low camera count invalidates a frame even if coordinates are finite.
    low_camera = np.isfinite(camera_count) & (camera_count < min_cameras)
    # Missing reprojection error invalidates a frame because reconstruction quality is unknown.
    error_missing = ~np.isfinite(error)
    # High reprojection error invalidates a frame because triangulation quality is poor.
    error_high = np.isfinite(error) & (error > threshold)

    # Score/likelihood is optional unless the caller explicitly requires it.
    score, score_column = _optional_score_array(point)
    score_column_missing = score is None
    if score is not None:
        # Score channels, when present, must align to the same frame index as xyz.
        if len(score) != n_frames:
            raise ValueError(
                f"Frame-length mismatch for keypoint '{keypoint}' score column "
                f"'{score_column}': score={len(score)}, required_channels={n_frames}"
            )
        # Missing score values are invalid when a score column exists.
        score_missing = ~np.isfinite(score)
        # Scores below the caller threshold are invalid.
        score_low = np.isfinite(score) & (score < score_min)
    elif require_score:
        # If score is mandatory and absent, mark all frames as score-missing.
        score_missing = np.ones(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)
    else:
        # If score is optional and absent, do not let it affect QC.
        score_missing = np.zeros(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)

    # A frame is invalid if any required coordinate, camera, error, or score rule fails.
    invalid = (
        xyz_missing
        | camera_missing
        | low_camera
        | error_missing
        | error_high
        | score_missing
        | score_low
    )
    # Return xyz together so callers can mask/interpolate the same frames.
    xyz = np.column_stack([x, y, z])
    # Keep separate masks so QC summaries can report why frames failed.
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
    # Metadata records exactly which thresholds and score column were used.
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
        max_interp_gap_frames=None,
        max_interp_gap_s=0.02,
        fps=None,
        min_valid_fraction=0.7,
        require_start_end_valid=False,
):
    """Summarize one frame-wise invalid mask using the current QC rule."""
    # Resolve the time-based interpolation rule once so all gap statistics use
    # the same effective frame threshold for this trial.
    if max_interp_gap_frames is None:
        max_interp_gap_frames = interp_gap_frames_from_fps(max_interp_gap_s, fps)
    # Normalize the mask type before slicing and counting.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    n_frames = len(invalid_mask)
    # Default to the full trace if no window is supplied.
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames - 1
    # Clamp requested windows so summaries never index outside the available frames.
    start_frame = max(int(start_frame), 0)
    end_frame = min(int(end_frame), n_frames - 1)
    # Extract the analysis-window mask; use an empty mask for invalid windows.
    window_invalid = (
        invalid_mask[start_frame:end_frame + 1]
        if end_frame >= start_frame and n_frames
        else np.array([], dtype=bool)
    )
    # Valid frames are the complement of the combined invalid mask.
    window_valid = ~window_invalid
    # Consecutive invalid-frame runs drive the interpolation and long-gap checks.
    gap_lengths = _true_run_lengths(window_invalid)
    # Count window size and invalid burden in the selected analysis window.
    total_frames = len(window_invalid)
    invalid_frames = _count_true(window_invalid)
    invalid_fraction = _fraction(invalid_frames, total_frames)
    valid_fraction = _fraction(total_frames - invalid_frames, total_frames)
    # Convert valid-fraction requirement into the maximum tolerated invalid fraction.
    max_invalid_fraction = 1.0 - float(min_valid_fraction)
    # Track the longest invalid run for the long-gap exclusion rule.
    max_gap = int(max(gap_lengths)) if gap_lengths else 0
    # Count invalid runs that exceed the interpolation threshold.
    long_gap_count = int(sum(gap > max_interp_gap_frames for gap in gap_lengths))
    # Count invalid frames that are short enough to be candidates for interpolation.
    interpolatable_count = int(sum(
        gap for gap in gap_lengths if gap <= max_interp_gap_frames
    ))
    # Optionally require the first frame of the analysis window to be valid.
    start_valid = (
        bool(not invalid_mask[start_frame])
        if n_frames and 0 <= start_frame < n_frames
        else False
    )
    # Optionally require the last frame of the analysis window to be valid.
    end_valid = (
        bool(not invalid_mask[end_frame])
        if n_frames and 0 <= end_frame < n_frames
        else False
    )

    # Collect explicit exclusion reasons instead of returning only a boolean.
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

    # Main summary used by analysis functions and QC diagnostic plots.
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
        "Max_Interp_Gap_s": max_interp_gap_s,
        "Max_Interp_Gap_Frames": max_interp_gap_frames,
        "Min_Valid_Fraction": min_valid_fraction,
        "Max_Invalid_Fraction": max_invalid_fraction,
        "QC_Passed": len(exclusion_reasons) == 0,
        "QC_Exclusion_Reason": ";".join(exclusion_reasons),
    }

    if components is not None:
        # Add reason-specific counts/fractions for every provided component mask.
        for name, mask in components.items():
            # The combined invalid mask is already represented by the main summary.
            if name == "invalid":
                continue
            # Align each component to the same window used for the combined mask.
            mask = np.asarray(mask, dtype=bool)
            window = (
                mask[start_frame:end_frame + 1]
                if end_frame >= start_frame and len(mask)
                else np.array([], dtype=bool)
            )
            # Convert snake_case component names into compact dataframe column prefixes.
            column = "".join(part.capitalize() for part in name.split("_"))
            summary[f"{column}_Frame_Count"] = _count_true(window)
            summary[f"{column}_Fraction"] = _fraction(_count_true(window), total_frames)

    return summary


def interpolate_invalid_xyz_gaps(xyz, invalid_mask, max_gap_frames=5):
    """Set invalid xyz frames to NaN and linearly interpolate short invalid runs."""
    # Work on a copy so callers keep access to the raw coordinates.
    xyz = np.asarray(xyz, dtype=float).copy()
    # Normalize the invalid mask before using it to overwrite coordinates.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    # Invalid frames are removed first; only eligible short gaps are filled below.
    xyz[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(xyz)
    # Frame numbers are the x-axis for interpolation.
    x_index = np.arange(n_frames)
    runs = []
    i = 0
    # Identify contiguous invalid-frame runs as half-open [start, stop) intervals.
    while i < n_frames:
        if not invalid_mask[i]:
            i += 1
            continue
        start = i
        while i < n_frames and invalid_mask[i]:
            i += 1
        runs.append((start, i))

    # Only fill invalid runs that are short and bounded by valid finite endpoints.
    for start, stop in runs:
        gap_len = stop - start
        left = start - 1
        right = stop
        # Do not interpolate long gaps or edge gaps without both neighboring frames.
        if gap_len > max_gap_frames or left < 0 or right >= n_frames:
            continue
        # All xyz dimensions must be finite at both endpoints for interpolation.
        if not np.all(np.isfinite(xyz[left])) or not np.all(np.isfinite(xyz[right])):
            continue
        # Interpolate each coordinate dimension independently across the gap.
        for dim in range(xyz.shape[1]):
            xyz[start:stop, dim] = np.interp(
                x_index[start:stop],
                [left, right],
                [xyz[left, dim], xyz[right, dim]],
            )
        # Report how many invalid frames were actually filled.
        interpolated_total += gap_len
    return xyz, interpolated_total


def interpolate_invalid_trace_gaps(values, invalid_mask, max_gap_frames=5):
    """Set invalid scalar frames to NaN and linearly interpolate short invalid runs."""
    # Work on a numeric copy so the original scalar trace is unchanged.
    values = np.asarray(values, dtype=float).copy()
    # Normalize the invalid mask before applying it to the trace.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    # Invalid scalar values are blanked before short-gap interpolation.
    values[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(values)
    # Frame numbers provide the interpolation x-axis.
    x_index = np.arange(n_frames)
    i = 0
    # Scan contiguous invalid runs and handle each run immediately.
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
        # Fill only short internal gaps with finite values on both sides.
        if (
                gap_len <= max_gap_frames
                and left >= 0
                and right < n_frames
                and np.isfinite(values[left])
                and np.isfinite(values[right])
        ):
            # Linear interpolation preserves the trace length and frame alignment.
            values[start:stop] = np.interp(
                x_index[start:stop],
                [left, right],
                [values[left], values[right]],
            )
            interpolated_total += gap_len
    return values, interpolated_total

def _count_true(values):
    # Centralize boolean counting so masks are converted consistently.
    values = np.asarray(values, dtype=bool)
    return int(np.count_nonzero(values))


def _fraction(count, total):
    # Empty windows do not have a meaningful fraction.
    if total == 0:
        return np.nan
    return float(count) / float(total)


def _true_run_lengths(values):
    # Convert any boolean-like sequence into a clean mask.
    values = np.asarray(values, dtype=bool)
    lengths = []
    run_length = 0
    # Walk the mask and count consecutive True runs.
    for value in values:
        if value:
            run_length += 1
        elif run_length:
            # A False value terminates the current invalid run.
            lengths.append(run_length)
            run_length = 0
    # Preserve a run that reaches the last frame.
    if run_length:
        lengths.append(run_length)
    return lengths


