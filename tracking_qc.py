"""Configuration helpers shared by tracking-QC-aware plotting workflows."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackingQCConfig:
    """Tracking-QC options without coupling them to a plotting workflow."""

    # Master switch: when False, callers can bypass all QC filtering.
    enabled: bool = False
    # Reprojection-error cutoff used to mark a frame invalid.
    error_max: float = 50
    # Minimum accepted score for the required PointName_score channel.
    score_min: float = 0.8
    # Minimum number of cameras contributing to the reconstructed 3D point.
    min_cameras: int = 2
    # Maximum invalid gap that can be linearly interpolated, expressed in seconds.
    max_interp_gap_s: float = 0.02
    # Minimum fraction of valid frames required within the analysis window.
    min_valid_fraction: float = 0.7

    @property
    def max_invalid_fraction(self):
        # Convert the valid-frame threshold into the equivalent invalid-frame limit.
        return 1.0 - float(self.min_valid_fraction)

    def output_metadata(self):
        """Return compact QC metadata for result dataframes."""
        # Routine outputs now keep QC diagnostics in summarize_invalid_mask()
        # instead of repeating threshold/provenance columns in every result row.
        return {}


def build_config(
        apply_tracking_qc=False,
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
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
    )


def interp_gap_frames_from_fps(max_interp_gap_s, fps):
    """Convert the time-based interpolation threshold into trial-local frames."""
    # A finite positive FPS is required because the interpolation rule is time-based.
    if fps is None or pd.isna(fps) or float(fps) <= 0:
        raise ValueError("fps must be finite and > 0 to resolve time-based QC interpolation.")
    # Round to the nearest whole-frame gap and keep at least one frame interpolable.
    return max(1, int(round(float(max_interp_gap_s) * float(fps))))


def point_invalid_components(
        point,
        keypoint,
        min_cameras=2,
        error_max=50,
        score_min=0.8,
):
    """Return frame-wise invalid components for one 3D keypoint trace."""
    # Convert point attributes to numeric arrays so NaN/finite checks are consistent.
    x = np.asarray(point.x_coord, dtype=float)
    y = np.asarray(point.y_coord, dtype=float)
    z = np.asarray(point.z_coord, dtype=float)
    camera_count = np.asarray(point.camera_count, dtype=float)
    error = np.asarray(point.error, dtype=float)
    score = np.asarray(point.score, dtype=float)
    # Required 3D channels must be frame-aligned; missing values should be NaN,
    # not shorter arrays that silently change the analyzed window.
    required_lengths = {
        "x": len(x),
        "y": len(y),
        "z": len(z),
        "camera_count": len(camera_count),
        "error": len(error),
        "score": len(score),
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

    # Missing score values invalidate a frame because confidence quality is unknown.
    score_missing = ~np.isfinite(score)
    # Scores below the caller threshold invalidate a frame.
    score_low = np.isfinite(score) & (score < score_min)

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
        "Min_Cameras": min_cameras,
        "Score_Min": score_min,
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

    # Main summary used by analysis functions and QC diagnostic plots. This is
    # intentionally limited to the compact diagnostic fields requested for the
    # active analysis workflow.
    summary = {
        "QC_Passed": len(exclusion_reasons) == 0,
        "QC_Exclusion_Reason": ";".join(exclusion_reasons),
        "Valid_Frame_Fraction": valid_fraction,
        "Invalid_Frame_Fraction": invalid_fraction,
        "Max_Invalid_Gap_Frames": max_gap,
        "Interpolated_Frame_Count": interpolatable_count,
        "Max_Interp_Gap_Frames": max_interp_gap_frames,
    }

    if components is not None:
        # Component masks are still accepted for API compatibility, but routine
        # summaries no longer export per-component diagnostic counts/fractions.
        pass

    return summary


def qc_keypoint_xyz(
        point,
        keypoint,
        fps,
        start_frame=None,
        end_frame=None,
        min_cameras=2,
        error_max=50,
        score_min=0.8,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        require_start_end_valid=False,
):
    """
    Run the active keypoint-first tracking QC rule for one xyz trace.

    This is the main public QC entry point: it validates one keypoint, records
    the compact diagnostic summary, and returns interpolated xyz only when the
    keypoint passes the analysis-window QC rule.
    """
    # Resolve the time-based interpolation threshold using this trial's FPS.
    max_interp_gap_frames = interp_gap_frames_from_fps(max_interp_gap_s, fps)
    # Build frame-wise invalid components from raw xyz/camera/error/score data.
    xyz, components, metadata = point_invalid_components(
        point=point,
        keypoint=keypoint,
        min_cameras=min_cameras,
        error_max=error_max,
        score_min=score_min,
    )
    # The combined invalid mask is the single source of truth for pass/fail.
    invalid_mask = components["invalid"]
    # Default the QC window to the full trace when no analysis window is supplied.
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(invalid_mask) - 1
    # Clamp the requested QC window to the available frame range.
    start_frame = max(int(start_frame), 0)
    end_frame = min(int(end_frame), len(invalid_mask) - 1)
    # Summarize the keypoint with only the compact diagnostic fields used by analysis.
    summary = summarize_invalid_mask(
        invalid_mask,
        start_frame=start_frame,
        end_frame=end_frame,
        max_interp_gap_frames=max_interp_gap_frames,
        max_interp_gap_s=max_interp_gap_s,
        fps=fps,
        min_valid_fraction=min_valid_fraction,
        require_start_end_valid=require_start_end_valid,
    )
    # Interpolate eligible short xyz gaps so passed keypoints are ready for geometry.
    interpolated_xyz, interpolated_count = interpolate_invalid_xyz_gaps(
        xyz,
        invalid_mask,
        max_gap_frames=max_interp_gap_frames,
    )
    # Record the actual number of frames filled rather than the candidate count.
    summary["Interpolated_Frame_Count"] = int(interpolated_count)
    # Identify this keypoint without reintroducing excessive per-component diagnostics.
    summary["Keypoint"] = keypoint
    # Keep threshold provenance separate so qc_summary remains compact and readable.
    qc_metadata = {
        "Min_Cameras": min_cameras,
        "Error_Max": error_max,
        "Score_Min": score_min,
        "Max_Interp_Gap_s": max_interp_gap_s,
    }
    # Downstream analysis receives cleaned xyz only when this keypoint passes QC.
    clean_xyz = interpolated_xyz if summary["QC_Passed"] else None
    # Return a dictionary so callers can access data, mask, and diagnostics explicitly.
    return {
        "clean_xyz": clean_xyz,
        "invalid_mask": invalid_mask,
        "qc_summary": summary,
        "qc_metadata": qc_metadata,
    }


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
