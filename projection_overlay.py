#!/usr/bin/env python
"""
Specialized overlay helper for the 2D projection example.

It reads one frame from the example MP4, reads the matching row from the
example H5 file, and saves a PNG with the 2D projected points overlaid.
"""

from pathlib import Path
import re

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent / "2D projection"
EXAMPLE_STEM = "2025-01-30-11-23-30.38_LPAcrossLegsJoints_T3-TiTa_Fly_1_Trial_20_Cam5"
DEFAULT_VIDEO_PATH = EXAMPLE_DIR / f"{EXAMPLE_STEM}.mp4"
DEFAULT_H5_PATH = EXAMPLE_DIR / f"{EXAMPLE_STEM}.h5"
DEFAULT_OUTPUT_PATH = EXAMPLE_DIR / "projection_overlay_frame.pdf"
DEFAULT_KINEMATIC_CSV_PATH = None


SKELETON = [
    # ("L-wing", "L-wing-hinge"),
    # ("R-wing", "R-wing-hinge"),
    # ("R-fBC", "R-fCT"),
    # ("R-fCT", "R-fFT"),
    # ("R-fFT", "R-fTT"),
    # ("R-fTT", "R-fLT"),
    # ("R-mBC", "R-mCT"),
    # ("R-mCT", "R-mFT"),
    # ("R-mFT", "R-mTT"),
    # ("R-mTT", "R-mLT"),
    # ("R-hBC", "R-hCT"),
    # ("R-hCT", "R-hFT"),
    # ("R-hFT", "R-hTT"),
    # ("R-hTT", "R-hLT"),
    # ("L-fBC", "L-fCT"),
    # ("L-fCT", "L-fFT"),
    # ("L-fFT", "L-fTT"),
    # ("L-fTT", "L-fLT"),
    # ("L-mBC", "L-mCT"),
    # ("L-mCT", "L-mFT"),
    # ("L-mFT", "L-mTT"),
    # ("L-mTT", "L-mLT"),
    ("L-hBC", "L-hCT"),
    ("L-hCT", "L-hFT"),
    ("L-hFT", "L-hTT"),
    ("L-hTT", "L-hLT"),
]

BODY_SEGMENT_GROUPS = [
    # ["L-wing", "L-wing-hinge"],
    # ["R-wing", "R-wing-hinge"],
    # ["abdomen-tip"],
    # ["platform-tip"],
    # ["L-platform-tip"],
    # ["R-platform-tip"],
    # ["platform-axis"],
    # ["R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT"],
    # ["R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT"],
    # ["R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT"],
    # ["L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT"],
    # ["L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT"],
    ["L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"],
]

DEFAULT_SEGMENT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#E45756",
    "#EECA3B",
    "#1F77B4",
    "#D62728",
    "#2CA02C",
    "#6BAED6",
    "#FB6A4A",
    "#74C476",
]

POINT_TO_SEGMENT_COLOR = {
    point: DEFAULT_SEGMENT_COLORS[group_idx]
    for group_idx, points in enumerate(BODY_SEGMENT_GROUPS)
    for point in points
}
SKELETON_COLORS = {
    (point_a, point_b): POINT_TO_SEGMENT_COLOR.get(point_a, "cyan")
    for point_a, point_b in SKELETON
}


def _read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= frame_count:
            raise ValueError(f"frame_idx {frame_idx} is outside video range 0-{frame_count - 1}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _read_projection_table(h5_path):
    table = pd.read_hdf(h5_path)
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"Expected the H5 file to contain a pandas DataFrame: {h5_path}")
    return table


def _extract_points(table, frame_idx, likelihood_threshold=0.0):
    if frame_idx < 0 or frame_idx >= len(table):
        raise ValueError(f"frame_idx {frame_idx} is outside H5 row range 0-{len(table) - 1}")

    row = table.iloc[int(frame_idx)]
    points = {}

    if isinstance(table.columns, pd.MultiIndex):
        columns = table.columns
        coord_level = None
        for level in range(columns.nlevels):
            values = {str(value).lower() for value in columns.get_level_values(level)}
            if {"x", "y"}.issubset(values):
                coord_level = level
                break
        if coord_level is None:
            raise ValueError("Could not identify x/y coordinate level in H5 MultiIndex columns.")

        bodypart_level = coord_level - 1 if coord_level > 0 else 0
        bodyparts = sorted(set(columns.get_level_values(bodypart_level)))
        for bodypart in bodyparts:
            try:
                x_col = next(col for col in columns if col[bodypart_level] == bodypart and str(col[coord_level]).lower() == "x")
                y_col = next(col for col in columns if col[bodypart_level] == bodypart and str(col[coord_level]).lower() == "y")
            except StopIteration:
                continue

            likelihood = 1.0
            likelihood_cols = [
                col for col in columns
                if col[bodypart_level] == bodypart and str(col[coord_level]).lower() in {"likelihood", "probability"}
            ]
            if likelihood_cols:
                likelihood = row[likelihood_cols[0]]

            x_value = row[x_col]
            y_value = row[y_col]
            if np.isfinite(x_value) and np.isfinite(y_value) and likelihood >= likelihood_threshold:
                points[str(bodypart)] = (float(x_value), float(y_value))
        return points

    raise ValueError("This helper expects a DeepLabCut-style H5 file with MultiIndex columns.")


def _collect_point_traces(table, start_idx, end_idx, likelihood_threshold=0.0):
    if start_idx > end_idx:
        raise ValueError("start_idx must be less than or equal to end_idx.")
    if start_idx < 0 or end_idx >= len(table):
        raise ValueError(f"Frame range {start_idx}-{end_idx} is outside H5 row range 0-{len(table) - 1}")

    traces = {}
    for frame_idx in range(int(start_idx), int(end_idx) + 1):
        points = _extract_points(table, frame_idx, likelihood_threshold=likelihood_threshold)
        for point_name, coord in points.items():
            traces.setdefault(point_name, []).append((frame_idx, coord[0], coord[1]))
    return traces


def _resolve_point_colors(point_colors=None, segment_colors=None):
    resolved = dict(POINT_TO_SEGMENT_COLOR)

    if segment_colors is not None:
        for segment_name, color in segment_colors.items():
            matching_groups = [
                group for group in BODY_SEGMENT_GROUPS
                if segment_name in group or segment_name == "-".join(group)
            ]
            for group in matching_groups:
                for point in group:
                    resolved[point] = color

    if point_colors is not None:
        resolved.update(point_colors)

    return resolved


def _resolve_skeleton_colors(point_colors):
    return {
        (point_a, point_b): point_colors.get(point_a, SKELETON_COLORS.get((point_a, point_b), "cyan"))
        for point_a, point_b in SKELETON
    }


def _segment_edges(segment):
    return list(zip(segment[:-1], segment[1:]))


def _trajectory_path_efficiency(rows):
    coords = np.asarray(rows, dtype=float)[:, 1:3]
    path_length = float(np.nansum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    displacement = float(np.linalg.norm(coords[-1] - coords[0]))
    if path_length <= 1e-12:
        return np.nan
    return displacement / path_length


def _infer_kinematic_csv_path(video_path):
    """
    Infer the matching old-style kinematic CSV path from a video path.

    Example:
    ..._Fly_2_Trial_19_Cam5.mp4 -> ..._Fly_2_Trial_19_.csv

    The helper checks the same folder first and returns None when no candidate
    exists.
    """
    video_path = Path(video_path)
    stem = video_path.stem
    base_no_cam = re.sub(r"_Cam\d+$", "", stem, flags=re.IGNORECASE)
    candidates = [
        video_path.with_name(base_no_cam + "_.csv"),
        video_path.with_name(base_no_cam + ".csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_kinematic_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find kinematic CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _kinematic_path_metrics(table, point_name, start_idx, end_idx):
    """
    Calculate 3D path metrics for one keypoint over start_idx:end_idx.
    Expected columns: {point_name}_x, {point_name}_y, {point_name}_z.
    """
    columns = [f"{point_name}_{axis}" for axis in ("x", "y", "z")]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        return {
            "path_efficiency": np.nan,
            "path_length": np.nan,
            "displacement": np.nan,
            "n_valid_frames": 0,
            "error": f"missing columns: {missing}",
        }

    if start_idx > end_idx:
        raise ValueError("start_idx must be less than or equal to end_idx.")
    if start_idx < 0 or end_idx >= len(table):
        raise ValueError(f"Frame range {start_idx}-{end_idx} is outside CSV row range 0-{len(table) - 1}")

    coords = table.loc[int(start_idx):int(end_idx), columns].to_numpy(dtype=float)
    valid = np.all(np.isfinite(coords), axis=1)
    coords = coords[valid]
    if len(coords) < 2:
        return {
            "path_efficiency": np.nan,
            "path_length": np.nan,
            "displacement": np.nan,
            "n_valid_frames": int(len(coords)),
            "error": "fewer than 2 valid frames",
        }

    steps = np.diff(coords, axis=0)
    path_length = float(np.sum(np.linalg.norm(steps, axis=1)))
    displacement = float(np.linalg.norm(coords[-1] - coords[0]))
    path_efficiency = np.nan
    if path_length > 1e-12:
        path_efficiency = displacement / path_length

    return {
        "path_efficiency": path_efficiency,
        "path_length": path_length,
        "displacement": displacement,
        "n_valid_frames": int(len(coords)),
        "error": "",
    }


def _time_gradient_color(base_color, fraction, start_mix=0.70):
    """
    Blend from a dark version of base_color to base_color across time.

    fraction should be 0 at the start frame and 1 at the end frame.
    """
    base_rgb = np.asarray(mcolors.to_rgb(base_color), dtype=float)
    black = np.zeros(3, dtype=float)
    early_rgb = black * start_mix + base_rgb * (1.0 - start_mix)
    return tuple(early_rgb * (1.0 - fraction) + base_rgb * fraction)


def create_projection_overlay(
        frame_idx,
        video_path=DEFAULT_VIDEO_PATH,
        h5_path=DEFAULT_H5_PATH,
        output_path=None,
        likelihood_threshold=0.0,
        draw_labels=True
):
    """
    Save a single video frame with matching 2D projection points overlaid.

    frame_idx is zero-based and must match the row index in the H5 file.
    """
    video_path = Path(video_path)
    h5_path = Path(h5_path)
    output_path = Path(output_path) if output_path is not None else (
        DEFAULT_OUTPUT_PATH.with_name(f"projection_overlay_frame_{frame_idx}.png")
    )

    frame = _read_video_frame(video_path, frame_idx)
    table = _read_projection_table(h5_path)
    points = _extract_points(table, frame_idx, likelihood_threshold=likelihood_threshold)
    if not points:
        raise ValueError(f"No valid projection points found for frame {frame_idx}.")

    fig, ax = plt.subplots(figsize=(8, 8 * frame.shape[0] / frame.shape[1]))
    ax.imshow(frame)

    for point_a, point_b in SKELETON:
        if point_a in points and point_b in points:
            x_values = [points[point_a][0], points[point_b][0]]
            y_values = [points[point_a][1], points[point_b][1]]
            ax.plot(x_values, y_values, color="cyan", linewidth=1.4, alpha=0.85)

    xs = [coord[0] for coord in points.values()]
    ys = [coord[1] for coord in points.values()]
    ax.scatter(xs, ys, s=22, color="yellow", edgecolor="black", linewidth=0.5, zorder=3)

    if draw_labels:
        for name, (x_value, y_value) in points.items():
            ax.text(x_value + 3, y_value + 3, name, color="white", fontsize=6, zorder=4)

    ax.set_title(f"2D projection overlay, frame {frame_idx}")
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def create_projection_trace_overlay(
        start_idx,
        end_idx,
        background_frame_idx=None,
        video_path=DEFAULT_VIDEO_PATH,
        h5_path=DEFAULT_H5_PATH,
        kinematic_csv_path=DEFAULT_KINEMATIC_CSV_PATH,
        output_path=None,
        likelihood_threshold=0.0,
        draw_labels=False,
        point_colors=None,
        segment_colors=None,
        fps=250,
        trace_alpha=0.75,
        trace_linewidth=1.4,
        segment_frame_step=1,
        trajectory_points=None,
        show_start_end=False,
        trace_offset_x=0.0,
        trace_offset_y=0.0,
        show_kinematic_metrics=True,
        metric_points=None,
        title=None
):
    """
    Save one frame with selected keypoint movement trajectories overlaid.

    start_idx and end_idx define the trace window. background_frame_idx chooses
    the video frame shown under the traces; by default it uses end_idx.

    point_colors can override individual bodypart colors:
        {"L-fTT": "magenta", "L-mTT": "orange"}

    segment_colors can override a whole body segment by naming one point in it:
        {"L-fBC": "magenta", "R-mBC": "orange"}

    trajectory_points controls which keypoints are traced. By default, all TT
    keypoints found in BODY_SEGMENT_GROUPS are traced. Set trajectory_points to
    a list such as ["L-fTT", "L-mTT", "L-hTT"] to draw only those trajectories.

    trace_offset_x and trace_offset_y shift the plotted 2D traces only. They do
    not affect 3D path-efficiency metrics.

    kinematic_csv_path can be provided directly. If omitted, the function tries
    to infer the old-style matching CSV by removing the trailing _Cam# token
    from the video filename and checking the same folder.
    """
    video_path = Path(video_path)
    h5_path = Path(h5_path)
    background_frame_idx = int(end_idx if background_frame_idx is None else background_frame_idx)
    output_path = Path(output_path) if output_path is not None else (
        DEFAULT_OUTPUT_PATH.with_name(f"projection_trace_overlay_{start_idx}_{end_idx}.pdf")
    )

    frame = _read_video_frame(video_path, background_frame_idx)
    table = _read_projection_table(h5_path)
    if trajectory_points is None:
        trajectory_points = [
            point
            for segment in BODY_SEGMENT_GROUPS
            for point in segment
            if point.endswith("TT")
        ]
    trajectory_points = list(dict.fromkeys(trajectory_points))

    point_traces = {point: [] for point in trajectory_points}
    for frame_idx in range(int(start_idx), int(end_idx) + 1, int(segment_frame_step)):
        points = _extract_points(table, frame_idx, likelihood_threshold=likelihood_threshold)
        for point_name in trajectory_points:
            if point_name in points:
                x_value, y_value = points[point_name]
                point_traces[point_name].append((frame_idx, x_value, y_value))

    point_color_map = _resolve_point_colors(point_colors=point_colors, segment_colors=segment_colors)

    metrics_by_point = {}
    metric_warning = None
    if show_kinematic_metrics:
        if metric_points is None:
            metric_points = trajectory_points
        metric_points = list(dict.fromkeys(metric_points))

        resolved_csv_path = Path(kinematic_csv_path) if kinematic_csv_path is not None else _infer_kinematic_csv_path(video_path)
        if resolved_csv_path is None:
            metric_warning = "Kinematic CSV not found"
        else:
            kinematic_table = _read_kinematic_csv(resolved_csv_path)
            for point_name in metric_points:
                metrics_by_point[point_name] = _kinematic_path_metrics(
                    kinematic_table,
                    point_name,
                    int(start_idx),
                    int(end_idx)
                )

    fig, ax = plt.subplots(figsize=(8, 8 * frame.shape[0] / frame.shape[1]))
    ax.imshow(frame)

    for point_name, rows in point_traces.items():
        if len(rows) < 2:
            continue
        rows = np.asarray(rows, dtype=float)
        base_color = point_color_map.get(point_name, "yellow")
        x_values = rows[:, 1] + float(trace_offset_x)
        y_values = rows[:, 2] + float(trace_offset_y)

        for i in range(len(rows) - 1):
            ax.plot(
                x_values[i:i + 2],
                y_values[i:i + 2],
                color=base_color,
                linewidth=trace_linewidth,
                alpha=trace_alpha,
                zorder=3
            )

        if show_start_end:
            ax.scatter(
                x_values[0],
                y_values[0],
                s=22,
                color=_time_gradient_color(base_color, 0.0),
                edgecolor="black",
                linewidth=0.4,
                zorder=4
            )
            ax.scatter(
                x_values[-1],
                y_values[-1],
                s=34,
                color=base_color,
                edgecolor="black",
                linewidth=0.5,
                zorder=5
            )
        if draw_labels:
            ax.text(
                x_values[-1] + 3,
                y_values[-1] + 3,
                point_name,
                color="white",
                fontsize=6,
                zorder=6
            )

    latency_s = (int(end_idx) - int(start_idx)) / float(fps)
    label_lines = [(f"Latency: {latency_s:.2f}s", "white")]
    if show_kinematic_metrics:
        if metric_warning is not None:
            label_lines.append((metric_warning, "white"))
        else:
            for point_name in (metric_points if metric_points is not None else []):
                metric = metrics_by_point.get(point_name, {})
                pe = metric.get("path_efficiency", np.nan)
                if np.isfinite(pe):
                    label = f"{point_name} path efficiency: {pe:.2f}"
                else:
                    label = f"{point_name} path efficiency: nan"
                label_lines.append((label, point_color_map.get(point_name, "white")))

    # Draw one background box, then colored text lines on top of it.
    ax.text(
        0.02,
        0.98,
        "\n".join(text for text, _ in label_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=(1, 1, 1, 0),
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.5),
        zorder=10
    )
    line_step = 0.038
    for line_idx, (text, color) in enumerate(label_lines):
        ax.text(
            0.025,
            0.975 - line_idx * line_step,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=11
        )

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Selected keypoint trajectories, frames {start_idx}-{end_idx}")
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def create_path_efficiency_schematic(
        start_idx,
        end_idx,
        video_path=DEFAULT_VIDEO_PATH,
        h5_path=DEFAULT_H5_PATH,
        output_path=None,
        background_frame_idx=None,
        likelihood_threshold=0.0,
        trajectory_points=None,
        point_colors=None,
        segment_colors=None,
        segment_frame_step=1,
        trace_alpha=0.95,
        trace_linewidth=2.2
):
    """
    Create a two-panel path-efficiency figure.

    Top: video frame with selected keypoint trajectories overlaid.
    Bottom: white-background schematic. The colored line is the trajectory; the
    black dashed line connects the start and end positions as displacement.
    """
    video_path = Path(video_path)
    h5_path = Path(h5_path)
    background_frame_idx = int(end_idx if background_frame_idx is None else background_frame_idx)
    output_path = Path(output_path) if output_path is not None else (
        DEFAULT_OUTPUT_PATH.with_name(f"path_efficiency_overlay_schematic_{start_idx}_{end_idx}.pdf")
    )

    frame = _read_video_frame(video_path, background_frame_idx)
    table = _read_projection_table(h5_path)
    if trajectory_points is None:
        trajectory_points = [
            point
            for segment in BODY_SEGMENT_GROUPS
            for point in segment
            if point.endswith("TT")
        ]
    trajectory_points = list(dict.fromkeys(trajectory_points))

    point_traces = {point: [] for point in trajectory_points}
    for frame_idx in range(int(start_idx), int(end_idx) + 1, int(segment_frame_step)):
        points = _extract_points(table, frame_idx, likelihood_threshold=likelihood_threshold)
        for point_name in trajectory_points:
            if point_name in points:
                x_value, y_value = points[point_name]
                point_traces[point_name].append((frame_idx, x_value, y_value))

    point_color_map = _resolve_point_colors(point_colors=point_colors, segment_colors=segment_colors)

    aspect = frame.shape[0] / frame.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(8, 16 * aspect), facecolor="white")
    ax_overlay, ax_schematic = axes
    ax_overlay.imshow(frame)
    ax_schematic.set_facecolor("white")
    metric_lines = []

    for point_name, rows in point_traces.items():
        if len(rows) < 2:
            continue

        rows = np.asarray(rows, dtype=float)
        coords = rows[:, 1:3]
        base_color = point_color_map.get(point_name, "black")
        path_efficiency = _trajectory_path_efficiency(rows)

        ax_overlay.plot(
            coords[:, 0],
            coords[:, 1],
            color=base_color,
            linewidth=trace_linewidth,
            alpha=trace_alpha
        )
        ax_schematic.plot(
            coords[:, 0],
            coords[:, 1],
            color=base_color,
            linewidth=trace_linewidth,
            alpha=trace_alpha,
            label=point_name
        )
        ax_schematic.plot(
            [coords[0, 0], coords[-1, 0]],
            [coords[0, 1], coords[-1, 1]],
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95
        )
        ax_schematic.scatter(coords[0, 0], coords[0, 1], color=base_color, s=34, edgecolor="black", zorder=3)
        ax_schematic.scatter(coords[-1, 0], coords[-1, 1], color=base_color, s=42, marker="s", edgecolor="black", zorder=3)
        metric_lines.append(f"{point_name}: PE={path_efficiency:.2f}")

    if not metric_lines:
        raise ValueError("No valid trajectories were found for the selected keypoints and frame window.")

    ax_overlay.set_title(f"Selected keypoint trajectories, frames {start_idx}-{end_idx}")
    ax_overlay.set_axis_off()

    ax_schematic.text(
        1.02,
        0.98,
        "\n".join(metric_lines),
        transform=ax_schematic.transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=10
    )
    ax_schematic.set_aspect("equal", adjustable="datalim")
    ax_schematic.invert_yaxis()
    ax_schematic.set_xlabel("Projected x")
    ax_schematic.set_ylabel("Projected y")
    ax_schematic.legend(frameon=False, loc="lower left")
    ax_schematic.spines["top"].set_visible(False)
    ax_schematic.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 0.78, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def create_synthetic_path_efficiency_schematic(
        output_path=None,
        colors=("tab:blue", "tab:orange", "tab:green"),
        random_state=3
):
    """
    Create a standalone schematic explaining path efficiency.

    This function uses artificial trajectories only. It does not read video,
    H5 files, or any external data.
    """
    output_path = Path(output_path) if output_path is not None else (
        EXAMPLE_DIR / "synthetic_path_efficiency_schematic.pdf"
    )

    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 1, 220)
    start = np.asarray([0.0, 0.0])
    end = np.asarray([5.0, 0.0])

    def smooth_random_path(amplitude, n_control_points):
        control_x = np.linspace(start[0], end[0], n_control_points)
        control_y = rng.normal(0.0, amplitude, size=n_control_points)
        control_y[0] = start[1]
        control_y[-1] = end[1]

        dense_x = np.linspace(start[0], end[0], len(t))
        dense_y = np.interp(dense_x, control_x, control_y)

        window = max(9, int(len(t) * 0.12))
        if window % 2 == 0:
            window += 1
        kernel_x = np.linspace(-2.5, 2.5, window)
        kernel = np.exp(-0.5 * kernel_x ** 2)
        kernel = kernel / kernel.sum()
        pad = window // 2
        dense_y = np.convolve(np.pad(dense_y, pad, mode="edge"), kernel, mode="valid")
        dense_y[0] = start[1]
        dense_y[-1] = end[1]
        return np.column_stack([dense_x, dense_y])

    paths = {
        "Mildly indirect": smooth_random_path(amplitude=0.35, n_control_points=5),
        "Moderately indirect": smooth_random_path(amplitude=0.75, n_control_points=7),
        "Inefficient": smooth_random_path(amplitude=1.25, n_control_points=10),
    }

    def metrics(coords):
        path_length = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        displacement = float(np.linalg.norm(coords[-1] - coords[0]))
        efficiency = displacement / path_length if path_length > 1e-12 else np.nan
        return path_length, displacement, efficiency

    fig, ax = plt.subplots(figsize=(9.2, 6.0), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="black",
        linestyle="--",
        linewidth=2.3
    )
    ax.text(
        5.15,
        0.08,
        "Displacement",
        ha="left",
        va="bottom",
        fontsize=11,
        color="black"
    )

    for i, (name, coords) in enumerate(paths.items()):
        color = colors[i % len(colors)]
        path_length, displacement, efficiency = metrics(coords)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=3.0
        )

    ax.set_title("Path efficiency = displacement / path length")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-0.25, 8.1)
    all_y = np.concatenate([coords[:, 1] for coords in paths.values()])
    y_pad = max(0.35, 0.12 * (float(np.max(all_y)) - float(np.min(all_y))))
    ax.set_ylim(float(np.min(all_y)) - y_pad, float(np.max(all_y)) + y_pad)
    ax.set_axis_off()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # Edit these indices for this specialized example.
    # create_projection_trace_overlay(start_idx=250, end_idx=298)

    create_projection_trace_overlay(
        start_idx=272,
        end_idx=428,
        background_frame_idx=272,
        title="HL-R-ti-ta",
        trajectory_points=["L-fTT", "L-mTT", "L-hTT"],
        point_colors={
            "L-fTT": "magenta",
            "L-mTT": "orange",
            "L-hTT": "cyan",
        },
        fps=200,
        trace_offset_x=-10,
        trace_offset_y=-15,
    )

    """create_path_efficiency_schematic(
        start_idx=362,
        end_idx=467,
        background_frame_idx=362,
        trajectory_points=["L-fTT", "L-mTT", "L-hTT"],
        point_colors={
            "L-fTT": "magenta",
            "L-mTT": "orange",
            "L-hTT": "cyan",
        }
    )"""
