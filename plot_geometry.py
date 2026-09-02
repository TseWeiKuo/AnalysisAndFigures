"""TT geometry and trajectory plotting workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""

import math
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tracking_qc as tqc
import trial_helpers as th
from survival_stats_runner import SurvivalStatsRunner


# Final WT contact-geometry figures use one fixed projected-TT geometry definition.
TT_TRAJECTORY_JOINTS = ("L-fTT", "L-mTT", "L-hTT")
TT_TRAJECTORY_PLANE_AXIS = ("R-mBC", "L-mBC")
TT_TRAJECTORY_ORIGIN_KEYPOINT = "R-mBC"
TT_GEOMETRY_TRIAL_TYPES = ("Landing", "Flying")
TT_PATH_EFFICIENCY_LEG = "L-h"


def _get_stats_runner(self):
    # PlotCreator provides the shared stats runner; direct module calls fall
    # back to a local instance without changing plotting behavior.
    return getattr(self, "stats_runner", SurvivalStatsRunner())


def _load_sc_lookup(calculator, sc_csv_path, legs):
    """Load a secondary-contact CSV keyed by the standard trial index."""
    # SLC-adjusted windows need a secondary-contact table because each leg can
    # have its own valid secondary-contact frame.
    if sc_csv_path is None:
        raise ValueError("sc_csv_path is required for the finalized SLC-adjusted TT window.")

    # The CSV must contain the trial Index column plus one column per leg,
    # for example L-f, L-m, and L-h.
    sc_df = pd.read_csv(sc_csv_path)
    required_columns = {"Index", *legs}
    missing_columns = required_columns.difference(sc_df.columns)
    if missing_columns:
        raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

    # Parse the human-readable Index cell into the same tuple form used by the
    # Group/Trial data structures, then store the whole CSV row for lookup.
    lookup = {}
    for _, sc_row in sc_df.iterrows():
        index = calculator.parse_index_cell(sc_row["Index"])
        lookup[tuple(index)] = sc_row
    return lookup


def _select_slc_adjusted_tt_window_end(
        calculator,
        sc_lookup,
        index,
        leg,
        moc,
        mol,
        fps,
        total_frames,
        outcome,
        tau
):
    """Choose the finalized SLC-adjusted TT endpoint."""
    # The finalized figures use SLC when present; otherwise success falls back
    # to MOL and failed/missing-MOL trials fall back to the tau-censored window.
    if outcome == "Success" and not pd.isna(mol) and mol > moc:
        valid_end = int(min(mol, total_frames - 1))
        no_sc_rule = "MOC_to_MOL_no_valid_SLC"
    elif outcome == "Success":
        valid_end = int(min(moc + tau * fps, total_frames - 1))
        no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC_missing_MOL"
    else:
        valid_end = int(min(moc + tau * fps, total_frames - 1))
        no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC"

    # Pull the trial's secondary-contact row. Missing rows or missing leg
    # columns fall back to the no-SLC endpoint.
    sc_row = sc_lookup.get(tuple(index))
    if sc_row is None or leg not in sc_row:
        return valid_end, no_sc_rule, np.nan, False

    # Accept the SLC frame only if it lies inside the MOC-to-valid_end window.
    is_valid, sc_frame = calculator.validate_sc_frame_window(
        sc_row[leg],
        moc,
        valid_end
    )
    if is_valid:
        return int(min(sc_frame, total_frames - 1)), "MOC_to_SLC", sc_frame, True
    return valid_end, no_sc_rule, np.nan, False


def _calculate_tt_metrics(tt_xyz, fps, min_frames=3, min_path_length=1e-6):
    """Calculate speed, efficiency, length, displacement, and duration."""
    # Drop frames with any non-finite x/y/z coordinate before measuring motion.
    tt_xyz = np.asarray(tt_xyz, dtype=float)
    valid = np.all(np.isfinite(tt_xyz), axis=1)
    tt_xyz = tt_xyz[valid]
    # Require enough samples to define at least a short trajectory.
    if len(tt_xyz) < min_frames:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Path length is the sum of frame-to-frame 3D step distances.
    steps = np.diff(tt_xyz, axis=0)
    path_length = np.sum(np.linalg.norm(steps, axis=1))
    # Displacement is the straight-line distance between the first and last
    # valid TT positions in the selected window.
    displacement = np.linalg.norm(tt_xyz[-1] - tt_xyz[0])
    # Duration uses the number of frame intervals, not the number of samples.
    duration_s = (len(tt_xyz) - 1) / fps
    average_speed = path_length / duration_s if duration_s > 0 else np.nan
    # Path efficiency approaches 1 for a straight path and decreases as the
    # trajectory becomes more circuitous.
    path_efficiency = (
        displacement / path_length
        if path_length > min_path_length
        else np.nan
    )
    return average_speed, path_efficiency, path_length, displacement, duration_s


def _empty_qc_summary():
    # Raw-mode rows keep the same compact QC columns without pretending QC ran.
    return {
        "QC_Passed": True,
        "QC_Exclusion_Reason": "",
        "Valid_Frame_Fraction": np.nan,
        "Invalid_Frame_Fraction": np.nan,
        "Max_Invalid_Gap_Frames": np.nan,
        "Interpolated_Frame_Count": np.nan,
        "Max_Interp_Gap_Frames": np.nan,
    }


def _compact_qc_fields(qc_summary):
    """Return only the compact QC fields used by analysis CSV outputs."""
    # Keep QC output stable and avoid reintroducing older verbose diagnostics.
    return {
        "QC_Passed": qc_summary.get("QC_Passed", True),
        "QC_Exclusion_Reason": qc_summary.get("QC_Exclusion_Reason", ""),
        "Valid_Frame_Fraction": qc_summary.get("Valid_Frame_Fraction", np.nan),
        "Invalid_Frame_Fraction": qc_summary.get("Invalid_Frame_Fraction", np.nan),
        "Max_Invalid_Gap_Frames": qc_summary.get("Max_Invalid_Gap_Frames", np.nan),
        "Interpolated_Frame_Count": qc_summary.get("Interpolated_Frame_Count", np.nan),
        "Max_Interp_Gap_Frames": qc_summary.get("Max_Interp_Gap_Frames", np.nan),
    }


def _qc_keypoint_xyz_for_window(
        trial_info,
        keypoint,
        start_frame,
        end_frame,
        min_cameras,
        max_interp_gap_s,
        min_valid_fraction,
        error_max,
        score_min,
        require_start_end_valid,
):
    """Run tracking QC directly on one keypoint for a geometry analysis window."""
    # Geometry uses the same keypoint-level QC entry point as the rest of the pipeline.
    qc_result = tqc.qc_keypoint_xyz(
        point=trial_info.trial_data[keypoint],
        keypoint=keypoint,
        fps=trial_info.fps,
        start_frame=start_frame,
        end_frame=end_frame,
        min_cameras=min_cameras,
        max_interp_gap_s=max_interp_gap_s,
        min_valid_fraction=min_valid_fraction,
        error_max=error_max,
        score_min=score_min,
        require_start_end_valid=require_start_end_valid,
    )
    # Keep only compact QC fields before returning to plotting/stat code.
    qc_summary = _compact_qc_fields(qc_result["qc_summary"])
    return qc_result["clean_xyz"], qc_summary


def _collect_TT_MOC_to_SLC_projected_data(
        self,
        group_info,
        sc_csv_paths,
        tau,
        axis_average_frames,
        apply_tracking_qc,
        min_cameras,
        max_interp_gap_s,
        min_valid_fraction,
        error_max=50,
        score_min=0.8,
):
    """Collect projected TT trajectory/endpoint data for combined TT plots."""
    # The finalized projected-TT workflow fixes the geometry definition and
    # keeps only the MOC-anchored axis-averaging frame count configurable.
    if axis_average_frames < 1:
        raise ValueError("axis_average_frames must be >= 1.")

    # Normalize group_info into a list of (plot label, Group object) pairs. This
    # allows callers to pass one Group, a list of Groups, or a label->Group dict.
    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    # A single SC CSV path is only unambiguous for a single group. Multi-group
    # plots need a dict so each group can use its own secondary-contact file.
    if not isinstance(sc_csv_paths, dict):
        if len(group_items) != 1:
            raise ValueError("sc_csv_paths must be a dict when plotting multiple groups.")
        sc_csv_paths = {group_items[0][0]: sc_csv_paths}

    # Every trial must contain TT joints, the plane-defining axis, the origin
    # keypoint, and platform-tip motion data for projection into 2D.
    required_points = (
        set(TT_TRAJECTORY_JOINTS)
        | set(TT_TRAJECTORY_PLANE_AXIS)
        | {TT_TRAJECTORY_ORIGIN_KEYPOINT, "platform-tip"}
    )
    rows = []
    trajectory_rows = []
    skipped_rows = []

    def unit(vector, name):
        # Convert any vector into a unit vector and reject degenerate axes,
        # because projection would be unstable with near-zero directions.
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{name} has near-zero length.")
        return vector / norm

    def average_slice(total_frames, moc, endpoint):
        # Finalized figures estimate the anatomical projection plane from the
        # fixed pre-MOC window only.
        start = max(moc - axis_average_frames, 0)
        stop = moc
        return None if stop <= start else slice(start, stop)

    def read_axis(point_a, point_b, trial_info, avg_slice):
        # Read two 3D keypoint traces, average each across avg_slice, and return
        # the mean location of point_a plus the point_a->point_b direction.
        coords_a = self.calculator.ReadAndTranspose(point_a, trial_info).astype(float)
        coords_b = self.calculator.ReadAndTranspose(point_b, trial_info).astype(float)
        mean_a = np.nanmean(coords_a[avg_slice], axis=0)
        mean_b = np.nanmean(coords_b[avg_slice], axis=0)
        return mean_a, mean_b - mean_a

    def platform_motion_axis(trial_info, start_frame=200, stop_frame=250):
        # Use platform-tip motion as the projected Y reference direction. The
        # best-fit motion axis is estimated by PCA/SVD over a fixed frame window.
        platform_xyz = self.calculator.ReadAndTranspose("platform-tip", trial_info).astype(float)
        start = max(int(start_frame), 0)
        stop = min(int(stop_frame) + 1, len(platform_xyz))
        if stop - start < 2:
            raise ValueError("platform-tip motion window has fewer than 2 frames.")

        coords = platform_xyz[start:stop]
        coords = coords[np.all(np.isfinite(coords), axis=1)]
        if len(coords) < 2:
            raise ValueError("platform-tip motion window has fewer than 2 finite coordinates.")

        # Center the coordinates before SVD so the first right-singular vector
        # describes direction of movement, not absolute position.
        centered = coords - np.nanmean(coords, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values[0] < 1e-8:
            raise ValueError("platform-tip motion window has near-zero movement.")
        motion = vh[0]

        # Orient the axis in the same direction as net platform movement so sign
        # is consistent across trials.
        net_motion = coords[-1] - coords[0]
        if np.dot(motion, net_motion) < 0:
            motion = -motion
        return motion, start, stop - 1

    def project_point(point, origin_3d, plane_normal, basis_x, basis_y):
        # Orthogonally project a 3D point onto the selected plane, then express
        # that projected point in the 2D basis defined by basis_x and basis_y.
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            return np.nan, np.nan
        point_on_plane = point - np.dot(point - origin_3d, plane_normal) * plane_normal
        relative = point_on_plane - origin_3d
        return float(np.dot(relative, basis_x)), float(np.dot(relative, basis_y))

    def lookup_sc_path(group_label, group):
        # Prefer the plotting label key, but fall back to the Group object's
        # internal name when looking up the SC CSV path.
        return sc_csv_paths.get(group_label, sc_csv_paths.get(group.group_name))

    for group_label, current_group in group_items:
        # Load this group's secondary-contact table and validate that it has
        # one column per TT leg being analyzed.
        sc_path = lookup_sc_path(group_label, current_group)
        if sc_path is None:
            raise ValueError(f"No SC CSV path provided for group '{group_label}'.")

        sc_df = pd.read_csv(sc_path)
        # The SC table must contain the finalized left-leg TT contact columns.
        required_sc_columns = {"Index"} | {joint.replace("TT", "") for joint in TT_TRAJECTORY_JOINTS}
        missing_sc_columns = required_sc_columns.difference(sc_df.columns)
        if missing_sc_columns:
            raise ValueError(f"{group_label} SC CSV is missing columns: {sorted(missing_sc_columns)}")
        sc_lookup = {
            tuple(self.calculator.parse_index_cell(sc_row["Index"])): sc_row
            for _, sc_row in sc_df.iterrows()
        }

        # Initialize metadata and kinematic traces on demand. This keeps the
        # plotting function usable from notebooks without requiring setup code.
        if len(current_group.trial_metadata) == 0:
            current_group.initialize_manual_data()
            current_group.filter_nan_fly()
        current_group.read_kinematic_data(list(TT_GEOMETRY_TRIAL_TYPES))

        for index in current_group.get_targeted_trials(list(TT_GEOMETRY_TRIAL_TYPES)):
            # Convert the fly/trial index into the key used by both metadata
            # and loaded kinematic data.
            index_tuple = tuple(index)
            key = current_group._trial_key(index[0], index[1])
            if key not in current_group.fly_kinematic_data or key not in current_group.trial_metadata:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing kinematic data or metadata"})
                continue

            # Pull trial objects and reject trials missing any points needed for
            # either projection geometry or TT trajectory extraction.
            trial_info = current_group.fly_kinematic_data[key]
            meta = current_group.trial_metadata[key]
            missing_points = [point for point in required_points if point not in trial_info.trial_data]
            if missing_points:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"missing required points: {missing_points}"})
                continue

            # MOC anchors the trajectory start. FPS converts frame differences
            # into seconds for trajectory averaging and output tables.
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing MOC/fps"})
                continue
            moc = int(moc)
            if moc < 0 or moc >= trial_info.total_frames_number:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"invalid MOC: {moc}"})
                continue

            # Define the maximum endpoint that can be considered for this trial:
            # successful trials use MOL; failed/censored trials use MOC + tau.
            outcome = th.classify_landing(meta, current_group.latency_threshold)
            if outcome == "Success":
                if pd.isna(mol) or mol <= moc:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"success trial missing valid MOL: MOL={mol}"})
                    continue
                valid_end = int(min(mol, trial_info.total_frames_number - 1))
                fallback_rule = "MOL_no_valid_SLC"
            else:
                valid_end = int(min(moc + tau * fps, trial_info.total_frames_number - 1))
                fallback_rule = "MOC_plus_tau_no_valid_SLC"

            if valid_end <= moc:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "endpoint window is empty"})
                continue

            # Choose the frames used to estimate the projection plane and axes.
            avg_slice = average_slice(trial_info.total_frames_number, moc, valid_end)
            if avg_slice is None:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "empty axis averaging window"})
                continue

            try:
                # The fixed plane axis defines the plane normal; platform-tip motion is
                # projected into that plane to define the vertical plotting axis.
                plane_origin, plane_vector = read_axis(
                    TT_TRAJECTORY_PLANE_AXIS[0],
                    TT_TRAJECTORY_PLANE_AXIS[1],
                    trial_info,
                    avg_slice
                )
                plane_normal = unit(plane_vector, "plane_axis")
                platform_motion, motion_start, motion_stop = platform_motion_axis(trial_info)
                platform_motion_on_plane = platform_motion - np.dot(platform_motion, plane_normal) * plane_normal
                basis_y = unit(platform_motion_on_plane, "platform-tip motion projected onto plane")
                # The x-axis is perpendicular to basis_y within the projection
                # plane, giving a right-handed 2D coordinate system.
                basis_x = unit(np.cross(basis_y, plane_normal), "platform-motion-derived projected x-axis")
            except ValueError as exc:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": str(exc)})
                continue

            # Set the coordinate origin from the selected origin keypoint at
            # MOC. All projected TT coordinates are reported relative to this
            # MOC-anchored origin.
            origin_xyz = self.calculator.ReadAndTranspose(TT_TRAJECTORY_ORIGIN_KEYPOINT, trial_info).astype(float)
            origin_3d = origin_xyz[moc]
            origin_x, origin_y = project_point(origin_3d, plane_origin, plane_normal, basis_x, basis_y)
            if not np.isfinite(origin_x) or not np.isfinite(origin_y):
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "invalid projected origin"})
                continue

            # For each TT joint, choose a leg-specific endpoint, collect the full
            # projected trajectory, and store endpoint/AEP/VEP landmark points.
            sc_row = sc_lookup.get(index_tuple)
            for joint in TT_TRAJECTORY_JOINTS:
                # Convert a keypoint name such as L-mTT into the SC CSV leg
                # column such as L-m.
                leg = joint.replace("TT", "")
                # Start from the trial-level fallback endpoint. This will be
                # replaced by SLC if the SC table provides a valid leg-specific
                # frame inside the trial window.
                endpoint_frame = valid_end
                endpoint_rule = fallback_rule
                slc_frame = np.nan
                slc_valid = False

                if sc_row is not None:
                    # Validate the candidate SLC frame against MOC and the
                    # trial's maximum valid endpoint.
                    slc_valid, candidate_slc_frame = self.calculator.validate_sc_frame_window(
                        sc_row[leg],
                        moc,
                        valid_end
                    )
                    if slc_valid:
                        endpoint_frame = int(min(candidate_slc_frame, trial_info.total_frames_number - 1))
                        endpoint_rule = "SLC"
                        slc_frame = int(candidate_slc_frame)

                if apply_tracking_qc:
                    # Geometry calls keypoint-level QC directly and receives cleaned xyz only on pass.
                    xyz, qc_summary = _qc_keypoint_xyz_for_window(
                        trial_info=trial_info,
                        keypoint=joint,
                        min_cameras=min_cameras,
                        error_max=error_max,
                        score_min=score_min,
                        max_interp_gap_s=max_interp_gap_s,
                        min_valid_fraction=min_valid_fraction,
                        start_frame=moc,
                        end_frame=endpoint_frame,
                        require_start_end_valid=True
                    )
                    if not qc_summary["QC_Passed"]:
                        skipped_rows.append({
                            "Group_Label": group_label,
                            "Index": str(index),
                            "Joint": joint,
                            "Reason": "failed TT tracking QC",
                            **qc_summary,
                        })
                        continue
                else:
                    # Without QC, use the raw 3D TT trace and fill QC columns
                    # with neutral/NaN values for consistent output schema.
                    xyz = self.calculator.ReadAndTranspose(joint, trial_info).astype(float)
                    qc_summary = _empty_qc_summary()

                # Project every finite TT coordinate from MOC through endpoint
                # into the 2D plane and store the trial-level trajectory rows.
                projected_trace = []
                for frame in range(moc, endpoint_frame + 1):
                    # Convert this frame's 3D TT position into plane coordinates.
                    x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    # Re-zero the projected coordinate system so the selected
                    # origin keypoint is at (0, 0).
                    projected_x = x - origin_x
                    projected_y = y - origin_y
                    # Keep an in-memory copy for finding AEP/VEP after the loop.
                    projected_trace.append((frame, projected_x, projected_y))
                    # Append one row per frame; these rows are later averaged
                    # by fly and also plotted as raw trial traces.
                    trajectory_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Outcome": outcome,
                        "TrialType": meta["TrialType"],
                        "Joint": joint,
                        "Leg": leg,
                        "Frame": int(frame),
                        "Time_From_MOC_s": (frame - moc) / fps,
                        "Projected_X": projected_x,
                        "Projected_Y": projected_y,
                        "Endpoint_Frame": int(endpoint_frame),
                        "Endpoint_Rule": endpoint_rule,
                        "SLC_Frame": slc_frame,
                        "SLC_Valid_For_Window": slc_valid,
                        "Reference_X_Source": "platform_tip_motion_best_fit_200_250",
                        "Platform_Motion_Start_Frame": motion_start,
                        "Platform_Motion_End_Frame": motion_stop,
                        **_compact_qc_fields(qc_summary),
                    })

                def append_point_row(point_type, frame, x, y, marker):
                    # Store special landmark points in a separate table from the
                    # full trajectory. These points drive radial displacement
                    # summaries and endpoint overlays.
                    rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Outcome": outcome,
                        "TrialType": meta["TrialType"],
                        "Joint": joint,
                        "Leg": leg,
                        "Point_Type": point_type,
                        "Marker": marker,
                        "Frame": int(frame),
                        "Projected_X": x,
                        "Projected_Y": y,
                        "Origin_Keypoint": TT_TRAJECTORY_ORIGIN_KEYPOINT,
                        "Origin_Frame_Mode": "moc",
                        "Endpoint_Frame": int(endpoint_frame),
                        "Endpoint_Rule": endpoint_rule,
                        "SLC_Frame": slc_frame,
                        "SLC_Valid_For_Window": slc_valid,
                        "Plane_Axis_A": TT_TRAJECTORY_PLANE_AXIS[0],
                        "Plane_Axis_B": TT_TRAJECTORY_PLANE_AXIS[1],
                        "Reference_X_Source": "platform_tip_motion_best_fit_200_250",
                        "Platform_Motion_Start_Frame": motion_start,
                        "Platform_Motion_End_Frame": motion_stop,
                        "Axis_Average_Anchor": "moc",
                        "Axis_Average_Start_Frame": avg_slice.start,
                        "Axis_Average_End_Frame": avg_slice.stop - 1,
                        **_compact_qc_fields(qc_summary),
                    })

                # Save the start and selected endpoint positions for radial
                # displacement calculations.
                for point_type, frame, marker in (("MOC", moc, "o"), ("Endpoint", endpoint_frame, "D")):
                    x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    append_point_row(point_type, frame, x - origin_x, y - origin_y, marker)

                if projected_trace:
                    # AEP is the most anterior point in this projected coordinate
                    # system: minimum projected X across the trajectory window.
                    aep_frame, aep_x, aep_y = min(projected_trace, key=lambda value: value[1])
                    # VEP is the most ventral point: minimum projected Y across
                    # the same trajectory window.
                    vep_frame, vep_x, vep_y = min(projected_trace, key=lambda value: value[2])
                    append_point_row("AEP", aep_frame, aep_x, aep_y, "<")
                    append_point_row("VEP", vep_frame, vep_x, vep_y, "v")

    # Convert collected lists to DataFrames for plotting, CSV export, and
    # downstream notebook inspection.
    point_df = pd.DataFrame(rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if point_df.empty:
        raise ValueError(f"No valid projected TT endpoint points were available. Skipped: {skipped_rows}")
    return point_df, trajectory_df, skipped_df


def _plot_flywise_radial_angle_stripplot(
        fly_radial_df,
        group_order,
        joint_colors,
        file_name,
        save_csv=True,
        angle_stats_df=None
):
    # Build a compact plotting table from fly-average 2D displacement vectors.
    angle_df = fly_radial_df.copy()
    if angle_df.empty:
        return None, None, angle_df

    # Convert each 2D vector into a circular direction angle in degrees using
    # the same displacement-origin coordinate system as the radial vector plot.
    angle_df["Vector_Angle_Deg"] = (
        np.degrees(np.arctan2(angle_df["Mean_Displacement_Y"], angle_df["Mean_Displacement_X"]))
        + 360.0
    ) % 360.0

    # Recompute magnitude from X/Y so the size encoding always matches the
    # plotted vector components even if the upstream column is absent/stale.
    angle_df["Vector_Magnitude"] = np.hypot(
        angle_df["Mean_Displacement_X"],
        angle_df["Mean_Displacement_Y"]
    )

    # Keep only finite angle and magnitude rows because missing vectors cannot
    # be represented as meaningful stripplot points.
    finite_mask = np.isfinite(angle_df["Vector_Angle_Deg"]) & np.isfinite(angle_df["Vector_Magnitude"])
    angle_df = angle_df[finite_mask].copy()
    if angle_df.empty:
        return None, None, angle_df

    # Use one x position per contact group and small fixed offsets per TT joint.
    group_positions = {group_label: i for i, group_label in enumerate(group_order)}
    joints = [joint for joint in TT_TRAJECTORY_JOINTS if joint in set(angle_df["Joint"])]
    joint_offsets = np.linspace(-0.18, 0.18, len(joints)) if len(joints) > 1 else np.array([0.0])
    offset_by_joint = dict(zip(joints, joint_offsets))

    # Scale point area against a fixed 0-to-1 magnitude reference so the legend
    # always shows the same interpretable magnitude examples.
    magnitudes = angle_df["Vector_Magnitude"].to_numpy(dtype=float)
    size_min = 28.0
    size_max = 160.0
    magnitude_size_reference = 1.0

    def magnitude_to_size(value):
        # Clip magnitudes to the fixed display reference so outlier vectors do
        # not make the rest of the stripplot unreadably small.
        clipped = np.clip(float(value), 0.0, magnitude_size_reference)
        return size_min + (clipped / magnitude_size_reference) * (size_max - size_min)

    angle_df["Point_Size"] = [magnitude_to_size(value) for value in magnitudes]

    def p_to_label(p_value):
        # Convert primary circular-test p values into compact bracket labels.
        if pd.isna(p_value):
            return "n.s."
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."

    def add_angle_bracket(ax, x1, x2, y, label):
        # Draw one horizontal comparison bracket inside the angle panel.
        bracket_height = 2.5
        ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color="black", linewidth=0.8)
        ax.text((x1 + x2) / 2, y + bracket_height, label, ha="center", va="bottom", fontsize=8)

    # Draw a stripplot-style figure with fly-level points jittered within each
    # group/joint bin and magnitude encoded by point size.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rng = np.random.default_rng(0)
    for _, row in angle_df.iterrows():
        base_x = group_positions[row["Group_Label"]]
        x = base_x + offset_by_joint.get(row["Joint"], 0.0) + rng.uniform(-0.035, 0.035)
        ax.scatter(
            x,
            row["Vector_Angle_Deg"],
            s=row["Point_Size"],
            color=joint_colors.get(row["Joint"], "0.35"),
            alpha=0.4,
            edgecolors="none",
            linewidth=0,
        )

    # Add the horizontal reference at 0 degrees so directional shifts are easy
    # to read against the displacement-origin coordinate system.
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xticks([group_positions[group_label] for group_label in group_order])
    ax.set_xticklabels(group_order)
    ax.set_ylabel("Fly mean radial vector angle (deg)")
    ax.set_xlabel("")
    ax.set_title("Fly-wise TT radial displacement angle")
    ax.set_ylim(150, 370)
    ax.set_yticks([180, 270, 360])

    # Create a joint-color legend separate from the magnitude-size legend.
    joint_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=joint_colors.get(joint, "0.35"),
            markeredgecolor="none",
            markersize=6,
            label=joint
        )
        for joint in joints
    ]
    joint_legend = ax.legend(handles=joint_handles, title="Joint", frameon=True, fontsize=8, loc="upper left")
    ax.add_artist(joint_legend)

    # Show fixed magnitude-to-size examples in a small legend.
    size_values = [0.1, 0.5, 1.0]
    size_handles = []
    for value in size_values:
        # Use the same fixed scaling for legend points and plotted fly points.
        marker_size = magnitude_to_size(value)
        size_handles.append(
            ax.scatter([], [], s=marker_size, color="0.6", alpha=0.4, edgecolors="none", linewidth=0, label=f"{value:g}")
        )
    ax.legend(handles=size_handles, title="Magnitude", frameon=True, fontsize=8, loc="upper right")

    # Add brackets from the primary circular-angle permutation rows without
    # running any statistics inside the plotting helper.
    if angle_stats_df is not None and not angle_stats_df.empty:
        # Use only the primary circular-angle rows from the shared stats table.
        primary_stats = angle_stats_df[angle_stats_df["test"] == "primary_circular_angle_permutation"]
        pair_order = list(itertools.combinations(group_order, 2))
        for _, stat_row in primary_stats.iterrows():
            joint = stat_row["joint"]
            if joint not in offset_by_joint:
                continue
            group_a = stat_row["group_a"]
            group_b = stat_row["group_b"]
            if group_a not in group_positions or group_b not in group_positions:
                continue
            pair_i = pair_order.index((group_a, group_b)) if (group_a, group_b) in pair_order else 0
            joint_i = joints.index(joint) if joint in joints else 0
            x1 = group_positions[group_a] + offset_by_joint[joint]
            x2 = group_positions[group_b] + offset_by_joint[joint]
            y = 330 + pair_i * 10 + joint_i * 2.5
            add_angle_bracket(ax, x1, x2, y, p_to_label(stat_row["p_value"]))

    # Draw a second stripplot organized by TT joint/leg rather than contact
    # group; contact group is encoded by marker shape.
    leg_fig, leg_ax = plt.subplots(figsize=(7.0, 4.6))
    joint_positions = {joint: i for i, joint in enumerate(joints)}
    group_markers = {"T1": "D", "T2": "o", "T3": "^"}
    group_offsets = np.linspace(-0.18, 0.18, len(group_order)) if len(group_order) > 1 else np.array([0.0])
    offset_by_group = dict(zip(group_order, group_offsets))

    # Plot one fly-level point per joint and contact group. Color continues to
    # identify the TT joint, while shape identifies the contact group.
    for _, row in angle_df.iterrows():
        base_x = joint_positions[row["Joint"]]
        group_label = row["Group_Label"]
        x = base_x + offset_by_group.get(group_label, 0.0) + rng.uniform(-0.035, 0.035)
        leg_ax.scatter(
            x,
            row["Vector_Angle_Deg"],
            s=row["Point_Size"],
            marker=group_markers.get(group_label, "o"),
            color=joint_colors.get(row["Joint"], "0.35"),
            alpha=0.4,
            edgecolors="none",
            linewidth=0,
        )

    # Match the angle-axis convention from the contact-group stripplot so both
    # panels are directly comparable.
    leg_ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    leg_ax.set_xticks([joint_positions[joint] for joint in joints])
    leg_ax.set_xticklabels(joints)
    leg_ax.set_ylabel("Fly mean radial vector angle (deg)")
    leg_ax.set_xlabel("")
    leg_ax.set_title("Fly-wise TT radial displacement angle by leg")
    leg_ax.set_ylim(150, 370)
    leg_ax.set_yticks([180, 270, 360])

    # Use marker-only handles for the contact-group legend while preserving the
    # requested T1 diamond, T2 circle, and T3 triangle mapping.
    group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=group_markers.get(group_label, "o"),
            linestyle="",
            color="0.45",
            markeredgecolor="none",
            markersize=7,
            label=group_label
        )
        for group_label in group_order
    ]
    group_legend = leg_ax.legend(handles=group_handles, title="Contact group", frameon=True, fontsize=8, loc="upper left")
    leg_ax.add_artist(group_legend)

    # Repeat the same fixed magnitude legend on the by-leg figure so it can
    # stand alone from the original contact-group stripplot.
    leg_size_handles = [
        leg_ax.scatter(
            [],
            [],
            s=magnitude_to_size(value),
            color="0.6",
            alpha=0.4,
            edgecolors="none",
            linewidth=0,
            label=f"{value:g}"
        )
        for value in size_values
    ]
    leg_ax.legend(handles=leg_size_handles, title="Magnitude", frameon=True, fontsize=8, loc="upper right")

    # Repeat primary-test brackets on the by-leg panel; each bracket connects
    # contact-group marker positions within the corresponding TT joint.
    if angle_stats_df is not None and not angle_stats_df.empty:
        # Reuse the same primary circular-angle rows for the by-leg bracket
        # overlay without recomputing statistics in the plotting helper.
        primary_stats = angle_stats_df[angle_stats_df["test"] == "primary_circular_angle_permutation"]
        pair_order = list(itertools.combinations(group_order, 2))
        for _, stat_row in primary_stats.iterrows():
            joint = stat_row["joint"]
            if joint not in joint_positions:
                continue
            group_a = stat_row["group_a"]
            group_b = stat_row["group_b"]
            if group_a not in offset_by_group or group_b not in offset_by_group:
                continue
            pair_i = pair_order.index((group_a, group_b)) if (group_a, group_b) in pair_order else 0
            x1 = joint_positions[joint] + offset_by_group[group_a]
            x2 = joint_positions[joint] + offset_by_group[group_b]
            y = 332 + pair_i * 9
            add_angle_bracket(leg_ax, x1, x2, y, p_to_label(stat_row["p_value"]))

    # Save the angle table and the standalone angle stripplot with a predictable
    # suffix tied to the parent radial-displacement output.
    sns.despine()
    fig.tight_layout()
    leg_fig.tight_layout()
    if file_name is not None:
        fig.savefig(f"{file_name}_radial_angle_stripplot.pdf", dpi=300, bbox_inches="tight")
        leg_fig.savefig(f"{file_name}_radial_angle_by_leg_stripplot.pdf", dpi=300, bbox_inches="tight")
        if save_csv:
            angle_df.to_csv(f"{file_name}_fly_average_radial_angles.csv", index=False)
    plt.close(fig)
    plt.close(leg_fig)
    return fig, ax, angle_df


def plot_TT_MOC_to_SLC_endpoint_projected_combined(
        self,
        group_info,
        sc_csv_paths,
        tau=0.71,
        axis_average_frames=100,
        file_name="TT_MOC_to_SLC_endpoint_projected_combined",
        colors=None,
        normalized_average_points=200,
        trial_color="0.55",
        trial_linewidth=0.25,
        trial_alpha=0.35,
        fly_linewidth=1.4,
        fly_alpha=0.95,
        radial_circle_diameter=None,
        n_perm=10000,
        apply_tracking_qc=False,
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
        save_csv=True
):
    """
    Plot projected TT trajectories and radial endpoint displacements in one
    3x2-style figure: one row per contact group, trajectory at left and
    MOC-to-endpoint displacement vectors at right.

    Trial-level traces/vectors are light gray. Fly-level averages use
    time-normalized MOC-to-endpoint trajectories from the finalized TT joints.
    """
    # Validate the remaining finalized resampling option before data collection.
    if normalized_average_points < 2:
        raise ValueError("normalized_average_points must be >= 2.")

    # Resolve one plotting color per TT joint. Dict input can be keyed either by
    # full keypoint name (L-mTT) or leg name (L-m).
    if colors is None:
        colors = {
            "L-fTT": "#1f77b4",
            "L-mTT": "#d62728",
            "L-hTT": "#2ca02c",
        }
    joint_colors = {joint: colors.get(joint, colors.get(joint.replace("TT", ""), "black"))
                    if isinstance(colors, dict) else colors[i % len(colors)]
                    for i, joint in enumerate(TT_TRAJECTORY_JOINTS)}

    # Normalize groups into explicit labels for row titles and group comparisons.
    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    # Build the core trial-level trajectory table and landmark-point table.
    # point_df stores MOC/endpoint/AEP/VEP points; trajectory_df stores one row
    # per projected TT frame.
    point_df, trajectory_df, skipped_df = _collect_TT_MOC_to_SLC_projected_data(
        self=self,
        group_info=group_info,
        sc_csv_paths=sc_csv_paths,
        tau=tau,
        axis_average_frames=axis_average_frames,
        apply_tracking_qc=apply_tracking_qc,
        min_cameras=min_cameras,
        max_interp_gap_s=max_interp_gap_s,
        min_valid_fraction=min_valid_fraction,
        error_max=error_max,
        score_min=score_min,
    )

    # Convert MOC and endpoint point rows into displacement vectors. Each vector
    # represents one trial/joint movement from MOC to the selected endpoint.
    radial_rows = []
    radial_group_cols = ["Group_Label", "Index", "Fly#", "Trial#", "Joint", "Leg"]
    for group_keys, sub in point_df.groupby(radial_group_cols):
        start = sub[sub["Point_Type"] == "MOC"]
        end = sub[sub["Point_Type"] == "Endpoint"]
        if start.empty or end.empty:
            continue
        start = start.iloc[0]
        end = end.iloc[0]
        dx = float(end["Projected_X"] - start["Projected_X"])
        dy = float(end["Projected_Y"] - start["Projected_Y"])
        radial_rows.append({
            "Group_Label": group_keys[0],
            "Index": group_keys[1],
            "Fly#": group_keys[2],
            "Trial#": group_keys[3],
            "Joint": group_keys[4],
            "Leg": group_keys[5],
            "Group_Name": end["Group_Name"],
            "Outcome": end["Outcome"],
            "TrialType": end["TrialType"],
            "Displacement_X": dx,
            "Displacement_Y": dy,
            "Displacement_Magnitude": float(np.hypot(dx, dy)),
            "Displacement_Angle_Deg": float(np.degrees(np.arctan2(dy, dx))),
            "Endpoint_Frame": end["Endpoint_Frame"],
            "Endpoint_Rule": end["Endpoint_Rule"],
            "SLC_Frame": end["SLC_Frame"],
            "SLC_Valid_For_Window": end["SLC_Valid_For_Window"],
        })
    radial_df = pd.DataFrame(radial_rows)

    def fly_average_trajectories():
        # Average trajectories within each fly, joint, and group. Averaging at
        # the fly level avoids letting flies with more trials dominate the
        # colored summary traces.
        average_rows = []
        if trajectory_df.empty:
            return pd.DataFrame()

        # group_cols identifies one fly-level average trace; trial_cols splits
        # that fly's raw trajectories into separate trials before resampling.
        group_cols = ["Group_Label", "Joint", "Leg", "Fly#"]
        trial_cols = ["Group_Label", "Joint", "Leg", "Fly#", "Trial#"]
        for fly_keys, fly_df in trajectory_df.groupby(group_cols):
            prepared = []
            max_time = 0
            for _, trial_df in fly_df.groupby(trial_cols):
                # Sort by time and extract numeric x/y coordinates for one
                # trial's projected TT trajectory.
                trial_df = trial_df.sort_values("Time_From_MOC_s")
                time_s = trial_df["Time_From_MOC_s"].to_numpy(dtype=float)
                x_values = trial_df["Projected_X"].to_numpy(dtype=float)
                y_values = trial_df["Projected_Y"].to_numpy(dtype=float)
                valid = np.isfinite(time_s) & np.isfinite(x_values) & np.isfinite(y_values)
                if np.sum(valid) < 2:
                    continue
                # Keep only finite samples; interpolation needs paired finite
                # time, x, and y values.
                time_s = time_s[valid]
                x_values = x_values[valid]
                y_values = y_values[valid]
                # Drop duplicate time stamps because np.interp expects a
                # monotonic set of x-coordinates.
                unique_time, unique_idx = np.unique(time_s, return_index=True)
                if len(unique_time) < 2:
                    continue
                prepared.append((unique_time, x_values[unique_idx], y_values[unique_idx]))
                max_time = max(max_time, float(unique_time[-1]))

            if not prepared or max_time <= 0:
                continue

            # Time-normalized averaging stretches each trial from MOC to endpoint
            # onto 0..1, matching the finalized WT projected-trajectory figure.
            average_time = np.linspace(0, 1, normalized_average_points)
            x_stack = []
            y_stack = []
            for time_s, x_values, y_values in prepared:
                normalized_time = time_s / time_s[-1]
                x_stack.append(np.interp(average_time, normalized_time, x_values))
                y_stack.append(np.interp(average_time, normalized_time, y_values))

            # Average the aligned x/y coordinates across this fly's trials.
            mean_x = np.nanmean(np.asarray(x_stack, dtype=float), axis=0)
            mean_y = np.nanmean(np.asarray(y_stack, dtype=float), axis=0)
            n_contributing = np.full(len(average_time), len(prepared), dtype=int)
            time_unit = "normalized_MOC_to_endpoint"

            # Store the fly-average trajectory in long-form rows for plotting
            # and optional CSV export.
            for time_value, x_value, y_value, n_value in zip(average_time, mean_x, mean_y, n_contributing):
                if not np.isfinite(x_value) or not np.isfinite(y_value):
                    continue
                average_rows.append({
                    "Group_Label": fly_keys[0],
                    "Joint": fly_keys[1],
                    "Leg": fly_keys[2],
                    "Fly#": fly_keys[3],
                    "Time_From_MOC_s": time_value,
                    "Mean_Projected_X": x_value,
                    "Mean_Projected_Y": y_value,
                    "n_trials_contributing": int(n_value),
                    "Average_Mode": "time_normalized",
                    "Average_Time_Unit": time_unit,
                })
        return pd.DataFrame(average_rows)

    # Build fly-average TT trajectories and fly-average displacement vectors.
    fly_trajectory_df = fly_average_trajectories()
    fly_radial_df = pd.DataFrame()
    if not radial_df.empty:
        fly_radial_df = (
            radial_df
            .groupby(["Group_Label", "Joint", "Leg", "Fly#"], as_index=False)
            .agg(
                Mean_Displacement_X=("Displacement_X", "mean"),
                Mean_Displacement_Y=("Displacement_Y", "mean"),
                n_trials=("Index", "nunique")
            )
        )
        fly_radial_df["Mean_Displacement_Magnitude"] = np.hypot(
            fly_radial_df["Mean_Displacement_X"],
            fly_radial_df["Mean_Displacement_Y"]
        )

    # Ask the stats runner for the primary circular-angle test and the secondary
    # original 2D vector permutation test; plotting code does not calculate p values.
    group_labels = [group_label for group_label, _ in group_items]
    radial_stats_df = _get_stats_runner(self).radial_direction_pairwise_tests(
        fly_vector_df=fly_radial_df,
        group_col="Group_Label",
        x_col="Mean_Displacement_X",
        y_col="Mean_Displacement_Y",
        trial_count_col="n_trials",
        joint_col="Joint",
        leg_col="Leg",
        group_pairs=list(itertools.combinations(group_labels, 2)),
        n_perm=n_perm
    )

    # Create separate fly-wise angle stripplots from the same 2D radial vectors;
    # the angle table stays local so the original return tuple remains stable.
    _, _, _radial_angle_df = _plot_flywise_radial_angle_stripplot(
        fly_radial_df=fly_radial_df,
        group_order=group_labels,
        joint_colors=joint_colors,
        file_name=file_name,
        save_csv=save_csv,
        angle_stats_df=radial_stats_df
    )

    # One row per group, two columns: projected TT trajectories at left and
    # MOC-to-endpoint displacement vectors at right.
    fig, axes = plt.subplots(
        len(group_items),
        2,
        figsize=(10.5, max(4.0, 3.6 * len(group_items))),
        squeeze=False
    )

    for row_i, (group_label, _) in enumerate(group_items):
        traj_ax = axes[row_i, 0]
        radial_ax = axes[row_i, 1]

        # Plot all raw trial trajectories in light gray to show the distribution
        # of movement paths without overpowering fly-average traces.
        group_traj = trajectory_df[trajectory_df["Group_Label"] == group_label]
        for _, trial_df in group_traj.groupby(["Joint", "Fly#", "Trial#"]):
            trial_df = trial_df.sort_values("Frame")
            traj_ax.plot(
                trial_df["Projected_X"],
                trial_df["Projected_Y"],
                color=trial_color,
                linewidth=trial_linewidth,
                alpha=trial_alpha,
                zorder=1
            )

        # Overlay each fly's average trajectory, colored by TT joint.
        group_fly_traj = fly_trajectory_df[fly_trajectory_df["Group_Label"] == group_label]
        for (joint, fly_num), fly_df in group_fly_traj.groupby(["Joint", "Fly#"]):
            fly_df = fly_df.sort_values("Time_From_MOC_s")
            traj_ax.plot(
                fly_df["Mean_Projected_X"],
                fly_df["Mean_Projected_Y"],
                color=joint_colors[joint],
                linewidth=fly_linewidth,
                alpha=fly_alpha,
                zorder=3
            )

        # Plot raw trial MOC-to-endpoint displacement vectors re-zeroed at (0, 0).
        group_radial = radial_df[radial_df["Group_Label"] == group_label]
        for _, row in group_radial.iterrows():
            start_x = 0.0
            start_y = 0.0
            end_x = float(row["Displacement_X"])
            end_y = float(row["Displacement_Y"])
            radial_ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=trial_color,
                linewidth=trial_linewidth,
                alpha=trial_alpha,
                zorder=1
            )

        # Overlay fly-average vectors and mark their endpoints.
        group_fly_radial = fly_radial_df[fly_radial_df["Group_Label"] == group_label]
        for _, row in group_fly_radial.iterrows():
            color = joint_colors[row["Joint"]]
            start_x = 0.0
            start_y = 0.0
            end_x = float(row["Mean_Displacement_X"])
            end_y = float(row["Mean_Displacement_Y"])
            radial_ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=color,
                linewidth=fly_linewidth,
                alpha=fly_alpha,
                zorder=3
            )
            radial_ax.scatter(
                end_x,
                end_y,
                color=color,
                s=14,
                edgecolors="none",
                zorder=4
            )

        # Add zero-reference axes and force equal scaling so distances are not
        # visually distorted.
        for ax in (traj_ax, radial_ax):
            ax.axhline(0, color="0.86", linewidth=0.7, zorder=0)
            ax.axvline(0, color="0.86", linewidth=0.7, zorder=0)
            ax.set_aspect("equal", adjustable="box")

        # Optional reference circle, useful when displacement should be compared
        # to a known platform or body-scale diameter.
        if radial_circle_diameter is not None:
            radial_ax.add_patch(
                plt.Circle(
                    (0, 0),
                    radial_circle_diameter / 2,
                    fill=False,
                    edgecolor="0.65",
                    linewidth=0.9,
                    zorder=0
                )
            )

        traj_ax.set_ylabel(group_label)
        if row_i == 0:
            traj_ax.set_title("Projected TT trajectory")
            radial_ax.set_title("MOC-to-endpoint displacement")
        # Add per-joint sample sizes directly on the trajectory panel.
        count_lines = []
        for joint in TT_TRAJECTORY_JOINTS:
            joint_traj = group_traj[group_traj["Joint"] == joint]
            n_joint_trials = joint_traj[["Fly#", "Trial#"]].drop_duplicates().shape[0]
            n_joint_flies = joint_traj["Fly#"].nunique()
            count_lines.append(f"{joint}: {n_joint_trials} tr, {n_joint_flies} flies")
        traj_ax.text(
            0.98,
            0.96,
            "\n".join(count_lines),
            transform=traj_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8
        )

    # Label only the bottom row's x-axes and every row's y-axes.
    axes[-1, 0].set_xlabel(f"Projected X from {TT_TRAJECTORY_ORIGIN_KEYPOINT}")
    axes[-1, 1].set_xlabel("Displacement X")
    for row_i in range(len(group_items)):
        axes[row_i, 0].set_ylabel(f"{group_items[row_i][0]}\nProjected Y")
        axes[row_i, 1].set_ylabel("Displacement Y")

    # Compute shared x/y limits across trajectory and radial panels so rows and
    # columns can be compared directly.
    axis_values = []
    for group_label, _ in group_items:
        group_traj = trajectory_df[trajectory_df["Group_Label"] == group_label]
        group_radial = radial_df[radial_df["Group_Label"] == group_label]
        if not group_traj.empty:
            axis_values.extend(group_traj["Projected_X"].to_numpy(dtype=float))
            axis_values.extend(group_traj["Projected_Y"].to_numpy(dtype=float))
        if not group_radial.empty:
            axis_values.extend([0.0])
            axis_values.extend(group_radial["Displacement_X"].to_numpy(dtype=float))
            axis_values.extend(group_radial["Displacement_Y"].to_numpy(dtype=float))

    axis_values = np.asarray(axis_values, dtype=float)
    axis_values = axis_values[np.isfinite(axis_values)]
    if len(axis_values) == 0:
        shared_min, shared_max = -0.5, 0.5
    else:
        shared_min = float(np.nanmin(axis_values))
        shared_max = float(np.nanmax(axis_values))
        shared_min = min(shared_min, 0.0)
        shared_max = max(shared_max, 0.0)
        if radial_circle_diameter is not None:
            radius = radial_circle_diameter / 2
            shared_min = min(shared_min, -radius)
            shared_max = max(shared_max, radius)
        if not np.isfinite(shared_min) or not np.isfinite(shared_max) or shared_min == shared_max:
            shared_min, shared_max = -0.5, 0.5

    # Round limits to half-unit ticks and apply the same limits to every panel.
    axis_pad = max((shared_max - shared_min) * 0.06, 0.05)
    tick_step = 0.5
    shared_min = math.floor((shared_min - axis_pad) / tick_step) * tick_step
    shared_max = math.ceil((shared_max + axis_pad) / tick_step) * tick_step
    if shared_min == shared_max:
        shared_min -= tick_step
        shared_max += tick_step
    shared_lim = (shared_min, shared_max)
    shared_ticks = np.arange(shared_min, shared_max + tick_step * 0.5, tick_step)
    for ax in axes.flatten():
        ax.set_xlim(shared_lim)
        ax.set_ylim(shared_lim)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)

    # Legend encodes colored fly-average TT joints plus gray raw trial traces.
    handles = [
        plt.Line2D([0], [0], color=joint_colors[joint], linewidth=fly_linewidth, label=joint)
        for joint in TT_TRAJECTORY_JOINTS
    ]
    handles.append(plt.Line2D([0], [0], color=trial_color, linewidth=trial_linewidth, label="trial"))
    axes[0, 1].legend(handles=handles, frameon=True, fontsize=8, loc="best")

    fig.suptitle(
        f"Projected TT trajectories and endpoint displacement using {TT_TRAJECTORY_PLANE_AXIS[0]}->{TT_TRAJECTORY_PLANE_AXIS[1]} normal"
    )
    sns.despine()
    fig.tight_layout()

    # Export every table needed to reproduce the figure: projected landmark
    # points, raw trajectories, radial vectors, fly averages, statistics, and
    # skipped-trial diagnostics.
    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        if save_csv:
            point_df.to_csv(f"{file_name}_projected_points.csv", index=False)
            trajectory_df.to_csv(f"{file_name}_projected_trajectories.csv", index=False)
            radial_df["Radial_Coordinate_Mode"] = "displacement_origin"
            radial_df.to_csv(f"{file_name}_radial_displacement_data.csv", index=False)
            fly_trajectory_df.to_csv(f"{file_name}_fly_average_trajectories.csv", index=False)
            fly_radial_df.to_csv(f"{file_name}_fly_average_radial_displacement.csv", index=False)
            radial_stats_df.to_csv(f"{file_name}_radial_direction_stats.csv", index=False)
            skipped_df.to_csv(f"{file_name}_skipped_trials.csv", index=False)
    plt.close(fig)
    return fig, axes, point_df, trajectory_df, radial_df, radial_stats_df, skipped_df


def plot_TT_summary_metrics_vs_LL(
        self,
        group_info,
        tau=0.71,
        min_frames=3,
        min_path_length=1e-6,
        sc_csv_path=None,
        file_name="TT_summary_metrics_vs_LL",
        save_csv=True,
        n_perm=20000,
        apply_tracking_qc=False,
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8
):
    """
    Plot L-hTT path efficiency against landing latency and landing outcome.

    The figure reports the trial-level Spearman correlation against landing
    latency and an unpaired label-shuffle comparison between successful and
    failed trials.
    """
    target_leg = TT_PATH_EFFICIENCY_LEG

    # Trial-level metric rows and QC skip diagnostics are accumulated first,
    # then converted into DataFrames.
    records = []
    qc_skipped_rows = []
    metric_name = "path_efficiency"
    y_col = "TT_Path_Efficiency"
    y_label = "L-hTT path efficiency (displacement/path)"
    metric_title = "Path efficiency"

    # Initialize metadata and kinematic traces if the group has not already been
    # prepared upstream.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()
        group_info.filter_nan_fly()

    group_info.read_kinematic_data(list(TT_GEOMETRY_TRIAL_TYPES))
    trial_indexes = group_info.get_targeted_trials(list(TT_GEOMETRY_TRIAL_TYPES))

    # The finalized summary plot always uses L-h SLC-adjusted windows.
    sc_lookup = _load_sc_lookup(self.calculator, sc_csv_path, (target_leg,))

    # Build one metric row for every valid trial and requested leg.
    for index in trial_indexes:
        key = group_info._trial_key(index[0], index[1])
        if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
            continue

        trial_info = group_info.fly_kinematic_data[key]
        meta = group_info.trial_metadata[key]
        fps = trial_info.fps
        moc = trial_info.moc
        mol = trial_info.mol

        if pd.isna(moc) or pd.isna(fps):
            continue

        # Landing latency supplies the x-axis value for every metric panel.
        ll_s, ll_censored, ll_source = th.landing_latency_seconds(meta, tau)
        if pd.isna(ll_s):
            continue

        moc_i = int(moc)
        outcome = th.classify_landing(meta, group_info.latency_threshold)
        for leg in (target_leg,):
            # Analyze the tibia-tarsus endpoint for this leg.
            point_name = f"{leg}TT"
            if point_name not in trial_info.trial_data:
                continue

            # Pick the finalized SLC-adjusted analysis endpoint.
            end_frame, window_rule, slc_frame, slc_valid = _select_slc_adjusted_tt_window_end(
                self.calculator,
                sc_lookup,
                index,
                leg,
                moc_i,
                mol,
                fps,
                trial_info.total_frames_number,
                outcome,
                tau
            )
            if moc_i < 0 or end_frame <= moc_i:
                continue

            if apply_tracking_qc:
                # Geometry calls keypoint-level QC directly for this TT analysis window.
                tt_xyz, qc_summary = _qc_keypoint_xyz_for_window(
                    trial_info=trial_info,
                    keypoint=point_name,
                    min_cameras=min_cameras,
                    error_max=error_max,
                    score_min=score_min,
                    max_interp_gap_s=max_interp_gap_s,
                    min_valid_fraction=min_valid_fraction,
                    start_frame=moc_i,
                    end_frame=end_frame,
                    require_start_end_valid=True
                )
                if not qc_summary["QC_Passed"]:
                    qc_skipped_rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "Keypoint": point_name,
                        "Outcome": outcome,
                        "Reason": "failed TT summary tracking QC",
                        "Analysis_Window_Start_Frame": moc_i,
                        "Analysis_Window_End_Frame": end_frame,
                        **qc_summary,
                    })
                    continue
            else:
                # Raw mode keeps the original tracked TT coordinates.
                tt_xyz = self.calculator.ReadAndTranspose(point_name, trial_info)
                qc_summary = _empty_qc_summary()

            # Slice from MOC through the selected endpoint. end_frame is
            # inclusive, so add 1 for Python slicing.
            end = min(end_frame + 1, len(tt_xyz))
            tt_seg = tt_xyz[moc_i:end]

            # Calculate the shared TT metrics and retain only path efficiency for this plot.
            _, path_efficiency, _, _, _ = _calculate_tt_metrics(
                tt_seg,
                fps,
                min_frames=min_frames,
                min_path_length=min_path_length
            )
            if pd.isna(path_efficiency):
                # Path-efficiency calculation can fail if too few finite
                # coordinates remain. Preserve this as a QC diagnostic when QC
                # is enabled.
                if apply_tracking_qc:
                    qc_skipped_rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "Keypoint": point_name,
                        "Outcome": outcome,
                        "Reason": "TT path efficiency unavailable after tracking QC",
                        "Analysis_Window_Start_Frame": moc_i,
                        "Analysis_Window_End_Frame": end_frame,
                        **qc_summary,
                    })
                continue

            # Store one long-form row per trial/leg. This table drives all
            # scatter panels and CSV exports.
            records.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Leg": leg,
                "Joint": "TT",
                "Outcome": outcome,
                "TrialType": meta["TrialType"],
                "Landing_Latency_s": ll_s,
                "LL_frame": meta["LL"],
                "Landing_Latency_Censored": ll_censored,
                "Landing_Latency_Source": ll_source,
                "TT_Path_Efficiency": path_efficiency,
                "Trajectory_Window_Mode": "SLC_adjusted",
                "Analysis_Window_Rule": window_rule,
                "Analysis_Window_Start_Frame": moc_i,
                "Analysis_Window_End_Frame": end_frame,
                "SLC_Frame": slc_frame,
                "SLC_Valid_For_Window": slc_valid,
                **_compact_qc_fields(qc_summary),
            })

    metric_df = pd.DataFrame(records)
    qc_skipped_df = pd.DataFrame(qc_skipped_rows)
    if metric_df.empty:
        print("No valid TT summary metric data found.")
        return None, None, metric_df, pd.DataFrame()


    # Compute one Spearman correlation for the plotted trial-level points using
    # the shared stats runner rather than local scipy calls.
    stats_runner = _get_stats_runner(self)
    clean = metric_df[["Landing_Latency_s", y_col]].dropna()
    trend_stat_df = stats_runner.spearman_correlation_test(
        clean["Landing_Latency_s"],
        clean[y_col],
        group_name=group_info.group_name,
        metric_x="landing_latency_s",
        metric_y=metric_name
    )
    # Add only figure-specific identifiers; the correlation result itself is
    # already stored as N/rho/p_value by the stats runner.
    trend_stat_df["leg"] = target_leg
    trend_stat_df["metric_column"] = y_col
    trend_stat_df["trajectory_window_mode"] = "SLC_adjusted"

    # The same trial-level path-efficiency table also supports the finalized
    # Success-vs-Failed stripplot, so no second data-collection pass is needed.
    success_df = metric_df[metric_df["Outcome"] == "Success"]
    failed_df = metric_df[metric_df["Outcome"] == "Failed"]
    outcome_stat_df = stats_runner.unpaired_permutation_test(
        success_df[y_col].to_numpy(dtype=float),
        failed_df[y_col].to_numpy(dtype=float),
        group_a="Success",
        group_b="Failed",
        metric=metric_name,
        n_perm=n_perm,
        n_fly_a=success_df["Fly#"].nunique(),
        n_fly_b=failed_df["Fly#"].nunique(),
        n_trials_a=len(success_df),
        n_trials_b=len(failed_df),
        test_name="success_vs_failed_unpaired_permutation"
    )
    # Keep the outcome stat table compact by adding only figure identifiers.
    outcome_stat_df.insert(0, "figure_group", group_info.group_name)
    outcome_stat_df["leg"] = target_leg
    outcome_stat_df["metric_column"] = y_col
    outcome_stat_df["trajectory_window_mode"] = "SLC_adjusted"

    # Save metric and statistics tables before drawing the multi-panel figure.
    if save_csv and file_name is not None:
        metric_df.to_csv(f"{file_name}_data.csv", index=False)
        trend_stat_df.to_csv(f"{file_name}_trend_stats.csv", index=False)
        outcome_stat_df.to_csv(f"{file_name}_outcome_stats.csv", index=False)
        if apply_tracking_qc:
            qc_skipped_df.to_csv(f"{file_name}_tracking_qc_skipped_trials.csv", index=False)

    palette = {
        "Success": "tab:blue",
        "Failed": "tab:red",
    }

    # The final figure has one latency-correlation panel and one outcome
    # comparison panel, both using the same QC-filtered path-efficiency rows.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    scatter_ax, outcome_ax = axes

    x_min = metric_df["Landing_Latency_s"].min()
    x_max = metric_df["Landing_Latency_s"].max()
    x_pad = max((x_max - x_min) * 0.05, 0.02)

    y_values = metric_df[y_col].to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    y_pad = max((y_max - y_min) * 0.05, 0.02)

    sns.scatterplot(
        data=metric_df,
        x="Landing_Latency_s",
        y=y_col,
        hue="Outcome",
        hue_order=["Success", "Failed"],
        palette=palette,
        s=45,
        alpha=0.75,
        ax=scatter_ax
    )

    rho = trend_stat_df.iloc[0]["rho"]
    p_value = trend_stat_df.iloc[0]["p_value"]
    n_points = int(trend_stat_df.iloc[0]["N"])
    # The title reports only the Spearman statistic that corresponds to the
    # plotted trial-level scatter points.
    
    def format_p_value(p_value):
        if pd.isna(p_value):
            return "p=NA"
        if p_value < 0.001:
            return "p<0.001"
        return f"p={p_value:.3f}"


    def format_rho_value(rho):
        if pd.isna(rho):
            return "rho=NA"
        return f"rho={rho:.2f}"


    def significance_label(p_value):
        # Convert the permutation p-value into the same compact annotation used
        # by the old standalone path-efficiency stripplot.
        if pd.isna(p_value):
            return "n.s."
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."


    def add_bracket(ax, x1, x2, y, text):
        # Draw one outcome-comparison bracket over the Success and Failed groups.
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        h = y_range * 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=11)
    
    stat_label = f"n={n_points}, {format_rho_value(rho)}, {format_p_value(p_value)}"

    scatter_ax.axvline(group_info.latency_threshold, color="black", linestyle="--", linewidth=1)
    scatter_ax.set_title(f"{target_leg}TT {metric_title} vs LL\n{stat_label}")
    scatter_ax.set_xlabel("Landing latency (s)")
    scatter_ax.set_ylabel(y_label)
    scatter_ax.set_xlim(x_min - x_pad, x_max + x_pad)
    scatter_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    scatter_ax.legend(frameon=False, fontsize=8)

    # Match plot_landing.plot_it_ot_landing_probability_and_latency: each
    # outcome gets a softened box and black-edged raw points offset to the side.
    outcome_order = ["Success", "Failed"]
    positions = np.arange(len(outcome_order), dtype=float)
    box_positions = positions - 0.10
    point_positions = positions + 0.10
    jitter_rng = np.random.default_rng(0)
    for i, outcome in enumerate(outcome_order):
        # Pull the trial-level path-efficiency values for one landing outcome.
        sub = metric_df[metric_df["Outcome"] == outcome]
        values = sub[y_col].astype(float).dropna().to_numpy()
        color = palette.get(outcome, "0.5")
        if len(values) > 0:
            # Draw the distribution summary with the same box, median, whisker,
            # and cap styling as the IT/OT landing-probability panel.
            outcome_ax.boxplot(
                values,
                positions=[box_positions[i]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": color,
                    "alpha": 0.25,
                    "edgecolor": color,
                },
                medianprops={"color": "black", "linewidth": 1.3},
                whiskerprops={"color": color},
                capprops={"color": color},
            )
        if not sub.empty:
            # Overlay raw trial points with light jitter so repeated values are
            # visible while preserving the Success/Failed categorical grouping.
            x = point_positions[i] + jitter_rng.uniform(-0.045, 0.045, size=len(sub))
            outcome_ax.scatter(
                x,
                sub[y_col],
                s=48,
                color=color,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.4,
            )
    outcome_p = outcome_stat_df.iloc[0]["p_value"]
    outcome_label = significance_label(outcome_p)
    outcome_ax.set_title(f"{target_leg}TT {metric_title} by outcome\n{format_p_value(outcome_p)}")
    outcome_ax.set_xticks(positions)
    outcome_ax.set_xticklabels(outcome_order)
    outcome_ax.set_xlabel("")
    outcome_ax.set_ylabel(y_label)
    outcome_ax.set_ylim(y_min - y_pad, min(1.05, y_max + max(y_pad, 0.08)))
    add_bracket(outcome_ax, 0, 1, outcome_ax.get_ylim()[1] - 0.06, outcome_label)

    # Final figure styling and export.
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Return one combined stats table so callers can inspect both tests from a
    # single object while separate CSVs stay easy to read.
    stat_df = pd.concat([trend_stat_df, outcome_stat_df], ignore_index=True, sort=False)
    return fig, axes, metric_df, stat_df


