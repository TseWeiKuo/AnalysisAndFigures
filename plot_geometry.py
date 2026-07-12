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
from scipy.stats import linregress

import plot_common as pc
import tracking_qc as tqc
import trial_helpers as th


def _load_sc_lookup(calculator, sc_csv_path, legs):
    """Load a secondary-contact CSV keyed by the standard trial index."""
    if sc_csv_path is None:
        raise ValueError("sc_csv_path is required when trajectory_window_mode='SLC_adjusted'.")

    sc_df = pd.read_csv(sc_csv_path)
    required_columns = {"Index", *legs}
    missing_columns = required_columns.difference(sc_df.columns)
    if missing_columns:
        raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

    lookup = {}
    for _, sc_row in sc_df.iterrows():
        index = calculator.parse_index_cell(sc_row["Index"])
        lookup[tuple(index)] = sc_row
    return lookup


def _select_tt_window_end(
        calculator,
        sc_lookup,
        index,
        leg,
        moc,
        mol,
        fps,
        total_frames,
        outcome,
        trajectory_window_mode,
        trajectory_window_s,
        tau
):
    """Choose the fixed, MOL-adjusted, or SLC-adjusted TT endpoint."""
    if trajectory_window_mode == "fixed":
        base_end = int(min(moc + trajectory_window_s * fps, total_frames - 1))
        base_rule = f"MOC_to_MOC_plus_{trajectory_window_s}s"
    elif outcome == "Success" and not pd.isna(mol) and mol > moc:
        base_end = int(min(mol, total_frames - 1))
        base_rule = "MOC_to_MOL"
    else:
        base_end = int(min(moc + tau * fps, total_frames - 1))
        base_rule = "MOC_to_MOC_plus_tau"

    if trajectory_window_mode != "SLC_adjusted":
        return base_end, base_rule, np.nan, False

    if outcome == "Success" and not pd.isna(mol) and mol > moc:
        valid_end = int(min(mol, total_frames - 1))
        no_sc_rule = "MOC_to_MOL_no_valid_SLC"
    elif outcome == "Success":
        valid_end = int(min(moc + tau * fps, total_frames - 1))
        no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC_missing_MOL"
    else:
        valid_end = int(min(moc + tau * fps, total_frames - 1))
        no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC"

    sc_row = sc_lookup.get(tuple(index))
    if sc_row is None or leg not in sc_row:
        return valid_end, no_sc_rule, np.nan, False

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
    tt_xyz = np.asarray(tt_xyz, dtype=float)
    valid = np.all(np.isfinite(tt_xyz), axis=1)
    tt_xyz = tt_xyz[valid]
    if len(tt_xyz) < min_frames:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    steps = np.diff(tt_xyz, axis=0)
    path_length = np.sum(np.linalg.norm(steps, axis=1))
    displacement = np.linalg.norm(tt_xyz[-1] - tt_xyz[0])
    duration_s = (len(tt_xyz) - 1) / fps
    average_speed = path_length / duration_s if duration_s > 0 else np.nan
    path_efficiency = (
        displacement / path_length
        if path_length > min_path_length
        else np.nan
    )
    return average_speed, path_efficiency, path_length, displacement, duration_s


def _empty_qc_summary():
    return {
        "Valid_Frame_Fraction": np.nan,
        "Max_Invalid_Gap_Frames": np.nan,
        "Interpolated_Frame_Count": np.nan,
        "QC_Passed": True,
        "QC_Exclusion_Reason": "",
    }


def _collect_TT_MOC_to_SLC_projected_data(
        self,
        group_info,
        sc_csv_paths,
        tt_joints,
        plane_axis,
        reference_axis,
        origin_keypoint,
        origin_frame,
        trial_types,
        tau,
        axis_average_frames,
        axis_average_anchor,
        apply_tracking_qc,
        tracking_error_thresholds,
        min_cameras,
        max_interp_gap_frames,
        min_valid_fraction,
):
    """Collect projected TT trajectory/endpoint data for combined TT plots."""
    if axis_average_anchor not in {"moc", "mol", "moc_to_endpoint"}:
        raise ValueError("axis_average_anchor must be 'moc', 'mol', or 'moc_to_endpoint'.")
    if origin_frame not in {"moc", "endpoint", "axis_average"}:
        raise ValueError("origin_frame must be 'moc', 'endpoint', or 'axis_average'.")
    if axis_average_frames < 1:
        raise ValueError("axis_average_frames must be >= 1.")
    if len(plane_axis) != 2 or len(reference_axis) != 2:
        raise ValueError("plane_axis and reference_axis must each contain two keypoint names.")

    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    if not isinstance(sc_csv_paths, dict):
        if len(group_items) != 1:
            raise ValueError("sc_csv_paths must be a dict when plotting multiple groups.")
        sc_csv_paths = {group_items[0][0]: sc_csv_paths}

    required_points = set(tt_joints) | set(plane_axis) | {origin_keypoint, "platform-tip"}
    rows = []
    trajectory_rows = []
    skipped_rows = []

    def unit(vector, name):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{name} has near-zero length.")
        return vector / norm

    def average_slice(total_frames, moc, endpoint):
        if axis_average_anchor == "moc":
            start = max(moc - axis_average_frames, 0)
            stop = moc
        elif axis_average_anchor == "mol":
            stop = min(endpoint + 1, total_frames)
            start = max(stop - axis_average_frames, 0)
        else:
            start = moc
            stop = min(endpoint + 1, total_frames)
        return None if stop <= start else slice(start, stop)

    def read_axis(point_a, point_b, trial_info, avg_slice):
        coords_a = self.calculator.ReadAndTranspose(point_a, trial_info).astype(float)
        coords_b = self.calculator.ReadAndTranspose(point_b, trial_info).astype(float)
        mean_a = np.nanmean(coords_a[avg_slice], axis=0)
        mean_b = np.nanmean(coords_b[avg_slice], axis=0)
        return mean_a, mean_b - mean_a

    def platform_motion_axis(trial_info, start_frame=200, stop_frame=250):
        platform_xyz = self.calculator.ReadAndTranspose("platform-tip", trial_info).astype(float)
        start = max(int(start_frame), 0)
        stop = min(int(stop_frame) + 1, len(platform_xyz))
        if stop - start < 2:
            raise ValueError("platform-tip motion window has fewer than 2 frames.")

        coords = platform_xyz[start:stop]
        coords = coords[np.all(np.isfinite(coords), axis=1)]
        if len(coords) < 2:
            raise ValueError("platform-tip motion window has fewer than 2 finite coordinates.")

        centered = coords - np.nanmean(coords, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values[0] < 1e-8:
            raise ValueError("platform-tip motion window has near-zero movement.")
        motion = vh[0]

        net_motion = coords[-1] - coords[0]
        if np.dot(motion, net_motion) < 0:
            motion = -motion
        return motion, start, stop - 1

    def project_point(point, origin_3d, plane_normal, basis_x, basis_y):
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            return np.nan, np.nan
        point_on_plane = point - np.dot(point - origin_3d, plane_normal) * plane_normal
        relative = point_on_plane - origin_3d
        return float(np.dot(relative, basis_x)), float(np.dot(relative, basis_y))

    def lookup_sc_path(group_label, group):
        return sc_csv_paths.get(group_label, sc_csv_paths.get(group.group_name))

    for group_label, current_group in group_items:
        sc_path = lookup_sc_path(group_label, current_group)
        if sc_path is None:
            raise ValueError(f"No SC CSV path provided for group '{group_label}'.")

        sc_df = pd.read_csv(sc_path)
        required_sc_columns = {"Index"} | {joint.replace("TT", "") for joint in tt_joints}
        missing_sc_columns = required_sc_columns.difference(sc_df.columns)
        if missing_sc_columns:
            raise ValueError(f"{group_label} SC CSV is missing columns: {sorted(missing_sc_columns)}")
        sc_lookup = {
            tuple(self.calculator.parse_index_cell(sc_row["Index"])): sc_row
            for _, sc_row in sc_df.iterrows()
        }

        if len(current_group.trial_metadata) == 0:
            current_group.initialize_manual_data()
            current_group.filter_nan_fly()
        current_group.read_kinematic_data(list(trial_types))

        for index in current_group.get_targeted_trials(list(trial_types)):
            index_tuple = tuple(index)
            key = current_group._trial_key(index[0], index[1])
            if key not in current_group.fly_kinematic_data or key not in current_group.trial_metadata:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing kinematic data or metadata"})
                continue

            trial_info = current_group.fly_kinematic_data[key]
            meta = current_group.trial_metadata[key]
            missing_points = [point for point in required_points if point not in trial_info.trial_data]
            if missing_points:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"missing required points: {missing_points}"})
                continue

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

            avg_slice = average_slice(trial_info.total_frames_number, moc, valid_end)
            if avg_slice is None:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "empty axis averaging window"})
                continue

            try:
                plane_origin, plane_vector = read_axis(plane_axis[0], plane_axis[1], trial_info, avg_slice)
                plane_normal = unit(plane_vector, "plane_axis")
                platform_motion, motion_start, motion_stop = platform_motion_axis(trial_info)
                platform_motion_on_plane = platform_motion - np.dot(platform_motion, plane_normal) * plane_normal
                basis_y = unit(platform_motion_on_plane, "platform-tip motion projected onto plane")
                basis_x = unit(np.cross(basis_y, plane_normal), "platform-motion-derived projected x-axis")
            except ValueError as exc:
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": str(exc)})
                continue

            origin_xyz = self.calculator.ReadAndTranspose(origin_keypoint, trial_info).astype(float)
            if origin_frame == "moc":
                origin_3d = origin_xyz[moc]
            elif origin_frame == "endpoint":
                origin_3d = origin_xyz[valid_end]
            else:
                origin_3d = np.nanmean(origin_xyz[avg_slice], axis=0)
            origin_x, origin_y = project_point(origin_3d, plane_origin, plane_normal, basis_x, basis_y)
            if not np.isfinite(origin_x) or not np.isfinite(origin_y):
                skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "invalid projected origin"})
                continue

            sc_row = sc_lookup.get(index_tuple)
            for joint in tt_joints:
                leg = joint.replace("TT", "")
                endpoint_frame = valid_end
                endpoint_rule = fallback_rule
                slc_frame = np.nan
                slc_valid = False

                if sc_row is not None:
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
                    xyz, _, qc_summary, _ = self.calculator.apply_xyz_tracking_qc(
                        trial_info=trial_info,
                        keypoint=joint,
                        min_cameras=min_cameras,
                        error_thresholds=tracking_error_thresholds,
                        max_interp_gap_frames=max_interp_gap_frames,
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
                    xyz = self.calculator.ReadAndTranspose(joint, trial_info).astype(float)
                    qc_summary = _empty_qc_summary()

                projected_trace = []
                for frame in range(moc, endpoint_frame + 1):
                    x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    projected_x = x - origin_x
                    projected_y = y - origin_y
                    projected_trace.append((frame, projected_x, projected_y))
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
                        "Apply_Tracking_QC": apply_tracking_qc,
                        "Min_Cameras": min_cameras if apply_tracking_qc else np.nan,
                        "Max_Interp_Gap_Frames": max_interp_gap_frames if apply_tracking_qc else np.nan,
                        "Min_Valid_Fraction": min_valid_fraction if apply_tracking_qc else np.nan,
                        "Valid_Frame_Fraction": qc_summary["Valid_Frame_Fraction"],
                        "Max_Invalid_Gap_Frames": qc_summary["Max_Invalid_Gap_Frames"],
                        "Interpolated_Frame_Count": qc_summary["Interpolated_Frame_Count"],
                    })

                def append_point_row(point_type, frame, x, y, marker):
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
                        "Origin_Keypoint": origin_keypoint,
                        "Origin_Frame_Mode": origin_frame,
                        "Endpoint_Frame": int(endpoint_frame),
                        "Endpoint_Rule": endpoint_rule,
                        "SLC_Frame": slc_frame,
                        "SLC_Valid_For_Window": slc_valid,
                        "Plane_Axis_A": plane_axis[0],
                        "Plane_Axis_B": plane_axis[1],
                        "Reference_X_Source": "platform_tip_motion_best_fit_200_250",
                        "Platform_Motion_Start_Frame": motion_start,
                        "Platform_Motion_End_Frame": motion_stop,
                        "Axis_Average_Anchor": axis_average_anchor,
                        "Axis_Average_Start_Frame": avg_slice.start,
                        "Axis_Average_End_Frame": avg_slice.stop - 1,
                        "Apply_Tracking_QC": apply_tracking_qc,
                        "Min_Cameras": min_cameras if apply_tracking_qc else np.nan,
                        "Max_Interp_Gap_Frames": max_interp_gap_frames if apply_tracking_qc else np.nan,
                        "Min_Valid_Fraction": min_valid_fraction if apply_tracking_qc else np.nan,
                        "Valid_Frame_Fraction": qc_summary["Valid_Frame_Fraction"],
                        "Max_Invalid_Gap_Frames": qc_summary["Max_Invalid_Gap_Frames"],
                        "Interpolated_Frame_Count": qc_summary["Interpolated_Frame_Count"],
                    })

                for point_type, frame, marker in (("MOC", moc, "o"), ("Endpoint", endpoint_frame, "D")):
                    x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    append_point_row(point_type, frame, x - origin_x, y - origin_y, marker)

                if projected_trace:
                    aep_frame, aep_x, aep_y = min(projected_trace, key=lambda value: value[1])
                    vep_frame, vep_x, vep_y = min(projected_trace, key=lambda value: value[2])
                    append_point_row("AEP", aep_frame, aep_x, aep_y, "<")
                    append_point_row("VEP", vep_frame, vep_x, vep_y, "v")

    point_df = pd.DataFrame(rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if point_df.empty:
        raise ValueError(f"No valid projected TT endpoint points were available. Skipped: {skipped_rows}")
    return point_df, trajectory_df, skipped_df


def plot_TT_MOC_to_SLC_endpoint_projected_combined(
        self,
        group_info,
        sc_csv_paths,
        tt_joints=("L-fTT", "L-mTT", "L-hTT"),
        plane_axis=("R-mBC", "L-mBC"),
        reference_axis=("R-mBC", "R-hBC"),
        origin_keypoint="R-mBC",
        origin_frame="moc",
        trial_types=("Landing", "Flying"),
        tau=0.71,
        axis_average_frames=100,
        axis_average_anchor="moc",
        file_name="TT_MOC_to_SLC_endpoint_projected_combined",
        colors=None,
        target_fps=250,
        trajectory_average_mode="absolute_time",
        normalized_average_points=200,
        trial_color="0.55",
        trial_linewidth=0.25,
        trial_alpha=0.35,
        fly_linewidth=1.4,
        fly_alpha=0.95,
        radial_circle_diameter=None,
        radial_coordinate_mode="displacement_origin",
        n_perm=20000,
        random_state=0,
        radial_stats_file_name=None,
        apply_tracking_qc=False,
        tracking_error_thresholds=None,
        min_cameras=2,
        max_interp_gap_frames=4,
        min_valid_fraction=0.8,
        save_csv=True
):
    """
    Plot projected TT trajectories and radial endpoint displacements in one
    3x2-style figure: one row per contact group, trajectory at left and
    MOC-to-endpoint displacement vectors at right.

    Trial-level traces/vectors are light gray. Fly-level averages are
    colored by TT joint and are resampled to target_fps before averaging
    each fly's trajectories.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0.")
    if trajectory_average_mode not in {"absolute_time", "time_normalized"}:
        raise ValueError("trajectory_average_mode must be 'absolute_time' or 'time_normalized'.")
    if normalized_average_points < 2:
        raise ValueError("normalized_average_points must be >= 2.")
    if radial_coordinate_mode not in {"displacement_origin", "trajectory_coordinates"}:
        raise ValueError(
            "radial_coordinate_mode must be 'displacement_origin' or 'trajectory_coordinates'."
        )
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1.")

    if isinstance(tt_joints, str):
        tt_joints = (tt_joints,)
    else:
        tt_joints = tuple(tt_joints)

    if colors is None:
        colors = {
            "L-fTT": "#1f77b4",
            "L-mTT": "#d62728",
            "L-hTT": "#2ca02c",
        }
    joint_colors = {joint: colors.get(joint, colors.get(joint.replace("TT", ""), "black"))
                    if isinstance(colors, dict) else colors[i % len(colors)]
                    for i, joint in enumerate(tt_joints)}

    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    point_df, trajectory_df, skipped_df = _collect_TT_MOC_to_SLC_projected_data(
        self=self,
        group_info=group_info,
        sc_csv_paths=sc_csv_paths,
        tt_joints=tt_joints,
        plane_axis=plane_axis,
        reference_axis=reference_axis,
        origin_keypoint=origin_keypoint,
        origin_frame=origin_frame,
        trial_types=trial_types,
        tau=tau,
        axis_average_frames=axis_average_frames,
        axis_average_anchor=axis_average_anchor,
        apply_tracking_qc=apply_tracking_qc,
        tracking_error_thresholds=tracking_error_thresholds,
        min_cameras=min_cameras,
        max_interp_gap_frames=max_interp_gap_frames,
        min_valid_fraction=min_valid_fraction,
    )

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
        average_rows = []
        if trajectory_df.empty:
            return pd.DataFrame()

        group_cols = ["Group_Label", "Joint", "Leg", "Fly#"]
        trial_cols = ["Group_Label", "Joint", "Leg", "Fly#", "Trial#"]
        for fly_keys, fly_df in trajectory_df.groupby(group_cols):
            prepared = []
            max_time = 0
            for _, trial_df in fly_df.groupby(trial_cols):
                trial_df = trial_df.sort_values("Time_From_MOC_s")
                time_s = trial_df["Time_From_MOC_s"].to_numpy(dtype=float)
                x_values = trial_df["Projected_X"].to_numpy(dtype=float)
                y_values = trial_df["Projected_Y"].to_numpy(dtype=float)
                valid = np.isfinite(time_s) & np.isfinite(x_values) & np.isfinite(y_values)
                if np.sum(valid) < 2:
                    continue
                time_s = time_s[valid]
                x_values = x_values[valid]
                y_values = y_values[valid]
                unique_time, unique_idx = np.unique(time_s, return_index=True)
                if len(unique_time) < 2:
                    continue
                prepared.append((unique_time, x_values[unique_idx], y_values[unique_idx]))
                max_time = max(max_time, float(unique_time[-1]))

            if not prepared or max_time <= 0:
                continue

            if trajectory_average_mode == "time_normalized":
                average_time = np.linspace(0, 1, normalized_average_points)
                x_stack = []
                y_stack = []
                for time_s, x_values, y_values in prepared:
                    normalized_time = time_s / time_s[-1]
                    x_stack.append(np.interp(average_time, normalized_time, x_values))
                    y_stack.append(np.interp(average_time, normalized_time, y_values))

                mean_x = np.nanmean(np.asarray(x_stack, dtype=float), axis=0)
                mean_y = np.nanmean(np.asarray(y_stack, dtype=float), axis=0)
                n_contributing = np.full(len(average_time), len(prepared), dtype=int)
                time_unit = "normalized_MOC_to_endpoint"
            else:
                average_time = np.arange(0, max_time + (0.5 / target_fps), 1 / target_fps)
                x_stack = []
                y_stack = []
                for time_s, x_values, y_values in prepared:
                    in_range = average_time <= time_s[-1]
                    x_interp = np.full(len(average_time), np.nan, dtype=float)
                    y_interp = np.full(len(average_time), np.nan, dtype=float)
                    x_interp[in_range] = np.interp(average_time[in_range], time_s, x_values)
                    y_interp[in_range] = np.interp(average_time[in_range], time_s, y_values)
                    x_stack.append(x_interp)
                    y_stack.append(y_interp)

                mean_x = np.nanmean(np.asarray(x_stack, dtype=float), axis=0)
                mean_y = np.nanmean(np.asarray(y_stack, dtype=float), axis=0)
                n_contributing = np.sum(np.isfinite(np.asarray(x_stack, dtype=float)), axis=0)
                time_unit = "seconds_from_MOC"

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
                    "Target_FPS": target_fps,
                    "Average_Mode": trajectory_average_mode,
                    "Average_Time_Unit": time_unit,
                })
        return pd.DataFrame(average_rows)

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

    radial_stats_df = pd.DataFrame()
    if not fly_radial_df.empty:
        rng = np.random.default_rng(random_state)

        def vector_permutation_test(vectors_a, vectors_b):
            vectors_a = np.asarray(vectors_a, dtype=float)
            vectors_b = np.asarray(vectors_b, dtype=float)
            valid_a = np.all(np.isfinite(vectors_a), axis=1)
            valid_b = np.all(np.isfinite(vectors_b), axis=1)
            vectors_a = vectors_a[valid_a]
            vectors_b = vectors_b[valid_b]
            if len(vectors_a) == 0 or len(vectors_b) == 0:
                return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

            mean_a = np.mean(vectors_a, axis=0)
            mean_b = np.mean(vectors_b, axis=0)
            observed_dx = float(mean_b[0] - mean_a[0])
            observed_dy = float(mean_b[1] - mean_a[1])
            observed_distance = float(np.hypot(observed_dx, observed_dy))

            pooled = np.vstack([vectors_a, vectors_b])
            n_a = len(vectors_a)
            perm_stats = np.empty(n_perm, dtype=float)
            for perm_i in range(n_perm):
                permuted = pooled[rng.permutation(len(pooled))]
                perm_mean_a = np.mean(permuted[:n_a], axis=0)
                perm_mean_b = np.mean(permuted[n_a:], axis=0)
                perm_stats[perm_i] = np.hypot(
                    perm_mean_b[0] - perm_mean_a[0],
                    perm_mean_b[1] - perm_mean_a[1]
                )

            p_value = (np.sum(perm_stats >= observed_distance) + 1) / (n_perm + 1)
            return (
                observed_distance,
                float(p_value),
                observed_dx,
                observed_dy,
                float(mean_a[0]),
                float(mean_a[1]),
                float(mean_b[0]),
                float(mean_b[1]),
            )

        def unpaired_or_nan(data_a, data_b, column):
            values_a = data_a[column].to_numpy(dtype=float)
            values_b = data_b[column].to_numpy(dtype=float)
            values_a = values_a[np.isfinite(values_a)]
            values_b = values_b[np.isfinite(values_b)]
            if len(values_a) == 0 or len(values_b) == 0:
                return np.nan, np.nan
            return self.calculator._permutation_test_unpaired(
                values_a,
                values_b,
                n_perm=n_perm,
                rng=rng
            )

        stat_rows = []
        group_labels = [label for label, _ in group_items]
        for joint in tt_joints:
            joint_fly_df = fly_radial_df[fly_radial_df["Joint"] == joint]
            for group_a, group_b in itertools.combinations(group_labels, 2):
                data_a = joint_fly_df[joint_fly_df["Group_Label"] == group_a]
                data_b = joint_fly_df[joint_fly_df["Group_Label"] == group_b]
                vectors_a = data_a[["Mean_Displacement_X", "Mean_Displacement_Y"]].to_numpy(dtype=float)
                vectors_b = data_b[["Mean_Displacement_X", "Mean_Displacement_Y"]].to_numpy(dtype=float)
                (
                    observed_vector_distance,
                    vector_p,
                    observed_dx,
                    observed_dy,
                    group_a_mean_x,
                    group_a_mean_y,
                    group_b_mean_x,
                    group_b_mean_y,
                ) = vector_permutation_test(vectors_a, vectors_b)

                x_diff, x_p = unpaired_or_nan(data_a, data_b, "Mean_Displacement_X")
                y_diff, y_p = unpaired_or_nan(data_a, data_b, "Mean_Displacement_Y")
                magnitude_diff, magnitude_p = unpaired_or_nan(
                    data_a,
                    data_b,
                    "Mean_Displacement_Magnitude"
                )

                stat_rows.append({
                    "Joint": joint,
                    "Leg": joint.replace("TT", ""),
                    "Group_A": group_a,
                    "Group_B": group_b,
                    "Test": "fly_mean_vector_label_shuffle",
                    "Primary_Statistic": "distance_between_group_mean_vectors",
                    "Observed_Vector_Distance": observed_vector_distance,
                    "Vector_Permutation_P": vector_p,
                    "Observed_Delta_X_GroupB_minus_GroupA": observed_dx,
                    "Observed_Delta_Y_GroupB_minus_GroupA": observed_dy,
                    "Group_A_Mean_X": group_a_mean_x,
                    "Group_A_Mean_Y": group_a_mean_y,
                    "Group_B_Mean_X": group_b_mean_x,
                    "Group_B_Mean_Y": group_b_mean_y,
                    "Secondary_X_Mean_Diff_GroupB_minus_GroupA": x_diff,
                    "Secondary_X_Permutation_P": x_p,
                    "Secondary_Y_Mean_Diff_GroupB_minus_GroupA": y_diff,
                    "Secondary_Y_Permutation_P": y_p,
                    "Secondary_Magnitude_Mean_Diff_GroupB_minus_GroupA": magnitude_diff,
                    "Secondary_Magnitude_Permutation_P": magnitude_p,
                    "Group_A_n_flies": int(data_a["Fly#"].nunique()),
                    "Group_B_n_flies": int(data_b["Fly#"].nunique()),
                    "Group_A_n_trials": int(data_a["n_trials"].sum()) if not data_a.empty else 0,
                    "Group_B_n_trials": int(data_b["n_trials"].sum()) if not data_b.empty else 0,
                    "n_perm": n_perm,
                    "random_state": random_state,
                })
        radial_stats_df = pd.DataFrame(stat_rows)

    fig, axes = plt.subplots(
        len(group_items),
        2,
        figsize=(10.5, max(4.0, 3.6 * len(group_items))),
        squeeze=False
    )

    for row_i, (group_label, _) in enumerate(group_items):
        traj_ax = axes[row_i, 0]
        radial_ax = axes[row_i, 1]

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

        group_radial = radial_df[radial_df["Group_Label"] == group_label]
        for _, row in group_radial.iterrows():
            if radial_coordinate_mode == "trajectory_coordinates":
                point_sub = point_df[
                    (point_df["Group_Label"] == row["Group_Label"])
                    & (point_df["Index"] == row["Index"])
                    & (point_df["Joint"] == row["Joint"])
                ]
                start_point = point_sub[point_sub["Point_Type"] == "MOC"]
                end_point = point_sub[point_sub["Point_Type"] == "Endpoint"]
                if start_point.empty or end_point.empty:
                    continue
                start_x = float(start_point.iloc[0]["Projected_X"])
                start_y = float(start_point.iloc[0]["Projected_Y"])
                end_x = float(end_point.iloc[0]["Projected_X"])
                end_y = float(end_point.iloc[0]["Projected_Y"])
            else:
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

        group_fly_radial = fly_radial_df[fly_radial_df["Group_Label"] == group_label]
        for _, row in group_fly_radial.iterrows():
            color = joint_colors[row["Joint"]]
            if radial_coordinate_mode == "trajectory_coordinates":
                fly_point_sub = point_df[
                    (point_df["Group_Label"] == row["Group_Label"])
                    & (point_df["Joint"] == row["Joint"])
                    & (point_df["Fly#"] == row["Fly#"])
                ]
                fly_moc = fly_point_sub[fly_point_sub["Point_Type"] == "MOC"]
                if fly_moc.empty:
                    continue
                start_x = float(fly_moc["Projected_X"].mean())
                start_y = float(fly_moc["Projected_Y"].mean())
                end_x = start_x + float(row["Mean_Displacement_X"])
                end_y = start_y + float(row["Mean_Displacement_Y"])
            else:
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

        for ax in (traj_ax, radial_ax):
            ax.axhline(0, color="0.86", linewidth=0.7, zorder=0)
            ax.axvline(0, color="0.86", linewidth=0.7, zorder=0)
            ax.set_aspect("equal", adjustable="box")

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
        count_lines = []
        for joint in tt_joints:
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

    axes[-1, 0].set_xlabel(f"Projected X from {origin_keypoint}")
    radial_xlabel = "Projected X from {0}".format(origin_keypoint) if radial_coordinate_mode == "trajectory_coordinates" else "Displacement X"
    radial_ylabel = "Projected Y" if radial_coordinate_mode == "trajectory_coordinates" else "Displacement Y"
    axes[-1, 1].set_xlabel(radial_xlabel)
    for row_i in range(len(group_items)):
        axes[row_i, 0].set_ylabel(f"{group_items[row_i][0]}\nProjected Y")
        axes[row_i, 1].set_ylabel(radial_ylabel)

    axis_values = []
    for group_label, _ in group_items:
        group_traj = trajectory_df[trajectory_df["Group_Label"] == group_label]
        group_radial = radial_df[radial_df["Group_Label"] == group_label]
        if not group_traj.empty:
            axis_values.extend(group_traj["Projected_X"].to_numpy(dtype=float))
            axis_values.extend(group_traj["Projected_Y"].to_numpy(dtype=float))
        if not group_radial.empty:
            if radial_coordinate_mode == "trajectory_coordinates":
                group_points = point_df[
                    (point_df["Group_Label"] == group_label)
                    & point_df["Point_Type"].isin(["MOC", "Endpoint"])
                ]
                axis_values.extend(group_points["Projected_X"].to_numpy(dtype=float))
                axis_values.extend(group_points["Projected_Y"].to_numpy(dtype=float))
            else:
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
        if radial_coordinate_mode == "displacement_origin":
            shared_min = min(shared_min, 0.0)
            shared_max = max(shared_max, 0.0)
        if radial_circle_diameter is not None:
            radius = radial_circle_diameter / 2
            shared_min = min(shared_min, -radius)
            shared_max = max(shared_max, radius)
        if not np.isfinite(shared_min) or not np.isfinite(shared_max) or shared_min == shared_max:
            shared_min, shared_max = -0.5, 0.5

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

    handles = [
        plt.Line2D([0], [0], color=joint_colors[joint], linewidth=fly_linewidth, label=joint)
        for joint in tt_joints
    ]
    handles.append(plt.Line2D([0], [0], color=trial_color, linewidth=trial_linewidth, label="trial"))
    axes[0, 1].legend(handles=handles, frameon=True, fontsize=8, loc="best")

    fig.suptitle(
        f"Projected TT trajectories and endpoint displacement using {plane_axis[0]}->{plane_axis[1]} normal"
    )
    sns.despine()
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        if save_csv:
            point_df.to_csv(f"{file_name}_projected_points.csv", index=False)
            trajectory_df.to_csv(f"{file_name}_projected_trajectories.csv", index=False)
            radial_df["Radial_Coordinate_Mode"] = radial_coordinate_mode
            radial_df.to_csv(f"{file_name}_radial_displacement_data.csv", index=False)
            fly_trajectory_df.to_csv(f"{file_name}_fly_average_trajectories.csv", index=False)
            fly_radial_df.to_csv(f"{file_name}_fly_average_radial_displacement.csv", index=False)
            radial_stats_output = radial_stats_file_name or f"{file_name}_fly_mean_vector_stats"
            radial_stats_df.to_csv(f"{radial_stats_output}.csv", index=False)
            skipped_df.to_csv(f"{file_name}_skipped_trials.csv", index=False)
    plt.close(fig)
    return fig, axes, point_df, trajectory_df, radial_df, radial_stats_df, skipped_df


def plot_left_TT_path_efficiency_grouped_stripplots(
        self,
        group_info,
        behavior_sources,
        file_name="left_TT_path_efficiency_grouped_stripplots",
        legs=("L-f", "L-m", "L-h"),
        trial_types=("Landing", "Flying"),
        tau=0.71,
        trajectory_window_mode="fixed",
        trajectory_window_s=0.10,
        min_frames=3,
        min_path_length=1e-6,
        sc_csv_path=None,
        colors=None,
        save_csv=True,
        n_perm=20000,
        apply_tracking_qc=False,
        tracking_error_thresholds=None,
        min_cameras=2,
        max_interp_gap_frames=4,
        min_valid_fraction=0.8
):
    """
    Plot left-leg TT path efficiency grouped by outcome and IT/OT behavior.

    Figure layout:
    - top panel: Success vs Failed within each leg
    - bottom panel: Inward touch vs Outward touch within each leg

    Window modes match plot_TT_summary_metrics_vs_LL:
    - fixed: MOC -> MOC + trajectory_window_s
    - mol_adjusted: success MOC -> MOL, failed MOC -> MOC + tau
    - SLC_adjusted: MOC -> leg-specific valid SC when present; otherwise
      success uses MOC -> MOL and failed uses MOC -> MOC + tau
    """
    if trajectory_window_mode not in {"fixed", "mol_adjusted", "SLC_adjusted"}:
        raise ValueError("trajectory_window_mode must be 'fixed', 'mol_adjusted', or 'SLC_adjusted'.")

    if isinstance(legs, str):
        legs = (legs,)
    else:
        legs = tuple(legs)

    if colors is None:
        colors = {
            "Success": "tab:blue",
            "Failed": "tab:red",
            "IT": "#8FD694",
            "OT": "#C7A0E8",
        }

    behavior_display_names = {
        "IT": "Inward touch",
        "OT": "Outward touch",
    }

    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()
        group_info.filter_nan_fly()

    group_info.read_kinematic_data(list(trial_types))

    sc_lookup = {}
    if trajectory_window_mode == "SLC_adjusted":
        sc_lookup = _load_sc_lookup(self.calculator, sc_csv_path, legs)

    behavior_trial_sets = th.trial_sets_from_behavior_sources(behavior_sources)
    behavior_by_index = {}
    for behavior_label, indexes in behavior_trial_sets.items():
        for index in indexes:
            behavior_by_index[tuple(index)] = behavior_label

    records = []
    qc_skipped_rows = []
    for index in group_info.get_targeted_trials(list(trial_types)):
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

        ll_s, ll_censored, ll_source = th.landing_latency_seconds(meta, tau)
        if pd.isna(ll_s):
            continue

        moc_i = int(moc)
        outcome = th.classify_landing(meta, group_info.latency_threshold)
        behavior_label = behavior_by_index.get(tuple(index))
        behavior_display = (
            behavior_display_names.get(behavior_label, behavior_label)
            if behavior_label is not None
            else np.nan
        )

        for leg in legs:
            tt_point = f"{leg}TT"
            if tt_point not in trial_info.trial_data:
                continue

            end_frame, window_rule, slc_frame, slc_valid = _select_tt_window_end(
                self.calculator,
                sc_lookup,
                index,
                leg,
                moc_i,
                mol,
                fps,
                trial_info.total_frames_number,
                outcome,
                trajectory_window_mode,
                trajectory_window_s,
                tau
            )
            if moc_i < 0 or end_frame <= moc_i:
                continue

            if apply_tracking_qc:
                tt_xyz, _, qc_summary, _ = self.calculator.apply_xyz_tracking_qc(
                    trial_info=trial_info,
                    keypoint=tt_point,
                    min_cameras=min_cameras,
                    error_thresholds=tracking_error_thresholds,
                    max_interp_gap_frames=max_interp_gap_frames,
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
                        "Keypoint": tt_point,
                        "Outcome": outcome,
                        "Reason": "failed TT path tracking QC",
                        "Analysis_Window_Start_Frame": moc_i,
                        "Analysis_Window_End_Frame": end_frame,
                        **qc_summary,
                    })
                    continue
            else:
                tt_xyz = self.calculator.ReadAndTranspose(tt_point, trial_info)
                qc_summary = _empty_qc_summary()
            tt_segment = tt_xyz[moc_i:min(end_frame + 1, len(tt_xyz))]
            _, path_efficiency, path_length, displacement, _ = _calculate_tt_metrics(
                tt_segment,
                fps,
                min_frames=min_frames,
                min_path_length=min_path_length
            )
            if pd.isna(path_efficiency):
                if apply_tracking_qc:
                    qc_skipped_rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "Keypoint": tt_point,
                        "Outcome": outcome,
                        "Reason": "path efficiency unavailable after tracking QC",
                        "Analysis_Window_Start_Frame": moc_i,
                        "Analysis_Window_End_Frame": end_frame,
                        **qc_summary,
                    })
                continue

            records.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Leg": leg,
                "Outcome": outcome,
                "Behavior_Label": behavior_label,
                "Behavior_Display": behavior_display,
                "TrialType": meta["TrialType"],
                "Landing_Latency_s": ll_s,
                "LL_frame": meta["LL"],
                "Landing_Latency_Censored": ll_censored,
                "Landing_Latency_Source": ll_source,
                "TT_Path_Efficiency": path_efficiency,
                "TT_Path_Length": path_length,
                "TT_Displacement": displacement,
                "Trajectory_Window_Mode": trajectory_window_mode,
                "Analysis_Window_Rule": window_rule,
                "Analysis_Window_Start_Frame": moc_i,
                "Analysis_Window_End_Frame": end_frame,
                "SLC_Frame": slc_frame,
                "SLC_Valid_For_Window": slc_valid,
                "Apply_Tracking_QC": apply_tracking_qc,
                "Min_Cameras": min_cameras if apply_tracking_qc else np.nan,
                "Max_Interp_Gap_Frames": max_interp_gap_frames if apply_tracking_qc else np.nan,
                "Min_Valid_Fraction": min_valid_fraction if apply_tracking_qc else np.nan,
                "Valid_Frame_Fraction": qc_summary["Valid_Frame_Fraction"],
                "Max_Invalid_Gap_Frames": qc_summary["Max_Invalid_Gap_Frames"],
                "Interpolated_Frame_Count": qc_summary["Interpolated_Frame_Count"],
            })

    path_df = pd.DataFrame(records)
    qc_skipped_df = pd.DataFrame(qc_skipped_rows)
    if path_df.empty:
        raise ValueError("No valid left-leg TT path efficiency rows were found.")

    def significance_label(p_value):
        if pd.isna(p_value):
            return "n.s."
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."

    def run_unpaired_test(data, leg, group_col, group_a, group_b, comparison_type):
        sub = data[data["Leg"] == leg]
        values_a = sub[sub[group_col] == group_a]["TT_Path_Efficiency"].astype(float).dropna().to_numpy()
        values_b = sub[sub[group_col] == group_b]["TT_Path_Efficiency"].astype(float).dropna().to_numpy()
        row = {
            "Comparison_Type": comparison_type,
            "Leg": leg,
            "Group_A": group_a,
            "Group_B": group_b,
            "n_A": len(values_a),
            "n_B": len(values_b),
            "mean_A": np.nan if len(values_a) == 0 else float(np.mean(values_a)),
            "mean_B": np.nan if len(values_b) == 0 else float(np.mean(values_b)),
            "mean_diff_B_minus_A": np.nan,
            "permutation_p": np.nan,
            "significance": "n.s.",
            "n_perm": n_perm,
        }
        if len(values_a) == 0 or len(values_b) == 0:
            return row

        observed, p_value = self.calculator._permutation_test_unpaired(
            values_a,
            values_b,
            n_perm=n_perm
        )
        row["mean_diff_B_minus_A"] = float(observed)
        row["permutation_p"] = float(p_value)
        row["significance"] = significance_label(p_value)
        return row

    stat_rows = []
    for leg in legs:
        stat_rows.append(run_unpaired_test(
            path_df,
            leg,
            "Outcome",
            "Success",
            "Failed",
            "success_vs_failed"
        ))

    behavior_df = path_df.dropna(subset=["Behavior_Label"]).copy()
    behavior_keys = list(behavior_sources.keys())
    if len(behavior_keys) >= 2:
        for leg in legs:
            stat_rows.append(run_unpaired_test(
                behavior_df,
                leg,
                "Behavior_Label",
                behavior_keys[0],
                behavior_keys[1],
                "behavior_group"
            ))

    stat_df = pd.DataFrame(stat_rows)

    if save_csv and file_name is not None:
        path_df.to_csv(f"{file_name}_data.csv", index=False)
        stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)
        if apply_tracking_qc:
            qc_skipped_df.to_csv(f"{file_name}_tracking_qc_skipped_trials.csv", index=False)

    fig, axes = plt.subplots(1, 1, figsize=(6.8, 7.0))

    def add_bracket(ax, x1, x2, y, text):
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        h = y_range * 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=11)

    sns.stripplot(
        data=path_df,
        x="Leg",
        y="TT_Path_Efficiency",
        hue="Outcome",
        order=list(legs),
        hue_order=["Success", "Failed"],
        palette={key: colors[key] for key in ("Success", "Failed")},
        dodge=True,
        jitter=True,
        size=5,
        alpha=0.4,
        ax=axes
    )
    axes.set_title("TT path efficiency by landing outcome")
    axes.set_xlabel("")
    axes.set_ylabel("TT path efficiency")
    axes.set_ylim(-0.05, 1.05)
    for leg_i, leg in enumerate(legs):
        stat_match = stat_df[
            (stat_df["Comparison_Type"] == "success_vs_failed")
            & (stat_df["Leg"] == leg)
        ]
        if stat_match.empty:
            continue
        add_bracket(axes, leg_i - 0.18, leg_i + 0.18, 0.94, stat_match.iloc[0]["significance"])
    axes.legend(
        frameon=False,
        title="Outcome",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )



    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return fig, axes, path_df, stat_df


def plot_TT_summary_metrics_vs_LL(
        self,
        group_info,
        legs=("L-f", "L-m", "L-h"),
        trial_types=("Landing", "Flying"),
        tau=0.71,
        trajectory_window_mode="mol_adjusted",
        trajectory_window_s=0.10,
        min_frames=3,
        min_path_length=1e-6,
        sc_csv_path=None,
        file_name="TT_summary_metrics_vs_LL",
        save_csv=True,
        n_perm=20000,
        random_state=0,
        apply_tracking_qc=False,
        tracking_error_thresholds=None,
        min_cameras=2,
        max_interp_gap_frames=4,
        min_valid_fraction=0.8
):
    """
    Plot TT average speed, path efficiency, and path length vs LL.

    Statistics are calculated separately for each metric and leg. Each panel
    reports trial-level Spearman correlation and fly-average Spearman
    correlation. Permutation p-values are computed by shuffling the y-values
    within that panel and recalculating Spearman rho.
    """
    rng = np.random.default_rng(random_state)
    qc_config = tqc.from_legacy_arguments(
        apply_tracking_qc=apply_tracking_qc,
        tracking_error_thresholds=tracking_error_thresholds,
        min_cameras=min_cameras,
        max_interp_gap_frames=max_interp_gap_frames,
        min_valid_fraction=min_valid_fraction
    )
    records = []
    qc_skipped_rows = []
    metric_options = {
        "average_speed": {
            "column": "TT_Average_Speed",
            "label": "TT average speed (mm/s)",
            "title": "Average speed",
        },
        "path_efficiency": {
            "column": "TT_Path_Efficiency",
            "label": "TT path efficiency (displacement/path)",
            "title": "Path efficiency",
        },
        "path_length": {
            "column": "TT_Path_Length",
            "label": "TT path length (mm)",
            "title": "Path length",
        },
    }

    if trajectory_window_mode not in {"fixed", "mol_adjusted", "SLC_adjusted"}:
        raise ValueError("trajectory_window_mode must be 'fixed', 'mol_adjusted', or 'SLC_adjusted'.")

    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()
        group_info.filter_nan_fly()

    group_info.read_kinematic_data(list(trial_types))
    trial_indexes = group_info.get_targeted_trials(list(trial_types))

    sc_lookup = {}
    if trajectory_window_mode == "SLC_adjusted":
        sc_lookup = _load_sc_lookup(self.calculator, sc_csv_path, legs)

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

        ll_s, ll_censored, ll_source = th.landing_latency_seconds(meta, tau)
        if pd.isna(ll_s):
            continue

        moc_i = int(moc)
        outcome = th.classify_landing(meta, group_info.latency_threshold)
        for leg in legs:
            point_name = f"{leg}TT"
            if point_name not in trial_info.trial_data:
                continue

            end_frame, window_rule, slc_frame, slc_valid = _select_tt_window_end(
                self.calculator,
                sc_lookup,
                index,
                leg,
                moc_i,
                mol,
                fps,
                trial_info.total_frames_number,
                outcome,
                trajectory_window_mode,
                trajectory_window_s,
                tau
            )
            if moc_i < 0 or end_frame <= moc_i:
                continue

            if apply_tracking_qc:
                tt_xyz, _, qc_summary, _ = self.calculator.apply_xyz_tracking_qc(
                    trial_info=trial_info,
                    keypoint=point_name,
                    min_cameras=min_cameras,
                    error_thresholds=tracking_error_thresholds,
                    max_interp_gap_frames=max_interp_gap_frames,
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
                tt_xyz = self.calculator.ReadAndTranspose(point_name, trial_info)
                qc_summary = _empty_qc_summary()
            end = min(end_frame + 1, len(tt_xyz))
            tt_seg = tt_xyz[moc_i:end]

            average_speed, path_efficiency, path_length, displacement, duration_s = _calculate_tt_metrics(
                tt_seg,
                fps,
                min_frames=min_frames,
                min_path_length=min_path_length
            )
            if all(pd.isna(value) for value in (average_speed, path_efficiency, path_length)):
                if apply_tracking_qc:
                    qc_skipped_rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "Keypoint": point_name,
                        "Outcome": outcome,
                        "Reason": "TT summary metrics unavailable after tracking QC",
                        "Analysis_Window_Start_Frame": moc_i,
                        "Analysis_Window_End_Frame": end_frame,
                        **qc_summary,
                    })
                continue

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
                "TT_Average_Speed": average_speed,
                "TT_Path_Efficiency": path_efficiency,
                "TT_Path_Length": path_length,
                "TT_Displacement": displacement,
                "Window_Duration_s": duration_s,
                "Trajectory_Window_Mode": trajectory_window_mode,
                "Analysis_Window_Rule": window_rule,
                "Analysis_Window_Start_Frame": moc_i,
                "Analysis_Window_End_Frame": end_frame,
                "SLC_Frame": slc_frame,
                "SLC_Valid_For_Window": slc_valid,
                **qc_config.output_metadata(),
                "Valid_Frame_Fraction": qc_summary["Valid_Frame_Fraction"],
                "Max_Invalid_Gap_Frames": qc_summary["Max_Invalid_Gap_Frames"],
                "Interpolated_Frame_Count": qc_summary["Interpolated_Frame_Count"],
            })

    metric_df = pd.DataFrame(records)
    qc_skipped_df = pd.DataFrame(qc_skipped_rows)
    if metric_df.empty:
        print("No valid TT summary metric data found.")
        return None, None, metric_df, pd.DataFrame()


    stat_rows = []
    for leg in legs:
        leg_df = metric_df[metric_df["Leg"] == leg]
        for metric_name, metric_info in metric_options.items():
            y_col = metric_info["column"]
            clean = leg_df[["Landing_Latency_s", y_col]].dropna()
            fly_average_clean = (
                leg_df
                .groupby("Fly#")[["Landing_Latency_s", y_col]]
                .mean()
                .dropna()
                .reset_index()
            )

            stat_row = {
                "Group_Name": group_info.group_name,
                "Leg": leg,
                "Metric": metric_name,
                "Metric_Column": y_col,
                "n": len(clean),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "spearman_permutation_p": np.nan,
                "fly_average_n": len(fly_average_clean),
                "fly_average_spearman_rho": np.nan,
                "fly_average_spearman_p": np.nan,
                "fly_average_spearman_permutation_p": np.nan,
                "linear_slope": np.nan,
                "linear_intercept": np.nan,
                "linear_r": np.nan,
                "linear_p": np.nan,
                "within_fly_permutation_slope": np.nan,
                "within_fly_permutation_p": np.nan,
                "n_perm": n_perm,
                "Trajectory_Window_Mode": trajectory_window_mode,
            }

            if len(clean) >= 3 and clean["Landing_Latency_s"].nunique() >= 2 and clean[y_col].nunique() >= 2:
                spearman_rho, spearman_p, spearman_perm_p = self.calculator.spearman_permutation_test(
                    clean["Landing_Latency_s"],
                    clean[y_col],
                    n_perm=n_perm,
                    rng=rng
                )
                linear = linregress(clean["Landing_Latency_s"], clean[y_col])

                stat_row.update({
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "spearman_permutation_p": spearman_perm_p,
                    "linear_slope": float(linear.slope),
                    "linear_intercept": float(linear.intercept),
                    "linear_r": float(linear.rvalue),
                    "linear_p": float(linear.pvalue),
                })

            if (
                    len(fly_average_clean) >= 3
                    and fly_average_clean["Landing_Latency_s"].nunique() >= 2
                    and fly_average_clean[y_col].nunique() >= 2
            ):
                fly_rho, fly_p, fly_perm_p = self.calculator.spearman_permutation_test(
                    fly_average_clean["Landing_Latency_s"],
                    fly_average_clean[y_col],
                    n_perm=n_perm,
                    rng=rng
                )
                stat_row.update({
                    "fly_average_spearman_rho": fly_rho,
                    "fly_average_spearman_p": fly_p,
                    "fly_average_spearman_permutation_p": fly_perm_p,
                })

            stat_rows.append(stat_row)

    stat_df = pd.DataFrame(stat_rows)

    if save_csv and file_name is not None:
        metric_df.to_csv(f"{file_name}_data.csv", index=False)
        stat_df.to_csv(f"{file_name}_trend_stats.csv", index=False)
        if apply_tracking_qc:
            qc_skipped_df.to_csv(f"{file_name}_tracking_qc_skipped_trials.csv", index=False)

    palette = {
        "Success": "tab:blue",
        "Failed": "tab:red",
    }

    fig, axes = plt.subplots(
        len(metric_options),
        len(legs),
        figsize=(4.2 * len(legs), 3.6 * len(metric_options)),
        sharex=True,
        squeeze=False
    )

    x_min = metric_df["Landing_Latency_s"].min()
    x_max = metric_df["Landing_Latency_s"].max()
    x_pad = max((x_max - x_min) * 0.05, 0.02)

    for row, (metric_name, metric_info) in enumerate(metric_options.items()):
        y_col = metric_info["column"]
        y_values = metric_df[y_col].to_numpy(dtype=float)
        y_values = y_values[np.isfinite(y_values)]
        y_min = np.nanmin(y_values)
        y_max = np.nanmax(y_values)
        y_pad = max((y_max - y_min) * 0.05, 0.02)

        for col, leg in enumerate(legs):
            ax = axes[row, col]
            leg_df = metric_df[metric_df["Leg"] == leg]
            sns.scatterplot(
                data=leg_df,
                x="Landing_Latency_s",
                y=y_col,
                hue="Outcome",
                hue_order=["Success", "Failed"],
                palette=palette,
                s=45,
                alpha=0.75,
                ax=ax
            )
            fly_average_df = (
                leg_df
                .groupby("Fly#")[["Landing_Latency_s", y_col]]
                .mean()
                .dropna()
                .reset_index()
            )
            if not fly_average_df.empty:
                ax.scatter(
                    fly_average_df["Landing_Latency_s"],
                    fly_average_df[y_col],
                    color="black",
                    marker="D",
                    s=52,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=5,
                    label="Fly average"
                )
                if (
                        len(fly_average_df) >= 3
                        and fly_average_df["Landing_Latency_s"].nunique() >= 2
                        and fly_average_df[y_col].nunique() >= 2
                ):
                    fit = linregress(fly_average_df["Landing_Latency_s"], fly_average_df[y_col])
                    x_fit = np.linspace(
                        fly_average_df["Landing_Latency_s"].min(),
                        fly_average_df["Landing_Latency_s"].max(),
                        100
                    )
                    ax.plot(
                        x_fit,
                        fit.intercept + fit.slope * x_fit,
                        color="black",
                        linewidth=1.6,
                        alpha=0.85,
                        label="Fly-average linear fit"
                    )

            stat_match = stat_df[
                (stat_df["Leg"] == leg)
                & (stat_df["Metric"] == metric_name)
            ]
            if not stat_match.empty:
                rho = stat_match.iloc[0]["spearman_rho"]
                p_value = stat_match.iloc[0]["spearman_permutation_p"]
                n_points = int(stat_match.iloc[0]["n"])
                fly_rho = stat_match.iloc[0]["fly_average_spearman_rho"]
                fly_p_value = stat_match.iloc[0]["fly_average_spearman_permutation_p"]
                fly_n = int(stat_match.iloc[0]["fly_average_n"])
                stat_label = (
                    f"trial n={n_points}, {pc.format_rho_value(rho)}, perm {pc.format_p_value(p_value)}\n"
                    f"fly n={fly_n}, {pc.format_rho_value(fly_rho)}, perm {pc.format_p_value(fly_p_value)}"
                )
            else:
                stat_label = "trial n=0, rho=NA, p=NA\nfly n=0, rho=NA, p=NA"

            ax.axvline(group_info.latency_threshold, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{leg} TT {metric_info['title']}\n{stat_label}")
            ax.set_xlabel("Landing latency (s)" if row == len(metric_options) - 1 else "")
            ax.set_ylabel(metric_info["label"] if col == 0 else "")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            if row == 0 and col == len(legs) - 1:
                ax.legend(frameon=False, fontsize=8)
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return fig, axes, metric_df, stat_df
