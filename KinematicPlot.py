import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.signal import find_peaks
from scipy.stats import linregress, ttest_rel
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kinematic_object import Group, Trial, Point
import kinematic_utilities as ku
import plot_common as pc
import tracking_qc as tqc
import trial_helpers as th
import plot_geometry as pg
import plot_landing as pl
import plot_optogenetics as po
import plot_secondary_contact as psc
import plot_angles as pa


class PlotCreator:
    def __init__(self, platform_offset, platform_height, radius=0, fps=0):
        self.calculator = ku.SimpleCalculation()
        self.detector = ku.DetectCharacteristics(radius, fps)
        self.analyzer = ku.GroupDataAnalyzer(platform_offset=platform_offset, radius=radius, FPS=fps)
        self.manipulator = ku.FileManipulation()

        self.platform_offset = platform_offset
        self.radius = radius
        self.fps = fps
        self.platform_height = platform_height

        self.key_point_pairs = [
            ["L-wing", "L-wing-hinge"],
            ["R-wing", "R-wing-hinge"],
            ["abdomen-tip"],
            ["platform-tip"],
            ["L-platform-tip"],
            ["R-platform-tip"],
            ["platform-axis"],
            ["R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT"],
            ["R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT"],
            ["R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT"],
            ["L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT"],
            ["L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT"],
            ["L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"]
        ]

        self.bodyparts = [
            "R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT",
            "R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT",
            "R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT",
            "L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT",
            "L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT",
            "L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"
        ]

    def formatting(self, ax, xticks=None, yticks=None, xlabel=None, ylabel=None,
                   yticklabel=None, xticklabel=None, ylabel_size=10, xlabel_size=10,
                   spine_width=3, tick_width=3):
        return pc.format_axes(
            ax,
            xticks=xticks,
            yticks=yticks,
            xlabel=xlabel,
            ylabel=ylabel,
            ylabel_size=ylabel_size,
            xlabel_size=xlabel_size,
            spine_width=spine_width,
            tick_width=tick_width
        )

    def centered_shades(self, color, n_shades=5, spread=0.6):
        return pc.centered_shades(color, n_shades=n_shades, spread=spread)

    def _get_trial_meta(self, group_info, index):
        return th.get_trial_metadata(group_info, index)

    def _get_trial_obj(self, group_info, index):
        return th.get_trial_object(group_info, index)

    def _ensure_trials_loaded(self, group_info, trial_types=None):
        return th.ensure_trials_loaded(group_info, trial_types=trial_types)


    def _resolve_evaluation_trial(
            self,
            groups,
            group_name,
            fly_number,
            trial_number,
            trial_types=("Landing", "Flying", "NF", "NA")
    ):
        if isinstance(groups, dict):
            if group_name in groups:
                group_info = groups[group_name]
            else:
                matches = [group for group in groups.values() if group.group_name == group_name]
                if len(matches) != 1:
                    raise KeyError(f"Could not resolve group_name '{group_name}' from groups.")
                group_info = matches[0]
        else:
            group_info = groups
            if group_name not in {None, group_info.group_name}:
                raise ValueError(f"Provided group is '{group_info.group_name}', not '{group_name}'.")

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
        group_info.read_kinematic_data(list(trial_types))

        key = group_info._trial_key(fly_number, trial_number)
        if key not in group_info.fly_kinematic_data:
            raise KeyError(f"{group_info.group_name} trial F{fly_number}T{trial_number} was not loaded.")
        return group_info, group_info.fly_kinematic_data[key]

    def _default_skeleton_segments(self):
        return [
            [group[i], group[i + 1]]
            for group in self.key_point_pairs
            for i in range(len(group) - 1)
        ]

    def _validate_segments(self, segments):
        if segments is None:
            return self._default_skeleton_segments()
        return [list(segment) for segment in segments]

    def _evaluation_unit(self, vector, name):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{name} has near-zero length.")
        return vector / norm

    def _evaluation_projection_basis(
            self,
            trial_info,
            plane_axis=("R-mBC", "L-mBC"),
            motion_start_frame=200,
            motion_stop_frame=250
    ):
        point_a = self.calculator.ReadAndTranspose(plane_axis[0], trial_info).astype(float)
        point_b = self.calculator.ReadAndTranspose(plane_axis[1], trial_info).astype(float)
        plane_vector = np.nanmean(point_b - point_a, axis=0)
        plane_normal = self._evaluation_unit(plane_vector, "projection plane normal")

        platform_xyz = self.calculator.ReadAndTranspose("platform-tip", trial_info).astype(float)
        start = max(int(motion_start_frame), 0)
        stop = min(int(motion_stop_frame) + 1, len(platform_xyz))
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

        basis_y = self._evaluation_unit(
            motion - np.dot(motion, plane_normal) * plane_normal,
            "platform-tip motion projected onto plane"
        )
        basis_x = self._evaluation_unit(
            np.cross(basis_y, plane_normal),
            "platform-motion-derived projected x-axis"
        )
        return plane_normal, basis_x, basis_y

    def _project_evaluation_point(self, point, origin_3d, plane_normal, basis_x, basis_y):
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            return np.nan, np.nan
        point_on_plane = point - np.dot(point - origin_3d, plane_normal) * plane_normal
        relative = point_on_plane - origin_3d
        return float(np.dot(relative, basis_x)), float(np.dot(relative, basis_y))



    def collect_tracking_quality_dataframe(
            self,
            groups,
            group_name,
            keypoints=None,
            trial_types=("Landing", "Flying"),
            tau=0.71,
            margin_s=0.2,
            min_cameras=2,
            max_error=None,
    ):
        """
        Collect frame-level tracking QC values for selected keypoints.

        The analysis window is MOC->MOL for successful trials and MOC->MOC+tau
        for failed/flying trials. The collected diagnostic window extends this
        by margin_s on each side.
        """
        if isinstance(groups, dict):
            if group_name in groups:
                group_info = groups[group_name]
            else:
                matches = [group for group in groups.values() if group.group_name == group_name]
                if len(matches) != 1:
                    raise KeyError(f"Could not resolve group_name '{group_name}' from groups.")
                group_info = matches[0]
        else:
            group_info = groups
            if group_name not in {None, group_info.group_name}:
                raise ValueError(f"Provided group is '{group_info.group_name}', not '{group_name}'.")

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
        group_info.read_kinematic_data(list(trial_types))

        rows = []
        skipped_rows = []

        def classify_trial(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if (
                    meta["TrialType"] == "Landing"
                    and not pd.isna(ll)
                    and ll != -1
                    and (ll / fps) <= group_info.latency_threshold
            ):
                return "Success"
            return "Failed"

        for index in group_info.get_targeted_trials(list(trial_types)):
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                skipped_rows.append({"Index": str(index), "Reason": "missing kinematic data or metadata"})
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                skipped_rows.append({"Index": str(index), "Reason": "missing MOC or fps"})
                continue

            moc = int(moc)
            fps = float(fps)
            outcome = classify_trial(meta)
            if outcome == "Success" and not pd.isna(mol) and mol > moc:
                analysis_end = int(min(mol, trial_info.total_frames_number - 1))
                endpoint_rule = "MOL"
            else:
                analysis_end = int(min(moc + tau * fps, trial_info.total_frames_number - 1))
                endpoint_rule = "MOC_plus_tau"

            if analysis_end <= moc:
                skipped_rows.append({"Index": str(index), "Reason": "empty analysis window"})
                continue

            margin_frames = int(round(margin_s * fps))
            qc_start = max(0, moc - margin_frames)
            qc_stop = min(trial_info.total_frames_number - 1, analysis_end + margin_frames)
            points_to_use = list(keypoints) if keypoints is not None else list(trial_info.trial_data.keys())

            for point_name in points_to_use:
                if point_name not in trial_info.trial_data:
                    skipped_rows.append({"Index": str(index), "Reason": f"missing keypoint: {point_name}"})
                    continue

                point = trial_info.trial_data[point_name]
                x = np.asarray(point.x_coord, dtype=float)
                y = np.asarray(point.y_coord, dtype=float)
                z = np.asarray(point.z_coord, dtype=float)
                camera_count = np.asarray(point.camera_count, dtype=float)
                error = np.asarray(point.error, dtype=float)

                for frame in range(qc_start, qc_stop + 1):
                    finite_coord = np.isfinite(x[frame]) and np.isfinite(y[frame]) and np.isfinite(z[frame])
                    finite_error = np.isfinite(error[frame])
                    camera_ok = np.isfinite(camera_count[frame]) and camera_count[frame] >= min_cameras
                    error_ok = True if max_error is None else (finite_error and error[frame] <= max_error)
                    rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "TrialType": meta["TrialType"],
                        "Outcome": outcome,
                        "Keypoint": point_name,
                        "Frame": frame,
                        "Time_From_MOC_s": (frame - moc) / fps,
                        "In_Analysis_Window": moc <= frame <= analysis_end,
                        "Analysis_Start_Frame": moc,
                        "Analysis_End_Frame": analysis_end,
                        "Endpoint_Rule": endpoint_rule,
                        "QC_Start_Frame": qc_start,
                        "QC_Stop_Frame": qc_stop,
                        "Camera_Count": camera_count[frame],
                        "Error": error[frame],
                        "Finite_Coordinates": finite_coord,
                        "Finite_Error": finite_error,
                        "Camera_OK": camera_ok,
                        "Error_OK": error_ok,
                        "Valid_QC_Frame": finite_coord and camera_ok and error_ok,
                        "Min_Cameras": min_cameras,
                        "Max_Error": max_error,
                        "Tau_s": tau,
                        "Margin_s": margin_s,
                    })

        qc_df = pd.DataFrame(rows)
        skipped_df = pd.DataFrame(skipped_rows)
        if qc_df.empty:
            raise ValueError(f"No tracking QC rows were collected. Skipped: {skipped_rows}")
        return qc_df, skipped_df


    def get_keypoints_error_threshold(
            self,
            group_infos,
            percentile=90,
            percentile_by_keypoint=None,
            min_cameras=2,
            trial_types=("Landing", "Flying"),
            tau=0.71,
            margin_s=0.2,
            keypoints=None,
            file_name=None,
            save_csv=True
    ):
        """
        Calculate keypoint-specific tracking error thresholds from multiple
        contact groups.

        Thresholds are computed from Landing/Flying frames in the analysis
        window plus margin_s on each side. The analysis window is MOC->MOL
        when MOL is available and after MOC, otherwise MOC->MOC+tau. Frames
        must have finite xyz, camera_count >= min_cameras, and finite error.
        percentile_by_keypoint can override the default percentile for
        specific keypoints.
        """
        if isinstance(group_infos, dict):
            group_items = list(group_infos.items())
        elif isinstance(group_infos, (list, tuple)):
            group_items = [(group.group_name, group) for group in group_infos]
        else:
            raise ValueError("group_infos must be a dict or list/tuple of Group objects.")

        percentile_by_keypoint = percentile_by_keypoint or {}
        rows = []
        skipped_rows = []

        requested_trial_types = tuple(trial_types)
        allowed_trial_types = {"Landing", "Flying"}
        active_trial_types = tuple(trial_type for trial_type in requested_trial_types if trial_type in allowed_trial_types)
        if not active_trial_types:
            raise ValueError("get_keypoints_error_threshold only analyzes Landing/Flying trials.")

        for contact_group, group_info in group_items:
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()

            try:
                group_info.read_kinematic_data(list(active_trial_types))
            except FileNotFoundError as exc:
                skipped_rows.append({
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": "",
                    "Reason": str(exc),
                })
                continue

            for index in group_info.get_targeted_trials(list(active_trial_types)):
                key = group_info._trial_key(index[0], index[1])
                if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                    skipped_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Reason": "missing kinematic data or metadata",
                    })
                    continue

                trial_info = group_info.fly_kinematic_data[key]
                meta = group_info.trial_metadata[key]
                if meta.get("TrialType") not in allowed_trial_types:
                    skipped_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Reason": f"excluded trial type: {meta.get('TrialType')}",
                    })
                    continue

                moc = trial_info.moc
                mol = trial_info.mol
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(fps):
                    skipped_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Reason": "missing MOC or fps",
                    })
                    continue

                moc = int(moc)
                fps = float(fps)
                total_frames = int(trial_info.total_frames_number)
                if total_frames <= 0:
                    skipped_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Reason": "missing total frame count",
                    })
                    continue

                mol_available = not pd.isna(mol) and int(mol) > moc
                if mol_available:
                    analysis_end = int(min(int(mol), total_frames - 1))
                    endpoint_rule = "MOL"
                else:
                    analysis_end = int(min(moc + tau * fps, total_frames - 1))
                    endpoint_rule = "MOC_plus_tau"

                if analysis_end <= moc:
                    skipped_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Reason": "empty analysis window",
                    })
                    continue

                margin_frames = int(round(margin_s * fps))
                qc_start = max(0, moc - margin_frames)
                qc_stop = min(total_frames - 1, analysis_end + margin_frames)
                points_to_use = list(keypoints) if keypoints is not None else list(trial_info.trial_data.keys())

                for keypoint in points_to_use:
                    if keypoint not in trial_info.trial_data:
                        skipped_rows.append({
                            "Contact_Group": contact_group,
                            "Group_Name": group_info.group_name,
                            "Index": str(index),
                            "Keypoint": keypoint,
                            "Reason": "missing keypoint",
                        })
                        continue

                    point = trial_info.trial_data[keypoint]
                    x = np.asarray(point.x_coord, dtype=float)
                    y = np.asarray(point.y_coord, dtype=float)
                    z = np.asarray(point.z_coord, dtype=float)
                    camera_count = np.asarray(point.camera_count, dtype=float)
                    error = np.asarray(point.error, dtype=float)
                    n_frames = min(len(x), len(y), len(z), len(camera_count), len(error))
                    point_qc_stop = min(qc_stop, n_frames - 1)
                    if point_qc_stop < qc_start:
                        skipped_rows.append({
                            "Contact_Group": contact_group,
                            "Group_Name": group_info.group_name,
                            "Index": str(index),
                            "Keypoint": keypoint,
                            "Reason": "QC window outside keypoint data",
                        })
                        continue

                    window_slice = slice(qc_start, point_qc_stop + 1)
                    valid = (
                            np.isfinite(x[window_slice])
                            & np.isfinite(y[window_slice])
                            & np.isfinite(z[window_slice])
                            & np.isfinite(camera_count[window_slice])
                            & (camera_count[window_slice] >= min_cameras)
                            & np.isfinite(error[window_slice])
                    )
                    valid_frames = np.arange(qc_start, point_qc_stop + 1, dtype=int)[valid]
                    valid_errors = error[valid_frames]
                    for frame, error_value in zip(valid_frames, valid_errors):
                        rows.append({
                            "Contact_Group": contact_group,
                            "Group_Name": group_info.group_name,
                            "Index": str(index),
                            "Fly#": index[0],
                            "Trial#": index[1],
                            "TrialType": meta["TrialType"],
                            "Keypoint": keypoint,
                            "Frame": int(frame),
                            "Time_From_MOC_s": (frame - moc) / fps,
                            "In_Analysis_Window": moc <= frame <= analysis_end,
                            "Analysis_Start_Frame": moc,
                            "Analysis_End_Frame": analysis_end,
                            "Endpoint_Rule": endpoint_rule,
                            "QC_Start_Frame": qc_start,
                            "QC_Stop_Frame": point_qc_stop,
                            "Tau_s": tau,
                            "Margin_s": margin_s,
                            "Camera_Count": camera_count[frame],
                            "Error": error_value,
                        })

        qc_df = pd.DataFrame(rows)
        skipped_df = pd.DataFrame(skipped_rows)
        if qc_df.empty:
            raise ValueError(f"No valid keypoint error values were available. Skipped: {skipped_rows}")

        threshold_rows = []
        for keypoint, sub in qc_df.groupby("Keypoint", sort=False):
            keypoint_percentile = percentile_by_keypoint.get(keypoint, percentile)
            threshold_rows.append({
                "Keypoint": keypoint,
                "Percentile": keypoint_percentile,
                "Error_Threshold": float(np.nanpercentile(sub["Error"].astype(float), keypoint_percentile)),
                "n_frames": int(len(sub)),
                "n_trials": int(sub[["Contact_Group", "Fly#", "Trial#"]].drop_duplicates().shape[0]),
                "n_flies": int(sub[["Contact_Group", "Fly#"]].drop_duplicates().shape[0]),
                "Contact_Groups": ",".join(map(str, sorted(sub["Contact_Group"].dropna().unique()))),
                "Min_Cameras": min_cameras,
                "Tau_s": tau,
                "Margin_s": margin_s,
                "Analysis_Window": "MOC_to_MOL_when_available_else_MOC_plus_tau",
            })
        threshold_df = pd.DataFrame(threshold_rows)

        fig, ax = plt.subplots(figsize=(max(8, 0.34 * qc_df["Keypoint"].nunique()), 5.2))
        sns.boxplot(data=qc_df, x="Keypoint", y="Error", color="white", fliersize=0, ax=ax)
        sns.stripplot(
            data=qc_df.sample(min(len(qc_df), 8000), random_state=0),
            x="Keypoint",
            y="Error",
            hue="Contact_Group",
            dodge=False,
            jitter=0.22,
            size=1.5,
            alpha=0.22,
            ax=ax
        )
        xtick_positions = {tick.get_text(): i for i, tick in enumerate(ax.get_xticklabels())}
        first_label = True
        for _, row in threshold_df.iterrows():
            keypoint = row["Keypoint"]
            if keypoint not in xtick_positions:
                continue
            x_pos = xtick_positions[keypoint]
            ax.scatter(
                x_pos,
                row["Error_Threshold"],
                marker="D",
                s=34,
                color="red",
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
                label="threshold" if first_label else None
            )
            ax.text(
                x_pos + 0.08,
                row["Error_Threshold"],
                f"{row['Error_Threshold']:.2g}",
                color="red",
                fontsize=7,
                ha="left",
                va="center",
                rotation=90,
                zorder=6
            )
            first_label = False
        ax.set_title("Keypoint tracking error thresholds")
        ax.set_xlabel("Keypoint")
        ax.set_ylabel("Tracking/reconstruction error")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(frameon=False, title="Contact group", loc="center left", bbox_to_anchor=(1.0, 0.5))
        sns.despine()
        plt.tight_layout(rect=[0, 0, 0.86, 1])

        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
            if save_csv:
                threshold_df.to_csv(f"{file_name}_thresholds.csv", index=False)
                qc_df.to_csv(f"{file_name}_data.csv", index=False)
                skipped_df.to_csv(f"{file_name}_skipped_trials.csv", index=False)
        plt.close(fig)
        return fig, ax, threshold_df, qc_df, skipped_df

    
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
        return pg.plot_TT_MOC_to_SLC_endpoint_projected_combined(**locals())
    def plot_LP_summary(
            self,
            data_to_plot,
            file_name,
            colors=None,
            markers=None,
            box_color=None,
            box_width=0.22,
            box_softness=0.65
    ):
        return pl.plot_LP_summary(**locals())
    def plot_LP_summary_from_groups(
            self,
            groups,
            file_name,
            colors=None,
            markers=None,
            box_color=None,
            box_width=0.22,
            box_softness=0.65
    ):
        return pl.plot_LP_summary_from_groups(**locals())
    def plot_LP_summary_light(self, combined_df, file_name, color):
        return pl.plot_LP_summary_light(**locals())
    def plot_LP_summary_light_from_group(self, group_info, file_name, color):
        return pl.plot_LP_summary_light_from_group(**locals())
    def plot_KM_curve(
            self,
            data_to_plot,
            file_name,
            colors=None,
            linestyles=None,
            markers=None,
            opto=False,
            marker_every=None
    ):
        return pl.plot_KM_curve(**locals())
    def plot_KM_curve_from_groups(
            self,
            groups,
            file_name,
            colors=None,
            linestyles=None,
            markers=None,
            opto=False,
            marker_every=None
    ):
        return pl.plot_KM_curve_from_groups(**locals())
    def plot_landing_latency_distribution(
            self,
            group_info,
            file_name=None,
            exclude_first_n_trials=3,
            bins=20,
            color="#d62728",
            save_csv=True
    ):
        return pl.plot_landing_latency_distribution(**locals())
    def plot_selected_chrimson_angle_traces(
            self,
            groups,
            angles=None,
            file_name="selected_CsChrimson_angle_traces",
            start=-0.5,
            end=3,
            condition="ON",
            colors=None,
            show_sem=True,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=True,
            smooth_window_frames=5,
            smooth_polyorder=2,
            qc_start=0,
            qc_end=2.0
    ):
        return pa.plot_selected_chrimson_angle_traces(**locals())
    def plot_wt_contact_group_angle_traces(
            self,
            groups_by_column,
            file_name="WT_FT_CT_angle_traces",
            contact_leg_map=None,
            contact_colors=None,
            start_s=-0.2,
            end_s=0.7,
            target_fps=250,
            trial_types=("Landing", "Flying"),
            show_sem=True,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=False,
            smooth_window_frames=5,
            smooth_polyorder=2,
            save_csv=True
    ):
        return pa.plot_wt_contact_group_angle_traces(**locals())
    def plot_angle_traces_by_trial_sets(
            self,
            group_info,
            angle_defs,
            trial_sets=None,
            behavior_sources=None,
            trial_types=("Landing", "Flying"),
            start_s=-0.2,
            end_s=0.71,
            target_fps=250,
            colors=None,
            show_sem=True,
            file_name="angle_traces_by_trial_sets",
            save_csv=True
    ):
        return pa.plot_angle_traces_by_trial_sets(**locals())
    def plot_manual_sc_inverted_km_from_csv(
            self,
            group_info,
            sc_csv_path,
            file_name="manual_SC_inverted_KM",
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            colors=None,
            save_csv=True
    ):
        return psc.plot_manual_sc_inverted_km_from_csv(**locals())
    def plot_it_ot_landing_probability_and_latency(
            self,
            group_info,
            behavior_sources,
            file_name="IT_OT_landing_probability_and_latency",
            behavior_labels=("IT", "OT"),
            behavior_display_names=None,
            trial_types=("Landing", "Flying"),
            tau=0.71,
            n_perm=20000,
            random_state=0,
            contacted_leg=None,
            angle_start_s=-0.1,
            angle_end_s=0.1,
            target_fps=200,
            min_angle_frames=3,
            use_absolute_angular_velocity=True,
            colors=None,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=False,
            smooth_window_frames=5,
            smooth_polyorder=2,
            save_csv=True
    ):
        return pl.plot_it_ot_landing_probability_and_latency(**locals())
    def compare_manual_sc_rmst_across_contact_groups(
            self,
            group_infos,
            sc_csv_paths,
            file_name="manual_SC_RMST_stats",
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            within_group_leg_pairs=None,
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            n_perm=10000,
            random_state=0
    ):
        return psc.compare_manual_sc_rmst_across_contact_groups(**locals())
    def plot_flywise_first_sc_probability_by_contact_group(
            self,
            group_infos,
            sc_csv_paths,
            file_name="flywise_secondary_contact_probability",
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            colors=None,
            save_csv=True,
            n_perm=10000
    ):
        return psc.plot_flywise_first_sc_probability_by_contact_group(**locals())
    def plot_valid_sc_count_vs_landing_latency(
            self,
            group_info,
            sc_csv_path,
            file_name="valid_SC_count_vs_landing_latency",
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            colors=None,
            subgroup_width=0.22,
            jitter=0.035,
            point_size=28,
            alpha=0.78,
            save_csv=True
    ):
        return psc.plot_valid_sc_count_vs_landing_latency(**locals())
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
        return pg.plot_left_TT_path_efficiency_grouped_stripplots(**locals())
    def _detect_chrimson_wing_mol(
            self,
            trial_info,
            wing_angle_def=None,
            window_start=750,
            window_stop=1250,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=True,
            smooth_window_frames=5,
            smooth_polyorder=2
    ):
        return po._detect_chrimson_wing_mol(**locals())
    def plot_chrimson_LP(
            self,
            group_info,
            color="red",
            threshold=0.71,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=True,
            smooth_window_frames=5,
            smooth_polyorder=2
    ):
        return po.plot_chrimson_LP(**locals())
    def plot_chrimson_LP_change_summary(
            self,
            groups,
            file_name="CsChrimson_LP_change_summary",
            threshold=0.71,
            n_perm=20000,
            random_state=0,
            intensity_colors=None,
            mean_color="black",
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=True,
            smooth_window_frames=5,
            smooth_polyorder=2
    ):
        return po.plot_chrimson_LP_change_summary(**locals())
    def plot_gtacr_LP_change_summary(
            self,
            groups,
            file_name="GtACR_LP_change_summary",
            n_perm=20000,
            random_state=0,
            color="#0B6E2E",
            box_color="#B7E1B0"
    ):
        return po.plot_gtacr_LP_change_summary(**locals())
    def plot_kmc_and_unpaired_rmst_perm(self,
            data_list,
            file_name,
            tau=0.71,
            n_perm=20000,
            random_state=0,
            colors=None,
            invert_curve=False,
    ):
        return po.plot_kmc_and_unpaired_rmst_perm(**locals())
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
        return pg.plot_TT_summary_metrics_vs_LL(**locals())
