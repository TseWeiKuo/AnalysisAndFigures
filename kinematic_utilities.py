import ast
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from scipy.signal import find_peaks, hilbert
from sklearn.utils import resample

from kinematic_object import Group, Trial
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

warnings.filterwarnings(action="ignore", category=FutureWarning)



# ------------------------------------------------------------
# General calculation helpers
# ------------------------------------------------------------

class SimpleCalculation:
    """
    These functions are responsible for preprocessing of angle data and 3D pose data.
    This version is written to match the new kinematic_object structure.
    """

    # ------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------

    def calculate_mean_diff(self, data1, data2):
        return np.mean(data1) - np.mean(data2)

    def calculate_median_diff(self, data1, data2):
        return np.median(data1) - np.median(data2)

    # ------------------------------------------------------------
    # Smoothing / normalization
    # ------------------------------------------------------------

    def exponential_moving_average(self, data, alpha):
        if isinstance(data, pd.Series):
            data = data.tolist()

        if len(data) == 0:
            return []

        smoothed_data = [data[0]]

        for i in range(1, len(data)):
            smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[-1])

        return smoothed_data

    def normalize_list(self, data, method="min-max"):
        if method == "min-max":
            min_val = min(data)
            max_val = max(data)

            if max_val == min_val:
                raise ValueError("Cannot perform Min-Max normalization when all values are the same.")

            return [(x - min_val) / (max_val - min_val) for x in data]

        if method == "z-score":
            signal = np.asarray(data)
            mean = np.mean(signal)
            std = np.std(signal)

            if std == 0:
                return np.zeros_like(signal)

            return (signal - mean) / std

    # ------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------

    def Calculate_distance_between_points(self, x, y, z, x1, y1, z1):
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)

    def calculate_angle(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        """
        Calculate the angle between pt1, pt2, and pt3 in 3D space.

        This is the same calculation you used before, but made safer:
        - avoids division by zero
        - clips cosine to [-1, 1] to avoid arccos nan from float error
        """
        pt1 = np.array([x1, y1, z1], dtype=float)
        pt2 = np.array([x2, y2, z2], dtype=float)
        pt3 = np.array([x3, y3, z3], dtype=float)

        vecA = pt1 - pt2
        vecB = pt3 - pt2

        magnitude_A = np.linalg.norm(vecA)
        magnitude_B = np.linalg.norm(vecB)

        if magnitude_A < 1e-8 or magnitude_B < 1e-8:
            return np.nan

        dot_product = np.dot(vecA, vecB)
        cos_theta = dot_product / (magnitude_A * magnitude_B)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    # ------------------------------------------------------------
    # Trial data helpers
    # ------------------------------------------------------------

    def TransposeData(self, df):
        df = pd.DataFrame(df)
        df = df.T
        return df

    def ReadAndTranspose(self, point, kinematic_data):
        """
        Read x/y/z coordinates of one point and return frame-wise coordinates.

        Output shape:
            n_frames x 3
        """
        return np.transpose(np.asarray([
            kinematic_data.trial_data[point].x_coord,
            kinematic_data.trial_data[point].y_coord,
            kinematic_data.trial_data[point].z_coord
        ]))

    def load_tracking_error_thresholds(self, threshold_source):
        if threshold_source is None:
            return None
        if isinstance(threshold_source, dict):
            return threshold_source

        threshold_df = pd.read_csv(threshold_source)
        key_col = "Keypoint"
        threshold_col = "Error_Threshold"
        if key_col not in threshold_df.columns or threshold_col not in threshold_df.columns:
            raise ValueError(f"Threshold file must contain '{key_col}' and '{threshold_col}' columns.")
        return dict(zip(threshold_df[key_col], threshold_df[threshold_col]))

    def get_tracking_qc_mask(
            self,
            trial_info,
            keypoints,
            min_cameras=2,
            error_thresholds=None,
            require_finite_error=True
    ):
        """
        Return one frame-wise QC mask requiring every listed keypoint to pass.

        A point/frame is valid when xyz are finite, camera count is at least
        min_cameras, and error is finite/below the keypoint threshold when a
        threshold is provided.
        """
        if isinstance(keypoints, str):
            keypoints = [keypoints]
        error_thresholds = self.load_tracking_error_thresholds(error_thresholds)

        n_frames = int(trial_info.total_frames_number)
        combined_mask = np.ones(n_frames, dtype=bool)
        point_summaries = []

        for keypoint in keypoints:
            if keypoint not in trial_info.trial_data:
                combined_mask &= False
                point_summaries.append({
                    "Keypoint": keypoint,
                    "Reason": "missing_keypoint",
                    "Valid_Frame_Fraction": 0.0,
                })
                continue

            point = trial_info.trial_data[keypoint]
            x = np.asarray(point.x_coord, dtype=float)
            y = np.asarray(point.y_coord, dtype=float)
            z = np.asarray(point.z_coord, dtype=float)
            camera_count = np.asarray(point.camera_count, dtype=float)
            error = np.asarray(point.error, dtype=float)

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            mask &= np.isfinite(camera_count) & (camera_count >= min_cameras)

            threshold = None if error_thresholds is None else error_thresholds.get(keypoint)
            if threshold is not None and not pd.isna(threshold):
                mask &= np.isfinite(error) & (error <= float(threshold))
            elif require_finite_error:
                mask &= np.isfinite(error)

            combined_mask &= mask
            point_summaries.append({
                "Keypoint": keypoint,
                "Reason": "ok",
                "Valid_Frame_Fraction": float(np.mean(mask)) if len(mask) else np.nan,
                "Min_Cameras": min_cameras,
                "Error_Threshold": threshold,
            })

        return combined_mask, pd.DataFrame(point_summaries)

    def invalid_gap_lengths(self, valid_mask):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        gaps = []
        current = 0
        for is_valid in valid_mask:
            if is_valid:
                if current > 0:
                    gaps.append(current)
                    current = 0
            else:
                current += 1
        if current > 0:
            gaps.append(current)
        return gaps

    def interpolate_short_nan_gaps(self, values, max_gap_frames=4):
        """
        Linearly interpolate NaN gaps up to max_gap_frames. Longer gaps remain NaN.
        """
        values = np.asarray(values, dtype=float).copy()
        finite = np.isfinite(values)
        if np.sum(finite) < 2:
            return values, 0

        interpolated_count = 0
        n = len(values)
        i = 0
        x = np.arange(n)
        while i < n:
            if finite[i]:
                i += 1
                continue
            start = i
            while i < n and not finite[i]:
                i += 1
            stop = i
            gap_len = stop - start
            left = start - 1
            right = stop
            if left >= 0 and right < n and gap_len <= max_gap_frames:
                values[start:stop] = np.interp(x[start:stop], [left, right], [values[left], values[right]])
                interpolated_count += gap_len
        return values, interpolated_count

    def smooth_trace(self, values, window_frames=5, polyorder=2):
        """
        Smooth finite contiguous trace segments with Savitzky-Golay filtering.
        NaN gaps are preserved.
        """
        values = np.asarray(values, dtype=float).copy()
        if window_frames is None or window_frames < 3:
            return values
        if window_frames % 2 == 0:
            window_frames += 1

        finite = np.isfinite(values)
        n = len(values)
        i = 0
        while i < n:
            if not finite[i]:
                i += 1
                continue
            start = i
            while i < n and finite[i]:
                i += 1
            stop = i
            seg_len = stop - start
            if seg_len >= window_frames:
                values[start:stop] = savgol_filter(
                    values[start:stop],
                    window_length=window_frames,
                    polyorder=min(polyorder, window_frames - 1),
                    mode="interp"
                )
        return values

    def apply_angle_tracking_qc(
            self,
            trial_info,
            angle_trace,
            angle_points,
            min_cameras=2,
            error_thresholds=None,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth=False,
            smooth_window_frames=5,
            smooth_polyorder=2
    ):
        """
        Apply tracking QC to one angle trace.

        Invalid frames are set to NaN. Short gaps are linearly interpolated;
        windows containing longer invalid gaps should be skipped by callers
        using the returned summary.
        """
        angle_trace = np.asarray(angle_trace, dtype=float).copy()
        qc_mask, point_summary = self.get_tracking_qc_mask(
            trial_info,
            angle_points,
            min_cameras=min_cameras,
            error_thresholds=error_thresholds,
            require_finite_error=True
        )
        qc_mask &= np.isfinite(angle_trace)
        filtered = angle_trace.copy()
        filtered[~qc_mask] = np.nan
        interpolated, interpolated_count = self.interpolate_short_nan_gaps(
            filtered,
            max_gap_frames=max_interp_gap_frames
        )
        if smooth:
            interpolated = self.smooth_trace(
                interpolated,
                window_frames=smooth_window_frames,
                polyorder=smooth_polyorder
            )

        gap_lengths = self.invalid_gap_lengths(qc_mask)
        summary = {
            "Valid_Frame_Fraction": float(np.mean(qc_mask)) if len(qc_mask) else np.nan,
            "Max_Invalid_Gap_Frames": int(max(gap_lengths)) if gap_lengths else 0,
            "Interpolated_Frame_Count": int(interpolated_count),
            "Long_Gap_Count": int(sum(gap > max_interp_gap_frames for gap in gap_lengths)),
            "Min_Cameras": min_cameras,
            "Max_Interp_Gap_Frames": max_interp_gap_frames,
            "Min_Valid_Fraction": min_valid_fraction,
            "Smooth_Angle": bool(smooth),
            "Smooth_Window_Frames": smooth_window_frames,
        }
        return interpolated, qc_mask, summary, point_summary

    def interpolate_short_xyz_gaps(self, xyz, max_gap_frames=4):
        xyz = np.asarray(xyz, dtype=float).copy()
        interpolated_total = 0
        for dim in range(xyz.shape[1]):
            xyz[:, dim], count = self.interpolate_short_nan_gaps(
                xyz[:, dim],
                max_gap_frames=max_gap_frames
            )
            interpolated_total += count
        return xyz, interpolated_total

    def apply_xyz_tracking_qc(
            self,
            trial_info,
            keypoint,
            min_cameras=2,
            error_thresholds=None,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            start_frame=None,
            end_frame=None,
            require_start_end_valid=False
    ):
        """
        Apply tracking QC to one keypoint's xyz trace.

        Invalid frames are set to NaN, and NaN gaps up to max_interp_gap_frames
        are linearly interpolated independently for x/y/z. Callers should skip
        windows with Max_Invalid_Gap_Frames > max_interp_gap_frames.
        """
        xyz = np.asarray(self.ReadAndTranspose(keypoint, trial_info), dtype=float)
        qc_mask, point_summary = self.get_tracking_qc_mask(
            trial_info,
            keypoint,
            min_cameras=min_cameras,
            error_thresholds=error_thresholds,
            require_finite_error=True
        )
        filtered = xyz.copy()
        filtered[~qc_mask] = np.nan
        filtered, interpolated_count = self.interpolate_short_xyz_gaps(
            filtered,
            max_gap_frames=max_interp_gap_frames
        )

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(qc_mask) - 1
        start_frame = max(int(start_frame), 0)
        end_frame = min(int(end_frame), len(qc_mask) - 1)
        window_mask = qc_mask[start_frame:end_frame + 1] if end_frame >= start_frame else np.array([], dtype=bool)
        gap_lengths = self.invalid_gap_lengths(window_mask)
        start_valid = bool(qc_mask[start_frame]) if len(qc_mask) and 0 <= start_frame < len(qc_mask) else False
        end_valid = bool(qc_mask[end_frame]) if len(qc_mask) and 0 <= end_frame < len(qc_mask) else False
        valid_fraction = float(np.mean(window_mask)) if len(window_mask) else np.nan
        max_gap = int(max(gap_lengths)) if gap_lengths else 0
        long_gap_count = int(sum(gap > max_interp_gap_frames for gap in gap_lengths))

        exclusion_reasons = []
        if len(window_mask) == 0:
            exclusion_reasons.append("empty_qc_window")
        if pd.isna(valid_fraction) or valid_fraction < min_valid_fraction:
            exclusion_reasons.append("valid_fraction_below_threshold")
        if max_gap > max_interp_gap_frames:
            exclusion_reasons.append("long_invalid_gap")
        if require_start_end_valid and not start_valid:
            exclusion_reasons.append("start_frame_invalid")
        if require_start_end_valid and not end_valid:
            exclusion_reasons.append("end_frame_invalid")

        summary = {
            "Keypoint": keypoint,
            "Valid_Frame_Fraction": valid_fraction,
            "Max_Invalid_Gap_Frames": max_gap,
            "Long_Gap_Count": long_gap_count,
            "Interpolated_Frame_Count": int(interpolated_count),
            "Start_Frame_Valid": start_valid,
            "End_Frame_Valid": end_valid,
            "Min_Cameras": min_cameras,
            "Max_Interp_Gap_Frames": max_interp_gap_frames,
            "Min_Valid_Fraction": min_valid_fraction,
            "QC_Passed": len(exclusion_reasons) == 0,
            "QC_Exclusion_Reason": ";".join(exclusion_reasons),
        }
        return filtered, qc_mask, summary, point_summary

    # ------------------------------------------------------------
    # Segment / angle calculations from Trial object
    # ------------------------------------------------------------

    def Calculate_segment_length(self, trial_info, skeletons):
        """
        Calculate length of specified segments for each frame.

        trial_info should be a Trial object from the new kinematic_object.
        skeletons example:
            [["L-fFT", "L-fTT"], ["L-fTT", "L-fLT"]]
        """
        collected_seg_length_data = dict()

        for seg in skeletons:
            seg_name = f"{seg[0]}_{seg[1]}"

            if seg_name not in collected_seg_length_data:
                collected_seg_length_data[seg_name] = []

            for f in range(trial_info.total_frames_number):
                length = self.Calculate_distance_between_points(
                    x=trial_info.trial_data[seg[0]].x_coord[f],
                    y=trial_info.trial_data[seg[0]].y_coord[f],
                    z=trial_info.trial_data[seg[0]].z_coord[f],
                    x1=trial_info.trial_data[seg[1]].x_coord[f],
                    y1=trial_info.trial_data[seg[1]].y_coord[f],
                    z1=trial_info.trial_data[seg[1]].z_coord[f]
                )
                collected_seg_length_data[seg_name].append(length)

        return collected_seg_length_data

    def Calculate_joint_angle(
            self,
            trial_info,
            angles,
            apply_tracking_qc=False,
            tracking_error_thresholds=None,
            min_cameras=2,
            max_interp_gap_frames=4,
            min_valid_fraction=0.8,
            smooth_angle=False,
            smooth_window_frames=5,
            smooth_polyorder=2,
            return_qc=False
    ):
        """
        Calculate specified joint angles for each frame.

        angles example:
            [["R-fBC", "R-fCT", "R-fFT"], ["R-fCT", "R-fFT", "R-fTT"]]
        """
        collected_angle_data = dict()
        qc_summaries = []

        for ag in angles:
            joint_name = ag[1]
            if "wing" in joint_name:
                collected_angle_data[joint_name] = self.calculate_wing_angle_trace(trial_info)
            else:
                if joint_name not in collected_angle_data:
                    collected_angle_data[joint_name] = []
                for f in range(trial_info.total_frames_number):
                    angle = self.calculate_angle(
                        x1=trial_info.trial_data[ag[0]].x_coord[f],
                        y1=trial_info.trial_data[ag[0]].y_coord[f],
                        z1=trial_info.trial_data[ag[0]].z_coord[f],
                        x2=trial_info.trial_data[ag[1]].x_coord[f],
                        y2=trial_info.trial_data[ag[1]].y_coord[f],
                        z2=trial_info.trial_data[ag[1]].z_coord[f],
                        x3=trial_info.trial_data[ag[2]].x_coord[f],
                        y3=trial_info.trial_data[ag[2]].y_coord[f],
                        z3=trial_info.trial_data[ag[2]].z_coord[f]
                    )
                    collected_angle_data[joint_name].append(angle)

                collected_angle_data[joint_name] = np.array(collected_angle_data[joint_name])
                if apply_tracking_qc:
                    filtered_trace, qc_mask, qc_summary, _ = self.apply_angle_tracking_qc(
                        trial_info=trial_info,
                        angle_trace=collected_angle_data[joint_name],
                        angle_points=ag,
                        min_cameras=min_cameras,
                        error_thresholds=tracking_error_thresholds,
                        max_interp_gap_frames=max_interp_gap_frames,
                        min_valid_fraction=min_valid_fraction,
                        smooth=smooth_angle,
                        smooth_window_frames=smooth_window_frames,
                        smooth_polyorder=smooth_polyorder
                    )
                    collected_angle_data[joint_name] = filtered_trace
                    qc_summary.update({
                        "Joint": joint_name,
                        "Angle_Definition": "|".join(ag),
                    })
                    qc_summaries.append(qc_summary)

        if return_qc:
            return collected_angle_data, pd.DataFrame(qc_summaries)
        return collected_angle_data

    # ------------------------------------------------------------
    # Plane / circle / line helpers
    # ------------------------------------------------------------

    def line_plane_intersection(self, p1, p2, normal, d):
        """
        Line represented as:
            p(t) = p1 + t * (p2 - p1)
        Plane represented as:
            normal . x + d = 0
        """
        direction = np.array(p2) - np.array(p1)
        denom = np.dot(normal, direction)

        if abs(denom) < 1e-6:
            return None, None

        t = -(np.dot(normal, p1) + d) / denom

        if 0 <= t <= 1:
            return p1 + t * direction, t

        return None, None

    def is_inside_circle(self, intersection, center, radius):
        return np.linalg.norm(intersection - center) <= radius

    def best_fit_line_3d(self, points):
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, Vt = np.linalg.svd(centered_points)
        direction = Vt[0]
        return centroid, direction

    def check_cylinder_side_intersection(self, A, B, P1, d, r, h, n_steps=100):
        """
        Check if segment AB intersects the side of a finite cylinder.
        P1 is the top of the cylinder axis.
        d is the cylinder axis direction.
        """
        A, B, P1, d = map(np.asarray, (A, B, P1, d))
        d = d / np.linalg.norm(d)

        P0 = P1 - h * d
        min_dist = np.inf

        for t in np.linspace(0, 1, n_steps):
            Pt = A + t * (B - A)
            v = Pt - P0
            proj_len = np.dot(v, d)
            dist_to_axis = np.linalg.norm(v - proj_len * d)

            if dist_to_axis < min_dist:
                min_dist = dist_to_axis

            if 0 <= proj_len <= h:
                if np.isclose(dist_to_axis, r) or dist_to_axis < r:
                    return True, Pt, min_dist

        return False, None, min_dist

    # ------------------------------------------------------------
    # Platform geometry
    # ------------------------------------------------------------

    def calculate_platform_surfaces(self, trial_info=None, platform_traces=None,
                                    platform_center=None, platform_offset=0.07,
                                    radius=0.07, platform_height=3, height=None,
                                    trace_range=(200, 250)):
        """
        Calculate platform line / plane / cylinder surface.

        You can use this in two ways:

        1. pass trial_info
           -> function will read platform-tip trace automatically

        2. pass platform_traces and platform_center directly
           -> useful if you already extracted them elsewhere

        Returns:
            line_points, plane_points, verts, cylinder_top, cylinder_bottom,
            direction, perp_vector1, perp_vector2
        """
        if height is None:
            height = platform_height

        # --------------------------------------------------------
        # Read platform trace from Trial object if needed
        # --------------------------------------------------------
        if trial_info is not None:
            center_points = self.ReadAndTranspose("platform-tip", trial_info)
            start_frame = trace_range[0]
            end_frame = trace_range[1]
            platform_traces = np.array(center_points[start_frame:end_frame])

            if platform_center is None:
                platform_center = platform_traces[-1]

        if platform_traces is None or platform_center is None:
            raise ValueError("Need either trial_info or both platform_traces and platform_center.")

        centroid, direction = self.best_fit_line_3d(platform_traces)
        motion_vec = np.asarray(platform_traces[-1]) - np.asarray(platform_traces[0])

        # Make direction consistent with motion
        if np.dot(direction, motion_vec) < 0:
            direction = -direction

        # Best-fit line for plotting
        t_vals = np.linspace(-10, 10, 100)
        line_points = centroid + np.outer(t_vals, direction)

        # Build orthogonal vectors for plane / circle
        normal_vector = direction

        if np.allclose(normal_vector, [1, 0, 0]):
            perp_vector1 = np.cross(normal_vector, [0, 1, 0])
        else:
            perp_vector1 = np.cross(normal_vector, [1, 0, 0])

        perp_vector1 /= np.linalg.norm(perp_vector1)
        perp_vector2 = np.cross(normal_vector, perp_vector1)
        perp_vector2 /= np.linalg.norm(perp_vector2)

        # Circular plane
        u_vals = np.linspace(-radius, radius, 50)
        v_vals = np.linspace(-radius, radius, 50)
        U, V = np.meshgrid(u_vals, v_vals)

        mask = U ** 2 + V ** 2 <= radius ** 2
        U = U[mask]
        V = V[mask]

        platform_plane_origin = platform_center + platform_offset * direction
        plane_points = platform_plane_origin + U[..., None] * perp_vector1 + V[..., None] * perp_vector2

        # Cylinder top/bottom
        cylinder_top = platform_plane_origin
        cylinder_bottom = cylinder_top - direction * height

        theta = np.linspace(0, 2 * np.pi, 60)

        circle_top = np.array([
            cylinder_top + radius * (np.cos(t) * perp_vector1 + np.sin(t) * perp_vector2)
            for t in theta
        ])

        circle_bottom = np.array([
            cylinder_bottom + radius * (np.cos(t) * perp_vector1 + np.sin(t) * perp_vector2)
            for t in theta
        ])

        verts = []
        for i in range(len(theta) - 1):
            quad = [circle_bottom[i], circle_bottom[i + 1], circle_top[i + 1], circle_top[i]]
            verts.append(quad)

        return line_points, plane_points, verts, cylinder_top, cylinder_bottom, direction, perp_vector1, perp_vector2

    # ------------------------------------------------------------
    # Bootstrap / interpolation / threshold helpers
    # ------------------------------------------------------------

    def Bootstrapping_test(self, data1, data2, n_samps):
        original_mean_diff = self.calculate_mean_diff(data1, data2)
        bootstrap_mean_diffs = []

        resample_data = np.concatenate((data1, data2))

        for i in range(n_samps):
            bootstrap_sample1 = resample(resample_data, n_samples=len(data1))
            bootstrap_sample2 = resample(resample_data, n_samples=len(data2))
            bootstrap_mean_diff = self.calculate_mean_diff(bootstrap_sample1, bootstrap_sample2)
            bootstrap_mean_diffs.append(bootstrap_mean_diff)

        Mean_diff_p_value = np.sum(np.abs(bootstrap_mean_diffs) >= np.abs(original_mean_diff)) / n_samps
        return Mean_diff_p_value

    def Normalized_time(self, data, length=250):
        from scipy.interpolate import interp1d

        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, data, kind='linear')
        signal = f(x_new)

        return signal

    def get_ag_vel_thresh(self, ag, on_set_window=4):
        vel_threshold = None
        min_change_threshold = None

        ag = np.asarray(ag, dtype=float)
        dag = np.gradient(ag)
        positive_idx = np.where(dag > 0)[0]

        if len(positive_idx) == 0:
            return np.nan, np.nan

        onset = positive_idx[0]

        if onset + on_set_window >= len(dag):
            vel_threshold = np.nanmean(dag[onset:])
            min_change_threshold = ag[-1] - ag[onset]
        else:
            vel_threshold = np.mean(dag[onset:onset + on_set_window])
            min_change_threshold = ag[onset + on_set_window] - ag[onset]

        return vel_threshold, min_change_threshold
    def get_angle_data_fn(self, trial_info, leg_name):
        """
        Example stub.
        trial_index might be (fly_num, trial_num)
        leg_name might be 'L-f', 'L-m', 'L-h'
        """
        angs = [[f"{leg_name}BC", f"{leg_name}CT", f"{leg_name}FT"],
                [f"{leg_name}CT", f"{leg_name}FT", f"{leg_name}TT"]]
        angle_data = self.Calculate_joint_angle(trial_info, angs)
        return angle_data[f"{leg_name}FT"], angle_data[f"{leg_name}CT"]

    def get_ct_start_threshold(self, ct):
        ct = np.asarray(ct, dtype=float)
        dct = np.gradient(ct)
        positive_idx = np.where(dct > 0)[0]

        if len(positive_idx) == 0:
            return np.nan

        onset = positive_idx[0]
        return np.mean(ct[:onset])

    def calculate_wing_angle_trace(self, trial_info:Trial):
        """
        Calculate angle trace from three 3D points across frames.

        Parameters
        ----------
        pt1, pt2, pt3 : array-like
            Each should have shape (3, n_frames), where rows are x, y, z.

        Returns
        -------
        angle_deg : np.ndarray
            Angle values in degrees, shape (n_frames,).
            Angle is measured at pt2.
        """
        LW_PT = trial_info.get_point("L-wing")
        MID_PT = (trial_info.get_point("L-wing-hinge") + trial_info.get_point("R-wing-hinge")) / 2
        RW_PT = trial_info.get_point("R-wing")
        import numpy as np

        pt1 = np.asarray(LW_PT, dtype=float)
        pt2 = np.asarray(MID_PT, dtype=float)
        pt3 = np.asarray(RW_PT, dtype=float)

        if pt1.shape[0] != 3 or pt2.shape[0] != 3 or pt3.shape[0] != 3:
            raise ValueError("Each point must have shape (3, n_frames).")

        if pt1.shape != pt2.shape or pt1.shape != pt3.shape:
            raise ValueError("pt1, pt2, and pt3 must have the same shape.")

        # vectors from middle point pt2
        vecA = pt1 - pt2
        vecB = pt3 - pt2

        # dot product for each frame
        dot_product = np.sum(vecA * vecB, axis=0)

        # vector magnitudes for each frame
        magA = np.linalg.norm(vecA, axis=0)
        magB = np.linalg.norm(vecB, axis=0)

        denominator = magA * magB

        # avoid divide-by-zero
        angle_deg = np.full(pt1.shape[1], np.nan)

        valid = denominator > 1e-8

        cos_theta = dot_product[valid] / denominator[valid]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        angle_deg[valid] = np.degrees(angle_rad)

        return angle_deg

    def _permutation_test_unpaired(self, x, y, n_perm=10000, rng=None, return_distribution=False):
        """
        Primary p-value test for independent groups.
        Uses fly-level RMST or LP values.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # remove NaNs
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        if len(x) == 0 or len(y) == 0:
            raise ValueError("One or both groups are empty after removing NaNs.")

        observed = np.mean(y) - np.mean(x)

        pooled = np.concatenate([x, y])
        n_x = len(x)

        perm_stats = np.empty(n_perm)

        for i in range(n_perm):
            perm = rng.permutation(pooled)
            x_perm = perm[:n_x]
            y_perm = perm[n_x:]
            perm_stats[i] = np.mean(y_perm) - np.mean(x_perm)

        # corrected p-value (IMPORTANT)
        p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)

        if return_distribution:
            return observed, p_value, perm_stats
        return observed, p_value

    def paired_signflip_permutation_test(
            self,
            values_a,
            values_b,
            n_perm=10000,
            rng=None,
            return_distribution=False
    ):
        """
        Paired two-sided permutation test using random sign flips.

        Returns the observed mean difference, mean(values_b - values_a), and
        the permutation p-value. NaN pairs are removed before testing.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        values_a = np.asarray(values_a, dtype=float)
        values_b = np.asarray(values_b, dtype=float)
        valid = np.isfinite(values_a) & np.isfinite(values_b)
        diff = values_b[valid] - values_a[valid]
        if len(diff) == 0:
            if return_distribution:
                return np.nan, np.nan, np.asarray([])
            return np.nan, np.nan

        observed = float(np.mean(diff))
        perm_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            signs = rng.choice([-1, 1], size=len(diff), replace=True)
            perm_stats[i] = np.mean(diff * signs)

        p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)
        if return_distribution:
            return observed, float(p_value), perm_stats
        return observed, float(p_value)

    def paired_signflip_diff_test(self, diff, n_perm=10000, rng=None, return_distribution=False):
        """
        Paired sign-flip permutation test when paired differences are already computed.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        diff = np.asarray(diff, dtype=float)
        diff = diff[np.isfinite(diff)]
        if len(diff) == 0:
            if return_distribution:
                return np.nan, np.nan, np.asarray([])
            return np.nan, np.nan

        observed = float(np.mean(diff))
        perm_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            signs = rng.choice([-1, 1], size=len(diff), replace=True)
            perm_stats[i] = np.mean(diff * signs)

        p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)
        if return_distribution:
            return observed, float(p_value), perm_stats
        return observed, float(p_value)

    def spearman_permutation_test(self, x_values, y_values, n_perm=10000, rng=None):
        """
        Two-sided permutation test for Spearman correlation.

        Keeps x fixed, shuffles y ranks, and compares absolute permuted rho to
        absolute observed rho.
        """
        from scipy.stats import spearmanr

        if rng is None:
            rng = np.random.default_rng(0)

        clean = pd.DataFrame({
            "x": np.asarray(x_values, dtype=float),
            "y": np.asarray(y_values, dtype=float),
        }).dropna()

        if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
            return np.nan, np.nan, np.nan

        spearman = spearmanr(clean["x"], clean["y"], nan_policy="omit")
        observed_rho = float(spearman.statistic)
        scipy_p = float(spearman.pvalue)

        x_rank = clean["x"].rank(method="average").to_numpy(dtype=float)
        y_rank = clean["y"].rank(method="average").to_numpy(dtype=float)
        perm_rhos = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm_rhos[i] = np.corrcoef(x_rank, rng.permutation(y_rank))[0, 1]

        permutation_p = (
            np.sum(np.abs(perm_rhos) >= np.abs(observed_rho)) + 1
        ) / (n_perm + 1)
        return observed_rho, scipy_p, float(permutation_p)

    def parse_index_cell(self, value):
        """
        Parse trial Index values from CSV/spreadsheet cells.

        Accepts tuple/list values or strings such as "(1, 2)" and returns a
        tuple. Raises ValueError for unsupported values.
        """
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, (tuple, list)):
            parsed = tuple(value)
            if len(parsed) == 2:
                return int(float(parsed[0])), int(float(parsed[1]))
            return parsed
        raise ValueError(f"Could not parse trial Index value: {value}")

    def validate_sc_timing(self, raw_sc, moc, mol, fps, threshold, missing_values=(0, 10000)):
        """
        Validate absolute-frame secondary contact timing.

        A valid SC must occur after MOC and within `threshold` seconds. If MOL is
        a real post-MOC frame, SC after MOL is invalid. MOL=-1 is treated as no
        MOL cutoff, which is useful for failed/flying trials.
        """
        invalid_result = {
            "is_valid": False,
            "event": 0,
            "duration": threshold,
            "sc_frame": np.nan,
            "sc_time_s": np.nan,
        }

        if pd.isna(raw_sc) or raw_sc in missing_values:
            return invalid_result.copy()

        sc_frame = float(raw_sc)
        sc_time_s = (sc_frame - moc) / fps
        result = invalid_result.copy()
        result["sc_frame"] = sc_frame

        if sc_time_s < 0 or sc_time_s > threshold:
            return result
        if not pd.isna(mol) and mol != -1 and mol > moc and sc_frame > mol:
            return result

        return {
            "is_valid": True,
            "event": 1,
            "duration": sc_time_s,
            "sc_frame": sc_frame,
            "sc_time_s": sc_time_s,
        }

    def validate_sc_frame_window(self, raw_sc, start_frame, end_frame, missing_values=(0, 10000)):
        """
        Validate an absolute SC frame against an explicit frame window.
        """
        if pd.isna(raw_sc) or raw_sc in missing_values:
            return False, np.nan

        sc_frame = float(raw_sc)
        if start_frame <= sc_frame <= end_frame:
            return True, sc_frame
        return False, sc_frame

    def parse_tuple_cell(self, x):
        """
        Parse tuple-like spreadsheet cell contents.
        Accepts tuple, list, string like '(12, 34)', or NaN.
        Returns tuple(start, stop) or None.
        """
        if pd.isna(x):
            return None
        if isinstance(x, tuple):
            return x
        if isinstance(x, list) and len(x) == 2:
            return tuple(x)
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() == "nan":
                return None
            try:
                val = ast.literal_eval(x)
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    return int(val[0]), int(val[1])
            except Exception:
                return None
        return None
# ------------------------------------------------------------
# Detection helpers
# ------------------------------------------------------------

class DetectCharacteristics:
    def __init__(self, radius=0, FPS=0):
        self.radius = radius
        self.calculator = SimpleCalculation()
        self.fps = FPS

    def ReadCoordsAll(self, kinematic_data, fnum):
        """
        Read all point coordinates at one frame from a Trial object.
        """
        points = [
            "L-wing", "L-wing-hinge", "R-wing", "R-wing-hinge", "abdomen-tip",
            "platform-tip", "L-platform-tip", "R-platform-tip", "platform-axis",

            "R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT",
            "R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT",
            "R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT",

            "L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT",
            "L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT",
            "L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"
        ]

        coords = dict()

        for p in points:
            coords[p] = np.asarray([
                kinematic_data.trial_data[p].x_coord[fnum],
                kinematic_data.trial_data[p].y_coord[fnum],
                kinematic_data.trial_data[p].z_coord[fnum]
            ])

        return coords

    def check_leg_platform_intersection(self, leg_p1, leg_p2, direction, center_point, platform_offset, radius):
        """
        Check if a leg segment intersects the circular top plane of the platform.
        """
        platform_plane_origin = center_point + platform_offset * direction
        d = -np.dot(direction, platform_plane_origin)

        intersection, intersect_proportion = self.calculator.line_plane_intersection(
            np.array(leg_p1),
            np.array(leg_p2),
            direction,
            d
        )

        if intersection is not None and self.calculator.is_inside_circle(
            intersection,
            np.array(platform_plane_origin),
            radius
        ):
            return True, intersect_proportion

        return False, None

    def detect_leg_search(self, ft, ct,
                          ft_vel_thresh=1.5,
                          ct_vel_thresh=1.5,
                          min_ft_change=20,
                          min_ct_change=10,
                          sync_window=4,
                          pattern_duration=20,
                          idle_reset_thresh=(0.5, 0.5),
                          ct_start_thresh=80):
        """
        Detect synchronized FT + CT rising events.
        """
        ft_s = np.asarray(ft, dtype=float)
        ct_s = np.asarray(ct, dtype=float)

        dft = np.gradient(ft_s)
        dct = np.gradient(ct_s)

        state = "idle"
        event_indices = []

        ft_start = None
        ct_start = None

        for i in range(len(dft)):
            if state == "idle":
                if ft_start is None and dft[i] > ft_vel_thresh:
                    ft_start = i

                if ct_start is None and dct[i] > ct_vel_thresh and ct_s[i] <= ct_start_thresh:
                    ct_start = i

                if ft_start is not None and ct_start is not None:
                    if abs(ft_start - ct_start) <= sync_window:
                        state = "rising"
                    else:
                        if ft_start < ct_start:
                            ft_start = None
                        else:
                            ct_start = None

            elif state == "rising":
                onset = min(ft_start, ct_start)

                if i - onset > pattern_duration:
                    state = "idle"
                    ft_start = None
                    ct_start = None
                    continue

                if (ft_s[i] - ft_s[ft_start] >= min_ft_change and
                        ct_s[i] - ct_s[ct_start] >= min_ct_change):
                    event_indices.append(onset)
                    state = "event"

            elif state == "event":
                if (abs(dft[i]) < idle_reset_thresh[0] and
                        abs(dct[i]) < idle_reset_thresh[1]):
                    state = "idle"
                    ft_start = None
                    ct_start = None

        return len(event_indices), event_indices

    def detect_landing(self, data, windsize=5):
        """
        Detect landing from normalized and smoothed signal.
        """
        data = self.calculator.normalize_list(data)

        from scipy.ndimage import gaussian_filter1d
        data = gaussian_filter1d(data, sigma=10)

        for i in range(len(data) - windsize):
            if np.mean(data[i:i + windsize]) < 0.2:
                return i

        return -1


# ------------------------------------------------------------
# File / Excel helpers
# ------------------------------------------------------------

class FileManipulation:
    """
    Keep file I/O helpers here.
    """

    def highlight_excel_cells(self, excel_path, sheet_name, rows, cols, fill_color="FFFF00"):
        wb = load_workbook(excel_path)
        ws = wb[sheet_name]
        fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

        for row, col in zip(rows, cols):
            ws.cell(row=row, column=col).fill = fill

        wb.save(excel_path)

    def save_dataframe(self, df, path):
        df.to_csv(path, index=False)

    # Extract the data path of the csv files and organize them in the order by fly number
    def read_secondary_contact_data(self, data, legs, filepath=None):
        if filepath is not None:
            data_to_read = pd.read_csv(filepath)
        else:
            data_to_read = data
        successful_landing_data = dict()
        failed_landing_data = dict()
        jointA = "FT"
        jointB = "TT"
        for row in data_to_read.iterrows():
            for l in legs:
                if l not in successful_landing_data.keys() and l not in failed_landing_data.keys():
                    successful_landing_data[l] = []
                    failed_landing_data[l] = []

                if row[1][l + jointA] > 0 and int(row[1][l + jointA]) != 10000:

                    if row[1]["Result"] != "Failed":
                        successful_landing_data[l].append(row[1][l + jointA])
                    else:
                        failed_landing_data[l].append(row[1][l + jointA])
                elif row[1][l + jointB] > 0 and int(row[1][l + jointB]) != 10000:

                    if row[1]["Result"] != "Failed":
                        successful_landing_data[l].append(row[1][l + jointB])
                    else:
                        failed_landing_data[l].append(row[1][l + jointB])
                else:

                    if row[1]["Result"] != "Failed":
                        successful_landing_data[l].append(np.nan)
                    else:
                        failed_landing_data[l].append(np.nan)
        successful_landing_data = pd.DataFrame(successful_landing_data)
        failed_landing_data = pd.DataFrame(failed_landing_data)
        return successful_landing_data, failed_landing_data

    def read_leg_search_data(self, data, legs, filepath=None):
        if filepath is not None:
            data_to_read = pd.read_csv(filepath)
        else:
            data_to_read = data
        failed_landing = data_to_read[data_to_read["Result"] == "Failed"]
        failed_landing = failed_landing[legs]

        successful_landing = data_to_read[data_to_read["Result"] == "Success"]
        successful_landing = successful_landing[legs]

        return successful_landing, failed_landing


# ------------------------------------------------------------
# Group-level analysis helpers
# ------------------------------------------------------------

class GroupDataAnalyzer:
    """
    Keep this class as the place for multi-trial or multi-fly analyses.

    I am intentionally leaving some methods blank because the existing file is very large
    and those methods are tightly tied to your own analysis logic. It is better for you
    to manually inspect and migrate them one by one.
    """

    def __init__(self, platform_offset=0, radius=0, FPS=0):
        self.platform_offset = platform_offset
        self.radius = radius
        self.fps = FPS
        self.calculator = SimpleCalculation()
        self.manipulator = FileManipulation()
        self.detector = DetectCharacteristics(radius=radius, FPS=FPS)


    # ------------------------------------------------------------
    # Internal helpers for new Group object
    # ------------------------------------------------------------

    def _ensure_metadata_ready(self, group_info):
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

    def _ensure_trials_loaded(self, group_info, trial_types=None):
        self._ensure_metadata_ready(group_info)

        if trial_types is None:
            trial_types = ["Landing", "Flying", "NF", "NA"]

        group_info.read_kinematic_data(trial_types=trial_types)

    def _get_trial_obj(self, group_info, index):
        key = f"F{index[0]}T{index[1]}"
        return group_info.fly_kinematic_data[key]

    def _get_trial_meta(self, group_info, index):
        key = f"F{index[0]}T{index[1]}"
        return group_info.trial_metadata[key]

    # ------------------------------------------------------------
    # Angle traces
    # ------------------------------------------------------------

    def Calculate_angle_traces(self, group_info, index_to_iterate, angles, threshold=None, start=-0.3, end=0.5, chrimson=False):
        """
        Calculate aligned angle traces for a set of trials.

        Notes:
        - This now ensures Trial objects are loaded before using fly_kinematic_data
        - Still uses MOC / MOL from Trial object, just like your original version
        """
        if index_to_iterate is None or len(index_to_iterate) == 0:
            return []

        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying"])

        collected_data = {}
        for a in angles:
            collected_data[a[1]] = []
        for index in index_to_iterate:
            trial_info = self._get_trial_obj(group_info, index)

            MOC = trial_info.moc
            if chrimson:
                MOC = 750
            if MOC < 0:
                print("Something wrong")
                continue

            angs = self.calculator.Calculate_joint_angle(trial_info, angles)

            for joint in angles:
                joint_name = joint[1]

                trace_start = int(MOC) - int(-start * trial_info.fps)
                trace_end = int(MOC) + int(end * trial_info.fps)
                Joint_signal = np.asarray(angs[joint_name][trace_start:trace_end])

                if trial_info.fps == 200:
                    target_len = int(round((end - start) * 250))
                    Joint_signal = self.calculator.Normalized_time(Joint_signal, target_len)

                collected_data[joint_name].append(Joint_signal)
        return collected_data

    # ------------------------------------------------------------
    # Secondary contact
    # ------------------------------------------------------------

    def AnalyzeSecondaryContact(self, index_to_iterate, group_info, threshold, radius, analysis_window=0.71, condition=""):
        """
        Analyze secondary contact timing for each segment.

        Output dataframe is preprocessed into:
            SC, Index, Result

        SC = minimum value among columns containing:
            L-f, L-m, L-h

        If all those values are 10000, SC is set to NaN.
        """
        if index_to_iterate is None or len(index_to_iterate) == 0:
            return pd.DataFrame(columns=["SC", "Index", "Result"])

        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying"])

        segs = [
            ["L-fFT", "L-fTT"], ["L-fTT", "L-fLT"],
            ["L-mFT", "L-mTT"], ["L-mTT", "L-mLT"],
            ["L-hFT", "L-hTT"], ["L-hTT", "L-hLT"]
        ]
        ra = [0.45, 0.4, 0.5, 0.4, 0.55, 0.4]
        pts = [
            "L-fFT", "L-fTT", "L-fLT",
            "L-mFT", "L-mTT", "L-mLT",
            "L-hFT", "L-hTT", "L-hLT"
        ]
        indi_leg_contact_event = dict()
        for j in segs:
            indi_leg_contact_event[j[0]] = []

        indi_leg_contact_event["Index"] = []
        indi_leg_contact_event["Result"] = []

        for index in index_to_iterate:
            pose_data = self._get_trial_obj(group_info, index)

            start = int(pose_data.moc)
            end = int(pose_data.mol)

            if start > 0:
                if end == -1:
                    end = start + int(analysis_window * pose_data.fps)
                    indi_leg_contact_event["Result"].append("Failed")
                elif (end - start) / pose_data.fps > threshold:
                    if analysis_window == threshold:
                        end = start + int(threshold * pose_data.fps)
                    indi_leg_contact_event["Result"].append("Failed")
                elif (end - start) / pose_data.fps <= threshold:
                    indi_leg_contact_event["Result"].append("Success")
                else:
                    print("Unable to categorize!")

            line_points, plane_points, verts, cylinder_top, cylinder_bottom, direction, perp_vector1, perp_vector2 = (
                self.calculator.calculate_platform_surfaces(
                    trial_info=pose_data,
                    platform_offset=0.03,
                    platform_height=3,
                    radius=radius
                )
            )

            Original_points = dict()
            center_points = self.calculator.ReadAndTranspose("platform-tip", pose_data)[start:end]

            for p in pts:
                Original_points[p] = self.calculator.ReadAndTranspose(p, pose_data)[start:end]

            for s, point in enumerate(segs):
                NoContact = True
                stable_contact = 0

                for current_frame in range(end - start):
                    A = Original_points[point[0]][current_frame]
                    B = Original_points[point[1]][current_frame]
                    P1 = center_points[current_frame]

                    d = direction
                    r = radius
                    h = 3

                    intersects_side, pt_side, min_dist = self.calculator.check_cylinder_side_intersection(
                        A, B, P1, d, r, h
                    )
                    intersects_top, pt_top = self.detector.check_leg_platform_intersection(
                        A, B, d, P1, 0.03, r
                    )

                    if intersects_side or intersects_top:
                        stable_contact += 1
                    else:
                        stable_contact = 0

                    if stable_contact >= 2:
                        indi_leg_contact_event[point[0]].append(current_frame / pose_data.fps)
                        NoContact = False
                        break

                if NoContact:
                    indi_leg_contact_event[point[0]].append(10000)

            if index not in indi_leg_contact_event["Index"]:
                indi_leg_contact_event["Index"].append(index)

        # ------------------------------------------------------------
        # preprocess before saving
        # ------------------------------------------------------------
        indi_leg_contact_event = pd.DataFrame(indi_leg_contact_event)

        # only use left leg columns for SC
        sc_cols = [col for col in indi_leg_contact_event.columns
                   if ("L-f" in col or "L-m" in col or "L-h" in col)]

        sc_df = indi_leg_contact_event[sc_cols].copy()

        # replace dummy code with NaN before taking minimum
        sc_df = sc_df.replace(10000, np.nan)

        processed_df = pd.DataFrame()
        processed_df["SC"] = sc_df.min(axis=1)
        processed_df["Index"] = indi_leg_contact_event["Index"]
        processed_df["Result"] = indi_leg_contact_event["Result"]

        if condition == "":
            indi_leg_contact_event.to_csv(f"{group_info.group_name}-{threshold}-SC_data.csv", index=False)
            return indi_leg_contact_event
        processed_df.to_csv(f"{group_info.group_name}-{condition}-{threshold}-SC_data.csv", index=False)
        return processed_df

    # ------------------------------------------------------------
    # Contact leg metrics
    # ------------------------------------------------------------

    def Calculate_contact_leg_metrices(self, group_info, index_to_iterate, joint_angle, threshold=0.71):
        """
        Compare average angular velocity between success and failed trials.
        """
        if index_to_iterate is None or len(index_to_iterate) == 0:
            return [], []

        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying"])

        joint_to_examine = joint_angle[0][1]
        failed_ang_v = []
        failed_posture = []
        success_ang_v = []
        success_posture = []
        for index in index_to_iterate:
            trial_info = self._get_trial_obj(group_info, index)
            start = trial_info.moc

            if start > 0:
                angs = self.calculator.Calculate_joint_angle(trial_info, joint_angle)
                Failed = False

                analysis_start = trial_info.moc

                if trial_info.mol < 0 or (trial_info.mol > 0 and ((trial_info.mol - trial_info.moc) / trial_info.fps) > threshold):
                    Failed = True

                ang_v = np.mean(np.gradient(angs[joint_to_examine][analysis_start:int(analysis_start + 0.1 * trial_info.fps)])) * trial_info.fps
                posture = np.mean(angs[joint_to_examine][:analysis_start])
                if Failed:
                    failed_ang_v.append(ang_v)
                    failed_posture.append(posture)
                else:
                    success_ang_v.append(ang_v)
                    success_posture.append(posture)

        return success_ang_v, success_posture, failed_ang_v, failed_posture

    # ------------------------------------------------------------
    # Leg search
    # ------------------------------------------------------------

    def Analyze_leg_search(self, group_info, index_to_iterate=None, condition="", threshold=0.71, analysis_window=0.71):
        print(threshold)
        """
        Analyze leg search counts.

        Default behavior:
        - if no index_to_iterate is passed, use Landing trials from Group metadata
        """
        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying"])
        Angles = [
            [["L-fBC", "L-fCT", "L-fFT"], ["L-fCT", "L-fFT", "L-fTT"]],
            [["L-mBC", "L-mCT", "L-mFT"], ["L-mCT", "L-mFT", "L-mTT"]],
            [["L-hBC", "L-hCT", "L-hFT"], ["L-hCT", "L-hFT", "L-hTT"]],
            [["R-fBC", "R-fCT", "R-fFT"], ["R-fCT", "R-fFT", "R-fTT"]],
            [["R-mBC", "R-mCT", "R-mFT"], ["R-mCT", "R-mFT", "R-mTT"]],
            [["R-hBC", "R-hCT", "R-hFT"], ["R-hCT", "R-hFT", "R-hTT"]]
        ]

        leg_search_data = dict()
        leg_search_data["L-f"] = []
        leg_search_data["L-m"] = []
        leg_search_data["L-h"] = []
        leg_search_data["R-f"] = []
        leg_search_data["R-m"] = []
        leg_search_data["R-h"] = []
        leg_search_data["Index"] = []
        leg_search_data["Result"] = []

        if index_to_iterate is None:
            self._ensure_metadata_ready(group_info)
            index_to_iterate = group_info.get_targeted_trials(["Landing"])

        for index in index_to_iterate:
            trial_info = self._get_trial_obj(group_info, index)
            start = trial_info.moc
            end = trial_info.mol
            leg_search_data["Index"].append(index)

            if end == -1:
                end = start + int(analysis_window * trial_info.fps)
                leg_search_data["Result"].append("Failed")
            elif (end - start) / trial_info.fps > threshold:
                if analysis_window == threshold:
                    end = start + int(threshold * trial_info.fps)
                leg_search_data["Result"].append("Failed")
            elif (end - start) / trial_info.fps <= threshold:
                leg_search_data["Result"].append("Success")
            else:
                print("Unable to categorize!")

            for pair in Angles:
                ags = self.calculator.Calculate_joint_angle(trial_info, pair)

                ct_trace = None
                ft_trace = None

                for ag in pair:
                    trace = np.array(ags[ag[1]][start:end + 1])
                    if len(ags[ag[1]]) == 1400:
                        trace = self.calculator.Normalized_time(trace, int(1.25 * len(trace)))
                    if "CT" in ag[1]:
                        ct_trace = trace
                    if "FT" in ag[1]:
                        ft_trace = trace

                if "f" in pair[0][0][:3]:
                    """counts, events = self.detector.detect_leg_search(
                        ft=ft_trace,
                        ct=ct_trace,
                        ft_vel_thresh=1.07,
                        ct_vel_thresh=0.295,
                        min_ft_change=6,
                        min_ct_change=1,
                        pattern_duration=20,
                        ct_start_thresh=32
                    )"""
                    peaks, _ = find_peaks(ct_trace, height=86.939, prominence=19.609)
                    leg_search_data[pair[0][0][:3]].append(len(peaks))

                elif "m" in pair[0][0][:3]:
                    """counts, events = self.detector.detect_leg_search(
                        ft=ft_trace,
                        ct=ct_trace,
                        ft_vel_thresh=1.94,
                        ct_vel_thresh=0.8,
                        min_ft_change=9,
                        min_ct_change=3,
                        pattern_duration=19,
                        ct_start_thresh=60
                    )"""
                    peaks, _ = find_peaks(ct_trace, height=84.355, prominence=15.267)
                    leg_search_data[pair[0][0][:3]].append(len(peaks))
                else:
                    # peaks, _ = find_peaks(ct_trace, height=95.878, prominence=10.697)
                    peaks, _ = find_peaks(ct_trace, height=105.215, prominence=7.418)
                    leg_search_data[pair[0][0][:3]].append(len(peaks))


        ls_cols = ["L-f", "L-m", "L-h"]
        leg_search_data = pd.DataFrame(leg_search_data)

        leg_search_data["LS_sum"] = leg_search_data[ls_cols].sum(axis=1)
        leg_search_data = leg_search_data[["LS_sum", "Index", "Result"]].copy()
        if condition == "":
            leg_search_data.to_csv(f"{group_info.group_name}-{threshold}-LS_data.csv", index=False)
            return leg_search_data
        leg_search_data.to_csv(f"{group_info.group_name}-{condition}-{threshold}-LS_data.csv", index=False)
        return leg_search_data


    def Analyze_leg_search_CHR(self, group_info, index_to_iterate=None, filename="", threshold=0.71):
        """
        Analyze leg search counts for Chr data.
        """
        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying", "NF", "NA"])

        Angles = [
            [["L-fBC", "L-fCT", "L-fFT"], ["L-fCT", "L-fFT", "L-fTT"]],
            [["L-mBC", "L-mCT", "L-mFT"], ["L-mCT", "L-mFT", "L-mTT"]],
            [["L-hBC", "L-hCT", "L-hFT"], ["L-hCT", "L-hFT", "L-hTT"]],
            [["R-fBC", "R-fCT", "R-fFT"], ["R-fCT", "R-fFT", "R-fTT"]],
            [["R-mBC", "R-mCT", "R-mFT"], ["R-mCT", "R-mFT", "R-mTT"]],
            [["R-hBC", "R-hCT", "R-hFT"], ["R-hCT", "R-hFT", "R-hTT"]]
        ]

        leg_search_data = dict()
        leg_search_data["L-f"] = []
        leg_search_data["L-m"] = []
        leg_search_data["L-h"] = []
        leg_search_data["R-f"] = []
        leg_search_data["R-m"] = []
        leg_search_data["R-h"] = []
        leg_search_data["Index"] = []
        leg_search_data["Result"] = []

        if index_to_iterate is None:
            self._ensure_metadata_ready(group_info)
            index_to_iterate = group_info.get_targeted_trials(["Landing"])

        for index in index_to_iterate:
            trial_info = self._get_trial_obj(group_info, index)
            angs = self.calculator.Calculate_joint_angle(trial_info, [["L-wing", "L-wing-hinge", "R-wing"]])

            start = 750
            end = self.detector.detect_landing(angs["L-wing-hinge"][750:1250]) + start

            plt.plot(angs["L-wing-hinge"][750:1250])

            if end == 749:
                end = 1250

            leg_search_data["Index"].append(index)

            if end == 1250:
                end = start + int(threshold * trial_info.fps)
                leg_search_data["Result"].append("Failed")
            elif (end - start) / trial_info.fps > threshold:
                end = start + int(threshold * trial_info.fps)
                leg_search_data["Result"].append("Failed")
            elif (end - start) / trial_info.fps <= threshold:
                leg_search_data["Result"].append("Success")
            else:
                print("Unable to categorize!")

            print(index, end)

            for pair in Angles:
                ags = self.calculator.Calculate_joint_angle(trial_info, pair)

                ct_trace = None
                ft_trace = None

                for ag in pair:
                    trace = np.array(ags[ag[1]][start:end + 1])

                    if "CT" in ag[1]:
                        ct_trace = trace
                    if "FT" in ag[1]:
                        ft_trace = trace

                if "h" not in pair[0][0][:3]:
                    counts, events = self.detector.detect_leg_search(ft_trace, ct_trace)
                    leg_search_data[pair[0][0][:3]].append(counts)
                else:
                    peaks, _ = find_peaks(ct_trace, prominence=15)
                    leg_search_data[pair[0][0][:3]].append(len(peaks))

        leg_search_data = pd.DataFrame(leg_search_data)
        leg_search_data.to_csv(f"{group_info.group_name}-{filename}-LS_data_.csv", index=False)

        return leg_search_data

    def _build_leg_search_path(self, group_info, threshold, condition=""):
        if condition == "":
            return f"{group_info.group_name}-{threshold}-LS_data.csv"
        return f"{group_info.group_name}-{condition}-{threshold}-LS_data.csv"

    def _build_secondary_contact_path(self, group_info, threshold, condition=""):
        if condition == "":
            return f"{group_info.group_name}-{threshold}-SC_data.csv"
        return f"{group_info.group_name}-{condition}-{threshold}-SC_data.csv"

    def get_or_run_leg_search(self, group_info, index_to_iterate=None, filename="", threshold=0.71, analysis_window=0.71,
                              force_recompute=False):
        """
        Load saved leg search data if it already exists.
        Otherwise run Analyze_leg_search and save it.

        Returns:
            DataFrame with columns like:
            LS_sum, Index, Result
        """
        save_path = self._build_leg_search_path(group_info, threshold, filename)
        if (not force_recompute) and os.path.exists(save_path):
            print(f"Using existing leg search data: {save_path}")
            return pd.read_csv(save_path)

        print(f"Running leg search analysis: {save_path}")
        return self.Analyze_leg_search(
            group_info=group_info,
            index_to_iterate=index_to_iterate,
            analysis_window=analysis_window,
            condition=filename,
            threshold=threshold
        )

    def get_or_run_secondary_contact(self, group_info, index_to_iterate=None, radius=0.4, threshold=0.71, filename="", analysis_window=0.71,
                                     force_recompute=False):
        """
        Load saved secondary contact data if it already exists.
        Otherwise run AnalyzeSecondaryContact and save it.

        Returns:
            DataFrame with columns:
            SC, Index, Result
        """
        save_path = self._build_secondary_contact_path(group_info, threshold, filename)

        if (not force_recompute) and os.path.exists(save_path):
            print(f"Using existing secondary contact data: {save_path}")
            return pd.read_csv(save_path)

        print(f"Running secondary contact analysis: {save_path}")
        return self.AnalyzeSecondaryContact(
            index_to_iterate=index_to_iterate,
            group_info=group_info,
            threshold=threshold,
            condition=filename,
            analysis_window=analysis_window,
            radius=radius
        )

    def get_secondary_contact_kmc_df(self, group_info,
                                     index_to_iterate=None,
                                     radius=0.4,
                                     threshold=0.71,
                                     filename="",
                                     analysis_window=0.71,
                                     force_recompute=False):
        """
        Return a KMC-ready dataframe for secondary contact.

        It will:
        - load saved SC analysis if available
        - otherwise run SC analysis
        - convert SC / Index / Result into latency / event format
        """
        if index_to_iterate is None:
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()
                group_info.filter_nan_fly()

            index_to_iterate = group_info.get_targeted_trials(["Landing", "Flying"])

        sc_df = self.get_or_run_secondary_contact(
            group_info=group_info,
            index_to_iterate=index_to_iterate,
            radius=radius,
            threshold=threshold,
            filename=filename,
            analysis_window=analysis_window,
            force_recompute=force_recompute
        )

        if sc_df is None or len(sc_df) == 0:
            return pd.DataFrame(columns=["SC", "Index", "Result", "Latency", "Event", "Group_Name"])

        kmc_df = sc_df.copy()
        kmc_df["Event"] = kmc_df["SC"].notna().astype(int)
        kmc_df["Latency"] = kmc_df["SC"].fillna(threshold)
        kmc_df["Group_Name"] = group_info.group_name

        return kmc_df

    def extract_leg_search_parameters(self, group_info:Group, label_df, leg_columns=("L-f", "L-m", "L-h"), smooth_window=1, margin=0):
        """
        Extract parameter values from labeled search bouts.

        Parameters
        ----------
        label_df : pd.DataFrame
            Must contain:
                - 'Index' column with trial identifier
                - leg columns storing (start, stop) tuples
        get_angle_data_fn : callable
            Function with signature:
                ft, ct = get_angle_data_fn(trial_index, leg_name)
            where:
                trial_index is the value from label_df['Index']
                leg_name is one of leg_columns
            and ft, ct are 1D angle arrays.
        leg_columns : tuple[str]
            Label columns to parse.
        smooth_window : int
            Moving average window before gradient calculation.
        margin : int
            Optional number of frames to expand around labeled interval.
            Useful if labels are tight and you want to capture onset context.

        Returns
        -------
        param_df : pd.DataFrame
            One row per labeled bout, containing extracted parameters.
        """
        group_info.initialize_manual_data()
        group_info.read_kinematic_data()
        records = []

        for _, row in label_df.iterrows():
            trial_index = self.calculator.parse_tuple_cell(row["Index"]) if isinstance(row["Index"], str) else row["Index"]

            for leg in leg_columns:
                if leg not in row:
                    continue

                bout = self.calculator.parse_tuple_cell(row[leg])
                if bout is None:
                    continue

                start, stop = bout
                if stop <= start:
                    continue

                trial_info = group_info.fly_kinematic_data[f"F{trial_index[0]}T{trial_index[1]}"]
                ft, ct = self.calculator.get_angle_data_fn(trial_info, leg)
                ft = np.asarray(ft, dtype=float)
                ct = np.asarray(ct, dtype=float)

                if len(ft) != len(ct):
                    raise ValueError(f"FT and CT length mismatch for {trial_index}, {leg}")

                # Expand interval if desired
                s = max(0, start - margin)
                e = min(len(ft), stop + margin)

                if e - s < 2:
                    continue

                # Segment restricted to labeled bout
                ft_seg = ft[s:e]
                ct_seg = ct[s:e]

                ft_vel_threshold, ft_min_ag_threshold = self.calculator.get_ag_vel_thresh(ft_seg)
                ct_vel_threshold, ct_min_ag_threshold = self.calculator.get_ag_vel_thresh(ct_seg)
                ct_start_threshold = self.calculator.get_ct_start_threshold(ct_seg)
                bout_duration = e - s

                records.append({
                    "Index": trial_index,
                    "Leg": leg,
                    "LabelStart": start,
                    "LabelStop": stop,
                    "UsedStart": s,
                    "UsedStop": e,
                    "FT_vel_threshold": ft_vel_threshold,
                    "CT_vel_threshold": ct_vel_threshold,
                    "FT_min_change": ft_min_ag_threshold,
                    "CT_min_change": ct_min_ag_threshold,
                    "CT_start_threshold": ct_start_threshold,
                    "Bout_duration": bout_duration,
                })
        param_df = pd.DataFrame(records)
        return param_df

    def extract_ct_peak_finder_parameters(
            self,
            group_info,
            label_df,
            leg_columns=("L-f", "L-m", "L-h"),
            smooth_window=5,
            margin_before=5,
            margin_after=20,
            loose_peak_distance=3,
    ):
        """
        Extract CT extension parameters from positive labeled bouts.

        Assumes each label cell stores:
            (rise_start_frame, peak_frame)

        Returns one row per labeled extension event.
        """

        group_info.initialize_manual_data()
        group_info.read_kinematic_data()

        records = []

        for _, row in label_df.iterrows():

            trial_index = (
                self.calculator.parse_tuple_cell(row["Index"])
                if isinstance(row["Index"], str)
                else row["Index"]
            )

            for leg in leg_columns:

                if leg not in row:
                    continue

                bout = self.calculator.parse_tuple_cell(row[leg])

                if bout is None:
                    continue

                start, peak = bout
                start = int(start)
                peak = int(peak)

                if peak <= start:
                    continue

                key = f"F{trial_index[0]}T{trial_index[1]}"

                if key not in group_info.fly_kinematic_data:
                    print(f"Missing trial: {key}")
                    continue

                trial_info = group_info.fly_kinematic_data[key]

                ft, ct = self.calculator.get_angle_data_fn(trial_info, leg)
                ct = np.asarray(ct, dtype=float)

                if len(ct) < peak + 1:
                    print(f"Peak frame outside trace length: {key}, {leg}")
                    continue

                # Smooth CT trace
                ct_s = self.calculator.exponential_moving_average(ct, alpha=0.4)
                ct_s = np.asarray(ct_s, dtype=float)

                # If you prefer your old moving average instead, use:
                # ct_s = self.moving_average(ct, smooth_window)

                dct = np.gradient(ct_s)

                # Main labeled interval
                labeled_seg = ct_s[start:peak + 1]
                labeled_vel = dct[start:peak + 1]

                # Wider window around label for estimating true peak properties
                s = max(0, start - margin_before)
                e = min(len(ct_s), peak + margin_after + 1)
                search_seg = ct_s[s:e]

                # Loose peak detection inside label-centered window
                candidate_peaks, _ = find_peaks(
                    search_seg,
                    distance=loose_peak_distance
                )

                # Convert local peak indices to global frame indices
                candidate_peaks_global = candidate_peaks + s

                # Pick the detected peak closest to your labeled peak frame
                if len(candidate_peaks_global) > 0:
                    closest_idx = np.argmin(np.abs(candidate_peaks_global - peak))
                    detected_peak_global = int(candidate_peaks_global[closest_idx])
                    detected_peak_local = int(candidate_peaks[closest_idx])

                    prom = peak_prominences(search_seg, [detected_peak_local])[0][0]
                    width = peak_widths(search_seg, [detected_peak_local], rel_height=0.5)[0][0]
                else:
                    detected_peak_global = peak
                    detected_peak_local = peak - s

                    # fallback: your label-based prominence proxy
                    prom = ct_s[peak] - ct_s[start]
                    width = np.nan

                ct_start_angle = ct_s[start]
                ct_peak_angle = ct_s[peak]
                ct_detected_peak_angle = ct_s[detected_peak_global]

                rise_amplitude = ct_peak_angle - ct_start_angle
                abs_rise_amplitude = abs(rise_amplitude)

                peak_velocity = np.nanmax(labeled_vel)
                peak_abs_velocity = np.nanmax(np.abs(labeled_vel))

                rise_duration_frames = peak - start
                rise_duration_sec = rise_duration_frames / trial_info.fps

                records.append({
                    "Index": trial_index,
                    "Fly": trial_index[0],
                    "Trial": trial_index[1],
                    "Leg": leg,

                    "Rise_Start": start,
                    "Labeled_Peak": peak,
                    "Detected_Peak": detected_peak_global,
                    "Peak_Frame_Error": detected_peak_global - peak,

                    "CT_start_angle": ct_start_angle,
                    "CT_labeled_peak_angle": ct_peak_angle,
                    "CT_detected_peak_angle": ct_detected_peak_angle,

                    "CT_rise_amplitude": rise_amplitude,
                    "CT_abs_rise_amplitude": abs_rise_amplitude,

                    "CT_peak_velocity": peak_velocity,
                    "CT_peak_abs_velocity": peak_abs_velocity,

                    "Rise_duration_frames": rise_duration_frames,
                    "Rise_duration_sec": rise_duration_sec,

                    "Peak_prominence": prom,
                    "Peak_width": width,
                })

        return pd.DataFrame(records)

    def extract_labeled_ct_peak_parameters(
            self,
            label_df,
            group_info,
            get_angle_data_fn,
            smooth_window=7,
            peak_correction_window=5,
            target_len=1750,
            debug_plot=False,
    ):
        """
        Extract peak-finder parameters from manually labeled CT extension peaks.

        label_df columns:
            Fly, Trial, Leg, Event_ID, Peak_Frame

        Notes:
            - If trace length is 1400, it is normalized to target_len.
            - Peak_Frame is converted into the normalized frame coordinate.
            - Local maxima are searched within Peak_Frame ± peak_correction_window.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks, peak_prominences, peak_widths

        records = []

        for _, row in label_df.iterrows():

            fly = int(row["Fly"])
            trial = int(row["Trial"])
            leg = row["Leg"]
            event_id = int(row["Event_ID"])

            labeled_peak_original = int(row["Peak_Frame"])
            labeled_peak = labeled_peak_original

            key = f"F{fly}T{trial}"

            if key not in group_info.fly_kinematic_data:
                print(f"Missing trial: {key}")
                continue

            trial_info = group_info.fly_kinematic_data[key]

            ft, ct = get_angle_data_fn(trial_info, leg)

            original_len = len(ct)
            normalized = False

            peak_correction_window_used = peak_correction_window

            if original_len == 1400:
                normalized = True

                labeled_peak = int(round(labeled_peak_original * target_len / original_len))
                peak_correction_window_used = int(round(peak_correction_window * target_len / original_len))

                ft = self.calculator.Normalized_time(ft, target_len)
                ct = self.calculator.Normalized_time(ct, target_len)

            ct = np.asarray(ct, dtype=float)

            if labeled_peak < 0 or labeled_peak >= len(ct):
                print(
                    f"Peak outside trace: {key}, {leg}, "
                    f"original={labeled_peak_original}, normalized={labeled_peak}"
                )
                continue

            # Smooth CT trace exactly as you will for detection
            sw = smooth_window
            if sw % 2 == 0:
                sw += 1

            if len(ct) > sw:
                ct_s = self.calculator.exponential_moving_average(ct, alpha=0.4)
                ct_s = np.asarray(ct_s, dtype=float)
            else:
                ct_s = ct.copy()

            if debug_plot:
                plt.figure(figsize=(8, 3))
                plt.plot(ct, color="orange", alpha=0.5, label="raw / normalized CT")
                plt.plot(ct_s, color="blue", label="smoothed CT")
                plt.scatter(x=labeled_peak, y=ct_s[labeled_peak], color="red")
                # plt.axvline(labeled_peak, color="black", linestyle="--", label="labeled peak")
                plt.title(f"{key} {leg} Event {event_id}")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # Search for real local maxima near labeled peak
            search_left = max(0, labeled_peak - peak_correction_window_used)
            search_right = min(len(ct_s), labeled_peak + peak_correction_window_used + 1)

            search_segment = ct_s[search_left:search_right]

            local_peaks, _ = find_peaks(search_segment, distance=1)

            if len(local_peaks) > 0:
                local_labeled = labeled_peak - search_left

                closest_local_peak = local_peaks[np.argmin(np.abs(local_peaks - local_labeled))]

                corrected_peak = search_left + int(closest_local_peak)
                correction_status = "local_peak_found"

                prominence = peak_prominences(ct_s, [corrected_peak])[0][0]
                width = peak_widths(ct_s, [corrected_peak], rel_height=0.5)[0][0]

            else:
                corrected_peak = labeled_peak
                correction_status = "no_local_peak_found"

                prominence = np.nan
                width = np.nan

            peak_height = ct_s[corrected_peak]

            records.append({
                "Fly": fly,
                "Trial": trial,
                "Leg": leg,
                "Event_ID": event_id,

                "Original_Trace_Length": original_len,
                "Normalized": normalized,
                "Target_Length": target_len if normalized else original_len,

                "Labeled_Peak_Frame_Original": labeled_peak_original,
                "Labeled_Peak_Frame_Normalized": labeled_peak,
                "Corrected_Peak_Frame": corrected_peak,
                "Peak_Frame_Error": corrected_peak - labeled_peak,
                "Peak_Correction_Window_Used": peak_correction_window_used,
                "Correction_Status": correction_status,

                "CT_peak_height": peak_height,
                "Peak_prominence": prominence,
                "Peak_width": width,
            })

        param_df = pd.DataFrame(records)

        if len(param_df) == 0:
            return param_df

        param_df = param_df.sort_values(
            ["Fly", "Trial", "Leg", "Corrected_Peak_Frame"]
        ).reset_index(drop=True)

        # Optional diagnostic, even if you are not using distance as detector parameter
        param_df["Peak_to_peak_distance"] = (
            param_df
            .groupby(["Fly", "Trial", "Leg"])["Corrected_Peak_Frame"]
            .diff()
        )

        print(param_df["Correction_Status"].value_counts())

        return param_df
