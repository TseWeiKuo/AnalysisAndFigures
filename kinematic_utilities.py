import ast
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tracking_qc as tqc

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

    def get_tracking_qc_mask(
            self,
            trial_info,
            keypoints,
            min_cameras=2,
            require_finite_error=True,
            error_max=50,
            score_min=0.8,
            require_score=False,
    ):
        """
        Return one frame-wise QC mask requiring every listed keypoint to pass.

        A point/frame is valid when xyz are finite, camera count is at least
        min_cameras, reprojection error is finite and below threshold, and
        optional score fields pass score_min.
        """
        if isinstance(keypoints, str):
            keypoints = [keypoints]

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
            _, components, metadata = tqc.point_invalid_components(
                point=point,
                keypoint=keypoint,
                min_cameras=min_cameras,
                error_max=error_max,
                score_min=score_min,
                require_score=require_score,
            )
            if not require_finite_error:
                invalid = components["invalid"] & ~components["error_missing"]
            else:
                invalid = components["invalid"]
            mask = ~invalid
            combined_mask &= mask
            point_summaries.append({
                "Keypoint": keypoint,
                "Reason": "ok",
                "Valid_Frame_Fraction": float(np.mean(mask)) if len(mask) else np.nan,
                "Min_Cameras": min_cameras,
                "Error_Threshold": metadata["Error_Threshold"],
                "Score_Min": score_min,
                "Score_Column": metadata["Score_Column"],
                "Score_Column_Missing": metadata["Score_Column_Missing"],
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

    def smooth_trace_ema(self, values, alpha=0.4):
        """Smooth finite contiguous trace segments with exponential moving average."""
        values = np.asarray(values, dtype=float).copy()
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
            values[start:stop] = np.asarray(
                self.exponential_moving_average(values[start:stop], alpha),
                dtype=float,
            )
        return values

    def apply_angle_tracking_qc(
            self,
            trial_info,
            angle_trace,
            angle_points,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False,
            start_frame=None,
            end_frame=None,
            smooth=False,
            smooth_method="savgol",
            smooth_window_frames=5,
            smooth_polyorder=2,
            smooth_alpha=0.4
    ):
        """
        Apply tracking QC to one angle trace.

        Invalid frames are set to NaN. Short gaps are linearly interpolated
        using a time-based threshold resolved from the trial FPS.
        """
        # Convert the 20 ms interpolation rule to this trial's native frame count.
        max_interp_gap_frames = tqc.interp_gap_frames_from_fps(max_interp_gap_s, trial_info.fps)
        angle_trace = np.asarray(angle_trace, dtype=float).copy()
        qc_mask, point_summary = self.get_tracking_qc_mask(
            trial_info,
            angle_points,
            min_cameras=min_cameras,
            require_finite_error=True,
            error_max=error_max,
            score_min=score_min,
            require_score=require_score,
        )
        invalid_mask = (~qc_mask) | ~np.isfinite(angle_trace)
        interpolated, interpolated_count = tqc.interpolate_invalid_trace_gaps(
            angle_trace,
            invalid_mask,
            max_gap_frames=max_interp_gap_frames
        )
        if smooth:
            interpolated = self.smooth_trace_ema(
                interpolated,
                alpha=smooth_alpha,
            )

        summary = tqc.summarize_invalid_mask(
            invalid_mask,
            start_frame=start_frame,
            end_frame=end_frame,
            max_interp_gap_frames=max_interp_gap_frames,
            max_interp_gap_s=max_interp_gap_s,
            min_valid_fraction=min_valid_fraction,
        )
        summary.update({
            "Interpolated_Frame_Count": int(interpolated_count),
            "Min_Cameras": min_cameras,
            "Max_Interp_Gap_s": max_interp_gap_s,
            "Error_Max": error_max,
            "Score_Min": score_min,
            "Require_Score": bool(require_score),
            "Smooth_Angle": bool(smooth),
            "Smooth_Method": smooth_method if smooth else "",
            "Smooth_Window_Frames": smooth_window_frames,
            "Smooth_Alpha": smooth_alpha if smooth else np.nan,
        })
        return interpolated, ~invalid_mask, summary, point_summary

    def apply_xyz_tracking_qc(
            self,
            trial_info,
            keypoint,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False,
            start_frame=None,
            end_frame=None,
            require_start_end_valid=False
    ):
        """
        Apply tracking QC to one keypoint's xyz trace.

        Invalid frames are set to NaN, and invalid gaps up to max_interp_gap_s
        are linearly interpolated independently for x/y/z.
        """
        # Convert the 20 ms interpolation rule to this trial's native frame count.
        max_interp_gap_frames = tqc.interp_gap_frames_from_fps(max_interp_gap_s, trial_info.fps)
        point = trial_info.trial_data[keypoint]
        xyz, components, metadata = tqc.point_invalid_components(
            point=point,
            keypoint=keypoint,
            min_cameras=min_cameras,
            error_max=error_max,
            score_min=score_min,
            require_score=require_score,
        )
        invalid_mask = components["invalid"]
        filtered, interpolated_count = tqc.interpolate_invalid_xyz_gaps(
            xyz,
            invalid_mask,
            max_gap_frames=max_interp_gap_frames
        )

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(invalid_mask) - 1
        start_frame = max(int(start_frame), 0)
        end_frame = min(int(end_frame), len(invalid_mask) - 1)
        summary = tqc.summarize_invalid_mask(
            invalid_mask,
            components=components,
            start_frame=start_frame,
            end_frame=end_frame,
            max_interp_gap_frames=max_interp_gap_frames,
            max_interp_gap_s=max_interp_gap_s,
            min_valid_fraction=min_valid_fraction,
            require_start_end_valid=require_start_end_valid,
        )
        summary.update({
            "Keypoint": keypoint,
            "Interpolated_Frame_Count": int(interpolated_count),
            "Min_Cameras": min_cameras,
            "Max_Interp_Gap_s": max_interp_gap_s,
            "Error_Max": error_max,
            "Score_Min": score_min,
            "Require_Score": bool(require_score),
            **metadata,
        })
        point_summary = pd.DataFrame([{
            "Keypoint": keypoint,
            "Reason": "ok",
            "Valid_Frame_Fraction": summary["Valid_Frame_Fraction"],
            "Invalid_Frame_Fraction": summary["Invalid_Frame_Fraction"],
            "Min_Cameras": min_cameras,
            "Error_Threshold": metadata["Error_Threshold"],
            "Score_Min": score_min,
            "Score_Column": metadata["Score_Column"],
            "Score_Column_Missing": metadata["Score_Column_Missing"],
        }])
        return filtered, ~invalid_mask, summary, point_summary

    def Calculate_joint_angle(
            self,
            trial_info,
            angles,
            apply_tracking_qc=False,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False,
            smooth_angle=False,
            smooth_method="savgol",
            smooth_window_frames=5,
            smooth_polyorder=2,
            smooth_alpha=0.4,
            qc_start=None,
            qc_end=None,
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
                    max_interp_gap_s=max_interp_gap_s,
                    min_valid_fraction=min_valid_fraction,
                    error_max=error_max,
                    score_min=score_min,
                    require_score=require_score,
                    start_frame=qc_start,
                    end_frame=qc_end,
                    smooth=smooth_angle,
                    smooth_method=smooth_method,
                    smooth_window_frames=smooth_window_frames,
                    smooth_polyorder=smooth_polyorder,
                    smooth_alpha=smooth_alpha,
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

    def Normalized_time(self, data, length=250):
        from scipy.interpolate import interp1d

        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, data, kind='linear')
        signal = f(x_new)

        return signal

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

