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
            if smooth_method == "ema":
                interpolated = self.smooth_trace_ema(
                    interpolated,
                    alpha=smooth_alpha,
                )
            else:
                interpolated = self.smooth_trace(
                    interpolated,
                    window_frames=smooth_window_frames,
                    polyorder=smooth_polyorder
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
            if "wing" in joint_name:
                collected_angle_data[joint_name] = self.calculate_wing_angle_trace(trial_info)
                if apply_tracking_qc:
                    wing_points = ["L-wing", "L-wing-hinge", "R-wing"]
                    filtered_trace, qc_mask, qc_summary, _ = self.apply_angle_tracking_qc(
                        trial_info=trial_info,
                        angle_trace=collected_angle_data[joint_name],
                        angle_points=wing_points,
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
                        "Angle_Definition": "|".join(wing_points),
                    })
                    qc_summaries.append(qc_summary)
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

    def __init__(self):
        self.calculator = SimpleCalculation()


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

    def Calculate_angle_traces(
            self,
            group_info,
            index_to_iterate,
            angles,
            threshold=None,
            start=-0.3,
            end=0.5,
            chrimson=False,
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
        qc_rows = []
        skipped_rows = []
        for index in index_to_iterate:
            trial_info = self._get_trial_obj(group_info, index)

            MOC = trial_info.moc
            if chrimson:
                MOC = 750
            if MOC < 0:
                print("Something wrong")
                skipped_rows.append({
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Joint": "",
                    "Reason": "invalid alignment frame",
                    "Alignment_Frame": MOC,
                })
                continue

            # Convert the requested QC window from seconds around the alignment
            # frame into absolute frame indices before angle QC is calculated.
            qc_start_frame = None
            qc_end_frame = None
            if apply_tracking_qc:
                qc_window_start_s = start if qc_start is None else qc_start
                qc_window_end_s = end if qc_end is None else qc_end
                qc_start_frame = int(MOC) + int(qc_window_start_s * trial_info.fps)
                qc_end_frame = int(MOC) + int(qc_window_end_s * trial_info.fps) - 1

            angle_result = self.calculator.Calculate_joint_angle(
                trial_info,
                angles,
                apply_tracking_qc=apply_tracking_qc,
                min_cameras=min_cameras,
                max_interp_gap_s=max_interp_gap_s,
                min_valid_fraction=min_valid_fraction,
                error_max=error_max,
                score_min=score_min,
                require_score=require_score,
                smooth_angle=smooth_angle,
                smooth_method=smooth_method,
                smooth_window_frames=smooth_window_frames,
                smooth_polyorder=smooth_polyorder,
                smooth_alpha=smooth_alpha,
                # The angle QC summary and pass/fail rule should describe the
                # same analysis window that the downstream trace uses.
                qc_start=qc_start_frame,
                qc_end=qc_end_frame,
                return_qc=apply_tracking_qc
            )
            if apply_tracking_qc:
                angs, trial_qc_df = angle_result
                if not trial_qc_df.empty:
                    trial_qc_df = trial_qc_df.copy()
                    trial_qc_df["Index"] = str(index)
                    trial_qc_df["Fly#"] = index[0]
                    trial_qc_df["Trial#"] = index[1]
                    trial_qc_df["Group_Name"] = group_info.group_name
                    qc_rows.extend(trial_qc_df.to_dict("records"))
            else:
                angs = angle_result

            for joint in angles:
                joint_name = joint[1]

                trace_start = int(MOC) - int(-start * trial_info.fps)
                trace_end = int(MOC) + int(end * trial_info.fps)
                Joint_signal = np.asarray(angs[joint_name][trace_start:trace_end])

                if apply_tracking_qc:
                    # Resolve the time-based gap rule in the trial's native FPS
                    # before checking plotted-window validity.
                    max_interp_gap_frames = tqc.interp_gap_frames_from_fps(max_interp_gap_s, trial_info.fps)
                    # Reuse the absolute QC frames passed into angle
                    # calculation so the secondary finite-sample guard is
                    # aligned with the primary QC summary.
                    qc_window_start_s = start if qc_start is None else qc_start
                    qc_window_end_s = end if qc_end is None else qc_end
                    qc_trace_start = qc_start_frame
                    qc_trace_end = qc_end_frame + 1
                    qc_trace_start = max(qc_trace_start, trace_start)
                    qc_trace_end = min(qc_trace_end, trace_end)
                    qc_start_offset = max(qc_trace_start - trace_start, 0)
                    qc_end_offset = max(qc_trace_end - trace_start, qc_start_offset)
                    qc_signal = Joint_signal[qc_start_offset:qc_end_offset]
                    window_valid = np.isfinite(qc_signal)
                    window_valid_fraction = float(np.mean(window_valid)) if len(window_valid) else np.nan
                    window_gaps = self.calculator.invalid_gap_lengths(window_valid) if len(window_valid) else []
                    max_window_gap = int(max(window_gaps)) if window_gaps else 0
                    finite_count = int(np.sum(window_valid))
                    skip_reason = ""
                    if pd.isna(window_valid_fraction):
                        skip_reason = "empty plotted window"
                    elif window_valid_fraction < min_valid_fraction:
                        skip_reason = "valid_fraction_below_threshold"
                    elif max_window_gap > max_interp_gap_frames:
                        skip_reason = "long_invalid_gap"
                    elif finite_count < 2:
                        skip_reason = "fewer_than_two_finite_frames"
                    if skip_reason:
                        skipped_rows.append({
                            "Group_Name": group_info.group_name,
                            "Index": str(index),
                            "Fly#": index[0],
                            "Trial#": index[1],
                            "Joint": joint_name,
                            "Angle_Definition": "|".join(joint),
                            "Reason": skip_reason,
                            "Alignment_Frame": MOC,
                            "Trace_Start_Frame": trace_start,
                            "Trace_End_Frame": trace_end - 1,
                            "QC_Start_Frame": qc_trace_start,
                            "QC_End_Frame": qc_trace_end - 1,
                            "Requested_Start_s": start,
                            "Requested_End_s": end,
                            "QC_Start_s": qc_window_start_s,
                            "QC_End_s": qc_window_end_s,
                            "Window_Frame_Count": int(len(qc_signal)),
                            "Finite_Frame_Count": finite_count,
                            "Window_Valid_Frame_Fraction": window_valid_fraction,
                            "Window_Max_Invalid_Gap_Frames": max_window_gap,
                            "Min_Valid_Fraction": min_valid_fraction,
                            "Max_Interp_Gap_s": max_interp_gap_s,
                            "Max_Interp_Gap_Frames": max_interp_gap_frames,
                            "Min_Cameras": min_cameras,
                        })
                        continue

                if trial_info.fps == 200:
                    target_len = int(round((end - start) * 250))
                    Joint_signal = self.calculator.Normalized_time(Joint_signal, target_len)

                collected_data[joint_name].append(Joint_signal)
        if return_qc:
            return collected_data, pd.DataFrame(qc_rows), pd.DataFrame(skipped_rows)
        return collected_data


