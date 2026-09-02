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

    def apply_xyz_tracking_qc(
            self,
            trial_info,
            keypoint,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            start_frame=None,
            end_frame=None,
            require_start_end_valid=False
    ):
        """
        Apply tracking QC to one keypoint's xyz trace.

        Invalid frames are set to NaN, and invalid gaps up to max_interp_gap_s
        are linearly interpolated independently for x/y/z.
        """
        # Use the new single keypoint-level QC entry point so xyz QC has one source of truth.
        point = trial_info.trial_data[keypoint]
        qc_result = tqc.qc_keypoint_xyz(
            point=point,
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
        # Keep the compatibility wrapper behavior: failed keypoints return NaN xyz traces.
        filtered = qc_result["clean_xyz"]
        if filtered is None:
            raw_n_frames = len(qc_result["invalid_mask"])
            filtered = np.full((raw_n_frames, 3), np.nan, dtype=float)
        # Reuse the compact summary emitted by tracking_qc instead of rebuilding it here.
        summary = dict(qc_result["qc_summary"])
        # Keep threshold and score-column provenance separate from the compact pass/fail summary.
        metadata = dict(qc_result["qc_metadata"])
        # The valid mask remains useful for existing geometry code that expects this return value.
        valid_mask = ~qc_result["invalid_mask"]
        point_summary = pd.DataFrame([{
            "Keypoint": keypoint,
            "Reason": summary["QC_Exclusion_Reason"] or "ok",
            "Valid_Frame_Fraction": summary["Valid_Frame_Fraction"],
            "Invalid_Frame_Fraction": summary["Invalid_Frame_Fraction"],
        }])
        return filtered, valid_mask, summary, point_summary

    def _angle_trace_from_xyz(self, xyz_a, xyz_b, xyz_c, n_frames):
        """Calculate one angle trace from three frame-aligned cleaned xyz arrays."""
        # Allocate the output up front so failed/undefined frames remain explicit NaNs.
        angle_trace = np.full(n_frames, np.nan, dtype=float)
        # Calculate each frame from cleaned xyz; any non-finite coordinate yields NaN.
        for f in range(n_frames):
            if (
                    np.all(np.isfinite(xyz_a[f]))
                    and np.all(np.isfinite(xyz_b[f]))
                    and np.all(np.isfinite(xyz_c[f]))
            ):
                angle_trace[f] = self.calculate_angle(
                    x1=xyz_a[f, 0],
                    y1=xyz_a[f, 1],
                    z1=xyz_a[f, 2],
                    x2=xyz_b[f, 0],
                    y2=xyz_b[f, 1],
                    z2=xyz_b[f, 2],
                    x3=xyz_c[f, 0],
                    y3=xyz_c[f, 1],
                    z3=xyz_c[f, 2],
                )
        return angle_trace

    def _aggregate_keypoint_qc_for_angle(self, keypoint_summaries, joint_name, angle_definition):
        """Collapse three keypoint QC rows into one compact angle-level summary."""
        # Convert the keypoint dictionaries into a dataframe for clear aggregation.
        keypoint_df = pd.DataFrame(keypoint_summaries)
        # The angle passes only when every required keypoint passes independently.
        qc_passed = bool(keypoint_df["QC_Passed"].all()) if not keypoint_df.empty else False
        # Failed keypoint reasons are preserved with the keypoint name for debugging.
        failed_reasons = []
        for _, row in keypoint_df.iterrows():
            if not bool(row.get("QC_Passed", False)):
                reason = row.get("QC_Exclusion_Reason", "") or "qc_failed"
                failed_reasons.append(f"{row.get('Keypoint', 'unknown')}:{reason}")
        # Angle-level fractions use conservative keypoint-first aggregation, not a combined mask.
        valid_fraction = (
            float(keypoint_df["Valid_Frame_Fraction"].min())
            if "Valid_Frame_Fraction" in keypoint_df and not keypoint_df.empty
            else np.nan
        )
        invalid_fraction = (
            float(keypoint_df["Invalid_Frame_Fraction"].max())
            if "Invalid_Frame_Fraction" in keypoint_df and not keypoint_df.empty
            else np.nan
        )
        # The longest invalid gap across the three keypoints determines the angle burden.
        max_invalid_gap = (
            int(keypoint_df["Max_Invalid_Gap_Frames"].max())
            if "Max_Invalid_Gap_Frames" in keypoint_df and not keypoint_df.empty
            else 0
        )
        # Sum interpolated frames across keypoints to record total xyz correction burden.
        interpolated_count = (
            int(keypoint_df["Interpolated_Frame_Count"].sum())
            if "Interpolated_Frame_Count" in keypoint_df and not keypoint_df.empty
            else 0
        )
        # The interpolation threshold is shared across keypoints within the same trial.
        max_interp_gap = (
            int(keypoint_df["Max_Interp_Gap_Frames"].max())
            if "Max_Interp_Gap_Frames" in keypoint_df and not keypoint_df.empty
            else 0
        )
        # Return the same compact QC fields expected by downstream plotting/stat code.
        return {
            "Joint": joint_name,
            "Angle_Definition": angle_definition,
            "QC_Passed": qc_passed,
            "QC_Exclusion_Reason": ";".join(failed_reasons),
            "Valid_Frame_Fraction": valid_fraction,
            "Invalid_Frame_Fraction": invalid_fraction,
            "Max_Invalid_Gap_Frames": max_invalid_gap,
            "Interpolated_Frame_Count": interpolated_count,
            "Max_Interp_Gap_Frames": max_interp_gap,
        }

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
            smooth_angle=False,
            smooth_method="savgol",
            smooth_window_frames=5,
            smooth_polyorder=2,
            smooth_alpha=0.4,
            qc_start=None,
            qc_end=None,
            return_qc=False,
            return_keypoint_qc=False
    ):
        """
        Calculate specified joint angles for each frame.

        angles example:
            [["R-fBC", "R-fCT", "R-fFT"], ["R-fCT", "R-fFT", "R-fTT"]]
        """
        collected_angle_data = dict()
        qc_summaries = []
        keypoint_qc_summaries = []
        n_frames = int(trial_info.total_frames_number)

        for ag in angles:
            joint_name = ag[1]
            angle_definition = "|".join(ag)

            if apply_tracking_qc:
                # Keypoint-first QC: each point is cleaned independently before angle calculation.
                cleaned_xyz_by_keypoint = {}
                angle_keypoint_summaries = []
                for keypoint in ag:
                    if keypoint not in trial_info.trial_data:
                        # Missing keypoints fail the angle trace immediately but still produce diagnostics.
                        summary = {
                            "Keypoint": keypoint,
                            "QC_Passed": False,
                            "QC_Exclusion_Reason": "missing_keypoint",
                            "Valid_Frame_Fraction": 0.0,
                            "Invalid_Frame_Fraction": 1.0,
                            "Max_Invalid_Gap_Frames": n_frames,
                            "Interpolated_Frame_Count": 0,
                            "Max_Interp_Gap_Frames": tqc.interp_gap_frames_from_fps(max_interp_gap_s, trial_info.fps),
                        }
                        cleaned_xyz_by_keypoint[keypoint] = None
                    else:
                        # Run the single point-level QC function through the compatibility wrapper.
                        cleaned_xyz, _, summary, _ = self.apply_xyz_tracking_qc(
                            trial_info=trial_info,
                            keypoint=keypoint,
                            min_cameras=min_cameras,
                            max_interp_gap_s=max_interp_gap_s,
                            min_valid_fraction=min_valid_fraction,
                            error_max=error_max,
                            score_min=score_min,
                            start_frame=qc_start,
                            end_frame=qc_end,
                            require_start_end_valid=False,
                        )
                        cleaned_xyz_by_keypoint[keypoint] = cleaned_xyz if summary["QC_Passed"] else None
                    # Annotate keypoint diagnostics with the angle they contributed to.
                    summary = dict(summary)
                    summary["Joint"] = joint_name
                    summary["Angle_Definition"] = angle_definition
                    angle_keypoint_summaries.append(summary)
                    keypoint_qc_summaries.append(summary)

                # Collapse the three point summaries into one angle-level pass/fail row.
                qc_summary = self._aggregate_keypoint_qc_for_angle(
                    angle_keypoint_summaries,
                    joint_name=joint_name,
                    angle_definition=angle_definition,
                )
                qc_summaries.append(qc_summary)

                if qc_summary["QC_Passed"]:
                    # Passed traces are calculated from interpolated xyz, not from raw angle values.
                    angle_trace = self._angle_trace_from_xyz(
                        cleaned_xyz_by_keypoint[ag[0]],
                        cleaned_xyz_by_keypoint[ag[1]],
                        cleaned_xyz_by_keypoint[ag[2]],
                        n_frames=n_frames,
                    )
                    # EMA smoothing
                    if smooth_angle:
                        angle_trace = self.smooth_trace_ema(angle_trace, alpha=smooth_alpha)
                    collected_angle_data[joint_name] = angle_trace
                else:
                    # Failed angle traces remain frame-aligned but contain no usable angle values.
                    collected_angle_data[joint_name] = np.full(n_frames, np.nan, dtype=float)
                continue

            # Without tracking QC, preserve the raw frame-wise angle calculation behavior.
            angle_trace = np.full(n_frames, np.nan, dtype=float)
            for f in range(n_frames):
                angle_trace[f] = self.calculate_angle(
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
            collected_angle_data[joint_name] = angle_trace

        if return_qc:
            if return_keypoint_qc:
                # Optional third return exposes keypoint-level diagnostics without changing default callers.
                return (
                    collected_angle_data,
                    pd.DataFrame(qc_summaries),
                    pd.DataFrame(keypoint_qc_summaries),
                )
            return collected_angle_data, pd.DataFrame(qc_summaries)
        return collected_angle_data

    def Normalized_time(self, data, length=250):
        from scipy.interpolate import interp1d

        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, data, kind='linear')
        signal = f(x_new)

        return signal

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

