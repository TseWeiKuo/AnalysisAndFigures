"""Angle-trace collection and plotting workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tracking_qc as tqc
import trial_helpers as th

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
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
        require_score=False,
        smooth_angle=True,
        smooth_window_frames=5,
        smooth_polyorder=2,
        qc_start=0,
        qc_end=2.0
):
    """
    Plot selected CsChrimson ON angle traces in one panel.

    The first angle definition is plotted as a solid leg-angle trace, and
    the second angle definition is plotted as a dashed wing-angle trace.
    """
    if isinstance(groups, dict):
        group_items = list(groups.items())
    else:
        group_items = [(group.group_name, group) for group in groups]

    if angles is None:
        angles = [
            ["R-mCT", "R-mFT", "R-mTT"],
        ]
    if len(angles) < 2:
        raise ValueError("Provide at least two angle definitions: leg angle and wing angle.")

    if colors is None:
        colors = sns.color_palette("tab10", len(group_items))

    condition = str(condition).upper()
    frames = np.arange(int(start * 250), int(end * 250)) / 250
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    stat_rows = []
    qc_rows = []
    skipped_rows = []

    for group_idx, (group_label, group_info) in enumerate(group_items):
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying"])
        on_index, off_index = group_info.get_ON_OFF_index()
        index_to_iterate = on_index if condition == "ON" else off_index

        if len(index_to_iterate) == 0:
            print(f"No {condition} trials found for {group_info.group_name}")
            continue

        angle_result = self.analyzer.Calculate_angle_traces(
            group_info=group_info,
            index_to_iterate=index_to_iterate,
            angles=angles,
            start=start,
            end=end,
            chrimson=True,
            apply_tracking_qc=apply_tracking_qc,
            min_cameras=min_cameras,
            max_interp_gap_s=max_interp_gap_s,
            min_valid_fraction=min_valid_fraction,
            # Forward the current fixed-threshold QC rule instead of relying on
            # calculator defaults when optogenetic angle QC is enabled.
            error_max=error_max,
            score_min=score_min,
            require_score=require_score,
            smooth_angle=smooth_angle,
            smooth_window_frames=smooth_window_frames,
            smooth_polyorder=smooth_polyorder,
            qc_start=qc_start,
            qc_end=qc_end,
            return_qc=apply_tracking_qc
        )
        if apply_tracking_qc:
            group_data, group_qc_df, group_skipped_df = angle_result
            if not group_qc_df.empty:
                group_qc_df = group_qc_df.copy()
                group_qc_df["Plot_Label"] = group_label
                group_qc_df["Condition"] = condition
                qc_rows.extend(group_qc_df.to_dict("records"))
            if not group_skipped_df.empty:
                group_skipped_df = group_skipped_df.copy()
                group_skipped_df["Plot_Label"] = group_label
                group_skipped_df["Condition"] = condition
                skipped_rows.extend(group_skipped_df.to_dict("records"))
        else:
            group_data = angle_result

        color = colors[group_idx % len(colors)]
        for angle_idx, angle_def in enumerate(angles[:2]):
            ax = axes[angle_idx]
            joint_name = angle_def[1]
            traces = group_data.get(joint_name, [])
            total_trials = len(index_to_iterate)
            valid_trials = len(traces)
            if len(traces) == 0:
                print(f"No valid {condition} trace found for {group_info.group_name}-{joint_name}")
                continue

            traces = np.asarray(traces, dtype=float)
            mean_trace = np.nanmean(traces, axis=0)
            valid_n = np.sum(np.isfinite(traces), axis=0)
            sem_trace = np.full_like(mean_trace, np.nan, dtype=float)
            valid_sem = valid_n > 1
            sem_trace[valid_sem] = (
                    np.nanstd(traces[:, valid_sem], axis=0, ddof=1)
                    / np.sqrt(valid_n[valid_sem])
            )

            line_style = "solid" if angle_idx == 0 else "dashed"
            trace_label = "leg angle" if angle_idx == 0 else "wing angle"
            ax.plot(
                frames[:len(mean_trace)],
                mean_trace,
                color=color,
                linestyle=line_style,
                linewidth=2.2,
                label=f"{group_label} {trace_label} ({valid_trials}/{total_trials})",
            )
            if show_sem:
                ax.fill_between(
                    frames[:len(mean_trace)],
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    color=color,
                    alpha=0.12,
                    linewidth=0
                )

            stat_rows.append({
                "Group": group_info.group_name,
                "Plot_Label": group_label,
                "Condition": condition,
                "Joint": joint_name,
                "Trace_Type": trace_label,
                "n_trials": valid_trials,
                "total_trials": total_trials,
                "valid_total_label": f"{valid_trials}/{total_trials}",
                "start": start,
                "end": end,
                "qc_start": qc_start,
                "qc_end": qc_end,
                "Apply_Tracking_QC": bool(apply_tracking_qc),
            })

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", linestyle="solid", linewidth=2.2, label="solid = leg angle"),
        Line2D([0], [0], color="black", linestyle="dashed", linewidth=2.2, label="dashed = wing angle"),
    ]
    for axis_idx, ax in enumerate(axes):
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(2, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlim(start, end)
        ax.set_ylabel("Leg angle (deg)" if axis_idx == 0 else "Wing angle (deg)")
        self.formatting(
            ax,
            xticks=[start, 0, 2, end],
            xlabel="Time from light ON (s)" if axis_idx == 1 else None
        )
    axes[0].set_title(f"Selected CsChrimson {condition} leg and wing angle traces")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + style_handles, labels=labels + [handle.get_label() for handle in style_handles],
                  frameon=True, fontsize=7, loc="upper right")
    sns.despine(trim=True)
    plt.tight_layout()

    stat_df = pd.DataFrame(stat_rows)
    qc_df = pd.DataFrame(qc_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        stat_df.to_csv(f"{file_name}_summary.csv", index=False)
        if apply_tracking_qc:
            qc_df.to_csv(f"{file_name}_angle_qc_summary.csv", index=False)
            skipped_df.to_csv(f"{file_name}_angle_qc_skipped_trials.csv", index=False)
    plt.close(fig)
    if apply_tracking_qc:
        return fig, axes, stat_df, qc_df, skipped_df
    return fig, axes, stat_df

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
        save_csv=True
):
    """
    Plot baseline-corrected WT contact-leg CT/FT angle traces aligned to MOC.

    The 2x2 layout is fixed for Figure 2C:
    - columns are contact materials/conditions, e.g. TiTa and CxTr
    - row 1 is CT angle change, row 2 is FT angle change
    - T1/T2/T3 traces use their corresponding right-side leg.

    Each trial is extracted using its native fps, aligned with MOC as time
    zero, then interpolated onto a 250 Hz target grid. Each resampled trial
    is baseline-corrected by subtracting its own mean angle from start_s up
    to, but not including, MOC before the group mean and SEM are calculated.
    """
    if contact_leg_map is None:
        contact_leg_map = {
            "T1": "R-f",
            "T2": "R-m",
            "T3": "R-h",
        }

    if contact_colors is None:
        contact_colors = {
            "T1": "tab:blue",
            "T2": "tab:red",
            "T3": "tab:green",
        }

    column_labels = list(groups_by_column.keys())
    row_defs = [
        ("CT", "CT angle change (deg)"),
        ("FT", "FT angle change (deg)"),
    ]

    target_n = int(round((end_s - start_s) * target_fps)) + 1
    target_time = np.linspace(start_s, end_s, target_n)
    baseline_mask = (target_time >= start_s) & (target_time < 0)
    if not np.any(baseline_mask):
        raise ValueError("The plotting window must include at least one sample before MOC.")

    def angle_definition(leg, joint_type):
        if joint_type == "CT":
            return [f"{leg}BC", f"{leg}CT", f"{leg}FT"]
        if joint_type == "FT":
            return [f"{leg}CT", f"{leg}FT", f"{leg}TT"]
        raise ValueError(f"Unsupported joint_type: {joint_type}")

    def collect_resampled_traces(group_info, leg, joint_type, column_label, contact_group):
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(list(trial_types))

        angle_def = angle_definition(leg, joint_type)
        joint_name = angle_def[1]
        traces = []
        qc_rows = []
        skipped_rows = []

        for index in group_info.get_targeted_trials(list(trial_types)):
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data:
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Joint_Type": joint_type,
                    "Reason": "missing kinematic data",
                })
                continue

            trial_info = group_info.fly_kinematic_data[key]
            moc = trial_info.moc
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Joint_Type": joint_type,
                    "Reason": "missing MOC or fps",
                })
                continue

            start_frame = int(round(moc + start_s * fps))
            end_frame = int(round(moc + end_s * fps))

            # Skip incomplete windows so every averaged trace represents
            # the same MOC-centered interval.
            if start_frame < 0 or end_frame >= trial_info.total_frames_number:
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Joint_Type": joint_type,
                    "Reason": "incomplete MOC-centered window",
                })
                continue
            if any(point not in trial_info.trial_data for point in angle_def):
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Joint_Type": joint_type,
                    "Reason": "missing angle keypoint",
                })
                continue

            angle_result = self.calculator.Calculate_joint_angle(
                trial_info,
                [angle_def],
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
                qc_start=start_frame,
                qc_end=end_frame,
                return_qc=apply_tracking_qc
            )
            if apply_tracking_qc:
                angle_data, angle_qc_df = angle_result
                if not angle_qc_df.empty:
                    qc_record = angle_qc_df.iloc[0].to_dict()
                    qc_record.update({
                        "Column": column_label,
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Joint_Type": joint_type,
                    })
                    qc_rows.append(qc_record)
                    if not bool(qc_record.get("QC_Passed", True)):
                        skipped_rows.append({
                            "Column": column_label,
                            "Contact_Group": contact_group,
                            "Group_Name": group_info.group_name,
                            "Index": str(index),
                            "Fly#": index[0],
                            "Trial#": index[1],
                            "Joint_Type": joint_type,
                            "Reason": "failed angle tracking QC",
                            **qc_record,
                        })
                        continue
            else:
                angle_data = angle_result
            angle_trace = angle_data[joint_name]
            source_frames = np.arange(start_frame, end_frame + 1)
            source_time = (source_frames - moc) / fps
            source_trace = np.asarray(angle_trace[start_frame:end_frame + 1], dtype=float)

            valid = np.isfinite(source_time) & np.isfinite(source_trace)
            window_valid_fraction = float(np.mean(valid)) if len(valid) else np.nan
            max_invalid_gap = max(self.calculator.invalid_gap_lengths(valid), default=0)
            # Resolve the time-based interpolation threshold in this trial's FPS
            # before applying the final plotted-window QC check.
            max_interp_gap_frames = tqc.interp_gap_frames_from_fps(max_interp_gap_s, fps)
            if apply_tracking_qc and (
                    window_valid_fraction < min_valid_fraction
                    or max_invalid_gap > max_interp_gap_frames
            ):
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Joint_Type": joint_type,
                    "Reason": "failed angle tracking QC",
                    "Valid_Frame_Fraction": window_valid_fraction,
                    "Max_Invalid_Gap_Frames": max_invalid_gap,
                    "Min_Valid_Fraction": min_valid_fraction,
                    "Max_Interp_Gap_s": max_interp_gap_s,
                    "Max_Interp_Gap_Frames": max_interp_gap_frames,
                })
                continue
            if np.sum(valid) < 2:
                skipped_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Joint_Type": joint_type,
                    "Reason": "fewer than two valid angle samples",
                })
                continue

            resampled = np.interp(
                target_time,
                source_time[valid],
                source_trace[valid],
                left=np.nan,
                right=np.nan
            )
            baseline_values = resampled[baseline_mask]
            baseline_values = baseline_values[np.isfinite(baseline_values)]
            if len(baseline_values) == 0:
                continue

            baseline_mean = float(np.nanmean(baseline_values))
            traces.append(resampled - baseline_mean)

        return np.asarray(traces, dtype=float), pd.DataFrame(qc_rows), pd.DataFrame(skipped_rows)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(column_labels),
        figsize=(5.2 * len(column_labels), 7.2),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    summary_rows = []
    qc_summary_tables = []
    skipped_tables = []
    plotted_angle_values = []
    for col, column_label in enumerate(column_labels):
        contact_groups = groups_by_column[column_label]

        for row, (joint_type, ylabel) in enumerate(row_defs):
            ax = axes[row, col]
            panel_count_lines = []

            for contact_group, group_info in contact_groups.items():
                leg = contact_leg_map[contact_group]
                traces, angle_qc_df, skipped_df = collect_resampled_traces(
                    group_info,
                    leg,
                    joint_type,
                    column_label,
                    contact_group
                )
                if not angle_qc_df.empty:
                    qc_summary_tables.append(angle_qc_df)
                if not skipped_df.empty:
                    skipped_tables.append(skipped_df)
                n_trials = int(traces.shape[0]) if traces.ndim == 2 else 0

                summary_rows.append({
                    "Column": column_label,
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Leg": leg,
                    "Joint_Type": joint_type,
                    "n_trials": n_trials,
                    "start_s": start_s,
                    "end_s": end_s,
                    "target_fps": target_fps,
                    "baseline_start_s": start_s,
                    "baseline_end_s": 0.0,
                    "baseline_end_inclusive": False,
                    "trace_value": "angle_change_from_pre_MOC_mean_deg",
                    "apply_tracking_qc": apply_tracking_qc,
                    "min_cameras": min_cameras if apply_tracking_qc else np.nan,
                    "max_interp_gap_s": max_interp_gap_s if apply_tracking_qc else np.nan,
                    "min_valid_fraction": min_valid_fraction if apply_tracking_qc else np.nan,
                    "smooth_angle": smooth_angle,
                    "smooth_method": smooth_method if smooth_angle else "",
                    "smooth_window_frames": smooth_window_frames if smooth_angle else np.nan,
                    "smooth_alpha": smooth_alpha if smooth_angle else np.nan,
                })

                if n_trials == 0:
                    continue

                panel_count_lines.append((
                    f"{contact_group} {leg}: n={n_trials}",
                    contact_colors.get(contact_group, "black")
                ))
                mean_trace = np.nanmean(traces, axis=0)
                valid_n = np.sum(np.isfinite(traces), axis=0)
                sem_trace = np.full_like(mean_trace, np.nan, dtype=float)
                valid_sem = valid_n > 1
                sem_trace[valid_sem] = (
                        np.nanstd(traces[:, valid_sem], axis=0, ddof=1)
                        / np.sqrt(valid_n[valid_sem])
                )
                plotted_angle_values.append(mean_trace)
                if show_sem:
                    plotted_angle_values.extend([mean_trace - sem_trace, mean_trace + sem_trace])

                color = contact_colors.get(contact_group, "black")
                ax.plot(
                    target_time,
                    mean_trace,
                    color=color,
                    linewidth=2.4,
                    label=f"{contact_group} ({leg}, n={n_trials})"
                )
                if show_sem:
                    ax.fill_between(
                        target_time,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=color,
                        alpha=0.18,
                        linewidth=0
                    )

            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(0, color="0.75", linestyle="-", linewidth=0.8)
            ax.set_title(f"{column_label}: {joint_type}")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xlabel("Time from MOC (s)" if row == len(row_defs) - 1 else "")
            for line_i, (count_text, count_color) in enumerate(panel_count_lines):
                ax.text(
                    0.03,
                    0.95 - line_i * 0.08,
                    count_text,
                    transform=ax.transAxes,
                    color=count_color,
                    fontsize=8,
                    ha="left",
                    va="top",
                )
            self.formatting(
                ax,
                xticks=[start_s, 0, end_s],
                xlabel=ax.get_xlabel(),
                ylabel=ax.get_ylabel()
            )
            if row == 0 and col == len(column_labels) - 1:
                ax.legend(frameon=False, fontsize=8)
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    if plotted_angle_values:
        finite_arrays = [
            np.asarray(values, dtype=float)[np.isfinite(values)]
            for values in plotted_angle_values
            if np.any(np.isfinite(values))
        ]
        if finite_arrays:
            finite_values = np.concatenate(finite_arrays)
            y_min = float(np.nanmin(finite_values))
            y_max = float(np.nanmax(finite_values))
            y_pad = max((y_max - y_min) * 0.08, 2.0)
            for ax in axes.flatten():
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

    sns.despine(trim=True)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")

    summary_df = pd.DataFrame(summary_rows)
    qc_summary_df = pd.concat(qc_summary_tables, ignore_index=True) if qc_summary_tables else pd.DataFrame()
    skipped_df = pd.concat(skipped_tables, ignore_index=True) if skipped_tables else pd.DataFrame()
    if save_csv and file_name is not None:
        summary_df.to_csv(f"{file_name}_summary.csv", index=False)
        if apply_tracking_qc:
            qc_summary_df.to_csv(f"{file_name}_angle_qc_summary.csv", index=False)
            skipped_df.to_csv(f"{file_name}_angle_qc_skipped_trials.csv", index=False)

    plt.close()
    return fig, axes, summary_df

def flight_postural_change(
        self,
        group_info,
        angle_def=("R-mCT", "R-mFT", "R-mTT"),
        trial_types=("Landing", "Flying"),
        pre_moc_window_s=0.5,
        max_trial_num=20,
        colors=None,
        show_sem=True,
        file_name="flight_postural_change",
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
        save_csv=True
):
    """
    Plot trial-by-trial pre-MOC flight posture from one joint angle.

    Each fly contributes at most one value per trial number: the mean angle
    from pre_moc_window_s before MOC up to MOC. The plotted line is the mean
    across flies for each trial number, and the shade is fly-level SEM.
    """
    # Normalize the input so callers can pass one Group, a list of Groups, or a
    # label-to-Group mapping while sharing the same aggregation code.
    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    # Use a stable categorical palette unless the notebook supplies group
    # colors keyed by label or group name.
    if colors is None:
        colors = sns.color_palette("tab10", len(group_items))

    angle_def = tuple(angle_def)
    joint_name = angle_def[1]
    value_rows = []
    summary_rows = []
    qc_rows = []
    skipped_rows = []

    for group_idx, (group_label, current_group) in enumerate(group_items):
        # Prepare metadata and kinematic traces in the same lazy style as the
        # other WT plotting functions.
        if len(current_group.trial_metadata) == 0:
            current_group.initialize_manual_data()
        current_group.filter_nan_fly()
        current_group.read_kinematic_data(list(trial_types))

        for fly_num in current_group.good_fly_index:
            for trial_num in range(1, int(max_trial_num) + 1):
                key = current_group._trial_key(fly_num, trial_num)
                if key not in current_group.trial_metadata:
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "missing metadata",
                    })
                    continue

                meta = current_group.trial_metadata[key]
                if meta.get("TrialType") not in trial_types:
                    continue
                if key not in current_group.fly_kinematic_data:
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "missing kinematic data",
                    })
                    continue

                trial_info = current_group.fly_kinematic_data[key]
                moc = trial_info.moc
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(fps):
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "missing MOC or fps",
                    })
                    continue

                # Define the exact pre-MOC frame window used for both QC and
                # the final posture value. MOC itself is excluded.
                moc = int(moc)
                fps = float(fps)
                start_frame = int(round(moc - pre_moc_window_s * fps))
                end_frame = moc - 1
                if start_frame < 0 or end_frame <= start_frame or end_frame >= trial_info.total_frames_number:
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "incomplete pre-MOC window",
                        "MOC_Frame": moc,
                        "Window_Start_Frame": start_frame,
                        "Window_End_Frame": end_frame,
                    })
                    continue
                if any(point not in trial_info.trial_data for point in angle_def):
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "missing angle keypoint",
                        "Angle_Definition": "|".join(angle_def),
                    })
                    continue

                # Calculate the angle trace and, when requested, apply the
                # current tracking QC only to the pre-MOC posture window.
                angle_result = self.calculator.Calculate_joint_angle(
                    trial_info,
                    [list(angle_def)],
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
                    qc_start=start_frame,
                    qc_end=end_frame,
                    return_qc=apply_tracking_qc
                )
                if apply_tracking_qc:
                    angle_data, angle_qc_df = angle_result
                    if not angle_qc_df.empty:
                        qc_record = angle_qc_df.iloc[0].to_dict()
                        qc_record.update({
                            "Group_Label": group_label,
                            "Group_Name": current_group.group_name,
                            "Fly#": fly_num,
                            "Trial#": trial_num,
                            "Window_Start_Frame": start_frame,
                            "Window_End_Frame": end_frame,
                        })
                        qc_rows.append(qc_record)
                        if not bool(qc_record.get("QC_Passed", True)):
                            skipped_rows.append({
                                "Group_Label": group_label,
                                "Group_Name": current_group.group_name,
                                "Fly#": fly_num,
                                "Trial#": trial_num,
                                "Reason": "failed posture tracking QC",
                                **qc_record,
                            })
                            continue
                else:
                    angle_data = angle_result

                # Average the finite angle samples inside the pre-MOC window to
                # produce one posture value for this fly/trial number.
                window_trace = np.asarray(angle_data[joint_name][start_frame:end_frame + 1], dtype=float)
                finite_trace = window_trace[np.isfinite(window_trace)]
                if len(finite_trace) == 0:
                    skipped_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Fly#": fly_num,
                        "Trial#": trial_num,
                        "Reason": "no finite pre-MOC angle samples",
                        "Window_Start_Frame": start_frame,
                        "Window_End_Frame": end_frame,
                    })
                    continue

                value_rows.append({
                    "Group_Label": group_label,
                    "Group_Name": current_group.group_name,
                    "Fly#": fly_num,
                    "Trial#": trial_num,
                    "TrialType": meta.get("TrialType"),
                    "Joint": joint_name,
                    "Angle_Definition": "|".join(angle_def),
                    "MOC_Frame": moc,
                    "FPS": fps,
                    "Window_Start_Frame": start_frame,
                    "Window_End_Frame": end_frame,
                    "Pre_MOC_Window_s": pre_moc_window_s,
                    "Mean_Pre_MOC_Angle_deg": float(np.nanmean(finite_trace)),
                    "Finite_Frame_Count": int(len(finite_trace)),
                    "Apply_Tracking_QC": bool(apply_tracking_qc),
                })

    value_df = pd.DataFrame(value_rows)
    if value_df.empty:
        raise ValueError("No valid pre-MOC posture values were available for plotting.")

    # Collapse fly-trial values by trial number. The SEM is across flies within
    # each trial number, not across pooled frames or pooled trials.
    for (group_label, group_name, trial_num), sub in value_df.groupby(["Group_Label", "Group_Name", "Trial#"]):
        values = sub["Mean_Pre_MOC_Angle_deg"].astype(float).dropna().to_numpy()
        n_flies = len(values)
        sem = np.nan
        if n_flies > 1:
            sem = float(np.nanstd(values, ddof=1) / np.sqrt(n_flies))
        summary_rows.append({
            "Group_Label": group_label,
            "Group_Name": group_name,
            "Trial#": trial_num,
            "Joint": joint_name,
            "Mean_Pre_MOC_Angle_deg": float(np.nanmean(values)) if n_flies else np.nan,
            "SEM_Pre_MOC_Angle_deg": sem,
            "SD_Pre_MOC_Angle_deg": float(np.nanstd(values, ddof=1)) if n_flies > 1 else np.nan,
            "n_flies": n_flies,
            "Pre_MOC_Window_s": pre_moc_window_s,
            "Apply_Tracking_QC": bool(apply_tracking_qc),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["Group_Label", "Trial#"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted_values = []
    for group_idx, (group_label, current_group) in enumerate(group_items):
        sub = summary_df[summary_df["Group_Label"] == group_label].sort_values("Trial#")
        if sub.empty:
            continue

        # Resolve group color by label, by internal group name, or by plotting
        # order so notebook-level color dictionaries work naturally.
        if isinstance(colors, dict):
            color = colors.get(group_label, colors.get(current_group.group_name, "black"))
        else:
            color = colors[group_idx % len(colors)]

        x = sub["Trial#"].to_numpy(dtype=float)
        y = sub["Mean_Pre_MOC_Angle_deg"].to_numpy(dtype=float)
        sem = sub["SEM_Pre_MOC_Angle_deg"].to_numpy(dtype=float)
        plotted_values.append(y)
        if show_sem:
            plotted_values.extend([y - sem, y + sem])

        ax.plot(x, y, color=color, linewidth=2.4, marker="o", markersize=4, label=str(group_label))
        if show_sem:
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)

    # Format the trial-number axis with integer ticks and scale the y-axis to
    # include both the mean trace and SEM ribbon.
    ax.set_xlim(0.5, max_trial_num + 0.5)
    ax.set_xticks(np.arange(1, int(max_trial_num) + 1))
    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"{joint_name} pre-MOC mean angle (deg)")
    ax.set_title(f"{joint_name} flight posture before MOC")
    if plotted_values:
        finite_arrays = [
            np.asarray(values, dtype=float)[np.isfinite(values)]
            for values in plotted_values
            if np.any(np.isfinite(values))
        ]
        if finite_arrays:
            finite_values = np.concatenate(finite_arrays)
            y_min = float(np.nanmin(finite_values))
            y_max = float(np.nanmax(finite_values))
            y_pad = max((y_max - y_min) * 0.08, 2.0)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.legend(frameon=False, fontsize=8)
    self.formatting(ax)
    sns.despine(trim=True)
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    qc_df = pd.DataFrame(qc_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if save_csv and file_name is not None:
        value_df.to_csv(f"{file_name}_fly_trial_values.csv", index=False)
        summary_df.to_csv(f"{file_name}_trial_summary.csv", index=False)
        if apply_tracking_qc:
            qc_df.to_csv(f"{file_name}_tracking_qc_summary.csv", index=False)
            skipped_df.to_csv(f"{file_name}_tracking_qc_skipped_trials.csv", index=False)

    return fig, ax, value_df, summary_df, qc_df, skipped_df

