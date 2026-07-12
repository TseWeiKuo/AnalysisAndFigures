"""Optogenetic landing and latency plotting workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time


def _significance_label(p_value, missing_label=""):
    if pd.isna(p_value):
        return missing_label
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


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
    if wing_angle_def is None:
        wing_angle_def = [["L-wing", "L-wing-hinge", "R-wing"]]

    angle_result = self.calculator.Calculate_joint_angle(
        trial_info,
        wing_angle_def,
        apply_tracking_qc=apply_tracking_qc,
        tracking_error_thresholds=tracking_error_thresholds,
        min_cameras=min_cameras,
        max_interp_gap_frames=max_interp_gap_frames,
        min_valid_fraction=min_valid_fraction,
        smooth_angle=smooth_angle,
        smooth_window_frames=smooth_window_frames,
        smooth_polyorder=smooth_polyorder,
        return_qc=apply_tracking_qc
    )
    if apply_tracking_qc:
        angle_data, qc_df = angle_result
    else:
        angle_data = angle_result
        qc_df = pd.DataFrame()

    wing_trace = np.asarray(angle_data["L-wing-hinge"], dtype=float)
    start = max(int(window_start), 0)
    stop = min(int(window_stop), len(wing_trace))
    wing_window = wing_trace[start:stop]
    finite_window = np.isfinite(wing_window)
    window_valid_fraction = float(np.mean(finite_window)) if len(finite_window) else np.nan
    window_gaps = self.calculator.invalid_gap_lengths(finite_window) if len(finite_window) else []
    max_window_gap = int(max(window_gaps)) if window_gaps else 0

    qc_passed = True
    exclusion_reason = ""
    if len(wing_window) == 0:
        qc_passed = False
        exclusion_reason = "empty wing detection window"
    elif apply_tracking_qc:
        if pd.isna(window_valid_fraction) or window_valid_fraction < min_valid_fraction:
            qc_passed = False
            exclusion_reason = "valid_fraction_below_threshold"
        elif max_window_gap > max_interp_gap_frames:
            qc_passed = False
            exclusion_reason = "long_invalid_gap"
        elif not np.all(finite_window):
            qc_passed = False
            exclusion_reason = "unfilled_invalid_frames_in_detection_window"

    if not qc_passed:
        return -1, {
            "Wing_QC_Passed": False,
            "Wing_QC_Exclusion_Reason": exclusion_reason,
            "Wing_Window_Valid_Fraction": window_valid_fraction,
            "Wing_Window_Max_Invalid_Gap_Frames": max_window_gap,
            "Wing_Window_Start_Frame": start,
            "Wing_Window_Stop_Frame": stop - 1,
            "Apply_Tracking_QC": bool(apply_tracking_qc),
        }, qc_df

    mol = self.detector.detect_landing(wing_window)
    return mol, {
        "Wing_QC_Passed": True,
        "Wing_QC_Exclusion_Reason": "",
        "Wing_Window_Valid_Fraction": window_valid_fraction,
        "Wing_Window_Max_Invalid_Gap_Frames": max_window_gap,
        "Wing_Window_Start_Frame": start,
        "Wing_Window_Stop_Frame": stop - 1,
        "Apply_Tracking_QC": bool(apply_tracking_qc),
    }, qc_df

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
    """
    Plot paired ON/OFF landing probability for Chrimson data.

    This version is integrated into the new framework:
    - prepares metadata if needed
    - filters opto flies
    - loads kinematic data if needed
    - uses get_ON_OFF_index()
    """
    from scipy.stats import ttest_rel

    def chrimson_intensity_color(group_name):
        normalized = str(group_name).replace("_", "").replace(" ", "").lower()
        if normalized.startswith("adxchr"):
            return "#8c8c8c"
        if "12mw" in normalized:
            return "#7F0000"
        if "4mw" in normalized:
            return "#D73027"
        if "400uw" in normalized:
            return "#F4A3A3"
        return color

    mean_color = chrimson_intensity_color(group_info.group_name)
    angs = [["L-wing", "L-wing-hinge", "R-wing"]]
    # ------------------------------------------------------------
    # Prepare group for kinematic-dependent opto analysis
    # ------------------------------------------------------------
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_Chr_manual_data()

    group_info.filter_opto_data()
    group_info.read_kinematic_data(["Landing", "Flying", "NF", "NA"])

    Fly_ON_LP = []
    Fly_OFF_LP = []
    Fly_ON_LL = []
    Fly_ON_event = []
    Fly_ON_Idx = []
    wing_qc_rows = []
    wing_qc_skipped_rows = []

    ON_index, OFF_index = group_info.get_ON_OFF_index()

    ON_index = sorted(ON_index, key=lambda x: (x[0], x[1]))
    OFF_index = sorted(OFF_index, key=lambda x: (x[0], x[1]))

    # ------------------------------------------------------------
    # Calculate ON landing probability
    # ------------------------------------------------------------
    if len(ON_index) > 0:
        flying_num = 0
        landing_num = 0

        for i, index in enumerate(ON_index):
            trial_info = group_info.fly_kinematic_data[f"F{index[0]}T{index[1]}"]
            if trial_info.LL == -1:
                flying_num += 1
                Fly_ON_LL.append(threshold)
                Fly_ON_event.append(0)
                Fly_ON_Idx.append(index[0])

            elif trial_info.LL == 1:
                MOL, wing_qc_summary, wing_qc_df = _detect_chrimson_wing_mol(
                    self,
                    trial_info,
                    wing_angle_def=angs,
                    apply_tracking_qc=apply_tracking_qc,
                    tracking_error_thresholds=tracking_error_thresholds,
                    min_cameras=min_cameras,
                    max_interp_gap_frames=max_interp_gap_frames,
                    min_valid_fraction=min_valid_fraction,
                    smooth_angle=smooth_angle,
                    smooth_window_frames=smooth_window_frames,
                    smooth_polyorder=smooth_polyorder
                )
                wing_qc_summary.update({
                    "Group": group_info.group_name,
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Condition": "ON",
                })
                wing_qc_rows.append(wing_qc_summary)
                if not wing_qc_df.empty:
                    wing_qc_df = wing_qc_df.copy()
                    wing_qc_df["Group"] = group_info.group_name
                    wing_qc_df["Fly#"] = index[0]
                    wing_qc_df["Trial#"] = index[1]
                    wing_qc_df["Condition"] = "ON"
                    wing_qc_rows.extend(wing_qc_df.to_dict("records"))
                if apply_tracking_qc and not wing_qc_summary["Wing_QC_Passed"]:
                    wing_qc_skipped_rows.append(wing_qc_summary.copy())
                elif MOL == -1 or (MOL / 250) > threshold:
                    flying_num += 1
                    Fly_ON_LL.append(threshold)
                    Fly_ON_event.append(0)
                    Fly_ON_Idx.append(index[0])
                else:
                    landing_num += 1
                    Fly_ON_LL.append(MOL / 250)
                    Fly_ON_event.append(1)
                    Fly_ON_Idx.append(index[0])

            else:
                pass

            is_last_trial = (i == len(ON_index) - 1)
            next_fly_starts = (not is_last_trial and ON_index[i + 1][0] != index[0])

            if is_last_trial or next_fly_starts:
                total = landing_num + flying_num
                lp = landing_num / total if total > 0 else np.nan
                Fly_ON_LP.append((index[0], lp))

                flying_num = 0
                landing_num = 0

    # ------------------------------------------------------------
    # Calculate OFF landing probability
    # ------------------------------------------------------------
    if len(OFF_index) > 0:
        flying_num = 0
        landing_num = 0

        for i, index in enumerate(OFF_index):
            trial_info = group_info.fly_kinematic_data[f"F{index[0]}T{index[1]}"]

            if trial_info.LL == -1:
                flying_num += 1
            else:
                landing_num += 1

            is_last_trial = (i == len(OFF_index) - 1)
            next_fly_starts = (not is_last_trial and OFF_index[i + 1][0] != index[0])

            if is_last_trial or next_fly_starts:
                total = landing_num + flying_num
                lp = landing_num / total if total > 0 else np.nan
                Fly_OFF_LP.append((index[0], lp))

                flying_num = 0
                landing_num = 0
    # ------------------------------------------------------------
    # Build dataframe for paired ON/OFF plotting
    # ------------------------------------------------------------
    on_df = pd.DataFrame(Fly_ON_LP, columns=["Fly#", "LandingProb"])
    on_df["Group_Name"] = "ON"

    off_df = pd.DataFrame(Fly_OFF_LP, columns=["Fly#", "LandingProb"])
    off_df["Group_Name"] = "OFF"

    combined_df = pd.concat([on_df, off_df], ignore_index=True)

    fly_counts = combined_df["Fly#"].value_counts()
    paired_flies = fly_counts[fly_counts == 2].index
    combined_df = combined_df[combined_df["Fly#"].isin(paired_flies)].copy()

    combined_df["Group_Name"] = pd.Categorical(
        combined_df["Group_Name"],
        categories=["OFF", "ON"],
        ordered=True
    )
    combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

    # ------------------------------------------------------------
    # Paired sign-flip test
    # ------------------------------------------------------------
    paired_df = combined_df.pivot(index="Fly#", columns="Group_Name", values="LandingProb")
    paired_df = paired_df.dropna(subset=["OFF", "ON"]).copy()

    if len(paired_df) >= 2:
        paired_df["Diff_ON_minus_OFF"] = paired_df["ON"] - paired_df["OFF"]
        observed_diff, p_val = self.calculator.paired_signflip_permutation_test(
            paired_df["OFF"].values,
            paired_df["ON"].values,
            n_perm=20000,
            rng=np.random.default_rng(0)
        )

        stat_df = pd.DataFrame([{
            "Group": group_info.group_name,
            "Test": "paired sign-flip permutation",
            "Metric": "Landing probability",
            "n_paired_flies": len(paired_df),
            "mean_OFF": paired_df["OFF"].mean() if "OFF" in paired_df else np.nan,
            "mean_ON": paired_df["ON"].mean() if "ON" in paired_df else np.nan,
            "mean_diff_ON_minus_OFF": observed_diff,
            "p_value": p_val,
            "n_perm": 20000,
            "threshold": threshold
        }])

        stat_df.to_csv(
            f"{group_info.group_name}-chr-LP-signflip-stat.csv",
            index=False
        )
    else:
        paired_df["Diff_ON_minus_OFF"] = np.nan
        observed_diff, p_val = np.nan, np.nan

    paired_df.to_csv(f"{group_info.group_name}-paired_values.csv")
    if apply_tracking_qc:
        pd.DataFrame(wing_qc_rows).to_csv(f"{group_info.group_name}-chr-wing-qc-summary.csv", index=False)
        pd.DataFrame(wing_qc_skipped_rows).to_csv(
            f"{group_info.group_name}-chr-wing-qc-skipped-trials.csv",
            index=False
        )

    # ------------------------------------------------------------
    # Plot paired landing probabilities
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 8))

    ax = sns.pointplot(
        data=combined_df,
        x="Group_Name",
        y="LandingProb",
        errorbar=None,
        color=color,
        linestyles=" ",
        markers="o"
    )

    for fly_id, group in combined_df.groupby("Fly#"):
        plt.plot(
            group["Group_Name"],
            group["LandingProb"],
            marker="o",
            markersize=12,
            color="lightgrey",
            linewidth=3,
            zorder=1
        )

    mean_df = combined_df.groupby("Group_Name", as_index=False)["LandingProb"].mean()
    plt.plot(
        mean_df["Group_Name"],
        mean_df["LandingProb"],
        color=mean_color,
        marker="o",
        markersize=12,
        linewidth=3,
        label="Mean",
        zorder=9
    )

    # ------------------------------------------------------------
    # Add significance stars
    # ------------------------------------------------------------
    y_max = combined_df["LandingProb"].max()
    bracket_y = min(1.05, y_max + 0.10)
    text_y = min(1.09, bracket_y + 0.03)
    h = 0.02

    ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + h, bracket_y + h, bracket_y], lw=2.5, c="black")

    if p_val < 1e-4:
        sig = "****"
    elif p_val < 1e-3:
        sig = "***"
    elif p_val < 1e-2:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "ns"

    ax.text(0.5, text_y, sig, ha="center", va="bottom", fontsize=14)

    plt.title("Change in Landing Probability Across Light Conditions")
    plt.xlabel(group_info.group_name, fontsize=20)
    plt.ylabel("Landing Probability", fontsize=20)

    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    plt.tick_params(axis="y", labelsize=18)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(width=3, length=8)

    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.5, 1.5)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{group_info.group_name}-chr-LP.pdf")
    # plt.show()
    plt.close()

    Fly_ON_LL_data = pd.DataFrame({
        "Group": [group_info.group_name] * len(Fly_ON_LL),
        "Latency": Fly_ON_LL,
        "Event": Fly_ON_event,
        "Fly#": Fly_ON_Idx,
        "Apply_Tracking_QC": bool(apply_tracking_qc),
    })

    return Fly_ON_LL_data

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
    """
    Plot fly-wise CsChrimson landing-probability change.

    Each point is one paired fly: Delta LP = LP_ON - LP_OFF. ON landing
    probability uses the same wing-folding/MOL validation as
    plot_chrimson_LP; OFF landing probability uses manual landing/flying
    labels from the same paired fly. Rows are arranged by genotype, and
    point colors encode light intensity.
    """
    if isinstance(groups, dict):
        group_items = list(groups.items())
    else:
        group_items = [(group.group_name, group) for group in groups]

    if intensity_colors is None:
        intensity_colors = {
            "low": "#F4A3A3",
            "med": "#D73027",
            "high": "#7F0000",
        }

    display_order = [
        "AD-low",
        "AN-low",
        "AN-med",
        "AN-high",
        "IAV-low",
        "IAV-med",
        "IAV-high",
        "HP2-low",
        "HP2-med",
        "HP2-high",
        "ALLCS-low",
        "BiCS-med",
        "BiCS-high",
        "BiCS-Halt-med",
        "BiCS-Halt-high",
        "BiCS-Halt-WG-high",
        "CS0048-med",
        "CS0048-high",
        "CS0021-high",
        "TaCS-low",
        "TaCS-med",
        "TaCS-high",
        "TaBri-med",
        "TaBri-high",
    ]
    display_map = {
        "ad x chr 400": ("AD-low", "AD", "low"),
        "adxchr-400uw": ("AD-low", "AD", "low"),
        "anxchr-400uw": ("AN-low", "AN", "low"),
        "anxchr-4mw": ("AN-med", "AN", "med"),
        "anxchr-12mw": ("AN-high", "AN", "high"),
        "iavxchr-400uw": ("IAV-low", "IAV", "low"),
        "iavxchr-4mw": ("IAV-med", "IAV", "med"),
        "iavxchr-12mw": ("IAV-high", "IAV", "high"),
        "hp2xchr-400uw": ("HP2-low", "HP2", "low"),
        "hp2xchr-4mw": ("HP2-med", "HP2", "med"),
        "hp2xchr-12mw": ("HP2-high", "HP2", "high"),
        "allcsxchr-400uw": ("ALLCS-low", "ALLCS", "low"),
        "bicsxchr-4mw": ("BiCS-med", "BiCS", "med"),
        "bicsxchr-12mw": ("BiCS-high", "BiCS", "high"),
        "bics-haltxchr-4mw": ("BiCS-Halt-med", "BiCS-Halt", "med"),
        "bicshaltxchr-12mw": ("BiCS-Halt-high", "BiCS-Halt", "high"),
        "bics-haltwgxchr-12mw": ("BiCS-Halt-WG-high", "BiCS-Halt-WG", "high"),
        "css0048xchr-4mw": ("CS0048-med", "CS0048", "med"),
        "css0048xchr-12mw": ("CS0048-high", "CS0048", "high"),
        "css0021xchr-12mw": ("CS0021-high", "CS0021", "high"),
        "tacsxchr-400uw": ("TaCS-low", "TaCS", "low"),
        "tacsxchr-4mw": ("TaCS-med", "TaCS", "med"),
        "tacsxchr-12mw": ("TaCS-high", "TaCS", "high"),
        "tabrilexar-4mw": ("TaBri-med", "TaBri", "med"),
        "tabrilexar-12mw": ("TaBri-high", "TaBri", "high"),
    }

    def normalize_group_name(name):
        return str(name).replace("_", "").replace(" ", "").lower()

    def group_plot_info(group_label):
        normalized = normalize_group_name(group_label)
        if normalized in display_map:
            return display_map[normalized]
        intensity = "high" if "12mw" in normalized else "med" if "4mw" in normalized else "low"
        genotype = str(group_label).split("x")[0].split("X")[0]
        return f"{genotype}-{intensity}", genotype, intensity

    rng = np.random.default_rng(random_state)
    angle_defs = [["L-wing", "L-wing-hinge", "R-wing"]]
    delta_rows = []
    stat_rows = []
    wing_qc_rows = []
    wing_qc_skipped_rows = []

    def success_after_light_on(trial_info, index, group_label):
        if trial_info.LL == -1:
            return False, False
        if trial_info.LL == 1:
            mol, wing_qc_summary, wing_qc_df = _detect_chrimson_wing_mol(
                self,
                trial_info,
                wing_angle_def=angle_defs,
                apply_tracking_qc=apply_tracking_qc,
                tracking_error_thresholds=tracking_error_thresholds,
                min_cameras=min_cameras,
                max_interp_gap_frames=max_interp_gap_frames,
                min_valid_fraction=min_valid_fraction,
                smooth_angle=smooth_angle,
                smooth_window_frames=smooth_window_frames,
                smooth_polyorder=smooth_polyorder
            )
            wing_qc_summary.update({
                "Group": group_label,
                "Fly#": index[0],
                "Trial#": index[1],
                "Condition": "ON",
            })
            wing_qc_rows.append(wing_qc_summary)
            if not wing_qc_df.empty:
                wing_qc_df = wing_qc_df.copy()
                wing_qc_df["Group"] = group_label
                wing_qc_df["Fly#"] = index[0]
                wing_qc_df["Trial#"] = index[1]
                wing_qc_df["Condition"] = "ON"
                wing_qc_rows.extend(wing_qc_df.to_dict("records"))
            if apply_tracking_qc and not wing_qc_summary["Wing_QC_Passed"]:
                wing_qc_skipped_rows.append(wing_qc_summary.copy())
                return np.nan, True
            return mol != -1 and (mol / 250) <= threshold, False
        return np.nan, True

    for group_label, group_info in group_items:
        plot_label, genotype, intensity = group_plot_info(group_label)
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_Chr_manual_data()

        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying", "NF", "NA"])

        on_index, off_index = group_info.get_ON_OFF_index()
        on_index = sorted(on_index, key=lambda item: (item[0], item[1]))
        off_index = sorted(off_index, key=lambda item: (item[0], item[1]))

        on_rows = []
        for index in on_index:
            key = f"F{index[0]}T{index[1]}"
            if key not in group_info.fly_kinematic_data:
                continue
            success, skip_trial = success_after_light_on(group_info.fly_kinematic_data[key], index, group_label)
            if skip_trial:
                continue
            on_rows.append({
                "Fly#": index[0],
                "Success": int(success),
            })

        off_rows = []
        for index in off_index:
            key = f"F{index[0]}T{index[1]}"
            if key not in group_info.fly_kinematic_data:
                continue
            trial_info = group_info.fly_kinematic_data[key]
            off_rows.append({
                "Fly#": index[0],
                "Success": int(trial_info.LL != -1),
            })

        on_df = pd.DataFrame(on_rows)
        off_df = pd.DataFrame(off_rows)
        if on_df.empty or off_df.empty:
            continue

        on_lp = on_df.groupby("Fly#", as_index=False)["Success"].mean().rename(columns={"Success": "LP_ON"})
        off_lp = off_df.groupby("Fly#", as_index=False)["Success"].mean().rename(columns={"Success": "LP_OFF"})
        paired_df = on_lp.merge(off_lp, on="Fly#", how="inner").dropna(subset=["LP_ON", "LP_OFF"])
        if paired_df.empty:
            continue

        paired_df["Delta_LP_ON_minus_OFF"] = paired_df["LP_ON"] - paired_df["LP_OFF"]
        paired_df["Group"] = group_label
        paired_df["Plot_Label"] = plot_label
        paired_df["Genotype"] = genotype
        paired_df["Intensity"] = intensity
        paired_df["n_ON_trials"] = paired_df["Fly#"].map(on_df.groupby("Fly#").size())
        paired_df["n_OFF_trials"] = paired_df["Fly#"].map(off_df.groupby("Fly#").size())
        delta_rows.extend(paired_df.to_dict("records"))

        if len(paired_df) >= 2:
            observed_diff, p_value = self.calculator.paired_signflip_permutation_test(
                paired_df["LP_OFF"].to_numpy(dtype=float),
                paired_df["LP_ON"].to_numpy(dtype=float),
                n_perm=n_perm,
                rng=rng
            )
        else:
            observed_diff = float(paired_df["Delta_LP_ON_minus_OFF"].mean())
            p_value = np.nan

        stat_rows.append({
            "Group": group_label,
            "Plot_Label": plot_label,
            "Genotype": genotype,
            "Intensity": intensity,
            "Test": "paired sign-flip permutation",
            "Metric": "LP_ON_minus_LP_OFF",
            "n_paired_flies": int(len(paired_df)),
            "mean_LP_OFF": float(paired_df["LP_OFF"].mean()),
            "mean_LP_ON": float(paired_df["LP_ON"].mean()),
            "mean_delta_LP_ON_minus_OFF": float(observed_diff),
            "p_value": p_value,
            "n_perm": n_perm,
            "threshold": threshold,
        })

    delta_df = pd.DataFrame(delta_rows)
    stat_df = pd.DataFrame(stat_rows)
    if delta_df.empty:
        raise ValueError("No paired CsChrimson ON/OFF landing-probability data were found.")

    present_labels = set(delta_df["Plot_Label"])
    ordered_labels = [label for label in display_order if label in present_labels]
    ordered_labels.extend(sorted(present_labels - set(ordered_labels)))
    y_positions = {label: i for i, label in enumerate(ordered_labels)}

    fig_height = max(6.0, 0.42 * len(ordered_labels))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    ytick_labels = []
    for plot_label in ordered_labels:
        stat_sub = stat_df[stat_df["Plot_Label"] == plot_label]
        if not stat_sub.empty:
            sig = _significance_label(stat_sub.iloc[0]["p_value"])
            n_flies = int(stat_sub.iloc[0]["n_paired_flies"])
            ytick_labels.append(f"{plot_label} (n = {n_flies}, {sig})")
        else:
            ytick_labels.append(plot_label)

    for plot_label in ordered_labels:
        sub = delta_df[delta_df["Plot_Label"] == plot_label]
        if sub.empty:
            continue
        y = np.full(len(sub), y_positions[plot_label], dtype=float)
        jitter = rng.normal(0, 0.055, size=len(sub))
        intensity = sub["Intensity"].iloc[0]
        genotype = sub["Genotype"].iloc[0]
        point_color = "#8c8c8c" if genotype == "AD" else intensity_colors.get(intensity, "0.5")
        ax.scatter(
            sub["Delta_LP_ON_minus_OFF"],
            y + jitter,
            color=point_color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.45,
            s=42,
            zorder=3
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Delta landing probability (ON - OFF)")
    ax.set_ylabel("CsChrimson group")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(len(ordered_labels) - 0.5, -0.5)
    self.formatting(ax)
    from matplotlib.lines import Line2D
    intensity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#8c8c8c",
            markeredgecolor="black",
            label="AD"
        )
    ]
    intensity_handles.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=intensity_colors[key],
            markeredgecolor="black",
            label=label
        )
        for key, label in [("low", "AN 400 uW"), ("med", "AN 4 mW"), ("high", "AN 12 mW")]
    ])
    ax.legend(
        handles=intensity_handles,
        frameon=False,
        loc="upper right"
    )
    sns.despine(trim=True)
    plt.tight_layout()

    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        delta_df.to_csv(f"{file_name}_flywise_delta_LP.csv", index=False)
        stat_df.to_csv(f"{file_name}_paired_signflip_stats.csv", index=False)
        if apply_tracking_qc:
            pd.DataFrame(wing_qc_rows).to_csv(f"{file_name}_wing_qc_summary.csv", index=False)
            pd.DataFrame(wing_qc_skipped_rows).to_csv(f"{file_name}_wing_qc_skipped_trials.csv", index=False)

    plt.close(fig)
    return fig, ax, delta_df, stat_df

def plot_gtacr_LP_change_summary(
        self,
        groups,
        file_name="GtACR_LP_change_summary",
        n_perm=20000,
        random_state=0,
        color="#0B6E2E",
        box_color="#B7E1B0"
):
    """
    Plot fly-wise GtACR landing-probability change.

    Each point is one paired fly: Delta LP = LP_ON - LP_OFF. Rows are
    ordered by the input group order, and each row includes a shifted
    horizontal boxplot summarizing the fly-wise deltas.
    """
    if isinstance(groups, dict):
        group_items = list(groups.items())
    else:
        group_items = [(group.group_name, group) for group in groups]

    rng = np.random.default_rng(random_state)
    delta_rows = []
    stat_rows = []

    for group_label, group_info in group_items:
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
        group_info.filter_opto_data()

        combined_df = group_info.get_paired_LP_df().copy()
        if combined_df.empty:
            continue

        combined_df["Group_Name"] = pd.Categorical(
            combined_df["Group_Name"],
            categories=["OFF", "ON"],
            ordered=True
        )
        combined_df = combined_df.sort_values(["Fly#", "Group_Name"])
        paired_df = (
            combined_df
            .pivot(index="Fly#", columns="Group_Name", values="LandingProb")
            .dropna(subset=["OFF", "ON"])
            .reset_index()
        )
        if paired_df.empty:
            continue

        paired_df["Delta_LP_ON_minus_OFF"] = paired_df["ON"] - paired_df["OFF"]
        paired_df["Group"] = group_label
        delta_rows.extend(paired_df.to_dict("records"))

        if len(paired_df) >= 2:
            observed_diff, p_value = self.calculator.paired_signflip_permutation_test(
                paired_df["OFF"].to_numpy(dtype=float),
                paired_df["ON"].to_numpy(dtype=float),
                n_perm=n_perm,
                rng=rng
            )
        else:
            observed_diff = float(paired_df["Delta_LP_ON_minus_OFF"].mean())
            p_value = np.nan

        stat_rows.append({
            "Group": group_label,
            "Test": "paired sign-flip permutation",
            "Metric": "LP_ON_minus_LP_OFF",
            "n_paired_flies": int(len(paired_df)),
            "mean_LP_OFF": float(paired_df["OFF"].mean()),
            "mean_LP_ON": float(paired_df["ON"].mean()),
            "mean_delta_LP_ON_minus_OFF": float(observed_diff),
            "p_value": p_value,
            "n_perm": n_perm,
        })

    delta_df = pd.DataFrame(delta_rows)
    stat_df = pd.DataFrame(stat_rows)
    if delta_df.empty:
        raise ValueError("No paired GtACR ON/OFF landing-probability data were found.")

    group_order = [label for label, _ in group_items if label in set(delta_df["Group"])]
    y_positions = {group: i for i, group in enumerate(group_order)}
    fig_height = max(4.5, 0.55 * len(group_order))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))

    ytick_labels = []
    for group in group_order:
        stat_sub = stat_df[stat_df["Group"] == group]
        if stat_sub.empty:
            ytick_labels.append(group)
            continue
        n_flies = int(stat_sub.iloc[0]["n_paired_flies"])
        sig = _significance_label(stat_sub.iloc[0]["p_value"], missing_label="n/a")
        ytick_labels.append(f"{group} (n = {n_flies}, {sig})")

    for group in group_order:
        sub = delta_df[delta_df["Group"] == group]
        if sub.empty:
            continue

        y_center = y_positions[group]
        values = sub["Delta_LP_ON_minus_OFF"].dropna().to_numpy(dtype=float)
        box = ax.boxplot(
            values,
            vert=False,
            positions=[y_center - 0.18],
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            zorder=2
        )
        for patch in box["boxes"]:
            patch.set(
                facecolor=box_color,
                edgecolor="black",
                linewidth=1.1,
                alpha=0.55
            )
        for line in box["medians"]:
            line.set(color="black", linewidth=1.2)
        for line in box["whiskers"] + box["caps"]:
            line.set(color="black", linewidth=1.0)

        y = np.full(len(sub), y_center, dtype=float)
        jitter = rng.normal(0, 0.055, size=len(sub))
        ax.scatter(
            sub["Delta_LP_ON_minus_OFF"],
            y + jitter,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.45,
            s=42,
            zorder=3
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Delta landing probability (ON - OFF)")
    ax.set_ylabel("GtACR group")
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(len(group_order) - 0.5, -0.5)
    self.formatting(ax)
    sns.despine(trim=True)
    plt.tight_layout()

    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        delta_df.to_csv(f"{file_name}_flywise_delta_LP.csv", index=False)
        stat_df.to_csv(f"{file_name}_paired_signflip_stats.csv", index=False)

    plt.close(fig)
    return fig, ax, delta_df, stat_df

def plot_kmc_and_unpaired_rmst_perm(self,
        data_list,
        file_name,
        tau=0.71,
        n_perm=20000,
        random_state=0,
        colors=None,
        invert_curve=False,
):
    if colors is None:
        colors = sns.color_palette("tab20", 20)
    else:
        colors = list(colors)
    combined_df = pd.concat(data_list, ignore_index=True).copy()

    required_cols = {"Group", "Latency", "Event", "Fly#"}
    missing = required_cols - set(combined_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    combined_df["Latency"] = pd.to_numeric(combined_df["Latency"], errors="coerce")
    combined_df["Event"] = pd.to_numeric(combined_df["Event"], errors="coerce")
    combined_df = combined_df.dropna(subset=["Group", "Latency", "Event", "Fly#"])

    combined_df["Event"] = combined_df["Event"].astype(int)
    combined_df["Latency"] = combined_df["Latency"].clip(upper=tau)

    # ------------------------------------------------------------
    # Plot KM curves at trial level
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    kmf = KaplanMeierFitter()

    group_order = list(pd.unique(combined_df["Group"]))

    for i, group_name in enumerate(group_order):
        sub = combined_df[combined_df["Group"] == group_name]

        kmf.fit(
            durations=sub["Latency"],
            event_observed=sub["Event"],
            label=f"{group_name} (n trials={len(sub)})"
        )

        kmf.plot(
            ax=ax,
            ci_show=False,
            color=colors[i],
            linewidth=3
        )
        if invert_curve and len(ax.lines) > 0:
            # lifelines plots survival by default; flip this trace to cumulative landing probability.
            y_data = ax.lines[-1].get_ydata()
            ax.lines[-1].set_ydata(1 - y_data)

    ylabel = "Landing probability" if invert_curve else "Probability of no wing folding"
    self.formatting(ax, xticks=[0, 0.35, 0.71], yticks=[0, 0.5, 1], xlabel="Time (s)", ylabel=ylabel, xlabel_size=18, ylabel_size=18)
    ax.set_xlim(0, tau)
    ax.set_ylim(-0.05, 1.05)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}-KMC.pdf")
    # plt.show()
    plt.close()

    # ------------------------------------------------------------
    # Compute fly-level RMST
    # ------------------------------------------------------------
    fly_rows = []

    for (group_name, fly), sub in combined_df.groupby(["Group", "Fly#"]):
        if len(sub) == 0:
            continue

        kmf.fit(
            durations=sub["Latency"],
            event_observed=sub["Event"],
            label=f"{group_name}-Fly{fly}"
        )

        rmst = float(restricted_mean_survival_time(kmf, t=tau))

        fly_rows.append({
            "Group": group_name,
            "Fly#": fly,
            "RMST": rmst,
            "n_trials": len(sub),
            "n_events": int(sub["Event"].sum()),
            "event_rate": float(sub["Event"].mean())
        })

    fly_rmst_df = pd.DataFrame(fly_rows)
    fly_rmst_df.to_csv(f"{file_name}-fly_rmst.csv", index=False)

    # ------------------------------------------------------------
    # Pairwise unpaired permutation tests on fly-level RMST
    # ------------------------------------------------------------
    stat_rows = []

    for group_a, group_b in itertools.combinations(group_order, 2):
        x = fly_rmst_df.loc[fly_rmst_df["Group"] == group_a, "RMST"].values
        y = fly_rmst_df.loc[fly_rmst_df["Group"] == group_b, "RMST"].values

        if len(x) == 0 or len(y) == 0:
            continue

        observed_diff, p_value = self.calculator._permutation_test_unpaired(
            x,
            y,
            n_perm=n_perm,
            rng=np.random.default_rng(random_state)
        )

        stat_rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "group_a": group_a,
            "group_b": group_b,
            "n_fly_a": len(x),
            "n_fly_b": len(y),
            "mean_rmst_a": np.mean(x),
            "mean_rmst_b": np.mean(y),
            "estimate_b_minus_a": observed_diff,
            "permutation_p": p_value,
            "tau": tau,
            "n_perm": n_perm,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(f"{file_name}-pairwise_rmst_permutation.csv", index=False)

    return stat_df, fly_rmst_df
