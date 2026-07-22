"""Landing-probability and Kaplan-Meier plotting workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from lifelines import KaplanMeierFitter

import trial_helpers as th


def _get_style(style_input, index, label, default):
    """Resolve a style supplied as None, a label mapping, or a sequence."""
    if style_input is None:
        return default
    if isinstance(style_input, dict):
        return style_input.get(label, default)
    return style_input[index % len(style_input)]


def _soften_color(color, softness):
    """Blend a color toward white while retaining its group identity."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    softness = float(np.clip(softness, 0, 1))
    return tuple(rgb + (1.0 - rgb) * softness)


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
    # Combine the per-group landing-probability DataFrames into one table so
    # the boxplot can be built from a single consistent set of group labels.
    combined_df = pd.concat(data_to_plot, ignore_index=True)

    # Scale the figure width with the number of groups. Each group gets one
    # categorical x-position, then boxplots and raw points are offset around it.
    fig, ax = plt.subplots(figsize=(len(data_to_plot) * 2, 8))

    group_names = combined_df["Group_Name"].unique()
    x_positions = np.arange(len(group_names))

    # Default to a broad categorical palette and one marker style per group
    # unless the caller passes explicit colors/markers.
    if colors is None:
        colors = sns.color_palette("tab20", len(data_to_plot))

    if markers is None:
        markers = ["o"] * len(data_to_plot)

    # Resolve each group's color once. This accepts either a sequence indexed
    # by plotting order or a dict keyed by group label.
    group_colors = []
    for group in group_names:
        matching_index = next(
            (
                i for i, data in enumerate(data_to_plot)
                if not data.empty and data["Group_Name"].iloc[0] == group
            ),
            0
        )
        group_colors.append(_get_style(colors, matching_index, group, "black"))

    # Build one distribution of fly-level landing probabilities per group.
    # The boxplot is shifted slightly left so the summary distribution and
    # individual fly points remain visually separate.
    box_offset = -0.18
    box_data = []

    for group in group_names:
        values = combined_df.loc[
            combined_df["Group_Name"] == group,
            "LandingProb"
        ].values
        box_data.append(values)

    bp = ax.boxplot(
        box_data,
        positions=x_positions + box_offset,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        zorder=1
    )

    # Style boxes with softened group colors by default. A caller can override
    # this with one shared color, a list of colors, or a dict keyed by group.
    for i, patch in enumerate(bp["boxes"]):
        group_color = group_colors[i]
        if box_color is None:
            face_color = _soften_color(group_color, box_softness)
        elif isinstance(box_color, dict):
            face_color = box_color.get(group_names[i], _soften_color(group_color, box_softness))
        elif isinstance(box_color, (list, tuple, np.ndarray)) and not mcolors.is_color_like(box_color):
            face_color = box_color[i % len(box_color)]
        else:
            face_color = box_color
        patch.set(
            facecolor=face_color,
            edgecolor=group_color,
            linewidth=2
        )

    for line in bp["medians"]:
        line.set(color="black", linewidth=2)

    # Match whiskers and caps to the group color so the box components stay
    # readable even when the fill is softened.
    for i, group_color in enumerate(group_colors):
        for line in bp["whiskers"][2 * i:2 * i + 2]:
            line.set(color=group_color, linewidth=2)
        for line in bp["caps"][2 * i:2 * i + 2]:
            line.set(color=group_color, linewidth=2)

    # Overlay individual fly-level points. Small random x-jitter prevents
    # points with identical landing probabilities from fully overlapping.
    strip_offset = 0.08

    for i, d in enumerate(data_to_plot):
        group = d["Group_Name"].iloc[0]

        group_index = np.where(group_names == group)[0][0]

        yvals = d["LandingProb"].values

        xvals = np.random.normal(
            loc=group_index + strip_offset,
            scale=0.02,
            size=len(yvals)
        )

        point_color = _get_style(colors, i, group, "black")
        marker = _get_style(markers, i, group, "o")

        ax.scatter(
            xvals,
            yvals,
            alpha=0.5,
            s=100,
            marker=marker,
            color=point_color,
            zorder=10
        )

    # Apply shared axis formatting from PlotCreator, then save the figure.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names, rotation=45)

    ax.set_ylim(-0.1, 1.1)

    self.formatting(ax, yticks=[0, 0.5, 1], ylabel="Landing Probability")

    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    sns.despine(trim=True)

    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf")

    plt.show()
    plt.close()

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
    data_to_plot = []

    # This wrapper accepts Group objects rather than prebuilt DataFrames. It
    # lazily initializes metadata, filters invalid flies, then reuses the core
    # plotting function above.
    for group_info in groups:
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()
        data_to_plot.append(group_info.get_LP_df())

    plot_LP_summary(
        self,
        data_to_plot=data_to_plot,
        file_name=file_name,
        colors=colors,
        markers=markers,
        box_color=box_color,
        box_width=box_width,
        box_softness=box_softness
    )

def plot_LP_summary_light(self, combined_df, file_name, color):
    # Work on a copy so sorting and categorical conversion do not mutate the
    # caller's DataFrame.
    combined_df = combined_df.copy()
    combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

    # Keep only flies that have both OFF and ON rows. The paired plot assumes
    # each fly contributes one value to each light condition.
    fly_counts = combined_df["Fly#"].value_counts()
    paired_flies = fly_counts[fly_counts == 2].index

    # Force OFF before ON so lines connect conditions in the intended order.
    combined_df = combined_df[combined_df["Fly#"].isin(paired_flies)].copy()
    combined_df["Group_Name"] = pd.Categorical(combined_df["Group_Name"], categories=["OFF", "ON"], ordered=True)
    combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

    # The pivot table is a paired-data check: rows with missing OFF or ON are
    # removed. The CSV export is left disabled but can be useful for inspection.
    paired_df = combined_df.pivot(index="Fly#", columns="Group_Name", values="LandingProb")
    paired_df = paired_df.dropna(subset=["OFF", "ON"])
    # paired_df.to_csv(f"{file_name}-paired_values.csv")

    fig, ax = plt.subplots(figsize=(4, 7))

    # Optional boxplot code kept here for manual reactivation if a paired-line
    # plot should also show condition distributions.
    """shift = 0.25
    width = 0.12

    off_vals = combined_df.loc[combined_df["Group_Name"] == "OFF", "LandingProb"].dropna().values
    on_vals = combined_df.loc[combined_df["Group_Name"] == "ON", "LandingProb"].dropna().values

    common_box_kwargs = dict(
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=2),
        whiskerprops=dict(color=color, linewidth=2),
        capprops=dict(color=color, linewidth=2),
        medianprops=dict(color=color, linewidth=2),
    )

    ax.boxplot([off_vals], positions=[0 - shift], **common_box_kwargs)
    ax.boxplot([on_vals], positions=[1 + shift], **common_box_kwargs)"""

    # Draw one grey connected line per fly to show the within-fly OFF-to-ON
    # change in landing probability.
    for fly_id, group in combined_df.groupby("Fly#"):
        group = group.sort_values("Group_Name")
        if len(group) == 2:
            ax.plot(
                [0, 1],
                group["LandingProb"].values,
                marker="o",
                markersize=10,
                color="lightgrey",
                linewidth=3,
                zorder=2
            )

    # Overlay the condition means as a colored line so the average trend is
    # visible on top of the individual paired fly trajectories.
    mean_df = combined_df.groupby("Group_Name", as_index=False)["LandingProb"].mean()
    ax.plot(
        [0, 1],
        mean_df["LandingProb"].values,
        color=color,
        marker="o",
        markersize=8,
        linewidth=2.5,
        zorder=11
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["OFF", "ON"])

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.5, 1.5)

    self.formatting(ax, yticks=[0, 0.5, 1], ylabel="Landing Probability")
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}-LP.pdf")
    # plt.show()
    plt.close()

def plot_LP_summary_light_from_group(self, group_info, file_name, color):
    # Prepare an optogenetic group and delegate the actual plotting to the
    # DataFrame-based paired-light function.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()
        group_info.filter_opto_data()

    combined_df = group_info.get_paired_LP_df()
    plot_LP_summary_light(self, combined_df, file_name, color)

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
    # Fit and plot one inverted Kaplan-Meier curve per input DataFrame. The
    # lifelines object returns survival probability; this plot displays
    # landing probability as 1 - survival.
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    kmf = KaplanMeierFitter()

    # Choose default visual styles. Optogenetic OFF/ON curves default to black
    # so line style or marker can carry the condition difference.
    if colors is None:
        if opto:
            colors = ["black", "black"]
        else:
            colors = sns.color_palette("tab20", len(data_to_plot))

    if linestyles is None:
        if opto:
            linestyles = ["solid", "solid"]  # OFF, ON if from_groups uses OFF then ON
        else:
            linestyles = ["solid"] * len(data_to_plot)

    if markers is None:
        markers = [None] * len(data_to_plot)

    stat_out = []

    # Each DataFrame should contain Latency, Event, and Group_Name columns.
    # Event marks actual landings; non-events are right-censored observations.
    for i, d in enumerate(data_to_plot):
        if d is None or len(d) == 0:
            continue

        label = d["Group_Name"].iloc[0]

        kmf.fit(
            d["Latency"],
            event_observed=d["Event"],
            label=label
        )

        # Convert the Kaplan-Meier survival curve into a cumulative landing
        # curve for easier interpretation in landing-probability figures.
        surv_df = kmf.survival_function_
        time = surv_df.index.values
        survival_prob = surv_df[label].values
        landing_prob = 1 - survival_prob

        line_color = _get_style(colors, i, label, "black")
        line_style = _get_style(linestyles, i, label, "solid")
        marker = _get_style(markers, i, label, None)

        ax.step(
            time,
            landing_prob,
            where="post",
            color=line_color,
            linestyle=line_style,
            linewidth=3,
            marker=marker,
            markevery=marker_every,
            label=label
        )

        # Record simple sample-size and censoring counts alongside the figure.
        stat_out.append({
            "Group": label,
            "n": len(d),
            "event_num": int(np.sum(d["Event"])),
            "censored_num": int(len(d) - np.sum(d["Event"]))
        })

    pd.DataFrame(stat_out).to_csv(f"{file_name}-KM_stat.csv", index=False)

    # Apply common axis styling, export the PDF, and close the figure to avoid
    # keeping matplotlib state alive across batch figure generation.
    ax.legend(frameon=False)

    self.formatting(
        ax,
        xticks=[0, 0.35, 0.71],
        yticks=[0, 0.5, 1],
        xlabel="Time (s)",
        ylabel="Landing probability"
    )

    ax.set_ylim(-0.05, 1.05)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}-LL-KMC-flipped.pdf")
    # plt.show()
    plt.close()

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
    data_to_plot = []

    # Convert each Group object into an LL DataFrame. In regular mode each
    # biological group contributes one curve; in optogenetic mode each group is
    # split into OFF and ON curves.
    for group_info in groups:
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        if not opto:
            group_info.filter_nan_fly()

            ll_df = group_info.get_LL(return_df=True)
            data_to_plot.append(ll_df)

        else:
            group_info.filter_opto_data()

            ll_df = group_info.get_LL(return_df=True)

            if ll_df is None or len(ll_df) == 0:
                continue

            off_df = ll_df[ll_df["Light"] == "OFF"].copy()
            on_df = ll_df[ll_df["Light"] == "ON"].copy()

            if len(off_df) > 0:
                off_df["Group_Name"] = group_info.group_name + "-OFF"
                data_to_plot.append(off_df)

            if len(on_df) > 0:
                on_df["Group_Name"] = group_info.group_name + "-ON"
                data_to_plot.append(on_df)

    plot_KM_curve(
        self,
        data_to_plot=data_to_plot,
        file_name=file_name,
        colors=colors,
        linestyles=linestyles,
        markers=markers,
        opto=opto,
        marker_every=marker_every
    )


def plot_landing_latency_distribution(
        self,
        group_info,
        file_name=None,
        exclude_first_n_trials=3,
        bins=20,
        color="#d62728",
        save_csv=True
):
    """Plot all valid raw LL values with a robust median + 2 scaled MAD cutoff."""
    # Load metadata if this group has not been initialized yet.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()

    # Extract raw landing-latency frame counts from metadata, drop early trials,
    # and convert valid LL values to seconds using each trial's FPS.
    rows = []
    for meta in group_info.trial_metadata.values():
        trial_number = meta.get("Trial#")
        ll_frame = pd.to_numeric(meta.get("LL"), errors="coerce")
        fps = pd.to_numeric(meta.get("fps"), errors="coerce")
        if pd.isna(trial_number) or int(trial_number) <= int(exclude_first_n_trials):
            continue
        if pd.isna(ll_frame) or pd.isna(fps) or fps <= 0 or ll_frame < 0:
            continue

        rows.append({
            "Group_Name": group_info.group_name,
            "Fly#": meta.get("Fly#"),
            "Trial#": int(trial_number),
            "Metadata_TrialType": meta.get("TrialType"),
            "LL_Frame": float(ll_frame),
            "FPS": float(fps),
            "Landing_Latency_s": float(ll_frame / fps),
        })

    latency_df = pd.DataFrame(rows)
    # Stop early if the requested exclusion/validity filters remove every row.
    if latency_df.empty:
        raise ValueError(
            f"No valid raw LL values remain for {group_info.group_name} after excluding "
            f"trials 1-{exclude_first_n_trials}."
        )

    # Use median absolute deviation rather than standard deviation so the
    # threshold is less sensitive to a few unusually long latency trials.
    latency = latency_df["Landing_Latency_s"].to_numpy(dtype=float)
    median = float(np.median(latency))
    raw_mad = float(np.median(np.abs(latency - median)))
    scaled_mad = 1.483 * raw_mad
    threshold = median + 2.0 * scaled_mad

    summary_df = pd.DataFrame([{
        "Group_Name": group_info.group_name,
        "Excluded_First_N_Trials": int(exclude_first_n_trials),
        "n_trials": int(len(latency_df)),
        "n_flies": int(latency_df["Fly#"].nunique()),
        "Median_s": median,
        "Raw_MAD_s": raw_mad,
        "Scaled_MAD_s": scaled_mad,
        "Threshold_Median_Plus_2Scaled_MAD_s": threshold,
        "Threshold_In_Frames": threshold * float(latency_df["FPS"].median()),
    }])

    # Plot the latency distribution with vertical reference lines for the
    # robust center and cutoff used to flag long-latency values.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sns.histplot(
        latency,
        bins=bins,
        color=color,
        alpha=0.45,
        edgecolor="white",
        stat="probability",
        kde=True,
        ax=ax,
    )
    ax.axvline(median, color="black", linewidth=2.2, label=f"Median = {median:.3f} s")
    ax.axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.9,
        label=f"Median + 2 scaled MAD = {threshold:.3f} s",
    )
    ax.text(
        0.98,
        0.96,
        (
            f"Median: {median:.3f} s\n"
            f"Raw MAD: {raw_mad:.3f} s\n"
            f"Scaled MAD: {scaled_mad:.3f} s\n"
            f"Threshold: {threshold:.3f} s"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.set_xlabel("Raw landing latency, LL / FPS (s)")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"{group_info.group_name} raw landing latency\n"
        f"excluding trials 1-{exclude_first_n_trials}"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    sns.despine()
    plt.tight_layout()

    # Save raw values and the summary threshold table when an output stem is
    # provided. The figure object is returned for callers that want inspection.
    if file_name is not None:
        fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        if save_csv:
            latency_df.to_csv(f"{file_name}_data.csv", index=False)
            summary_df.to_csv(f"{file_name}_summary.csv", index=False)
    plt.close(fig)
    return fig, ax, latency_df, summary_df

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
        max_interp_gap_frames=5,
        min_valid_fraction=0.7,
        smooth_angle=False,
        smooth_window_frames=5,
        smooth_polyorder=2,
        save_csv=True
):
    """
    Plot IT/OT fly-wise landing probability and landing-latency inverted KM.

    Fly-wise LP is # successful behavior trials / # total behavior trials.
    The trial-level permutation test shuffles success/fail outcomes while
    preserving the IT/OT sample sizes, then recalculates mean difference.
    """
    # Use a local random generator so jitter and permutation tests are
    # reproducible without changing NumPy's global random state.
    rng = np.random.default_rng(random_state)

    # Default colors and display names are keyed by behavior labels. Callers can
    # pass their own mappings when plotting different behavior categories.
    if colors is None:
        colors = {
            "IT": "#8FD694",
            "OT": "#C7A0E8",
        }

    if behavior_display_names is None:
        behavior_display_names = {
            "IT": "Inward touch",
            "OT": "Outward touch",
        }

    # A single behavior source path can be expanded into one marker-selection
    # source per behavior label. More complex callers can pass the full mapping.
    if isinstance(behavior_sources, (str, os.PathLike)):
        behavior_sources = {
            label: {
                "path": behavior_sources,
                "selection_mode": "marker",
                "marker_label": label,
            }
            for label in behavior_labels
        }

    # Ensure metadata and kinematic traces are available before building
    # trial-level landing and angle summaries.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()
        group_info.filter_nan_fly()

    group_info.read_kinematic_data(list(trial_types))

    # If the contacted leg is not supplied, infer it from the group name. T1,
    # T2, and T3 contact groups map to front, middle, and hind right legs.
    if contacted_leg is None:
        contact_leg_map = {
            "T1": "R-f",
            "T2": "R-m",
            "T3": "R-h",
        }
        contacted_leg = None
        for contact_group, leg in contact_leg_map.items():
            if contact_group in group_info.group_name:
                contacted_leg = leg
                break
        if contacted_leg is None:
            raise ValueError(
                "Could not infer contacted_leg from group name. "
                "Pass contacted_leg explicitly, for example contacted_leg='R-m'."
            )

    # Convert behavior-source definitions into a fast lookup from trial index
    # tuple, such as (fly, trial), to behavior label.
    behavior_trial_sets = th.trial_sets_from_behavior_sources(behavior_sources)
    behavior_by_index = {}
    for behavior_label, indexes in behavior_trial_sets.items():
        for index in indexes:
            behavior_by_index[tuple(index)] = behavior_label

    # Define the three points used for the contacted-leg joint angle and build
    # the common time axis used to resample all angle traces around MOC.
    angle_def = [f"{contacted_leg}CT", f"{contacted_leg}FT", f"{contacted_leg}TT"]
    target_n = int(round((angle_end_s - angle_start_s) * target_fps)) + 1
    target_time = np.linspace(angle_start_s, angle_end_s, target_n)

    # Collect trial-level landing outcomes, angle traces, velocity summaries,
    # and optional tracking-QC diagnostics in separate row lists. They become
    # DataFrames after the trial loop.
    trial_rows = []
    angle_trace_rows = []
    angular_velocity_rows = []
    angle_qc_rows = []
    angle_skipped_rows = []
    for index in group_info.get_targeted_trials(list(trial_types)):
        # Only analyze trials that appear in the IT/OT behavior annotations.
        index_tuple = tuple(index)
        behavior_label = behavior_by_index.get(index_tuple)
        if behavior_label not in behavior_labels:
            continue

        key = group_info._trial_key(index[0], index[1])
        if key not in group_info.trial_metadata:
            continue

        # Convert metadata into an inverted-KM duration/event pair. Successful
        # landings are events; censored trials contribute duration but no event.
        meta = group_info.trial_metadata[key]
        duration, event = th.inverted_km_latency_event(
            meta,
            group_info.latency_threshold,
            tau,
            allow_missing_sentinel=True
        )
        trial_rows.append({
            "Group_Name": group_info.group_name,
            "Index": str(index),
            "Fly#": index[0],
            "Trial#": index[1],
            "Behavior_Label": behavior_label,
            "Behavior_Display": behavior_display_names.get(behavior_label, behavior_label),
            "TrialType": meta["TrialType"],
            "LL_frame": meta["LL"],
            "Success": event,
            "Duration": duration,
            "Event": event,
            "Threshold_s": tau,
        })

        # Angle analysis is optional per trial. Missing kinematic data or
        # incomplete MOC-centered windows are recorded so skipped trials can be
        # audited in the exported QC CSVs.
        if key not in group_info.fly_kinematic_data:
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Reason": "missing kinematic data",
            })
            continue

        # Pull trial-specific timing and frame-rate metadata needed to convert
        # the MOC-centered time window into frame indices.
        trial_info = group_info.fly_kinematic_data[key]
        moc = trial_info.moc
        fps = trial_info.fps
        if pd.isna(moc) or pd.isna(fps):
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Reason": "missing MOC or fps",
            })
            continue
        if any(point not in trial_info.trial_data for point in angle_def):
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Reason": "missing angle keypoint",
            })
            continue

        # Convert the requested seconds around MOC into inclusive frame bounds,
        # then require the full window to be present in the kinematic trace.
        start_frame = int(round(moc + angle_start_s * fps))
        end_frame = int(round(moc + angle_end_s * fps))
        if start_frame < 0 or end_frame >= trial_info.total_frames_number or end_frame <= start_frame:
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Reason": "incomplete MOC-centered angle window",
            })
            continue

        # Calculate the contacted-leg joint angle. When tracking QC is enabled,
        # the calculator also returns a per-angle QC summary.
        angle_result = self.calculator.Calculate_joint_angle(
            trial_info,
            [angle_def],
            apply_tracking_qc=apply_tracking_qc,
            tracking_error_thresholds=tracking_error_thresholds,
            min_cameras=min_cameras,
            max_interp_gap_frames=max_interp_gap_frames,
            min_valid_fraction=min_valid_fraction,
            smooth_angle=smooth_angle,
            smooth_window_frames=smooth_window_frames,
            smooth_polyorder=smooth_polyorder,
            qc_start=start_frame,
            qc_end=end_frame,
            return_qc=apply_tracking_qc
        )
        if apply_tracking_qc:
            angle_data, angle_qc_df = angle_result
            if not angle_qc_df.empty:
                qc_record = angle_qc_df.iloc[0].to_dict()
                qc_record.update({
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Behavior_Label": behavior_label,
                    "Contacted_Leg": contacted_leg,
                })
                angle_qc_rows.append(qc_record)
                if not bool(qc_record.get("QC_Passed", True)):
                    angle_skipped_rows.append({
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Behavior_Label": behavior_label,
                        "Contacted_Leg": contacted_leg,
                        "Joint": angle_def[1],
                        "Reason": "failed angle tracking QC",
                        **qc_record,
                    })
                    continue
        else:
            angle_data = angle_result

        # Extract the FT-centered joint angle inside the MOC-centered window
        # and evaluate how much usable data remains after any QC/interpolation.
        angle_trace = angle_data[angle_def[1]]
        source_frames = np.arange(start_frame, end_frame + 1)
        source_time = (source_frames - moc) / fps
        source_trace = np.asarray(angle_trace[start_frame:end_frame + 1], dtype=float)
        valid = np.isfinite(source_time) & np.isfinite(source_trace)
        window_valid_fraction = float(np.mean(valid)) if len(valid) else np.nan
        max_invalid_gap = max(self.calculator.invalid_gap_lengths(valid), default=0)
        if apply_tracking_qc and (
                window_valid_fraction < min_valid_fraction
                or max_invalid_gap > max_interp_gap_frames
        ):
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Contacted_Leg": contacted_leg,
                "Joint": angle_def[1],
                "Reason": "failed angle tracking QC",
                "Valid_Frame_Fraction": window_valid_fraction,
                "Max_Invalid_Gap_Frames": max_invalid_gap,
                "Min_Valid_Fraction": min_valid_fraction,
                "Max_Interp_Gap_Frames": max_interp_gap_frames,
            })
            continue

        # Enforce a minimum number of valid samples so interpolation and
        # velocity calculations are not based on an underdetermined trace.
        if np.sum(valid) < min_angle_frames:
            angle_skipped_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Contacted_Leg": contacted_leg,
                "Joint": angle_def[1],
                "Reason": "fewer than min_angle_frames valid samples",
                "Valid_Sample_Count": int(np.sum(valid)),
                "Min_Angle_Frames": min_angle_frames,
            })
            continue

        # Resample each trial onto the common time axis. This makes traces from
        # trials with different FPS directly comparable for averaging.
        resampled_trace = np.interp(
            target_time,
            source_time[valid],
            source_trace[valid],
            left=np.nan,
            right=np.nan
        )
        for time_s, angle_value in zip(target_time, resampled_trace):
            angle_trace_rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Behavior_Label": behavior_label,
                "Behavior_Display": behavior_display_names.get(behavior_label, behavior_label),
                "Contacted_Leg": contacted_leg,
                "Joint": angle_def[1],
                "Time_From_MOC_s": time_s,
                "Angle_deg": angle_value,
                "Apply_Tracking_QC": apply_tracking_qc,
                "Smooth_Angle": smooth_angle,
            })

        # Summarize angular velocity in the original sampled window. The
        # absolute-value option reports speed regardless of movement direction.
        clean_angle = source_trace[valid]
        angular_velocity = np.diff(clean_angle) * fps
        if use_absolute_angular_velocity:
            angular_velocity = np.abs(angular_velocity)
        if len(angular_velocity) == 0:
            continue
        angular_velocity_rows.append({
            "Group_Name": group_info.group_name,
            "Index": str(index),
            "Fly#": index[0],
            "Trial#": index[1],
            "Behavior_Label": behavior_label,
            "Behavior_Display": behavior_display_names.get(behavior_label, behavior_label),
            "Contacted_Leg": contacted_leg,
            "Joint": angle_def[1],
            "Mean_Angular_Velocity_deg_s": float(np.nanmean(angular_velocity)),
            "Angle_Start_s": angle_start_s,
            "Angle_End_s": angle_end_s,
            "Use_Absolute_Angular_Velocity": use_absolute_angular_velocity,
            "Apply_Tracking_QC": apply_tracking_qc,
            "Smooth_Angle": smooth_angle,
        })

    # Materialize all collected rows as DataFrames. The trial table is required;
    # angle-related tables may be empty if traces were missing or filtered out.
    trial_df = pd.DataFrame(trial_rows)
    if trial_df.empty:
        raise ValueError("No IT/OT-labeled Landing/Flying trials were found.")
    angle_trace_df = pd.DataFrame(angle_trace_rows)
    angular_velocity_df = pd.DataFrame(angular_velocity_rows)
    angle_qc_df = pd.DataFrame(angle_qc_rows)
    angle_skipped_df = pd.DataFrame(angle_skipped_rows)

    # Collapse trial-level success/failure values into fly-wise landing
    # probabilities for the IT/OT summary plot.
    fly_lp_df = (
        trial_df
        .groupby(["Fly#", "Behavior_Label", "Behavior_Display"])
        .agg(
            Successful_Trials=("Success", "sum"),
            Total_Trials=("Success", "size"),
            Landing_Probability=("Success", "mean"),
        )
        .reset_index()
    )

    def trial_mean(label):
        values = trial_df.loc[trial_df["Behavior_Label"] == label, "Success"].astype(float)
        return np.nan if values.empty else float(values.mean())

    # Trial-level permutation test: keep the observed IT/OT sample sizes, shuffle
    # success outcomes, and compare shuffled mean differences to the observed
    # OT-minus-IT difference.
    observed_diff = trial_mean(behavior_labels[1]) - trial_mean(behavior_labels[0])
    label_counts = [int((trial_df["Behavior_Label"] == label).sum()) for label in behavior_labels]
    outcomes = trial_df["Success"].astype(float).to_numpy()

    perm_diffs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(outcomes)
        split_means = []
        start = 0
        for count in label_counts:
            stop = start + count
            split_means.append(np.mean(shuffled[start:stop]) if count > 0 else np.nan)
            start = stop
        perm_diffs.append(split_means[1] - split_means[0])

    perm_diffs = np.asarray(perm_diffs, dtype=float)
    p_value = (np.sum(np.abs(perm_diffs) >= abs(observed_diff)) + 1) / (np.sum(np.isfinite(perm_diffs)) + 1)

    stat_df = pd.DataFrame([{
        "Group_Name": group_info.group_name,
        "Group_A": behavior_labels[0],
        "Group_B": behavior_labels[1],
        "n_A": label_counts[0],
        "n_B": label_counts[1],
        "mean_success_A": trial_mean(behavior_labels[0]),
        "mean_success_B": trial_mean(behavior_labels[1]),
        "mean_diff_B_minus_A": observed_diff,
        "permutation_p": p_value,
        "n_perm": n_perm,
    }])

    # Export all intermediate tables so the plotted values, statistics, angle
    # traces, and QC decisions can be inspected outside matplotlib.
    if save_csv and file_name is not None:
        trial_df.to_csv(f"{file_name}_trial_data.csv", index=False)
        fly_lp_df.to_csv(f"{file_name}_fly_landing_probability.csv", index=False)
        stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)
        if not angle_trace_df.empty:
            angle_trace_df.to_csv(f"{file_name}_FT_angle_traces.csv", index=False)
        if not angular_velocity_df.empty:
            angular_velocity_df.to_csv(f"{file_name}_FT_angular_velocity.csv", index=False)
        if apply_tracking_qc:
            angle_qc_df.to_csv(f"{file_name}_FT_angle_qc_summary.csv", index=False)
            angle_skipped_df.to_csv(f"{file_name}_FT_angle_qc_skipped_trials.csv", index=False)

    # Convert p-values into compact annotations for the figures.
    def significance_label(p):
        if pd.isna(p):
            return "n.s."
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    # Figure 1: fly-wise landing probability. Boxes show the distribution of
    # fly means; jittered points show individual flies and point area reflects
    # how many trials contributed to that fly's probability.
    fig_lp, ax_lp = plt.subplots(figsize=(4.8, 4.2))
    positions = np.arange(len(behavior_labels), dtype=float)
    box_positions = positions - 0.16
    point_positions = positions + 0.12

    for i, label in enumerate(behavior_labels):
        sub = fly_lp_df[fly_lp_df["Behavior_Label"] == label]
        values = sub["Landing_Probability"].astype(float).dropna().to_numpy()
        if len(values) > 0:
            ax_lp.boxplot(
                values,
                positions=[box_positions[i]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": colors.get(label, "0.5"),
                    "alpha": 0.25,
                    "edgecolor": colors.get(label, "0.5"),
                },
                medianprops={"color": "black", "linewidth": 1.3},
                whiskerprops={"color": colors.get(label, "0.5")},
                capprops={"color": colors.get(label, "0.5")},
            )

        if not sub.empty:
            denom = sub["Total_Trials"].astype(float).to_numpy()
            if np.nanmax(denom) > np.nanmin(denom):
                sizes = 28 + (denom - np.nanmin(denom)) / (np.nanmax(denom) - np.nanmin(denom)) * 92
            else:
                sizes = np.full_like(denom, 60, dtype=float)
            x = point_positions[i] + rng.uniform(-0.045, 0.045, size=len(sub))
            ax_lp.scatter(
                x,
                sub["Landing_Probability"],
                s=sizes,
                color=colors.get(label, "0.5"),
                alpha=0.78,
                edgecolor="black",
                linewidth=0.4,
            )

    # Draw a simple bracket and annotate it with the permutation-test result.
    y_bracket = 1.03
    ax_lp.plot(
        [point_positions[0], point_positions[0], point_positions[1], point_positions[1]],
        [y_bracket, y_bracket + 0.03, y_bracket + 0.03, y_bracket],
        color="black",
        linewidth=1
    )
    ax_lp.text(np.mean(point_positions), y_bracket + 0.035, significance_label(p_value),
               ha="center", va="bottom", fontsize=12)
    ax_lp.set_xticks(positions)
    ax_lp.set_xticklabels([behavior_display_names.get(label, label) for label in behavior_labels])
    ax_lp.set_ylabel("Fly-wise landing probability")
    ax_lp.set_ylim(-0.05, 1.12)
    ax_lp.set_title(f"{group_info.group_name}: IT/OT landing probability")
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        fig_lp.savefig(f"{file_name}_landing_probability.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_lp)

    # Figure 2: inverted Kaplan-Meier landing latency. Each behavior label gets
    # one cumulative landing-probability curve and a count of events/trials.
    fig_km, ax_km = plt.subplots(figsize=(5.4, 4.2))
    kmf = KaplanMeierFitter()
    km_rows = []
    for label in behavior_labels:
        sub = trial_df[trial_df["Behavior_Label"] == label]
        if sub.empty:
            continue

        display = behavior_display_names.get(label, label)
        kmf.fit(
            durations=sub["Duration"],
            event_observed=sub["Event"],
            label=display
        )
        surv = kmf.survival_function_.iloc[:, 0]
        ax_km.step(
            surv.index.values,
            1 - surv.values,
            where="post",
            color=colors.get(label, "0.5"),
            linewidth=2.5,
            label=f"{display} ({int(sub['Event'].sum())}/{len(sub)})"
        )
        km_rows.append({
            "Behavior_Label": label,
            "Behavior_Display": display,
            "n_trials": len(sub),
            "n_events": int(sub["Event"].sum()),
            "event_fraction": float(sub["Event"].mean()),
            "median_survival_time": kmf.median_survival_time_,
        })

    # Run a log-rank test between the first two behavior labels when both have
    # data, then export the test result for reporting.
    km_stat_df = pd.DataFrame(km_rows)
    logrank_df = pd.DataFrame()
    logrank_p = np.nan
    if len(behavior_labels) >= 2:
        km_a = trial_df[trial_df["Behavior_Label"] == behavior_labels[0]]
        km_b = trial_df[trial_df["Behavior_Label"] == behavior_labels[1]]
        if not km_a.empty and not km_b.empty:
            logrank_result = logrank_test(
                km_a["Duration"],
                km_b["Duration"],
                event_observed_A=km_a["Event"],
                event_observed_B=km_b["Event"]
            )
            logrank_p = float(logrank_result.p_value)
            logrank_df = pd.DataFrame([{
                "Group_A": behavior_labels[0],
                "Group_B": behavior_labels[1],
                "n_A": len(km_a),
                "n_B": len(km_b),
                "events_A": int(km_a["Event"].sum()),
                "events_B": int(km_b["Event"].sum()),
                "test_statistic": float(logrank_result.test_statistic),
                "p_value": logrank_p,
            }])

    if save_csv and file_name is not None:
        km_stat_df.to_csv(f"{file_name}_km_stats.csv", index=False)
        if not logrank_df.empty:
            logrank_df.to_csv(f"{file_name}_km_logrank.csv", index=False)

    # Finish the KM axes, place the log-rank significance label, and save.
    ax_km.set_xlim(0, tau)
    ax_km.set_ylim(-0.05, 1.05)
    ax_km.set_xlabel("Landing latency after MOC (s)")
    ax_km.set_ylabel("Landing probability")
    ax_km.set_title(f"{group_info.group_name}: IT/OT landing latency")
    ax_km.text(
        0.5,
        0.96,
        significance_label(logrank_p),
        transform=ax_km.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold"
    )
    ax_km.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        fig_km.savefig(f"{file_name}_landing_latency_inverted_KM.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_km)

    # Figure 3: mean FT angle trace around MOC. Traces are averaged by behavior
    # label after resampling, with SEM ribbons when more than one trial
    # contributes at a time point.
    fig_angle, ax_angle = plt.subplots(figsize=(5.6, 4.0))
    plotted_angle_values = []
    angle_count_lines = []
    if not angle_trace_df.empty:
        for label in behavior_labels:
            sub = angle_trace_df[angle_trace_df["Behavior_Label"] == label]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index=["Index"],
                columns="Time_From_MOC_s",
                values="Angle_deg",
                aggfunc="mean"
            )
            traces = pivot.to_numpy(dtype=float)
            time_values = pivot.columns.to_numpy(dtype=float)
            mean_trace = np.nanmean(traces, axis=0)
            valid_n = np.sum(np.isfinite(traces), axis=0)
            sem_trace = np.full_like(mean_trace, np.nan, dtype=float)
            valid_sem = valid_n > 1
            sem_trace[valid_sem] = (
                    np.nanstd(traces[:, valid_sem], axis=0, ddof=1)
                    / np.sqrt(valid_n[valid_sem])
            )
            plotted_angle_values.extend([mean_trace, mean_trace - sem_trace, mean_trace + sem_trace])
            display = behavior_display_names.get(label, label)
            n_angle_trials = int(traces.shape[0])
            angle_count_lines.append((f"{display}: n={n_angle_trials}", colors.get(label, "0.5")))
            ax_angle.plot(
                time_values,
                mean_trace,
                color=colors.get(label, "0.5"),
                linewidth=2.4,
                label=f"{display} (n={n_angle_trials})"
            )
            ax_angle.fill_between(
                time_values,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=colors.get(label, "0.5"),
                alpha=0.20,
                linewidth=0
            )

    # Add MOC reference line, behavior-specific sample counts, and labels.
    ax_angle.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_angle.set_xlabel("Time from MOC (s)")
    ax_angle.set_ylabel(f"{contacted_leg} FT angle (deg)")
    behavior_title = "/".join(behavior_display_names.get(label, label) for label in behavior_labels)
    ax_angle.set_title(f"{group_info.group_name}: {behavior_title} {contacted_leg} FT angle trace")
    ax_angle.set_xlim(angle_start_s, angle_end_s)
    for line_i, (count_text, count_color) in enumerate(angle_count_lines):
        ax_angle.text(
            0.03,
            0.95 - line_i * 0.09,
            count_text,
            transform=ax_angle.transAxes,
            color=count_color,
            fontsize=8,
            ha="left",
            va="top",
        )

    # Autoscale y-limits to the data that were actually plotted, including SEM
    # bands, with a small padding so traces do not sit on the frame.
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
        ax_angle.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_angle.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        fig_angle.savefig(f"{file_name}_FT_angle_trace.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_angle)


    # Return figure handles and DataFrames so notebook callers can inspect or
    # reuse the outputs without re-reading the CSV files.
    return (
        fig_lp, ax_lp, fig_km, ax_km, fig_angle, ax_angle,
        fly_lp_df, trial_df, stat_df, km_stat_df, logrank_df,
        angle_trace_df, angular_velocity_df,
    )

