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
from openpyxl import load_workbook

import tracking_qc as tqc


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

def _initialize_chrimson_absolute_mol_metadata(
        group_info,
        light_on_frame=750,
        tau=0.71,
        require_kinematics=False
):
    """
    Initialize CsChrimson metadata from LL sheets that store absolute MOL frames.

    Numeric nonnegative values are treated as absolute MOL frames and converted
    to latency frames relative to light_on_frame. -1 remains Flying, "NF"
    remains NF, and blank cells remain NA.
    """
    if group_info.ll_data is None:
        raise ValueError(f"LL/MOL metadata is required for {group_info.group_name}.")
    if group_info.fps is None or len(group_info.fps) < group_info.total_fly_number:
        raise ValueError(
            f"FPS list for {group_info.group_name} must contain at least "
            f"{group_info.total_fly_number} values."
        )

    group_info.latency_threshold = tau
    group_info.landing_trial_index = []
    group_info.flying_trial_index = []
    group_info.not_flying_trial_index = []
    group_info.NA_trial_index = []
    group_info.trial_metadata = dict()
    missing_trials = []

    for i in range(group_info.total_fly_number):
        for t in range(group_info.trial_num):
            fly = i + 1
            trial = t + 1
            key = group_info._trial_key(fly, trial)

            path = None
            light = None
            if key in group_info.fly_kinematic_data_path:
                path = group_info.fly_kinematic_data_path[key]
                light = group_info._get_opto_label_from_path(path)
            else:
                missing_trials.append(key)

            mol_abs = group_info.ll_data.iloc[i, t]
            if isinstance(mol_abs, str) and mol_abs == "NF":
                trial_type = "NF"
                ll_val = mol_abs
            elif pd.isna(mol_abs):
                trial_type = "NA"
                ll_val = np.nan
            elif mol_abs == -1:
                trial_type = "Flying"
                ll_val = -1
            elif mol_abs >= 0:
                trial_type = "Landing"
                ll_val = int(round(mol_abs - light_on_frame))
            else:
                trial_type = "Unknown"
                ll_val = np.nan

            if trial_type == "Unknown":
                raise ValueError(
                    f"Cannot classify CsChrimson metadata value for "
                    f"{group_info.group_name} {key}: {mol_abs}"
                )

            group_info.trial_metadata[key] = {
                "Fly#": fly,
                "Trial#": trial,
                "LL": ll_val,
                "MOC": np.nan,
                "MOL": mol_abs,
                "fps": group_info.fps[i],
                "TrialType": trial_type,
                "Path": path,
                "Light": light,
            }

            idx = (fly, trial)
            if trial_type == "Landing":
                group_info.landing_trial_index.append(idx)
            elif trial_type == "Flying":
                group_info.flying_trial_index.append(idx)
            elif trial_type == "NF":
                group_info.not_flying_trial_index.append(idx)
            elif trial_type == "NA":
                group_info.NA_trial_index.append(idx)

    if require_kinematics and missing_trials:
        preview = ", ".join(missing_trials[:10])
        more = "" if len(missing_trials) <= 10 else f" ... and {len(missing_trials) - 10} more"
        raise FileNotFoundError(
            f"Missing kinematic CSVs for Chr group {group_info.group_name}: {preview}{more}"
        )


def get_chrimson_metadata_on_ll_data(
        group_info,
        tau=0.71,
        light_on_frame=750,
        min_trial_num=8
):
    _initialize_chrimson_absolute_mol_metadata(
        group_info,
        light_on_frame=light_on_frame,
        tau=tau
    )
    group_info.filter_opto_data(min_trial_num=min_trial_num)
    ll_df = group_info.get_LL(return_df=True)
    if ll_df.empty:
        return pd.DataFrame(columns=["Group", "Latency", "Event", "Fly#", "Apply_Tracking_QC"])

    on_df = ll_df[ll_df["Light"] == "ON"].copy()
    on_df = on_df.rename(columns={"Group_Name": "Group"})
    on_df["Apply_Tracking_QC"] = False
    return on_df[["Group", "Latency", "Event", "Fly#", "Apply_Tracking_QC"]]

def plot_chrimson_LP_metadata(
        self,
        group_info,
        color="red",
        tau=0.71,
        light_on_frame=750,
        min_trial_num=8
):
    """
    Plot paired ON/OFF CsChrimson LP using metadata only.

    The LL metadata sheet is interpreted as absolute MOL frame numbers for
    landing trials. Latency is computed as (MOL - light_on_frame) / fps, and
    tau is used as the censoring/landing threshold.
    """
    _initialize_chrimson_absolute_mol_metadata(
        group_info,
        light_on_frame=light_on_frame,
        tau=tau
    )
    group_info.filter_opto_data(min_trial_num=min_trial_num)

    combined_df = group_info.get_paired_LP_df().copy()
    if combined_df.empty:
        raise ValueError(f"No paired ON/OFF metadata LP data found for {group_info.group_name}.")

    combined_df["Group_Name"] = pd.Categorical(
        combined_df["Group_Name"],
        categories=["OFF", "ON"],
        ordered=True
    )
    combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

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
            "Metric": "Metadata landing probability",
            "n_paired_flies": len(paired_df),
            "mean_OFF": paired_df["OFF"].mean(),
            "mean_ON": paired_df["ON"].mean(),
            "mean_diff_ON_minus_OFF": observed_diff,
            "p_value": p_val,
            "n_perm": 20000,
            "tau": tau,
            "light_on_frame": light_on_frame,
        }])
    else:
        paired_df["Diff_ON_minus_OFF"] = np.nan
        p_val = np.nan
        stat_df = pd.DataFrame()

    mean_color = color
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

    y_max = combined_df["LandingProb"].max()
    bracket_y = min(1.05, y_max + 0.10)
    text_y = min(1.09, bracket_y + 0.03)
    h = 0.02
    ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + h, bracket_y + h, bracket_y], lw=2.5, c="black")
    ax.text(0.5, text_y, _significance_label(p_val, missing_label="n/a"), ha="center", va="bottom", fontsize=14)

    plt.title("Metadata-Based Landing Probability Across Light Conditions")
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
    plt.savefig(f"{group_info.group_name}-chr-LP-metadata.pdf")
    plt.close()

    paired_df.to_csv(f"{group_info.group_name}-metadata-paired_values.csv")
    if not stat_df.empty:
        stat_df.to_csv(f"{group_info.group_name}-chr-LP-metadata-signflip-stat.csv", index=False)

    on_ll_data = get_chrimson_metadata_on_ll_data(
        group_info,
        tau=tau,
        light_on_frame=light_on_frame,
        min_trial_num=min_trial_num
    )
    return on_ll_data

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

