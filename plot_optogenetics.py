"""Optogenetic landing and latency plotting workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

from survival_stats_runner import SurvivalStatsRunner


def _get_stats_runner(self):
    # Use PlotCreator's shared stats runner when available; direct calls get a
    # local runner so this module remains independently testable.
    return getattr(self, "stats_runner", SurvivalStatsRunner())

def get_opto_on_ll_data(
        group_info,
        tau=0.71,
        min_trial_num=8
):
    # Initialize normal LL/MOC/MOL metadata when the caller has not already
    # prepared the group in a notebook setup cell.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()

    # Apply the same ON/OFF fly filter used by the generic optogenetic plots.
    group_info.filter_opto_data(min_trial_num=min_trial_num)
    ll_df = group_info.get_LL(return_df=True)
    if ll_df.empty:
        # Landing latency is metadata-only, so this table intentionally carries
        # no tracking-QC fields.
        return pd.DataFrame(columns=["Group", "Latency", "Event", "Fly#"])

    # Keep only light-on trials for selected-group CsChrimson KM summaries.
    on_df = ll_df[ll_df["Light"] == "ON"].copy()
    on_df = on_df.rename(columns={"Group_Name": "Group"})
    return on_df[["Group", "Latency", "Event", "Fly#"]]

def plot_kmc_and_unpaired_rmst_perm(self,
        data_list,
        file_name,
        tau=0.71,
        n_perm=20000,
        random_state=0,
        colors=None,
        invert_curve=False,
        group_pairs=None,
        control_group=None,
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
    # Restrict RMST tests to planned comparisons when requested; by default the
    # legacy behavior still compares all groups.
    if group_pairs is None:
        if control_group is None:
            group_pairs = list(itertools.combinations(group_order, 2))
        else:
            # Require an exact group label match so planned control comparisons
            # cannot silently disappear because of a typo.
            if control_group not in group_order:
                raise ValueError(f"control_group '{control_group}' was not found in plotted groups: {group_order}")
            group_pairs = [
                (control_group, group_name)
                for group_name in group_order
                if group_name != control_group
            ]

    # The stats runner standardizes pairwise RMST comparisons and output columns.
    stat_df = _get_stats_runner(self).pairwise_flywise_rmst_permutation(
        fly_rmst_df,
        group_col="Group",
        value_col="RMST",
        trial_count_col="n_trials",
        group_pairs=group_pairs,
        metric="landing_latency_rmst",
        n_perm=n_perm
    )
    if not stat_df.empty:
        # RMST depends on tau, so keep tau as the only plot-level addition to
        # the standardized pairwise RMST table.
        stat_df["tau"] = tau
    stat_df.to_csv(f"{file_name}-pairwise_rmst_permutation.csv", index=False)

    return stat_df, fly_rmst_df

