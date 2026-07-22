"""Secondary-contact plotting and RMST comparison workflows.

Public callers should continue using KinematicPlot.PlotCreator.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

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
    """
    Plot inverted KM curves for manually labeled SC timing.

    SC labels are absolute frame numbers. A valid event must occur after MOC,
    within threshold seconds after MOC, and not after MOL when MOL exists.
    Invalid/missing events are censored at threshold.
    """
    # Default colors are keyed by leg labels. Callers can override this mapping
    # when plotting a different subset or desired palette.
    if colors is None:
        colors = {
            "L-f": "tab:blue",
            "L-m": "tab:orange",
            "L-h": "tab:green",
            "R-f": "tab:grey",
            "R-m": "tab:brown",
            "R-h": "tab:red"
        }

    # Load metadata and kinematic traces if the group has not already been
    # prepared by a notebook or upstream script.
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()

    group_info.filter_nan_fly()
    group_info.read_kinematic_data(list(trial_types))

    # Manual SC CSVs are keyed by Index and contain one absolute frame column
    # per requested leg.
    sc_df = pd.read_csv(sc_csv_path)
    required_columns = {"Index", *legs}
    missing_columns = required_columns.difference(sc_df.columns)
    if missing_columns:
        raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

    # Build a trial-index lookup so each trial can quickly find its manual SC
    # annotation row.
    sc_lookup = {}
    for _, row in sc_df.iterrows():
        index = self.calculator.parse_index_cell(row["Index"])
        sc_lookup[index] = row

    def classify_trial(meta):
        # Classify success using the same landing-latency rule used elsewhere:
        # a Landing trial with LL/fps within the group's success threshold.
        ll = meta["LL"]
        fps = meta["fps"]
        if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group_info.latency_threshold:
            return "Success"
        return "Failed"

    # Convert every trial/leg manual SC annotation into a KM event row.
    rows = []
    for index in group_info.get_targeted_trials(list(trial_types)):
        key = group_info._trial_key(index[0], index[1])
        if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
            print("Missing trials")
            continue

        trial_info = group_info.fly_kinematic_data[key]
        meta = group_info.trial_metadata[key]
        moc = trial_info.moc
        mol = trial_info.mol
        fps = trial_info.fps

        if pd.isna(moc) or pd.isna(fps):
            continue

        # Missing SC rows are allowed; those leg observations become censored at
        # the threshold after validate_sc_timing.
        sc_row = sc_lookup.get(tuple(index))
        outcome = classify_trial(meta)

        for leg in legs:
            # validate_sc_timing converts an absolute SC frame into duration and
            # event/censoring fields relative to MOC, MOL, and threshold.
            raw_sc = np.nan if sc_row is None or leg not in sc_row else sc_row[leg]
            sc_result = self.calculator.validate_sc_timing(raw_sc, moc, mol, fps, threshold)
            rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Leg": leg,
                "Outcome": outcome,
                "TrialType": meta["TrialType"],
                "MOC_frame": moc,
                "MOL_frame": mol,
                "Raw_SC_frame": sc_result["sc_frame"],
                "SC_time_after_MOC_s": sc_result["sc_time_s"],
                "Duration": sc_result["duration"],
                "Event": sc_result["event"],
                "Threshold_s": threshold,
            })

    km_df = pd.DataFrame(rows)
    if km_df.empty:
        raise ValueError("No valid trials were available for manual SC KM plotting.")

    if save_csv and file_name is not None:
        km_df.to_csv(f"{file_name}_trial_leg_events.csv", index=False)

    def plot_km_panel(ax, data, title, ylabel=True):
        # Plot one inverted KM curve per leg. Survival is converted to event
        # probability so the y-axis reads as probability of secondary contact.
        kmf = KaplanMeierFitter()
        stat_rows = []
        for leg in legs:
            leg_df = data[data["Leg"] == leg].copy()
            if leg_df.empty:
                continue

            kmf.fit(
                durations=leg_df["Duration"],
                event_observed=leg_df["Event"],
                label=leg
            )
            surv = kmf.survival_function_.iloc[:, 0]
            event_probability = 1 - surv
            ax.step(
                event_probability.index.values,
                event_probability.values,
                where="post",
                color=colors.get(leg, "black"),
                linewidth=2.5,
                label=f"{leg} ({int(leg_df['Event'].sum())}/{len(leg_df)})"
            )

            # Store sample size, event count, censoring count, and median
            # survival time for this panel/leg.
            stat_rows.append({
                "Panel": title,
                "Leg": leg,
                "n_observations": len(leg_df),
                "n_events": int(leg_df["Event"].sum()),
                "n_censored": int(len(leg_df) - leg_df["Event"].sum()),
                "event_fraction": float(leg_df["Event"].mean()),
                "median_survival_time": kmf.median_survival_time_,
            })

        ax.set_xlim(0, threshold)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel("Probability of SC")
        ax.legend(frameon=False, fontsize=8)
        return stat_rows

    # The split success/failed panel code is retained as a disabled option. The
    # active output below plots all trials together.
    stat_rows = []
    """fig, axes = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True, sharey=True)
    for ax, outcome in zip(axes, ["Success", "Failed"]):
        stat_rows.extend(
            plot_km_panel(
                ax,
                km_df[km_df["Outcome"] == outcome],
                f"{group_info.group_name}: {outcome} trials"
            )
        )
    axes[-1].set_xlabel("SC timing after MOC (s)")
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f"{file_name}_success_failed.pdf", dpi=300, bbox_inches="tight")
    plt.close()"""

    # Active figure: one panel containing all trial/leg observations.
    combined_fig, combined_ax = plt.subplots(figsize=(7, 4.5))
    stat_rows.extend(
        plot_km_panel(
            combined_ax,
            km_df,
            f"{group_info.group_name}: all trials"
        )
    )
    combined_ax.set_xlabel("SC timing after MOC (s)")
    sns.despine()
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f"{file_name}_all_trials.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save the KM summary stats after all panels have contributed rows.
    stat_df = pd.DataFrame(stat_rows)
    if save_csv and file_name is not None:
        stat_df.to_csv(f"{file_name}_stats.csv", index=False)

    return combined_fig, combined_ax, km_df, stat_df


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
    """
    Compute fly-wise SC RMST and run pairwise permutation tests.

    `legs` can be a shared sequence for all contact groups, or a dict
    mapping contact group names to the legs that should be analyzed.
    Example: {"T1": ("R-m", "R-h"), "T2": ("R-f", "R-h")}.

    Four CSV files are saved:
    - one within-group leg-comparison stats file per contact group
    - one across-group same-leg comparison stats file
    """
    # Use a local RNG so permutation tests are reproducible for this call.
    rng = np.random.default_rng(random_state)
    # Validate that every requested contact group has both a Group object and an
    # SC annotation CSV.
    for contact_group in contact_groups:
        if contact_group not in group_infos:
            raise ValueError(f"Missing group_infos entry for contact group: {contact_group}")
        if contact_group not in sc_csv_paths:
            raise ValueError(f"Missing sc_csv_paths entry for contact group: {contact_group}")

    # Allow one shared leg list or a contact-group-specific mapping of legs.
    if isinstance(legs, dict):
        legs_by_group = {
            contact_group: tuple(legs.get(contact_group, ()))
            for contact_group in contact_groups
        }
    else:
        shared_legs = tuple(legs)
        legs_by_group = {
            contact_group: shared_legs
            for contact_group in contact_groups
        }

    # Each contact group must analyze at least one leg.
    for contact_group, group_legs in legs_by_group.items():
        if len(group_legs) == 0:
            raise ValueError(f"No legs were selected for contact group: {contact_group}")

    # Decide which within-contact-group leg pairs will be tested. By default,
    # all pairwise combinations of selected legs are compared.
    if within_group_leg_pairs is None:
        within_pairs_by_group = {
            contact_group: list(itertools.combinations(group_legs, 2))
            for contact_group, group_legs in legs_by_group.items()
        }
    elif isinstance(within_group_leg_pairs, dict):
        within_pairs_by_group = {}
        for contact_group, group_legs in legs_by_group.items():
            if contact_group in within_group_leg_pairs:
                group_pairs = [tuple(pair) for pair in within_group_leg_pairs[contact_group]]
            else:
                group_pairs = list(itertools.combinations(group_legs, 2))

            invalid_pairs = [
                pair for pair in group_pairs
                if len(pair) != 2 or pair[0] not in group_legs or pair[1] not in group_legs
            ]
            if invalid_pairs:
                raise ValueError(
                    "within_group_leg_pairs must contain 2-item leg pairs from the selected "
                    f"legs for {contact_group}. Invalid pairs: {invalid_pairs}"
                )
            within_pairs_by_group[contact_group] = group_pairs
    else:
        shared_pairs = [tuple(pair) for pair in within_group_leg_pairs]
        within_pairs_by_group = {}
        for contact_group, group_legs in legs_by_group.items():
            group_pairs = [
                pair for pair in shared_pairs
                if len(pair) == 2 and pair[0] in group_legs and pair[1] in group_legs
            ]
            invalid_pairs = [
                pair for pair in shared_pairs
                if len(pair) != 2 or pair[0] not in group_legs or pair[1] not in group_legs
            ]
            if invalid_pairs:
                raise ValueError(
                    "A shared within_group_leg_pairs list must be valid for every contact "
                    f"group's selected legs. Invalid pairs for {contact_group}: {invalid_pairs}"
                )
            within_pairs_by_group[contact_group] = group_pairs

    event_rows = []
    for contact_group in contact_groups:
        # Prepare one contact group's metadata, kinematic traces, and SC lookup.
        group_info = group_infos[contact_group]
        group_legs = legs_by_group[contact_group]

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(list(trial_types))

        sc_df = pd.read_csv(sc_csv_paths[contact_group])
        required_columns = {"Index", *group_legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(
                f"{contact_group} SC CSV is missing required columns: {sorted(missing_columns)}"
            )

        # Parse SC CSV rows by standard trial index.
        sc_lookup = {}
        for _, row in sc_df.iterrows():
            index = self.calculator.parse_index_cell(row["Index"])
            sc_lookup[index] = row

        for index in group_info.get_targeted_trials(list(trial_types)):
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                print("Warning unrecognized trial type!")
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue

            # Add one event/censoring row for every selected leg in this trial.
            sc_row = sc_lookup.get(tuple(index))
            for leg in group_legs:
                raw_sc = np.nan if sc_row is None or leg not in sc_row else sc_row[leg]
                sc_result = self.calculator.validate_sc_timing(raw_sc, moc, mol, fps, threshold)
                event_rows.append({
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Leg": leg,
                    "TrialType": meta["TrialType"],
                    "Duration": sc_result["duration"],
                    "Event": sc_result["event"],
                    "Raw_SC_frame": sc_result["sc_frame"],
                    "SC_time_after_MOC_s": sc_result["sc_time_s"],
                })

    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        raise ValueError("No manual SC observations were available for RMST analysis.")

    # Compute fly-wise RMST separately for each contact group and leg.
    kmf = KaplanMeierFitter()
    fly_rows = []
    for (contact_group, fly_id, leg), sub in event_df.groupby(["Contact_Group", "Fly#", "Leg"]):
        # print(f"\n--- {contact_group}, Fly {fly_id}, {leg} ---")
        # print(sub.to_string(index=False))
        if sub.empty:
            continue

        kmf.fit(
            durations=sub["Duration"],
            event_observed=sub["Event"]
        )
        fly_rows.append({
            "Contact_Group": contact_group,
            "Fly#": fly_id,
            "Leg": leg,
            "RMST": float(restricted_mean_survival_time(kmf, t=threshold)),
            "n_trials": int(len(sub)),
            "n_events": int(sub["Event"].sum()),
            "event_fraction": float(sub["Event"].mean()),
        })

    fly_rmst_df = pd.DataFrame(fly_rows)
    if fly_rmst_df.empty:
        raise ValueError("No fly-wise SC RMST values could be computed.")

    # Within each contact group, compare leg RMST values using paired flies.
    within_results = {}
    for contact_group in contact_groups:
        group_df = fly_rmst_df[fly_rmst_df["Contact_Group"] == contact_group]
        stat_rows = []
        for leg_a, leg_b in within_pairs_by_group[contact_group]:
            paired = (
                group_df[group_df["Leg"].isin([leg_a, leg_b])]
                .pivot(index="Fly#", columns="Leg", values="RMST")
                .dropna(subset=[leg_a, leg_b])
                .reset_index()
            )

            # print(f"\n=== Within {contact_group}: {leg_a} vs {leg_b} ===")
            # print(paired.to_string(index=False))

            # Paired sign-flip test uses each fly's leg-B minus leg-A RMST
            # difference as the unit of analysis.
            observed, p_value = self.calculator.paired_signflip_permutation_test(
                paired[leg_a],
                paired[leg_b],
                n_perm=n_perm,
                rng=rng
            )
            stat_rows.append({
                "Comparison_Type": "within_contact_group_paired_leg_rmst",
                "Contact_Group": contact_group,
                "Leg_A": leg_a,
                "Leg_B": leg_b,
                "n_paired_flies": int(len(paired)),
                "mean_RMST_A": np.nan if len(paired) == 0 else float(paired[leg_a].mean()),
                "mean_RMST_B": np.nan if len(paired) == 0 else float(paired[leg_b].mean()),
                "mean_diff_B_minus_A": observed,
                "permutation_p": p_value,
                "n_perm": n_perm,
                "tau": threshold,
            })

        within_df = pd.DataFrame(stat_rows)
        within_results[contact_group] = within_df
        within_df.to_csv(f"{file_name}_{contact_group}_within_group.csv", index=False)

    # Across contact groups, compare the same leg with an unpaired permutation
    # test on fly-wise RMST.
    across_rows = []
    for group_a, group_b in itertools.combinations(contact_groups, 2):
        common_legs = sorted(set(legs_by_group[group_a]).intersection(legs_by_group[group_b]))
        for leg in common_legs:
            leg_df = fly_rmst_df[fly_rmst_df["Leg"] == leg]
            values_a = (
                leg_df[leg_df["Contact_Group"] == group_a]["RMST"]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            values_b = (
                leg_df[leg_df["Contact_Group"] == group_b]["RMST"]
                .astype(float)
                .dropna()
                .to_numpy()
            )

            observed = np.nan
            p_value = np.nan
            if len(values_a) > 0 and len(values_b) > 0:
                observed, p_value = self.calculator._permutation_test_unpaired(
                    values_a,
                    values_b,
                    n_perm=n_perm
                )

            across_rows.append({
                "Comparison_Type": "across_contact_group_unpaired_same_leg_rmst",
                "Leg": leg,
                "Contact_Group_A": group_a,
                "Contact_Group_B": group_b,
                "n_A": int(len(values_a)),
                "n_B": int(len(values_b)),
                "mean_RMST_A": np.nan if len(values_a) == 0 else float(np.mean(values_a)),
                "mean_RMST_B": np.nan if len(values_b) == 0 else float(np.mean(values_b)),
                "mean_diff_B_minus_A": observed,
                "permutation_p": p_value,
                "n_perm": n_perm,
                "tau": threshold,
            })

    across_df = pd.DataFrame(across_rows)
    across_df.to_csv(f"{file_name}_across_groups.csv", index=False)

    return fly_rmst_df, within_results, across_df



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
    """
    Plot fly-wise first secondary-contact probability for each leg.

    For each fly, the denominator is the number of valid trials. A leg
    contributes when it is among the first valid SC events among the
    selected legs in that trial. If multiple legs share the earliest valid
    SC time, all tied legs are counted as first contacts.
    """
    # Default leg palette for the first-contact stripplot.
    if colors is None:
        colors = {
            "L-f": "tab:blue",
            "L-m": "tab:orange",
            "L-h": "tab:green",
        }

    # Verify that every requested contact group has both data and annotations.
    for contact_group in contact_groups:
        if contact_group not in group_infos:
            raise ValueError(f"Missing group_infos entry for contact group: {contact_group}")
        if contact_group not in sc_csv_paths:
            raise ValueError(f"Missing sc_csv_paths entry for contact group: {contact_group}")

    # Build one trial/leg row indicating whether that leg was among the earliest
    # valid SC events in the trial.
    trial_rows = []
    for contact_group in contact_groups:
        group_info = group_infos[contact_group]

        # Initialize and load traces on demand.
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(list(trial_types))

        # Each contact group has its own manual SC CSV with Index plus leg
        # columns.
        sc_df = pd.read_csv(sc_csv_paths[contact_group])
        required_columns = {"Index", *legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(
                f"{contact_group} SC CSV is missing required columns: {sorted(missing_columns)}"
        )

        for _, sc_row in sc_df.iterrows():
            # Match the CSV row to a loaded kinematic trial.
            index = self.calculator.parse_index_cell(sc_row["Index"])
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps

            if pd.isna(moc) or pd.isna(fps):
                continue

            # Convert each leg's manual SC frame into seconds after MOC. Invalid
            # contacts are represented as NaN and excluded from first-time
            # selection.
            leg_times = {}
            for leg in legs:
                sc_result = self.calculator.validate_sc_timing(sc_row[leg], moc, mol, fps, threshold)
                leg_times[leg] = sc_result["sc_time_s"]
            valid_times = [
                (leg, sc_time)
                for leg, sc_time in leg_times.items()
                if not pd.isna(sc_time)
            ]
            # Find the earliest valid contact time, if any.
            first_time = min((sc_time for _, sc_time in valid_times), default=np.nan)
            # Count every leg tied at the earliest time as a first contact.
            first_legs = {
                leg for leg, sc_time in valid_times
                if not pd.isna(first_time) and np.isclose(sc_time, first_time, rtol=0, atol=1e-12)
            }

            # Store a row for every leg so each fly/leg denominator counts all
            # valid trials considered for first-contact participation.
            for leg in legs:
                is_first_sc = leg in first_legs
                trial_rows.append({
                    "Contact_Group": contact_group,
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Leg": leg,
                    "SC_time_after_MOC_s": leg_times[leg],
                    "First_SC": int(is_first_sc),
                    "First_SC_Tie_Size": len(first_legs),
                    "Tie_Handling": "all_tied_legs_counted",
                })

    trial_df = pd.DataFrame(trial_rows)
    if trial_df.empty:
        raise ValueError("No matching SC rows were found for the requested contact groups.")

    # Collapse trial rows into one first-contact participation probability per
    # contact group, fly, and leg.
    prob_df = (
        trial_df
        .groupby(["Contact_Group", "Fly#", "Leg"])
        .agg(
            First_SC_Count=("First_SC", "sum"),
            n_trials=("First_SC", "size")
        )
        .reset_index()
    )
    prob_df["Secondary_Contact_Probability"] = (
            prob_df["First_SC_Count"] / prob_df["n_trials"]
    )
    prob_df["First_Contact_Participation_Probability"] = prob_df["Secondary_Contact_Probability"]

    def run_permutation(group_a, group_b, comparison_type, group_a_label, group_b_label):
        # Compare two sets of fly-wise first-contact probabilities with an
        # unpaired permutation test.
        values_a = (
            group_a["First_Contact_Participation_Probability"]
            .astype(float)
            .dropna()
            .to_numpy()
        )
        values_b = (
            group_b["First_Contact_Participation_Probability"]
            .astype(float)
            .dropna()
            .to_numpy()
        )
        row = {
            "Comparison_Type": comparison_type,
            "Group_A": group_a_label,
            "Group_B": group_b_label,
            "n_A": len(values_a),
            "n_B": len(values_b),
            "mean_A": np.nan if len(values_a) == 0 else float(np.mean(values_a)),
            "mean_B": np.nan if len(values_b) == 0 else float(np.mean(values_b)),
            "mean_diff_B_minus_A": np.nan,
            "permutation_p": np.nan,
            "n_perm": n_perm,
        }
        if len(values_a) == 0 or len(values_b) == 0:
            return row

        observed, p_value = self.calculator._permutation_test_unpaired(
            values_a,
            values_b,
            n_perm=n_perm
        )
        row["mean_diff_B_minus_A"] = float(observed)
        row["permutation_p"] = float(p_value)
        return row

    # Within each contact group, compare legs against each other.
    stat_rows = []
    for contact_group in contact_groups:
        group_df = prob_df[prob_df["Contact_Group"] == contact_group]
        for leg_a, leg_b in itertools.combinations(legs, 2):
            stat_rows.append(run_permutation(
                group_df[group_df["Leg"] == leg_a],
                group_df[group_df["Leg"] == leg_b],
                "within_contact_group_between_legs",
                f"{contact_group}-{leg_a}",
                f"{contact_group}-{leg_b}"
            ))

    # For each leg, compare matching legs across contact groups.
    for leg in legs:
        leg_df = prob_df[prob_df["Leg"] == leg]
        for group_a, group_b in itertools.combinations(contact_groups, 2):
            stat_rows.append(run_permutation(
                leg_df[leg_df["Contact_Group"] == group_a],
                leg_df[leg_df["Contact_Group"] == group_b],
                "between_contact_groups_same_leg",
                f"{group_a}-{leg}",
                f"{group_b}-{leg}"
            ))

    stat_df = pd.DataFrame(stat_rows)

    # Export trial-level first-contact flags, fly-level probabilities, and
    # permutation test summaries.
    if save_csv and file_name is not None:
        trial_df.to_csv(f"{file_name}_trial_first_sc.csv", index=False)
        prob_df.to_csv(f"{file_name}_fly_probability.csv", index=False)
        stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)

    # Plot the fly-wise probabilities as dodged stripplots by contact group and
    # leg.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.stripplot(
        data=prob_df,
        x="Contact_Group",
        y="First_Contact_Participation_Probability",
        hue="Leg",
        order=list(contact_groups),
        hue_order=list(legs),
        palette=colors,
        dodge=True,
        jitter=True,
        size=6,
        alpha=0.85,
        ax=ax
    )
    ax.set_xlabel("Contact group")
    ax.set_ylabel("First-contact participation probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Fly-wise first-contact participation probability")
    ax.legend(
        frameon=False,
        title="Leg",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return fig, ax, trial_df, prob_df, stat_df


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
    """
    Plot trial-wise number of valid SC events against raw landing latency.

    Success is defined as Landing with LL/fps <= group latency threshold.
    Failed includes flying trials and landing trials above threshold.

    group_info can be one Group, a list/tuple of Groups, or a dict such as
    {"T1": groups["WT_T1_TTa"], "T2": ..., "T3": ...}. When plotting
    multiple contact groups, sc_csv_path must be a dict keyed by group label
    or group name. Contact groups are shown as offset subgroups at each
    valid-contact count; success is filled and failed is hollow.
    """
    # Normalize group input to labeled group items so the same plotting code
    # handles one group, a list of groups, or a dict of named groups.
    if isinstance(group_info, dict):
        group_items = list(group_info.items())
    elif isinstance(group_info, (list, tuple)):
        group_items = [(group.group_name, group) for group in group_info]
    else:
        group_items = [(group_info.group_name, group_info)]

    # A single SC CSV path is valid only for one group. Multi-group plots need a
    # path mapping keyed by group label or group name.
    if not isinstance(sc_csv_path, dict):
        if len(group_items) != 1:
            raise ValueError("sc_csv_path must be a dict when plotting multiple groups.")
        sc_csv_paths = {group_items[0][0]: sc_csv_path}
    else:
        sc_csv_paths = sc_csv_path

    # Resolve one color per plotted contact group.
    if colors is None:
        palette = sns.color_palette("tab10", len(group_items))
        colors = {label: palette[i] for i, (label, _) in enumerate(group_items)}
    elif not isinstance(colors, dict):
        colors = {label: colors[i % len(colors)] for i, (label, _) in enumerate(group_items)}
    elif not any(label in colors or group.group_name in colors for label, group in group_items):
        palette = sns.color_palette("tab10", len(group_items))
        colors = {label: palette[i] for i, (label, _) in enumerate(group_items)}

    def group_color(group_label, group):
        # Prefer explicit plot label colors, then Group.group_name colors.
        return colors.get(group_label, colors.get(group.group_name, "black"))

    def classify_trial(group, meta):
        # Success is a landing trial whose raw LL/fps is within the group's
        # landing latency threshold.
        ll = meta["LL"]
        fps = meta["fps"]
        if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group.latency_threshold:
            return "Success"
        return "Failed"

    # Build one row per annotated trial containing the number of valid SC legs
    # and the trial's raw landing latency.
    rows = []
    for group_label, current_group in group_items:
        # Find this group's SC CSV path.
        path = sc_csv_paths.get(group_label, sc_csv_paths.get(current_group.group_name))
        if path is None:
            raise ValueError(f"No SC CSV path provided for group '{group_label}'.")

        # Initialize metadata and kinematic data when needed.
        if len(current_group.trial_metadata) == 0:
            current_group.initialize_manual_data()
            current_group.filter_nan_fly()

        current_group.read_kinematic_data(list(trial_types))

        # Validate that the SC CSV contains all requested leg columns.
        sc_df = pd.read_csv(path)
        required_columns = {"Index", *legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(f"{group_label} SC CSV is missing columns: {sorted(missing_columns)}")

        for _, sc_row in sc_df.iterrows():
            # Match the annotation row to a trial with kinematic data and
            # metadata.
            index = self.calculator.parse_index_cell(sc_row["Index"])
            key = current_group._trial_key(index[0], index[1])
            if key not in current_group.fly_kinematic_data or key not in current_group.trial_metadata:
                continue

            trial_info = current_group.fly_kinematic_data[key]
            meta = current_group.trial_metadata[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue

            # Convert raw LL frame count to seconds. Trials without valid LL are
            # not useful for this latency scatterplot.
            ll = meta["LL"]
            latency_s = np.nan
            if not pd.isna(ll) and ll != -1:
                latency_s = ll / meta["fps"]
            if pd.isna(latency_s):
                continue

            # Count how many requested legs have valid SC timing in this trial.
            valid_sc_count = 0
            valid_legs = []
            for leg in legs:
                sc_result = self.calculator.validate_sc_timing(
                    sc_row[leg],
                    moc,
                    mol,
                    fps,
                    threshold
                )
                if sc_result["is_valid"]:
                    valid_sc_count += 1
                    valid_legs.append(leg)

            # Store the count, list of valid legs, and outcome for plotting.
            rows.append({
                "Contact_Group": group_label,
                "Group_Name": current_group.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Outcome": classify_trial(current_group, meta),
                "TrialType": meta["TrialType"],
                "LL_frame": ll,
                "Landing_Latency_s": latency_s,
                "Valid_SC_Count": valid_sc_count,
                "Valid_SC_Legs": ",".join(valid_legs),
            })

    count_df = pd.DataFrame(rows)
    if count_df.empty:
        raise ValueError("No matching SC rows with valid landing latency were found.")

    # Save the trial-level count table before plotting.
    if save_csv and file_name is not None:
        count_df.to_csv(f"{file_name}_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    group_labels = [label for label, _ in group_items]
    # Multiple contact groups are offset horizontally around each integer SC
    # count to avoid complete overlap.
    if len(group_labels) == 1:
        offsets = {group_labels[0]: 0.0}
    else:
        offsets = {
            label: offset
            for label, offset in zip(group_labels, np.linspace(-subgroup_width, subgroup_width, len(group_labels)))
        }

    # Plot success and failed trials separately so success can be filled and
    # failed can be hollow while sharing the same group color.
    rng = np.random.default_rng(0)
    for group_label, current_group in group_items:
        color = group_color(group_label, current_group)
        for outcome, filled in (("Success", True), ("Failed", False)):
            sub = count_df[
                (count_df["Contact_Group"] == group_label)
                & (count_df["Outcome"] == outcome)
            ]
            if sub.empty:
                continue
            x_values = (
                sub["Valid_SC_Count"].to_numpy(dtype=float)
                + offsets[group_label]
                + rng.uniform(-jitter, jitter, size=len(sub))
            )
            ax.scatter(
                x_values,
                sub["Landing_Latency_s"],
                s=point_size,
                marker="o",
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=0.8,
                alpha=alpha,
                label=f"{group_label} {outcome}",
            )

    # Show each group's success threshold as a horizontal reference line.
    for _, current_group in group_items:
        ax.axhline(current_group.latency_threshold, color="0.35", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("# valid leg contact events per trial")
    ax.set_ylabel("Landing latency (s)")
    ax.set_xticks(range(len(legs) + 1))
    ax.set_xlim(-0.5, len(legs) + 0.5)
    ax.set_title("Valid leg contact count vs landing latency")

    # Use separate legends for contact group colors and filled/hollow outcome
    # encoding.
    group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=group_color(label, group),
            markeredgecolor=group_color(label, group),
            label=label,
        )
        for label, group in group_items
    ]
    outcome_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="0.25", markerfacecolor="0.25", label="Success"),
        plt.Line2D([0], [0], marker="o", linestyle="none", color="0.25", markerfacecolor="none", label="Failed"),
    ]
    legend1 = ax.legend(
        handles=group_handles,
        frameon=False,
        title="Contact group",
        loc="center left",
        bbox_to_anchor=(1.0, 0.62)
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=outcome_handles,
        frameon=False,
        title="Outcome",
        loc="center left",
        bbox_to_anchor=(1.0, 0.28)
    )
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    return fig, ax, count_df
