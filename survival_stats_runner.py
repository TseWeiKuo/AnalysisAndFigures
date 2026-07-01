import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import restricted_mean_survival_time

import kinematic_utilities as ku


class SurvivalStatsRunner:
    """
    Standalone statistical analysis helper for your existing pipeline.

    Design goals:
    - DO NOT modify Group / PlotCreator / current plotting pipeline
    - Reuse your existing Group methods for landing latency
    - Reuse your existing GroupDataAnalyzer methods for secondary contact
    - Save fly-level tables and summary statistics to CSV

    Primary tests:
    - WT / independent groups:
        fly-level RMST difference + bootstrap CI + permutation p-value
    - Optogenetic paired ON/OFF:
        paired fly-level RMST difference + bootstrap CI + sign-flip p-value

    Secondary tests:
    - Mann-Whitney U (unpaired fly-level RMST)
    - Wilcoxon signed-rank (paired fly-level RMST)
    - Clustered Cox model at trial level (optional)
    """

    def __init__(self, tau=0.71, random_state=0, platform_offset=0.07, radius=0.07, fps=250):
        self.tau = tau
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.calculator = ku.SimpleCalculation()
        self.analyzer = ku.GroupDataAnalyzer(platform_offset=platform_offset, radius=radius, FPS=fps)

    # ------------------------------------------------------------
    # Preparation helpers
    # ------------------------------------------------------------
    def prepare_group(self, group_info, chr_data=False, use_opto_filter=False):
        if len(group_info.trial_metadata) == 0:
            if chr_data:
                group_info.initialize_Chr_manual_data()
            else:
                group_info.initialize_manual_data()

        if use_opto_filter:
            group_info.filter_opto_data()
        else:
            group_info.filter_nan_fly()

    # ------------------------------------------------------------
    # Trial-level dataframe builders
    # ------------------------------------------------------------
    def get_landing_trial_df(self, group_info, chr_data=False, use_opto_filter=False):
        """
        Reuse your existing Group.get_LL(return_df=True).
        Returns one row per trial with columns like:
        Fly#, Trial#, Latency, Event, Group_Name, TrialType, Light
        """
        self.prepare_group(group_info, chr_data=chr_data, use_opto_filter=use_opto_filter)
        df = group_info.get_LL(return_df=True).copy()
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["Fly#", "Trial#", "Latency", "Event", "Group_Name", "TrialType", "Light"])
        return df

    def get_secondary_contact_trial_df(
        self,
        group_info,
        radius=0.4,
        threshold=None,
        chr_data=False,
        use_opto_filter=False,
        force_recompute=False,
        filename="",
        index_to_iterate=None,
    ):
        """
        Reuse your existing GroupDataAnalyzer.get_secondary_contact_kmc_df().
        Output columns are standardized to:
        Fly#, Trial#, Latency, Event, Group_Name, Light
        """
        if threshold is None:
            threshold = self.tau

        self.prepare_group(group_info, chr_data=chr_data, use_opto_filter=use_opto_filter)
        group_info.read_kinematic_data(["Landing", "Flying"])

        if index_to_iterate is None:
            index_to_iterate = group_info.get_targeted_trials(["Landing", "Flying"])

        raw = self.analyzer.get_secondary_contact_kmc_df(
            group_info=group_info,
            index_to_iterate=index_to_iterate,
            radius=radius,
            threshold=threshold,
            filename=filename,
            force_recompute=force_recompute,
        )

        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["Fly#", "Trial#", "Latency", "Event", "Group_Name", "Light"])

        df = raw.copy()

        # Existing SC code uses lower-case latency/event and tuple-like Index column.
        if "latency" in df.columns and "Latency" not in df.columns:
            df["Latency"] = df["latency"]
        if "event" in df.columns and "Event" not in df.columns:
            df["Event"] = df["event"]

        if "Index" not in df.columns:
            raise ValueError("Secondary-contact dataframe must contain an 'Index' column with (Fly, Trial).")

        flys, trials, lights = [], [], []
        for idx in df["Index"]:
            fly, trial = self.calculator.parse_index_cell(idx)
            flys.append(fly)
            trials.append(trial)
            meta = group_info.trial_metadata[group_info._trial_key(fly, trial)]
            lights.append(meta["Light"])

        out = pd.DataFrame({
            "Fly#": flys,
            "Trial#": trials,
            "Latency": df["Latency"].astype(float),
            "Event": df["Event"].astype(int),
            "Group_Name": group_info.group_name,
            "Light": lights,
        })
        return out

    # ------------------------------------------------------------
    # RMST helpers
    # ------------------------------------------------------------
    def compute_fly_rmst(self, trial_df, fly_col="Fly#", time_col="Latency", event_col="Event"):
        kmf = KaplanMeierFitter()
        rows = []

        if trial_df is None or len(trial_df) == 0:
            return pd.DataFrame(columns=["Fly#", "RMST", "n_trials", "n_events", "event_fraction"])

        for fly, sub in trial_df.groupby(fly_col):
            kmf.fit(sub[time_col], event_observed=sub[event_col])
            rows.append({
                "Fly#": fly,
                "RMST": float(restricted_mean_survival_time(kmf, t=self.tau)),
                "n_trials": int(len(sub)),
                "n_events": int(sub[event_col].sum()),
                "event_fraction": float(sub[event_col].mean()),
            })

        return pd.DataFrame(rows).sort_values("Fly#").reset_index(drop=True)

    def _permutation_test_unpaired(self, x, y, n_perm=10000):
        """
        Primary p-value test for independent groups.
        Uses fly-level RMST or LP values.
        """
        return self.calculator._permutation_test_unpaired(
            x,
            y,
            n_perm=n_perm,
            rng=self.rng,
            return_distribution=True
        )

    def _signflip_test_paired(self, diff, n_perm=10000):
        """
        Primary p-value test for paired ON/OFF data.
        Uses fly-level RMST differences.
        """
        return self.calculator.paired_signflip_diff_test(
            diff,
            n_perm=n_perm,
            rng=self.rng,
            return_distribution=True
        )

    # ------------------------------------------------------------
    # Core comparisons
    # ------------------------------------------------------------
    def compare_unpaired_groups(self, df_a, df_b, out_prefix, label_a=None, label_b=None, n_perm=10000):
        """
        Primary analysis for WT / independent groups.
        Uses fly-level RMST as the statistical unit.
        Returns p-value from permutation test only.
        """
        fly_a = self.compute_fly_rmst(df_a)
        fly_b = self.compute_fly_rmst(df_b)

        x = fly_a["RMST"].values
        y = fly_b["RMST"].values

        observed_diff, perm_p, perm_dist = self._permutation_test_unpaired(x, y, n_perm=n_perm)

        if label_a is None:
            print(df_a)
            label_a = str(df_a["Group_Name"].iloc[0])
        if label_b is None:
            print(df_b)
            label_b = str(df_b["Group_Name"].iloc[0])

        fly_a = fly_a.copy()
        fly_b = fly_b.copy()
        fly_a["Group"] = label_a
        fly_b["Group"] = label_b
        fly_table = pd.concat([fly_a, fly_b], ignore_index=True)

        summary = pd.DataFrame([{
            "comparison_type": "unpaired_fly_level_rmst",
            "group_a": label_a,
            "group_b": label_b,
            "n_fly_a": len(fly_a),
            "n_fly_b": len(fly_b),
            "mean_rmst_a": np.mean(x),
            "mean_rmst_b": np.mean(y),
            "estimate_b_minus_a": observed_diff,
            "permutation_p": perm_p,
            "tau": self.tau,
        }])

        fly_table.to_csv(f"{out_prefix}-fly_rmst.csv", index=False)
        summary.to_csv(f"{out_prefix}-summary.csv", index=False)

        return summary, fly_table

    def compare_paired_opto(self, trial_df, out_prefix, on_label="ON", off_label="OFF", n_perm=10000):
        """
        Primary analysis for paired ON/OFF optogenetic data.
        Uses paired fly-level RMST difference.
        Returns p-value from sign-flip test only.
        """
        kmf = KaplanMeierFitter()
        rows = []

        for fly, sub in trial_df.groupby("Fly#"):
            on_df = sub[sub["Light"] == on_label]
            off_df = sub[sub["Light"] == off_label]

            if len(on_df) == 0 or len(off_df) == 0:
                continue

            kmf.fit(off_df["Latency"], event_observed=off_df["Event"])
            rmst_off = float(restricted_mean_survival_time(kmf, t=self.tau))

            kmf.fit(on_df["Latency"], event_observed=on_df["Event"])
            rmst_on = float(restricted_mean_survival_time(kmf, t=self.tau))

            rows.append({
                "Fly#": fly,
                "RMST_OFF": rmst_off,
                "RMST_ON": rmst_on,
                "Diff_ON_minus_OFF": rmst_on - rmst_off,
                "n_trials_OFF": len(off_df),
                "n_trials_ON": len(on_df),
                "n_events_OFF": int(off_df["Event"].sum()),
                "n_events_ON": int(on_df["Event"].sum()),
            })

        paired = pd.DataFrame(rows).sort_values("Fly#").reset_index(drop=True)

        if len(paired) == 0:
            raise ValueError("No paired ON/OFF flies found after filtering.")

        diff = paired["Diff_ON_minus_OFF"].values
        observed_diff, signflip_p, perm_dist = self._signflip_test_paired(diff, n_perm=n_perm)

        summary = pd.DataFrame([{
            "comparison_type": "paired_fly_level_rmst",
            "group": str(trial_df["Group_Name"].iloc[0]),
            "n_paired_flies": len(paired),
            "mean_rmst_off": paired["RMST_OFF"].mean(),
            "mean_rmst_on": paired["RMST_ON"].mean(),
            "estimate_on_minus_off": observed_diff,
            "signflip_p": signflip_p,
            "tau": self.tau,
        }])

        paired.to_csv(f"{out_prefix}-paired_fly_rmst.csv", index=False)
        summary.to_csv(f"{out_prefix}-summary.csv", index=False)

        return summary, paired

    # ------------------------------------------------------------
    # Convenience wrappers: landing
    # ------------------------------------------------------------
    def analyze_landing_unpaired(self, group_a, group_b, out_prefix, chr_data=False, n_perm=10000):
        df_a = self.get_landing_trial_df(group_a, chr_data=chr_data, use_opto_filter=False)
        df_b = self.get_landing_trial_df(group_b, chr_data=chr_data, use_opto_filter=False)
        return self.compare_unpaired_groups(
            df_a=df_a,
            df_b=df_b,
            out_prefix=out_prefix,
            label_a=group_a.group_name,
            label_b=group_b.group_name,
            n_perm=n_perm,
        )

    def analyze_landing_opto(self, group_info, out_prefix, chr_data=False, n_perm=10000):
        df = self.get_landing_trial_df(group_info, chr_data=chr_data, use_opto_filter=True)
        return self.compare_paired_opto(df, out_prefix=out_prefix, n_perm=n_perm)

    # ------------------------------------------------------------
    # Convenience wrappers: secondary contact
    # ------------------------------------------------------------
    def analyze_secondary_unpaired(
        self,
        group_a,
        group_b,
        out_prefix,
        radius_a=0.4,
        radius_b=0.4,
        chr_data=False,
        n_perm=10000,
        force_recompute=False,
    ):
        df_a = self.get_secondary_contact_trial_df(
            group_a,
            radius=radius_a,
            chr_data=chr_data,
            use_opto_filter=False,
            force_recompute=force_recompute,
        )
        df_b = self.get_secondary_contact_trial_df(
            group_b,
            radius=radius_b,
            chr_data=chr_data,
            use_opto_filter=False,
            force_recompute=force_recompute,
        )
        return self.compare_unpaired_groups(
            df_a=df_a,
            df_b=df_b,
            out_prefix=out_prefix,
            label_a=group_a.group_name,
            label_b=group_b.group_name,
            n_perm=n_perm,
        )

    def analyze_secondary_opto(
        self,
        group_info,
        out_prefix,
        radius=0.4,
        chr_data=False,
        n_perm=10000,
        force_recompute=False,
    ):
        self.prepare_group(group_info, chr_data=chr_data, use_opto_filter=True)
        on_index, off_index = group_info.get_ON_OFF_index()

        on_df = self.get_secondary_contact_trial_df(
            group_info,
            radius=radius,
            chr_data=chr_data,
            use_opto_filter=True,
            force_recompute=force_recompute,
            filename="ON",
            index_to_iterate=on_index,
        )
        off_df = self.get_secondary_contact_trial_df(
            group_info,
            radius=radius,
            chr_data=chr_data,
            use_opto_filter=True,
            force_recompute=force_recompute,
            filename="OFF",
            index_to_iterate=off_index,
        )

        trial_df = pd.concat([off_df, on_df], ignore_index=True)
        return self.compare_paired_opto(trial_df, out_prefix=out_prefix, n_perm=n_perm)

    def compare_lp_unpaired(self, group_a, group_b, out_prefix, label_a=None, label_b=None, n_perm=10000):
        """
        Primary p-value test for independent-group landing probability.
        Uses one landing probability value per fly.
        """
        if len(group_a.trial_metadata) == 0:
            group_a.initialize_manual_data()
        group_a.filter_nan_fly()

        if len(group_b.trial_metadata) == 0:
            group_b.initialize_manual_data()
        group_b.filter_nan_fly()

        df_a = group_a.get_LP_df()
        df_b = group_b.get_LP_df()

        x = df_a["LandingProb"].values
        y = df_b["LandingProb"].values

        observed_diff, p_value, perm_dist = self._permutation_test_unpaired(x, y, n_perm=n_perm)

        if label_a is None:
            label_a = group_a.group_name
        if label_b is None:
            label_b = group_b.group_name

        fly_table = pd.concat([
            df_a.assign(Group=label_a),
            df_b.assign(Group=label_b)
        ], ignore_index=True)

        summary = pd.DataFrame([{
            "comparison_type": "unpaired_landing_probability",
            "group_a": label_a,
            "group_b": label_b,
            "n_fly_a": len(df_a),
            "n_fly_b": len(df_b),
            "mean_lp_a": np.mean(x),
            "mean_lp_b": np.mean(y),
            "estimate_b_minus_a": observed_diff,
            "permutation_p": p_value
        }])

        fly_table.to_csv(f"{out_prefix}-lp_fly_values.csv", index=False)
        summary.to_csv(f"{out_prefix}-lp_summary.csv", index=False)

        return summary, fly_table

    def compare_lp_paired(self, group_info, out_prefix, on_label="ON", off_label="OFF", n_perm=10000):
        """
        Primary p-value test for paired ON/OFF landing probability.
        Uses one LP_ON and one LP_OFF per fly.
        """

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
        group_info.filter_opto_data()

        combined_df = group_info.get_paired_LP_df().copy()

        combined_df["Group_Name"] = pd.Categorical(
            combined_df["Group_Name"],
            categories=[off_label, on_label],
            ordered=True
        )
        combined_df = combined_df.sort_values(["Fly#", "Group_Name"])

        paired = combined_df.pivot(index="Fly#", columns="Group_Name", values="LandingProb")
        paired = paired.dropna(subset=[off_label, on_label]).reset_index()

        paired["Diff_ON_minus_OFF"] = paired[on_label] - paired[off_label]

        diff = paired["Diff_ON_minus_OFF"].values
        observed_diff, p_value, perm_dist = self._signflip_test_paired(diff, n_perm=n_perm)

        summary = pd.DataFrame([{
            "comparison_type": "paired_landing_probability",
            "group": group_info.group_name,
            "n_paired_flies": len(paired),
            "mean_lp_off": paired[off_label].mean(),
            "mean_lp_on": paired[on_label].mean(),
            "estimate_on_minus_off": observed_diff,
            "signflip_p": p_value
        }])

        paired.to_csv(f"{out_prefix}-lp_paired_fly_values.csv", index=False)
        summary.to_csv(f"{out_prefix}-lp_summary.csv", index=False)

        return summary, paired
