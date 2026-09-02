import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time


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
        # tau is the right-censoring horizon used for KM/RMST analyses.
        self.tau = tau
        self.random_state = random_state
        # Keep a runner-local RNG so permutation tests are reproducible without
        # changing global NumPy random state.
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Preparation helpers
    # ------------------------------------------------------------
    def prepare_group(self, group_info, chr_data=False, use_opto_filter=False):
        # Initialize the group's manual metadata only when it has not already
        # been loaded by an upstream notebook/script.
        if len(group_info.trial_metadata) == 0:
            if chr_data:
                group_info.initialize_Chr_manual_data()
            else:
                group_info.initialize_manual_data()

        # Choose the filter that matches the planned comparison: optogenetic
        # paired analyses keep ON/OFF structure; regular analyses remove NaN fly
        # records.
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
        # Prepare metadata/filtering first, then ask the Group object for the
        # existing landing-latency KM table.
        self.prepare_group(group_info, chr_data=chr_data, use_opto_filter=use_opto_filter)
        df = group_info.get_LL(return_df=True).copy()
        # Return a schema-stable empty DataFrame if no rows are available.
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["Fly#", "Trial#", "Latency", "Event", "Group_Name", "TrialType", "Light"])
        return df

    # ------------------------------------------------------------
    # RMST helpers
    # ------------------------------------------------------------
    def compute_fly_rmst(self, trial_df, fly_col="Fly#", time_col="Latency", event_col="Event"):
        # Compute one restricted mean survival time value per fly. The fly is
        # the statistical unit for the runner's primary group comparisons.
        kmf = KaplanMeierFitter()
        rows = []

        # Keep a stable output schema when callers pass an empty trial table.
        if trial_df is None or len(trial_df) == 0:
            return pd.DataFrame(columns=["Fly#", "RMST", "n_trials", "n_events", "event_fraction"])

        for fly, sub in trial_df.groupby(fly_col):
            # Fit a fly-specific KM curve and integrate it up to tau.
            kmf.fit(sub[time_col], event_observed=sub[event_col])
            rows.append({
                "Fly#": fly,
                "RMST": float(restricted_mean_survival_time(kmf, t=self.tau)),
                "n_trials": int(len(sub)),
                "n_events": int(sub[event_col].sum()),
                "event_fraction": float(sub[event_col].mean()),
            })

        return pd.DataFrame(rows).sort_values("Fly#").reset_index(drop=True)

    def flywise_rmst(
            self,
            trial_df,
            group_cols=None,
            fly_col="Fly#",
            duration_col="Duration",
            event_col="Event",
            value_name="RMST",
            tau=None
    ):
        # Compute one RMST value for each fly within optional grouping columns
        # such as Contact_Group/Leg or optogenetic condition.
        tau = self.tau if tau is None else tau
        group_cols = [] if group_cols is None else list(group_cols)
        output_cols = group_cols + [fly_col, value_name, "n_trials", "n_events", "event_fraction"]
        if trial_df is None or len(trial_df) == 0:
            return pd.DataFrame(columns=output_cols)

        kmf = KaplanMeierFitter()
        rows = []
        groupby_cols = group_cols + [fly_col]
        for group_key, sub in trial_df.groupby(groupby_cols):
            # Normalize pandas' scalar group key when only Fly# is grouped.
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(groupby_cols, group_key))
            kmf.fit(sub[duration_col], event_observed=sub[event_col])
            row.update({
                value_name: float(restricted_mean_survival_time(kmf, t=tau)),
                "n_trials": int(len(sub)),
                "n_events": int(sub[event_col].sum()),
                "event_fraction": float(sub[event_col].mean()),
            })
            rows.append(row)
        return pd.DataFrame(rows).sort_values(groupby_cols).reset_index(drop=True)

    def _permutation_test_unpaired(self, x, y, n_perm=10000):
        """
        Primary p-value test for independent groups.
        Uses fly-level RMST or LP values.
        """
        # Keep the primitive unpaired permutation kernel in this stats module so
        # kinematic utilities no longer own inferential p-value logic.
        x = self._clean_values(x)
        y = self._clean_values(y)
        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan, np.asarray([])

        # The observed effect is mean(group B) - mean(group A), matching the
        # legacy convention used throughout exported stat tables.
        observed = float(np.mean(y) - np.mean(x))
        pooled = np.concatenate([x, y])
        n_x = len(x)
        perm_stats = np.empty(n_perm, dtype=float)
        for perm_i in range(n_perm):
            # Shuffle labels while preserving the original group sizes.
            permuted = self.rng.permutation(pooled)
            perm_stats[perm_i] = float(np.mean(permuted[n_x:]) - np.mean(permuted[:n_x]))

        # Use the +1 correction so exact zero p-values are not reported.
        p_value = float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1))
        return observed, p_value, perm_stats

    def _signflip_test_paired(self, diff, n_perm=10000):
        """
        Primary p-value test for paired ON/OFF data.
        Uses fly-level RMST differences.
        """
        # Keep the primitive paired sign-flip kernel in this stats module so it
        # can serve both precomputed differences and paired value arrays.
        diff = self._clean_values(diff)
        if len(diff) == 0:
            return np.nan, np.nan, np.asarray([])

        # The observed effect is the mean paired difference.
        observed = float(np.mean(diff))
        perm_stats = np.empty(n_perm, dtype=float)
        for perm_i in range(n_perm):
            # Randomly flip the sign of each paired difference under the null
            # hypothesis that the direction of each pair is exchangeable.
            signs = self.rng.choice([-1, 1], size=len(diff), replace=True)
            perm_stats[perm_i] = float(np.mean(diff * signs))

        # Use the same two-sided +1 corrected p-value convention as unpaired tests.
        p_value = float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1))
        return observed, p_value, perm_stats

    # ------------------------------------------------------------
    # Shared inferential-test APIs
    # ------------------------------------------------------------
    def _clean_values(self, values):
        # Convert arbitrary numeric-like input into finite float values only.
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    def _mean_std(self, values):
        # Use sample standard deviation when at least two observations exist.
        values = self._clean_values(values)
        return {
            "Mean": np.nan if len(values) == 0 else float(np.mean(values)),
            "Std": np.nan if len(values) < 2 else float(np.std(values, ddof=1)),
        }

    def p_to_significance(self, p_value, missing_label="n.s."):
        # Centralize figure p-value labels so plots use one threshold convention.
        if pd.isna(p_value):
            return missing_label
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "n.s."

    def format_p_value(self, p_value):
        # Centralize compact p-value formatting for plot titles and annotations.
        if pd.isna(p_value):
            return "p=NA"
        if p_value < 0.001:
            return "p<0.001"
        return f"p={p_value:.3f}"

    def format_rho_value(self, rho):
        # Keep Spearman rho annotation formatting consistent across figures.
        if pd.isna(rho):
            return "rho=NA"
        return f"rho={rho:.2f}"

    def unpaired_permutation_test(
            self,
            values_a,
            values_b,
            group_a,
            group_b,
            metric,
            n_perm=10000,
            n_fly_a=None,
            n_fly_b=None,
            n_trials_a=None,
            n_trials_b=None,
            test_name="unpaired_permutation"
    ):
        # Compare independent group values with a two-sided label-shuffle test.
        values_a = self._clean_values(values_a)
        values_b = self._clean_values(values_b)
        stats_a = self._mean_std(values_a)
        stats_b = self._mean_std(values_b)
        row = {
            # Save one canonical row so plotting functions do not need legacy
            # aliases for the same statistical fields.
            "comparison": f"{group_a} vs {group_b}",
            "test": test_name,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "N_a": int(len(values_a)) if n_fly_a is None else int(n_fly_a),
            "mean_a": stats_a["Mean"],
            "std_a": stats_a["Std"],
            "N_b": int(len(values_b)) if n_fly_b is None else int(n_fly_b),
            "mean_b": stats_b["Mean"],
            "std_b": stats_b["Std"],
            "n_a": np.nan if n_trials_a is None else int(n_trials_a),
            "n_b": np.nan if n_trials_b is None else int(n_trials_b),
            "mean_diff_b_minus_a": np.nan,
            "p_value": np.nan,
            "n_perm": int(n_perm),
            "n_pairwise_comparison": 1,
        }
        if len(values_a) > 0 and len(values_b) > 0:
            # Run the local primitive kernel and standardize the returned stat
            # table schema for all plotting modules.
            observed, p_value, _ = self._permutation_test_unpaired(
                values_a,
                values_b,
                n_perm=n_perm
            )
            row["mean_diff_b_minus_a"] = float(observed)
            row["p_value"] = float(p_value)
        return pd.DataFrame([row])

    def paired_signflip_test(
            self,
            values_a,
            values_b,
            group_a,
            group_b,
            metric,
            n_perm=10000,
            n_trials_a=None,
            n_trials_b=None,
            test_name="paired_signflip_permutation"
    ):
        # Compare paired group values by randomly flipping within-pair differences.
        values_a = np.asarray(values_a, dtype=float)
        values_b = np.asarray(values_b, dtype=float)
        valid = np.isfinite(values_a) & np.isfinite(values_b)
        paired_a = values_a[valid]
        paired_b = values_b[valid]
        stats_a = self._mean_std(paired_a)
        stats_b = self._mean_std(paired_b)
        row = {
            # Paired summaries use N_paired because the fly pair, not each
            # condition separately, is the statistical unit.
            "comparison": f"{group_a} vs {group_b}",
            "test": test_name,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "N_paired": int(len(paired_a)),
            "mean_a": stats_a["Mean"],
            "std_a": stats_a["Std"],
            "mean_b": stats_b["Mean"],
            "std_b": stats_b["Std"],
            "n_a": np.nan if n_trials_a is None else int(n_trials_a),
            "n_b": np.nan if n_trials_b is None else int(n_trials_b),
            "mean_diff_b_minus_a": np.nan,
            "p_value": np.nan,
            "n_perm": int(n_perm),
            "n_pairwise_comparison": 1,
        }
        if len(paired_a) > 0:
            # The observed effect is mean(values_b - values_a), matching the
            # existing paired sign-flip implementation.
            diff = paired_b - paired_a
            observed, p_value, _ = self._signflip_test_paired(diff, n_perm=n_perm)
            row["mean_diff_b_minus_a"] = float(observed)
            row["p_value"] = float(p_value)
        return pd.DataFrame([row])

    def trial_label_shuffle_binary_rate_test(
            self,
            trial_df,
            group_col,
            outcome_col,
            group_a,
            group_b,
            metric,
            n_perm=10000
    ):
        # Shuffle binary outcomes across trial labels while preserving group sizes.
        sub_a = trial_df[trial_df[group_col] == group_a]
        sub_b = trial_df[trial_df[group_col] == group_b]
        outcomes_a = self._clean_values(sub_a[outcome_col].to_numpy(dtype=float))
        outcomes_b = self._clean_values(sub_b[outcome_col].to_numpy(dtype=float))
        combined = np.concatenate([outcomes_a, outcomes_b])
        observed = np.nan
        p_value = np.nan
        if len(outcomes_a) > 0 and len(outcomes_b) > 0:
            # Observed effect is group B landing/event probability minus group A.
            observed = float(np.mean(outcomes_b) - np.mean(outcomes_a))
            perm_stats = np.empty(n_perm, dtype=float)
            n_a = len(outcomes_a)
            for perm_i in range(n_perm):
                permuted = self.rng.permutation(combined)
                perm_stats[perm_i] = float(np.mean(permuted[n_a:]) - np.mean(permuted[:n_a]))
            p_value = float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1))
        return pd.DataFrame([{
            # Binary-rate tests report both trial counts and fly counts because
            # the permutation is trial-level while flies describe sampling.
            "comparison": f"{group_a} vs {group_b}",
            "test": "trial_label_shuffle_binary_rate",
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "N_a": int(sub_a["Fly#"].nunique()) if "Fly#" in sub_a else np.nan,
            "N_b": int(sub_b["Fly#"].nunique()) if "Fly#" in sub_b else np.nan,
            "n_a": int(len(outcomes_a)),
            "n_b": int(len(outcomes_b)),
            "success_a": int(np.nansum(outcomes_a)),
            "success_b": int(np.nansum(outcomes_b)),
            "mean_a": np.nan if len(outcomes_a) == 0 else float(np.mean(outcomes_a)),
            "std_a": np.nan if len(outcomes_a) < 2 else float(np.std(outcomes_a, ddof=1)),
            "mean_b": np.nan if len(outcomes_b) == 0 else float(np.mean(outcomes_b)),
            "std_b": np.nan if len(outcomes_b) < 2 else float(np.std(outcomes_b, ddof=1)),
            "mean_diff_b_minus_a": observed,
            "p_value": p_value,
            "n_perm": int(n_perm),
            "n_pairwise_comparison": 1,
        }])

    def spearman_correlation_test(self, x, y, group_name, metric_x, metric_y):
        # Run a trial-level Spearman correlation after dropping non-finite pairs.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid]
        y_clean = y[valid]
        rho = np.nan
        p_value = np.nan
        if len(x_clean) >= 3 and len(np.unique(x_clean)) >= 2 and len(np.unique(y_clean)) >= 2:
            rho, p_value = spearmanr(x_clean, y_clean)
            rho = float(rho)
            p_value = float(p_value)
        return pd.DataFrame([{
            # Correlations are not pairwise group tests, so use a compact
            # correlation-specific schema instead of group A/B summaries.
            "test": "spearman_correlation",
            "metric_x": metric_x,
            "metric_y": metric_y,
            "group": group_name,
            "N": int(len(x_clean)),
            "rho": rho,
            "p_value": p_value,
        }])

    def logrank_latency_test(
            self,
            trial_df,
            group_col,
            duration_col,
            event_col,
            group_a,
            group_b,
            metric,
            tau=None
    ):
        # Compare two KM latency distributions; censored rows remain included.
        sub_a = trial_df[trial_df[group_col] == group_a]
        sub_b = trial_df[trial_df[group_col] == group_b]
        row = {
            # KM/log-rank rows retain trial and event counts because censored
            # trials contribute to the survival comparison.
            "comparison": f"{group_a} vs {group_b}",
            "test": "logrank",
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "N_a": int(sub_a["Fly#"].nunique()) if "Fly#" in sub_a else np.nan,
            "n_a": int(len(sub_a)),
            "events_a": int(sub_a[event_col].sum()) if not sub_a.empty else 0,
            "mean_a": np.nan if sub_a.empty else float(sub_a[duration_col].mean()),
            "std_a": np.nan if len(sub_a) < 2 else float(sub_a[duration_col].std(ddof=1)),
            "N_b": int(sub_b["Fly#"].nunique()) if "Fly#" in sub_b else np.nan,
            "n_b": int(len(sub_b)),
            "events_b": int(sub_b[event_col].sum()) if not sub_b.empty else 0,
            "mean_b": np.nan if sub_b.empty else float(sub_b[duration_col].mean()),
            "std_b": np.nan if len(sub_b) < 2 else float(sub_b[duration_col].std(ddof=1)),
            "mean_diff_b_minus_a": np.nan,
            "p_value": np.nan,
            "n_perm": np.nan,
            "tau": np.nan if tau is None else float(tau),
            "n_pairwise_comparison": 1,
        }
        if not sub_a.empty and not sub_b.empty:
            result = logrank_test(
                sub_a[duration_col],
                sub_b[duration_col],
                event_observed_A=sub_a[event_col],
                event_observed_B=sub_b[event_col]
            )
            row["mean_diff_b_minus_a"] = row["mean_b"] - row["mean_a"]
            row["p_value"] = float(result.p_value)
        return pd.DataFrame([row])

    def pairwise_flywise_rmst_permutation(
            self,
            fly_df,
            group_col,
            value_col,
            trial_count_col,
            group_pairs,
            metric,
            n_perm=10000
    ):
        # Run only requested between-group fly-wise RMST comparisons.
        rows = []
        # Store the planned comparison count so downstream CSVs can apply a
        # correction later without changing the raw p values saved here.
        pairwise_comparison_count = len(group_pairs)
        for group_a, group_b in group_pairs:
            sub_a = fly_df[fly_df[group_col] == group_a]
            sub_b = fly_df[fly_df[group_col] == group_b]
            stat_df = self.unpaired_permutation_test(
                sub_a[value_col].to_numpy(dtype=float),
                sub_b[value_col].to_numpy(dtype=float),
                group_a=group_a,
                group_b=group_b,
                metric=metric,
                n_perm=n_perm,
                n_trials_a=sub_a[trial_count_col].sum() if trial_count_col in sub_a else None,
                n_trials_b=sub_b[trial_count_col].sum() if trial_count_col in sub_b else None,
                test_name="pairwise_flywise_rmst_unpaired_permutation"
            )
            row = stat_df.iloc[0].to_dict()
            # Store the correction-family size without altering the raw p value.
            row["n_pairwise_comparison"] = pairwise_comparison_count
            rows.append(row)
        return pd.DataFrame(rows)

    def vector_label_shuffle_test(
            self,
            vectors_a,
            vectors_b,
            group_a,
            group_b,
            metric,
            n_perm=10000,
            n_trials_a=None,
            n_trials_b=None,
            test_name="vector_label_shuffle"
    ):
        # Compare two groups of 2D vectors by shuffling labels and testing the
        # distance between group mean vectors.
        vectors_a = np.asarray(vectors_a, dtype=float)
        vectors_b = np.asarray(vectors_b, dtype=float)
        valid_a = np.all(np.isfinite(vectors_a), axis=1) if vectors_a.ndim == 2 else np.asarray([], dtype=bool)
        valid_b = np.all(np.isfinite(vectors_b), axis=1) if vectors_b.ndim == 2 else np.asarray([], dtype=bool)
        vectors_a = vectors_a[valid_a]
        vectors_b = vectors_b[valid_b]
        row = {
            # Vector tests need component-wise means/stds instead of ambiguous
            # scalar mean_a/std_a fields.
            "comparison": f"{group_a} vs {group_b}",
            "test": test_name,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "N_a": int(len(vectors_a)),
            "mean_x_a": np.nan,
            "mean_y_a": np.nan,
            "std_x_a": np.nan,
            "std_y_a": np.nan,
            "N_b": int(len(vectors_b)),
            "mean_x_b": np.nan,
            "mean_y_b": np.nan,
            "std_x_b": np.nan,
            "std_y_b": np.nan,
            "n_a": np.nan if n_trials_a is None else int(n_trials_a),
            "n_b": np.nan if n_trials_b is None else int(n_trials_b),
            "vector_distance": np.nan,
            "delta_x_b_minus_a": np.nan,
            "delta_y_b_minus_a": np.nan,
            "p_value": np.nan,
            "n_perm": int(n_perm),
            "n_pairwise_comparison": 1,
        }
        if len(vectors_a) == 0 or len(vectors_b) == 0:
            return pd.DataFrame([row])

        # Observed statistic is the Euclidean distance between mean vectors.
        mean_a = np.mean(vectors_a, axis=0)
        mean_b = np.mean(vectors_b, axis=0)
        delta = mean_b - mean_a
        observed_distance = float(np.hypot(delta[0], delta[1]))
        pooled = np.vstack([vectors_a, vectors_b])
        n_a = len(vectors_a)
        perm_stats = np.empty(n_perm, dtype=float)
        for perm_i in range(n_perm):
            # Shuffle full vector rows while preserving the original group sizes.
            permuted = pooled[self.rng.permutation(len(pooled))]
            perm_mean_a = np.mean(permuted[:n_a], axis=0)
            perm_mean_b = np.mean(permuted[n_a:], axis=0)
            perm_delta = perm_mean_b - perm_mean_a
            perm_stats[perm_i] = float(np.hypot(perm_delta[0], perm_delta[1]))

        # The vector-distance test is one-sided on distance magnitude.
        p_value = float((np.sum(perm_stats >= observed_distance) + 1) / (n_perm + 1))
        row.update({
            # Store the observed 2D mean vectors and their separation as the
            # effect-size fields for the secondary vector test.
            "mean_x_a": float(mean_a[0]),
            "mean_y_a": float(mean_a[1]),
            "std_x_a": np.nan if len(vectors_a) < 2 else float(np.std(vectors_a[:, 0], ddof=1)),
            "std_y_a": np.nan if len(vectors_a) < 2 else float(np.std(vectors_a[:, 1], ddof=1)),
            "mean_x_b": float(mean_b[0]),
            "mean_y_b": float(mean_b[1]),
            "std_x_b": np.nan if len(vectors_b) < 2 else float(np.std(vectors_b[:, 0], ddof=1)),
            "std_y_b": np.nan if len(vectors_b) < 2 else float(np.std(vectors_b[:, 1], ddof=1)),
            "vector_distance": observed_distance,
            "delta_x_b_minus_a": float(delta[0]),
            "delta_y_b_minus_a": float(delta[1]),
            "p_value": p_value,
        })
        return pd.DataFrame([row])

    def _circular_mean_rad(self, angles_rad):
        # Average directions on the unit circle so angles near 0/360 are treated
        # as neighboring values rather than opposite ends of a linear axis.
        resultant = np.mean(np.exp(1j * angles_rad))
        if not np.isfinite(resultant.real) or not np.isfinite(resultant.imag) or np.isclose(np.abs(resultant), 0.0):
            return np.nan
        return float(np.angle(resultant))

    def _angular_distance_deg(self, angle_a_rad, angle_b_rad):
        # Return the shortest absolute angular distance between two directions.
        if not np.isfinite(angle_a_rad) or not np.isfinite(angle_b_rad):
            return np.nan
        return float(np.degrees(np.abs(np.angle(np.exp(1j * (angle_b_rad - angle_a_rad))))))

    def _circular_std_deg(self, angles_rad):
        # Use circular standard deviation so angular spread respects wraparound.
        angles_rad = np.asarray(angles_rad, dtype=float)
        angles_rad = angles_rad[np.isfinite(angles_rad)]
        if len(angles_rad) == 0:
            return np.nan
        resultant_length = np.abs(np.mean(np.exp(1j * angles_rad)))
        if resultant_length <= 0:
            return np.nan
        return float(np.degrees(np.sqrt(-2 * np.log(resultant_length))))

    def circular_angle_permutation_test(
            self,
            vectors_a,
            vectors_b,
            group_a,
            group_b,
            n_perm=10000,
            n_trials_a=None,
            n_trials_b=None,
            test_name="primary_circular_angle_permutation"
    ):
        # Compare two groups of fly-level 2D vectors by testing the angular
        # distance between their circular mean directions.
        vectors_a = np.asarray(vectors_a, dtype=float)
        vectors_b = np.asarray(vectors_b, dtype=float)

        # Keep only complete 2D rows because angle calculation requires both
        # projected vector components.
        valid_a = np.all(np.isfinite(vectors_a), axis=1) if vectors_a.ndim == 2 else np.asarray([], dtype=bool)
        valid_b = np.all(np.isfinite(vectors_b), axis=1) if vectors_b.ndim == 2 else np.asarray([], dtype=bool)
        vectors_a = vectors_a[valid_a]
        vectors_b = vectors_b[valid_b]

        # Keep the primary output compact for supplementary statistics tables.
        row = {
            # Circular rows report angle-specific summary statistics rather
            # than linear mean/std values.
            "comparison": f"{group_a} vs {group_b}",
            "test": test_name,
            "metric": "radial_displacement_angle",
            "group_a": group_a,
            "group_b": group_b,
            "N_a": int(len(vectors_a)),
            "circular_mean_a_deg": np.nan,
            "circular_std_a_deg": np.nan,
            "N_b": int(len(vectors_b)),
            "circular_mean_b_deg": np.nan,
            "circular_std_b_deg": np.nan,
            "n_a": np.nan if n_trials_a is None else int(n_trials_a),
            "n_b": np.nan if n_trials_b is None else int(n_trials_b),
            "angular_distance_deg": np.nan,
            "p_value": np.nan,
            "n_perm": int(n_perm),
            "n_pairwise_comparison": 1,
        }
        if len(vectors_a) == 0 or len(vectors_b) == 0:
            return pd.DataFrame([row])

        # Convert vectors to radians with atan2 so direction is computed in the
        # projected radial-displacement coordinate system.
        angles_a = np.arctan2(vectors_a[:, 1], vectors_a[:, 0])
        angles_b = np.arctan2(vectors_b[:, 1], vectors_b[:, 0])
        mean_a = self._circular_mean_rad(angles_a)
        mean_b = self._circular_mean_rad(angles_b)
        observed_distance = self._angular_distance_deg(mean_a, mean_b)
        row["circular_mean_a_deg"] = np.nan if not np.isfinite(mean_a) else float((np.degrees(mean_a) + 360) % 360)
        row["circular_mean_b_deg"] = np.nan if not np.isfinite(mean_b) else float((np.degrees(mean_b) + 360) % 360)
        row["circular_std_a_deg"] = self._circular_std_deg(angles_a)
        row["circular_std_b_deg"] = self._circular_std_deg(angles_b)

        # Shuffle fly-level vector rows while preserving group sizes, then
        # recompute the circular mean-angle distance under the null labels.
        pooled = np.vstack([vectors_a, vectors_b])
        n_a = len(vectors_a)
        perm_stats = np.empty(n_perm, dtype=float)
        for perm_i in range(n_perm):
            permuted = pooled[self.rng.permutation(len(pooled))]
            perm_angles_a = np.arctan2(permuted[:n_a, 1], permuted[:n_a, 0])
            perm_angles_b = np.arctan2(permuted[n_a:, 1], permuted[n_a:, 0])
            perm_mean_a = self._circular_mean_rad(perm_angles_a)
            perm_mean_b = self._circular_mean_rad(perm_angles_b)
            perm_stats[perm_i] = self._angular_distance_deg(perm_mean_a, perm_mean_b)
        perm_stats = perm_stats[np.isfinite(perm_stats)]

        # The primary statistic is angular separation, so the permutation test is
        # right-tailed on larger circular mean-direction differences.
        p_value = np.nan
        if np.isfinite(observed_distance) and len(perm_stats) > 0:
            p_value = float((np.sum(perm_stats >= observed_distance) + 1) / (len(perm_stats) + 1))

        row["angular_distance_deg"] = observed_distance
        row["p_value"] = p_value
        return pd.DataFrame([row])

    def radial_direction_pairwise_tests(
            self,
            fly_vector_df,
            group_col,
            x_col,
            y_col,
            trial_count_col,
            joint_col,
            leg_col,
            group_pairs,
            n_perm=10000
    ):
        # Run the primary circular direction test and secondary 2D mean-vector
        # permutation test for each joint and requested group pair.
        rows = []
        if fly_vector_df is None or len(fly_vector_df) == 0:
            return pd.DataFrame()

        # Each Test type is corrected within the joint-by-group-pair family.
        comparison_count = fly_vector_df[joint_col].nunique() * len(group_pairs)
        for joint, joint_df in fly_vector_df.groupby(joint_col):
            # Resolve the leg label once per joint for a compact supplement row.
            leg_values = joint_df[leg_col].dropna().unique() if leg_col in joint_df else []
            leg = leg_values[0] if len(leg_values) > 0 else str(joint).replace("TT", "")
            for group_a, group_b in group_pairs:
                group_a_df = joint_df[joint_df[group_col] == group_a]
                group_b_df = joint_df[joint_df[group_col] == group_b]
                vectors_a = group_a_df[[x_col, y_col]].to_numpy(dtype=float)
                vectors_b = group_b_df[[x_col, y_col]].to_numpy(dtype=float)
                n_trials_a = group_a_df[trial_count_col].sum() if trial_count_col in group_a_df else np.nan
                n_trials_b = group_b_df[trial_count_col].sum() if trial_count_col in group_b_df else np.nan

                # Primary test: direction-only comparison on circular mean
                # angles derived from each fly's 2D radial displacement vector.
                primary = self.circular_angle_permutation_test(
                    vectors_a,
                    vectors_b,
                    group_a=group_a,
                    group_b=group_b,
                    n_perm=n_perm,
                    n_trials_a=n_trials_a,
                    n_trials_b=n_trials_b,
                    test_name="primary_circular_angle_permutation"
                ).iloc[0].to_dict()
                primary.update({"joint": joint, "leg": leg})
                primary["n_pairwise_comparison"] = comparison_count
                rows.append(primary)

                # Secondary test: original 2D vector mean-distance permutation,
                # retained as a direction-plus-magnitude support analysis.
                secondary = self.vector_label_shuffle_test(
                    vectors_a,
                    vectors_b,
                    group_a=group_a,
                    group_b=group_b,
                    metric="radial_displacement_2d_vector",
                    n_perm=n_perm,
                    n_trials_a=n_trials_a,
                    n_trials_b=n_trials_b,
                    test_name="secondary_2d_vector_mean_permutation"
                ).iloc[0].to_dict()
                rows.append({
                    # Keep the secondary vector row compact but preserve the
                    # vector-specific group summaries from the stats runner.
                    **secondary,
                    "joint": joint,
                    "leg": leg,
                    "n_pairwise_comparison": comparison_count,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Core comparisons
    # ------------------------------------------------------------
    def compare_unpaired_groups(
            self,
            df_a,
            df_b,
            out_prefix,
            label_a=None,
            label_b=None,
            n_perm=10000,
            pairwise_comparison_count=None
    ):
        """
        Primary analysis for WT / independent groups.
        Uses fly-level RMST as the statistical unit.
        Returns p-value from permutation test only.
        """
        # Collapse each trial-level table into fly-level RMST values before
        # testing group differences.
        fly_a = self.compute_fly_rmst(df_a)
        fly_b = self.compute_fly_rmst(df_b)

        x = fly_a["RMST"].values
        y = fly_b["RMST"].values

        # Primary statistic is mean(group B) - mean(group A); p-value comes from
        # an unpaired label-shuffle permutation test.
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

        # Summary table uses the same compact schema as other unpaired
        # permutation outputs, with tau retained because RMST depends on it.
        summary = pd.DataFrame([{
            "comparison": f"{label_a} vs {label_b}",
            "test": "unpaired_fly_level_rmst_permutation",
            "metric": "landing_latency_rmst",
            "group_a": label_a,
            "group_b": label_b,
            "N_a": len(fly_a),
            "mean_a": np.mean(x),
            "std_a": np.nan if len(x) < 2 else float(np.std(x, ddof=1)),
            "N_b": len(fly_b),
            "mean_b": np.mean(y),
            "std_b": np.nan if len(y) < 2 else float(np.std(y, ddof=1)),
            "n_a": int(fly_a["n_trials"].sum()) if "n_trials" in fly_a else np.nan,
            "n_b": int(fly_b["n_trials"].sum()) if "n_trials" in fly_b else np.nan,
            "mean_diff_b_minus_a": observed_diff,
            "p_value": perm_p,
            "n_perm": n_perm,
            "tau": self.tau,
            "n_pairwise_comparison": np.nan if pairwise_comparison_count is None else int(pairwise_comparison_count),
        }])

        # Save both the fly-level input table and the one-row summary table.
        fly_table.to_csv(f"{out_prefix}-fly_rmst.csv", index=False)
        summary.to_csv(f"{out_prefix}-summary.csv", index=False)

        return summary, fly_table

    def compare_paired_opto(self, trial_df, out_prefix, on_label="ON", off_label="OFF", n_perm=10000):
        """
        Primary analysis for paired ON/OFF optogenetic data.
        Uses paired fly-level RMST difference.
        Returns p-value from sign-flip test only.
        """
        # For paired optogenetic data, each fly must have both OFF and ON trial
        # subsets. RMST is calculated separately within each light condition.
        kmf = KaplanMeierFitter()
        rows = []

        for fly, sub in trial_df.groupby("Fly#"):
            # Split this fly's trials by light condition.
            on_df = sub[sub["Light"] == on_label]
            off_df = sub[sub["Light"] == off_label]

            if len(on_df) == 0 or len(off_df) == 0:
                continue

            # OFF RMST is the baseline condition.
            kmf.fit(off_df["Latency"], event_observed=off_df["Event"])
            rmst_off = float(restricted_mean_survival_time(kmf, t=self.tau))

            # ON RMST is compared to OFF within the same fly.
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

        # Test whether paired ON-minus-OFF differences are centered around zero.
        diff = paired["Diff_ON_minus_OFF"].values
        observed_diff, signflip_p, perm_dist = self._signflip_test_paired(diff, n_perm=n_perm)

        # Save one compact paired RMST summary row for the supplement table.
        summary = pd.DataFrame([{
            "comparison": f"{off_label} vs {on_label}",
            "test": "paired_fly_level_rmst_signflip",
            "metric": "landing_latency_rmst",
            "group": str(trial_df["Group_Name"].iloc[0]),
            "group_a": off_label,
            "group_b": on_label,
            "N_paired": len(paired),
            "mean_a": paired["RMST_OFF"].mean(),
            "std_a": np.nan if len(paired) < 2 else float(paired["RMST_OFF"].std(ddof=1)),
            "mean_b": paired["RMST_ON"].mean(),
            "std_b": np.nan if len(paired) < 2 else float(paired["RMST_ON"].std(ddof=1)),
            "n_a": int(paired["n_trials_OFF"].sum()) if "n_trials_OFF" in paired else np.nan,
            "n_b": int(paired["n_trials_ON"].sum()) if "n_trials_ON" in paired else np.nan,
            "mean_diff_b_minus_a": observed_diff,
            "p_value": signflip_p,
            "n_perm": n_perm,
            "tau": self.tau,
            "n_pairwise_comparison": 1,
        }])

        paired.to_csv(f"{out_prefix}-paired_fly_rmst.csv", index=False)
        summary.to_csv(f"{out_prefix}-summary.csv", index=False)

        return summary, paired

    # ------------------------------------------------------------
    # Convenience wrappers: landing
    # ------------------------------------------------------------
    def analyze_landing_unpaired(
            self,
            group_a,
            group_b,
            out_prefix,
            chr_data=False,
            n_perm=10000,
            pairwise_comparison_count=None
    ):
        # Build landing-latency trial tables for two independent groups, then
        # run the shared unpaired fly-level RMST comparison.
        df_a = self.get_landing_trial_df(group_a, chr_data=chr_data, use_opto_filter=False)
        df_b = self.get_landing_trial_df(group_b, chr_data=chr_data, use_opto_filter=False)
        return self.compare_unpaired_groups(
            df_a=df_a,
            df_b=df_b,
            out_prefix=out_prefix,
            label_a=group_a.group_name,
            label_b=group_b.group_name,
            n_perm=n_perm,
            pairwise_comparison_count=pairwise_comparison_count,
        )

    def analyze_landing_opto(self, group_info, out_prefix, chr_data=False, n_perm=10000):
        # Build one ON/OFF landing-latency trial table and run the paired RMST
        # comparison within flies.
        df = self.get_landing_trial_df(group_info, chr_data=chr_data, use_opto_filter=True)
        return self.compare_paired_opto(df, out_prefix=out_prefix, n_perm=n_perm)

    def compare_lp_unpaired(
            self,
            group_a,
            group_b,
            out_prefix,
            label_a=None,
            label_b=None,
            n_perm=10000,
            pairwise_comparison_count=None
    ):
        """
        Primary p-value test for independent-group landing probability.
        Uses one landing probability value per fly.
        """
        # Prepare both groups with the standard non-opto filtering and retrieve
        # fly-level landing probabilities.
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

        # Compare fly-level landing probabilities with the unpaired permutation
        # helper used elsewhere in this runner.
        observed_diff, p_value, perm_dist = self._permutation_test_unpaired(x, y, n_perm=n_perm)

        if label_a is None:
            label_a = group_a.group_name
        if label_b is None:
            label_b = group_b.group_name

        fly_table = pd.concat([
            df_a.assign(Group=label_a),
            df_b.assign(Group=label_b)
        ], ignore_index=True)

        # Save the raw fly-level values and a compact unpaired LP summary row.
        summary = pd.DataFrame([{
            "comparison": f"{label_a} vs {label_b}",
            "test": "unpaired_landing_probability_permutation",
            "metric": "landing_probability",
            "group_a": label_a,
            "group_b": label_b,
            "N_a": len(df_a),
            "mean_a": np.mean(x),
            "std_a": np.nan if len(x) < 2 else float(np.std(x, ddof=1)),
            "N_b": len(df_b),
            "mean_b": np.mean(y),
            "std_b": np.nan if len(y) < 2 else float(np.std(y, ddof=1)),
            "mean_diff_b_minus_a": observed_diff,
            "p_value": p_value,
            "n_perm": n_perm,
            "n_pairwise_comparison": np.nan if pairwise_comparison_count is None else int(pairwise_comparison_count),
        }])

        fly_table.to_csv(f"{out_prefix}-lp_fly_values.csv", index=False)
        summary.to_csv(f"{out_prefix}-lp_summary.csv", index=False)

        return summary, fly_table

    def compare_lp_paired(self, group_info, out_prefix, on_label="ON", off_label="OFF", n_perm=10000):
        """
        Primary p-value test for paired ON/OFF landing probability.
        Uses one LP_ON and one LP_OFF per fly.
        """

        # Prepare optogenetic metadata and get one OFF/ON landing probability
        # value per fly.
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
        group_info.filter_opto_data()

        combined_df = group_info.get_paired_LP_df().copy()

        # Force OFF before ON so the pivot and difference column have a stable
        # interpretation.
        combined_df["Group_Name"] = pd.Categorical(
            combined_df["Group_Name"],
            categories=[off_label, on_label],
            ordered=True
        )
        combined_df = combined_df.sort_values(["Fly#", "Group_Name"])

        paired = combined_df.pivot(index="Fly#", columns="Group_Name", values="LandingProb")
        paired = paired.dropna(subset=[off_label, on_label]).reset_index()

        # Compute the paired effect size for each fly, then run a sign-flip test
        # on those within-fly differences.
        paired["Diff_ON_minus_OFF"] = paired[on_label] - paired[off_label]

        diff = paired["Diff_ON_minus_OFF"].values
        observed_diff, p_value, perm_dist = self._signflip_test_paired(diff, n_perm=n_perm)

        summary = pd.DataFrame([{
            # Paired LP summaries mirror paired RMST summaries but omit tau
            # because ordinary landing probability has no time cutoff.
            "comparison": f"{off_label} vs {on_label}",
            "test": "paired_landing_probability_signflip",
            "metric": "landing_probability",
            "group": group_info.group_name,
            "group_a": off_label,
            "group_b": on_label,
            "N_paired": len(paired),
            "mean_a": paired[off_label].mean(),
            "std_a": np.nan if len(paired) < 2 else float(paired[off_label].std(ddof=1)),
            "mean_b": paired[on_label].mean(),
            "std_b": np.nan if len(paired) < 2 else float(paired[on_label].std(ddof=1)),
            "mean_diff_b_minus_a": observed_diff,
            "p_value": p_value,
            "n_perm": n_perm,
            "n_pairwise_comparison": 1,
        }])

        paired.to_csv(f"{out_prefix}-lp_paired_fly_values.csv", index=False)
        summary.to_csv(f"{out_prefix}-lp_summary.csv", index=False)

        return summary, paired
