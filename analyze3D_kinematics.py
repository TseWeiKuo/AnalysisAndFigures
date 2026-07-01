import os
import warnings


import KinematicPlot as kp
from group_config_new import build_groups
from survival_stats_runner import SurvivalStatsRunner
import pandas as pd
import itertools
import seaborn as sns
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    print(repo_root)
    figures_dir = os.path.join(repo_root, "Figures")
    sc_data_dir = os.path.join(repo_root, "SC data")
    other_landing_data_dir = os.path.join(repo_root, "LandingData", "Others")

    # Build all configured groups
    groups = build_groups()
    os.makedirs(figures_dir, exist_ok=True)
    os.chdir(figures_dir)

    def use_output_folder(*parts):
        """Create and enter a figure-output folder inside this repository."""
        output_folder = os.path.join(*parts)
        os.makedirs(output_folder, exist_ok=True)
        os.chdir(output_folder)
        return output_folder

    # Plotter
    plotter = kp.PlotCreator(
        platform_offset=0.07,
        platform_height=0.1,
        radius=0.03,
        fps=250
    )

    stats_runner = SurvivalStatsRunner(
        tau=0.71,
        random_state=0,
        platform_offset=0.03,
        radius=0.07,
        fps=250
    )

    def test_t2_it_ot_plots_with_old_and_new_kinematic_paths(
            old_kinematic_path,
            new_kinematic_path,
            output_root=None,
            n_perm=20000
    ):
        """
        Run the current T2 TiTa IT/OT plotting calls once with an old kinematic
        root and once with a reorganized/new-name kinematic root.

        This is intended as a parser regression test. It temporarily swaps only
        groups["WT_T2_TTa"].fly_kinematic_data_path, clears loaded Trial objects,
        and restores the original state afterward.
        """
        from kinematic_object import Get3D_path

        if output_root is None:
            output_root = use_output_folder(figures_dir, "T2_IT_OT_old_vs_new_name_parser_test")
        else:
            os.makedirs(output_root, exist_ok=True)

        group_info = groups["WT_T2_TTa"]
        original_path_map = group_info.fly_kinematic_data_path
        original_loaded_trials = group_info.fly_kinematic_data

        it_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_IT_filtered.xlsx")
        ot_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_OT_filtered.xlsx")
        wt_t2_tita_sc_path = os.path.join(sc_data_dir, "WT-T2-TiTa_LegContact.csv")
        behavior_sources = {
            "IT": {
                "path": ot_filtered_ll_path,
                "selection_mode": "numeric",
            },
            "OT": {
                "path": it_filtered_ll_path,
                "selection_mode": "numeric",
            },
        }

        summary_rows = []

        def run_one(label, kinematic_path):
            label_dir = use_output_folder(output_root, label)
            parsed_paths = Get3D_path(kinematic_path, required=True)

            group_info.fly_kinematic_data_path = parsed_paths
            group_info.fly_kinematic_data = {}

            summary_rows.append({
                "label": label,
                "kinematic_path": kinematic_path,
                "parsed_trial_count": len(parsed_paths),
                "first_keys": ",".join(sorted(parsed_paths.keys())[:10]),
            })

            plotter.plot_angle_traces_by_trial_sets(
                group_info=group_info,
                angle_defs=[
                    ["R-mCT", "R-mFT", "R-mTT"],
                ],
                behavior_sources=behavior_sources,
                trial_types=("Landing", "Flying"),
                start_s=-0.1,
                end_s=0.1,
                target_fps=250,
                file_name=f"{label}_WT_T2_TiTa_IT_OT_Rm_FT_angle_trace"
            )

            plotter.plot_it_ot_landing_probability_and_latency(
                group_info=group_info,
                behavior_sources=behavior_sources,
                file_name=f"{label}_WT_T2_TiTa_IT_OT_LP_latency_KM_FT_angle",
                behavior_labels=("IT", "OT"),
                behavior_display_names={
                    "IT": "Inward touch",
                    "OT": "Outward touch",
                },
                trial_types=("Landing", "Flying"),
                tau=0.71,
                n_perm=n_perm,
                contacted_leg="R-m",
                angle_start_s=-0.1,
                angle_end_s=0.1,
                target_fps=250,
                colors={
                    "IT": "#8FD694",
                    "OT": "#C7A0E8",
                }
            )

            plotter.plot_left_TT_path_efficiency_grouped_stripplots(
                group_info=group_info,
                behavior_sources=behavior_sources,
                file_name=f"{label}_WT_T2_TiTa_LhTT_path_efficiency_success_failed_IT_OT",
                legs=("L-h",),
                trial_types=("Landing", "Flying"),
                tau=0.71,
                trajectory_window_mode="mol_adjusted",
                n_perm=n_perm
            )

            plotter.plot_valid_sc_count_vs_landing_latency(
                group_info=group_info,
                sc_csv_path=wt_t2_tita_sc_path,
                file_name=f"{label}_WT_T2_TiTa_valid_SC_count_vs_LL",
                legs=("L-f", "L-m", "L-h"),
                threshold=0.71,
                trial_types=("Landing", "Flying")
            )

            plotter.plot_TT_summary_metrics_vs_LL(
                group_info=group_info,
                legs=("L-h",),
                trial_types=("Landing", "Flying"),
                tau=0.71,
                trajectory_window_mode="mol_adjusted",
                file_name=f"{label}_WT_T2_TiTa_LhTT_summary_metrics_vs_LL_mol_adjusted",
                n_perm=n_perm
            )

            os.chdir(output_root)

        try:
            run_one("old_names", old_kinematic_path)
            run_one("new_names", new_kinematic_path)
        finally:
            group_info.fly_kinematic_data_path = original_path_map
            group_info.fly_kinematic_data = original_loaded_trials
            os.chdir(figures_dir)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_root, "old_vs_new_parser_summary.csv"), index=False)
        return summary_df

    def save_unpaired_lp_ll_stats(group_pairs, out_prefix, n_perm=20000):
        """Save LP and LL pairwise stats using SurvivalStatsRunner."""
        lp_results = []
        ll_results = []
        for key_a, key_b in group_pairs:
            group_a = groups[key_a]
            group_b = groups[key_b]
            comparison_prefix = f"{out_prefix}_{key_a}_vs_{key_b}"

            res_lp, _ = stats_runner.compare_lp_unpaired(
                group_a=group_a,
                group_b=group_b,
                out_prefix=f"{comparison_prefix}_LP",
                n_perm=n_perm
            )
            res_lp["Comparison"] = f"{key_a} vs {key_b}"
            lp_results.append(res_lp)

            res_ll, _ = stats_runner.analyze_landing_unpaired(
                group_a=group_a,
                group_b=group_b,
                out_prefix=f"{comparison_prefix}_LL",
                n_perm=n_perm
            )
            res_ll["Comparison"] = f"{key_a} vs {key_b}"
            ll_results.append(res_ll)

        if lp_results:
            pd.concat(lp_results, ignore_index=True).to_csv(f"{out_prefix}_LP_pairwise_stats.csv", index=False)
        if ll_results:
            pd.concat(ll_results, ignore_index=True).to_csv(f"{out_prefix}_LL_pairwise_fly_RMST_stats.csv", index=False)

    def save_paired_opto_lp_ll_stats(group_keys, out_prefix, chr_data=False, n_perm=20000):
        """Save paired ON/OFF LP and LL stats using SurvivalStatsRunner."""
        lp_results = []
        ll_results = []
        for group_key in group_keys:
            group_info = groups[group_key]

            res_lp, _ = stats_runner.compare_lp_paired(
                group_info=group_info,
                out_prefix=f"{out_prefix}_{group_key}_LP_ON_OFF",
                on_label="ON",
                off_label="OFF",
                n_perm=n_perm
            )
            lp_results.append(res_lp)

            res_ll, _ = stats_runner.analyze_landing_opto(
                group_info=group_info,
                out_prefix=f"{out_prefix}_{group_key}_LL_ON_OFF",
                chr_data=chr_data,
                n_perm=n_perm
            )
            ll_results.append(res_ll)

        if lp_results:
            pd.concat(lp_results, ignore_index=True).to_csv(f"{out_prefix}_LP_ON_OFF_stats.csv", index=False)
        if ll_results:
            pd.concat(ll_results, ignore_index=True).to_csv(f"{out_prefix}_LL_ON_OFF_fly_RMST_stats.csv", index=False)


    """plotter.plot_TT_trajectories_moc_origin_projected_plane(
        group_info={
            "T1": groups["WT_T1_TTa"],
            "T2": groups["WT_T2_TTa"],
            "T3": groups["WT_T3_TTa"],
        },
        tt_joints=("L-fTT", "L-mTT", "L-hTT"),
        plane_axis=("R-mBC", "L-mBC"),
        reference_axis=("R-mBC", "R-hBC"),
        average_mode="time_normalized",  # or "time_normalized"
        target_fps=250,
        file_name="WT_T1_T2_T3_left_TT_projected_trajectories_tnorm"
    )"""


    r"""test_t2_it_ot_plots_with_old_and_new_kinematic_paths(
        r"C:\Users\agrawal-admin\Desktop\TibiaTarsusPlatformODLight-Wayne-2024-10-19\Network-01-18-2026\LPAcrossLegsJoints\T2-TiTa",
        r"D:\TibiaTarsusPlatformODLight-Wayne-2024-10-19\Network-01-18-2026\LPAcrossLegsJoints\T2-TiTa"
    )"""

    # ============================================================
    # WT Figure 2 workflow
    # ============================================================
    # Set this to True to generate the WT slide figures requested in
    # "Figures to make.docx". Figure 2D is intentionally omitted for now.
    RUN_WT_FIGURE_2 = False

    if RUN_WT_FIGURE_2:
        wt_figure_dir = use_output_folder(figures_dir, "WT_Figure_2")

        wt_tita_groups = {
            "T1": groups["WT_T1_TTa"],
            "T2": groups["WT_T2_TTa"],
            "T3": groups["WT_T3_TTa"],
        }
        wt_cxtr_groups = {
            "T1": groups["WT_T1_CTF"],
            "T2": groups["WT_T2_CTF"],
            "T3": groups["WT_T3_CTF"],
        }
        wt_tita_sc_paths = {
            "T1": os.path.join(sc_data_dir, "WT-T1-TiTa_LegContact.csv"),
            "T2": os.path.join(sc_data_dir, "WT-T2-TiTa_LegContact.csv"),
            "T3": os.path.join(sc_data_dir, "WT-T3-TiTa_LegContact.csv"),
        }
        wt_tita_manual_sc_legs = {
            "T1": ("L-f", "L-m", "L-h", "R-m", "R-h"),
            "T2": ("L-f", "L-m", "L-h", "R-f", "R-h"),
            "T3": ("L-f", "L-m", "L-h", "R-f", "R-m"),
        }

        wt_groups_ordered = [
            groups["WT_T1_TTa"],
            groups["WT_T1_CTF"],
            groups["WT_T2_TTa"],
            groups["WT_T2_CTF"],
            groups["WT_T3_TTa"],
            groups["WT_T3_CTF"],
        ]
        wt_group_colors = {
            "WT-T1-TiTa": "#1f77b4",
            "WT-T1-CxTr": "#8ecae6",
            "WT-T2-TiTa": "#d62728",
            "WT-T2-CxTr": "#ff9896",
            "WT-T3-TiTa": "#2ca02c",
            "WT-T3-CxTr": "#98df8a",
        }
        wt_group_markers = {group.group_name: "o" for group in wt_groups_ordered}
        wt_group_linestyles = {group.group_name: "solid" for group in wt_groups_ordered}
        n_perm = 20000

        # 2A: WT landing probability across T1/T2/T3, TiTa and CxTr.
        use_output_folder(wt_figure_dir, "2A_landing_probability")
        plotter.plot_LP_summary_from_groups(
            groups=wt_groups_ordered,
            file_name="Figure_2A_WT_landing_probability",
            colors=wt_group_colors,
            markers=wt_group_markers
        )
        wt_lp_stats = []
        for group_a, group_b in itertools.combinations(wt_groups_ordered, 2):
            out_prefix = f"Figure_2A_{group_a.group_name}_vs_{group_b.group_name}_LP"
            res_lp, _ = stats_runner.compare_lp_unpaired(
                group_a=group_a,
                group_b=group_b,
                out_prefix=out_prefix,
                n_perm=n_perm
            )
            wt_lp_stats.append(res_lp)
        pd.concat(wt_lp_stats, ignore_index=True).to_csv(
            "Figure_2A_WT_landing_probability_pairwise_stats.csv",
            index=False
        )

        # 2B: WT landing latency as inverted KM curves using the same groups.
        use_output_folder(wt_figure_dir, "2B_landing_latency_KM")
        plotter.plot_KM_curve_from_groups(
            groups=wt_groups_ordered,
            file_name="Figure_2B_WT_landing_latency_inverted_KM",
            colors=wt_group_colors,
            linestyles=wt_group_linestyles,
            markers={group.group_name: None for group in wt_groups_ordered}
        )
        wt_ll_stats = []
        for group_a, group_b in itertools.combinations(wt_groups_ordered, 2):
            out_prefix = f"Figure_2B_{group_a.group_name}_vs_{group_b.group_name}_LL"
            res_ll, _ = stats_runner.analyze_landing_unpaired(
                group_a=group_a,
                group_b=group_b,
                out_prefix=out_prefix,
                n_perm=n_perm
            )
            wt_ll_stats.append(res_ll)
        pd.concat(wt_ll_stats, ignore_index=True).to_csv(
            "Figure_2B_WT_landing_latency_pairwise_fly_RMST_stats.csv",
            index=False
        )

        # 2C: MOC-aligned CT/FT angle traces. Each contact group uses the
        # corresponding right-side leg and is resampled to 250 fps before averaging.
        use_output_folder(wt_figure_dir, "2C_CT_FT_angle_traces")
        plotter.plot_wt_contact_group_angle_traces(
            groups_by_column={
                "TiTa": wt_tita_groups,
                "CxTr": wt_cxtr_groups,
            },
            file_name="Figure_2C_WT_CT_FT_angle_traces",
            contact_leg_map={
                "T1": "R-f",
                "T2": "R-m",
                "T3": "R-h",
            },
            contact_colors={
                "T1": "#1f77b4",
                "T2": "#d62728",
                "T3": "#2ca02c",
            },
            start_s=-0.1,
            end_s=0.71,
            target_fps=250,
            trial_types=("Landing", "Flying"),
            show_sem=True
        )

        # 2D: Manual SC timing as inverted KM curves for TiTa contact groups.
        # This section also writes fly-wise RMST permutation statistics.
        use_output_folder(wt_figure_dir, "2D_manual_SC_KM")
        for contact_group, group_info in wt_tita_groups.items():
            plotter.plot_manual_sc_inverted_km_from_csv(
                group_info=group_info,
                sc_csv_path=wt_tita_sc_paths[contact_group],
                legs=wt_tita_manual_sc_legs[contact_group],
                threshold=0.71,
                file_name=f"Figure_2D_WT_{contact_group}_TiTa_manual_SC_inverted_KM"
            )
        plotter.compare_manual_sc_rmst_across_contact_groups(
            group_infos=wt_tita_groups,
            sc_csv_paths=wt_tita_sc_paths,
            contact_groups=("T1", "T2", "T3"),
            legs=wt_tita_manual_sc_legs,
            threshold=0.71,
            n_perm=20000,
            file_name="Figure_2D_WT_TiTa_manual_SC_RMST"
        )

        # 2E: Post-MOC left-side leg contact order heatmap for manually labeled TiTa data.
        use_output_folder(wt_figure_dir, "2E_SC_order_heatmap")
        plotter.plot_sc_order_heatmaps_by_contact_group(
            group_infos=wt_tita_groups,
            sc_csv_paths=wt_tita_sc_paths,
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            sort_trials=False,
            file_name="Figure_2E_WT_TiTa_left_leg_SC_order_heatmap"
        )

        # 2F: Fly-wise probability that each left-side leg makes first valid SC.
        use_output_folder(wt_figure_dir, "2F_SC_first_contact_probability")
        plotter.plot_flywise_first_sc_probability_by_contact_group(
            group_infos=wt_tita_groups,
            sc_csv_paths=wt_tita_sc_paths,
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            n_perm=n_perm,
            file_name="Figure_2F_WT_TiTa_flywise_secondary_contact_probability"
        )

        # Additional WT plot: TT path efficiency, path length, and average speed
        # versus raw landing latency. Each output CSV includes the panel-level
        # Spearman and permutation statistics.
        use_output_folder(wt_figure_dir, "WT_TT_metrics_vs_LL")
        for contact_group, group_info in wt_tita_groups.items():
            plotter.plot_TT_summary_metrics_vs_LL(
                group_info=group_info,
                legs=("L-f", "L-m", "L-h"),
                trial_types=("Landing", "Flying"),
                tau=0.71,
                trajectory_window_mode="mol_adjusted",
                file_name=f"WT_{contact_group}_TiTa_TT_summary_metrics_vs_LL_mol_adjusted",
                n_perm=n_perm
            )

        os.chdir(figures_dir)

    # ============================================================
    # WT TiTa behavior-subset plotting
    # ============================================================
    RUN_WT_T2_TITA_IT_OT_PLOTS = False

    if RUN_WT_T2_TITA_IT_OT_PLOTS:
        n_perm = 20000
        behavior_figure_dir = use_output_folder(figures_dir, "WT_TiTa_behavior_subsets")

        it_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_IT_filtered.xlsx")
        ot_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_OT_filtered.xlsx")
        wt_t2_tita_sc_path = os.path.join(sc_data_dir, "WT-T2-TiTa_LegContact.csv")

        # These filtered LL files keep numeric LL values for the retained subset
        # and mark the excluded subset with text. Therefore, numeric cells in the
        # OT-filtered file are used as IT trials, and numeric cells in the
        # IT-filtered file are used as OT trials. If you want to select the
        # literal text-marked cells instead, change selection_mode to "marker".
        wt_t2_tita_it_ot_sources = {
            "IT": {
                "path": ot_filtered_ll_path,
                "selection_mode": "numeric",
            },
            "OT": {
                "path": it_filtered_ll_path,
                "selection_mode": "numeric",
            },
        }

        # T2 TiTa IT/OT: FT angle trace from -0.1 to 0.1 s around MOC.
        use_output_folder(behavior_figure_dir, "T2_IT_OT_angle_trace")
        plotter.plot_angle_traces_by_trial_sets(
            group_info=groups["WT_T2_TTa"],
            angle_defs=[
                ["R-mCT", "R-mFT", "R-mTT"],
            ],
            behavior_sources=wt_t2_tita_it_ot_sources,
            trial_types=("Landing", "Flying"),
            start_s=-0.1,
            end_s=0.1,
            target_fps=250,
            file_name="WT_T2_TiTa_IT_OT_Rm_FT_angle_trace"
        )

        # T2 TiTa IT/OT: landing probability, inverted landing-latency KM,
        # FT angle trace, and FT angular velocity stripplot.
        use_output_folder(behavior_figure_dir, "T2_IT_OT_LP_KM_angle_velocity")
        plotter.plot_it_ot_landing_probability_and_latency(
            group_info=groups["WT_T2_TTa"],
            behavior_sources=wt_t2_tita_it_ot_sources,
            file_name="WT_T2_TiTa_IT_OT_LP_latency_KM_FT_angle",
            behavior_labels=("IT", "OT"),
            behavior_display_names={
                "IT": "Inward touch",
                "OT": "Outward touch",
            },
            trial_types=("Landing", "Flying"),
            tau=0.71,
            n_perm=n_perm,
            contacted_leg="R-m",
            angle_start_s=-0.1,
            angle_end_s=0.1,
            target_fps=250,
            colors={
                "IT": "#8FD694",
                "OT": "#C7A0E8",
            }
        )

        # T2 TiTa: L-hTT path efficiency grouped as Success/Failed and IT/OT.
        use_output_folder(behavior_figure_dir, "T2_IT_OT_LhTT_path_efficiency_stripplot")
        plotter.plot_left_TT_path_efficiency_grouped_stripplots(
            group_info=groups["WT_T2_TTa"],
            behavior_sources=wt_t2_tita_it_ot_sources,
            file_name="WT_T2_TiTa_LhTT_path_efficiency_success_failed_IT_OT",
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="mol_adjusted",
            n_perm=n_perm
        )

        # T2 TiTa: valid left-leg SC count versus raw landing latency.
        use_output_folder(behavior_figure_dir, "T2_SC_count_vs_LL")
        plotter.plot_valid_sc_count_vs_landing_latency(
            group_info=groups["WT_T2_TTa"],
            sc_csv_path=wt_t2_tita_sc_path,
            file_name="WT_T2_TiTa_valid_SC_count_vs_LL",
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying")
        )

        # T2 TiTa: L-hTT path efficiency versus raw landing latency.
        use_output_folder(behavior_figure_dir, "T2_LhTT_path_efficiency_vs_LL")
        plotter.plot_TT_summary_metrics_vs_LL(
            group_info=groups["WT_T2_TTa"],
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="mol_adjusted",
            file_name="WT_T2_TiTa_LhTT_summary_metrics_vs_LL_mol_adjusted",
            n_perm=n_perm
        )

        # ------------------------------------------------------------
        # T1 TiTa BO/NB behavior subset placeholders.
        # Replace these filenames with the final filtered Excel files once they
        # are available in LandingData/Others.
        # ------------------------------------------------------------
        bo_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T1-TiTa_new_BO_filtered.xlsx")
        nb_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T1-TiTa_new_NB_filtered.xlsx")
        wt_t1_tita_sc_path = os.path.join(sc_data_dir, "WT-T1-TiTa_LegContact.csv")

        # This mirrors the IT/OT filtered-file convention above: numeric cells
        # in the opposite filtered file define the retained behavior subset.
        wt_t1_tita_bo_nb_sources = {
            "BO": {
                "path": nb_filtered_ll_path,
                "selection_mode": "numeric",
            },
            "NB": {
                "path": bo_filtered_ll_path,
                "selection_mode": "numeric",
            },
        }

        # T1 TiTa BO/NB: FT angle trace from -0.1 to 0.1 s around MOC.
        use_output_folder(behavior_figure_dir, "T1_BO_NB_angle_trace")
        plotter.plot_angle_traces_by_trial_sets(
            group_info=groups["WT_T1_TTa"],
            angle_defs=[
                ["R-fCT", "R-fFT", "R-fTT"],
            ],
            behavior_sources=wt_t1_tita_bo_nb_sources,
            trial_types=("Landing", "Flying"),
            start_s=-0.1,
            end_s=0.1,
            target_fps=250,
            file_name="WT_T1_TiTa_BO_NB_Rf_FT_angle_trace"
        )

        # T1 TiTa BO/NB: landing probability, inverted landing-latency KM,
        # FT angle trace, and FT angular velocity stripplot.
        use_output_folder(behavior_figure_dir, "T1_BO_NB_LP_KM_angle_velocity")
        plotter.plot_it_ot_landing_probability_and_latency(
            group_info=groups["WT_T1_TTa"],
            behavior_sources=wt_t1_tita_bo_nb_sources,
            file_name="WT_T1_TiTa_BO_NB_LP_latency_KM_FT_angle",
            behavior_labels=("BO", "NB"),
            behavior_display_names={
                "BO": "BO",
                "NB": "NB",
            },
            trial_types=("Landing", "Flying"),
            tau=0.71,
            n_perm=n_perm,
            contacted_leg="R-f",
            angle_start_s=-0.1,
            angle_end_s=0.1,
            target_fps=250,
            colors={
                "BO": "#66C2A5",
                "NB": "#B39DDB",
            }
        )

        # T1 TiTa: L-hTT path efficiency grouped as Success/Failed and BO/NB.
        use_output_folder(behavior_figure_dir, "T1_BO_NB_LhTT_path_efficiency_stripplot")
        plotter.plot_left_TT_path_efficiency_grouped_stripplots(
            group_info=groups["WT_T1_TTa"],
            behavior_sources=wt_t1_tita_bo_nb_sources,
            file_name="WT_T1_TiTa_LhTT_path_efficiency_success_failed_BO_NB",
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="mol_adjusted",
            colors={
                "Success": "tab:blue",
                "Failed": "tab:red",
                "BO": "#66C2A5",
                "NB": "#B39DDB",
            },
            n_perm=n_perm
        )

        # T1 TiTa: valid left-leg SC count versus raw landing latency.
        use_output_folder(behavior_figure_dir, "T1_SC_count_vs_LL")
        plotter.plot_valid_sc_count_vs_landing_latency(
            group_info=groups["WT_T1_TTa"],
            sc_csv_path=wt_t1_tita_sc_path,
            file_name="WT_T1_TiTa_valid_SC_count_vs_LL",
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying")
        )

        # T1 TiTa: L-hTT path efficiency versus raw landing latency.
        use_output_folder(behavior_figure_dir, "T1_LhTT_path_efficiency_vs_LL")
        plotter.plot_TT_summary_metrics_vs_LL(
            group_info=groups["WT_T1_TTa"],
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="mol_adjusted",
            file_name="WT_T1_TiTa_LhTT_summary_metrics_vs_LL_mol_adjusted",
            n_perm=n_perm
        )

        os.chdir(figures_dir)

    # ============================================================
    # Opt-in SLC-adjusted TT metric test block
    # ============================================================
    RUN_T2_TITA_SLC_ADJUSTED_TT_METRIC_TEST = False

    if RUN_T2_TITA_SLC_ADJUSTED_TT_METRIC_TEST:
        # This block tests leg-specific SLC-adjusted trajectory windows without
        # changing the default mol_adjusted behavior-subset figure workflow above.
        n_perm = 1000
        slc_test_dir = use_output_folder(figures_dir, "T2_TiTa_SLC_adjusted_TT_metric_test")

        it_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_IT_filtered.xlsx")
        ot_filtered_ll_path = os.path.join(other_landing_data_dir, "WT-T2-TiTa_new_OT_filtered.xlsx")
        wt_t2_tita_sc_path = os.path.join(sc_data_dir, "WT-T2-TiTa_LegContact.csv")

        wt_t2_tita_it_ot_sources = {
            "IT": {
                "path": ot_filtered_ll_path,
                "selection_mode": "numeric",
            },
            "OT": {
                "path": it_filtered_ll_path,
                "selection_mode": "numeric",
            },
        }

        use_output_folder(slc_test_dir, "LhTT_path_efficiency_stripplot")
        plotter.plot_left_TT_path_efficiency_grouped_stripplots(
            group_info=groups["WT_T2_TTa"],
            behavior_sources=wt_t2_tita_it_ot_sources,
            file_name="WT_T2_TiTa_LhTT_path_efficiency_success_failed_IT_OT_SLC_adjusted",
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="SLC_adjusted",
            sc_csv_path=wt_t2_tita_sc_path,
            n_perm=n_perm
        )

        use_output_folder(slc_test_dir, "LhTT_summary_metrics_vs_LL")
        plotter.plot_TT_summary_metrics_vs_LL(
            group_info=groups["WT_T2_TTa"],
            legs=("L-h",),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="SLC_adjusted",
            sc_csv_path=wt_t2_tita_sc_path,
            file_name="WT_T2_TiTa_LhTT_summary_metrics_vs_LL_SLC_adjusted",
            n_perm=n_perm
        )

        os.chdir(figures_dir)

    # ============================================================
    # Opt-in projected TT MOC/end-point scatter test
    # ============================================================
    RUN_WT_TITA_TT_MOC_ENDPOINT_PROJECTED_SCATTER = True

    if RUN_WT_TITA_TT_MOC_ENDPOINT_PROJECTED_SCATTER:
        scatter_dir = use_output_folder(figures_dir, "WT_TiTa_TT_MOC_endpoint_projected_scatter")
        plotter.plot_TT_MOC_to_SLC_endpoint_projected_scatter(
            group_info={
                "T1": groups["WT_T1_TTa"],
                "T2": groups["WT_T2_TTa"],
                "T3": groups["WT_T3_TTa"],
            },
            sc_csv_paths={
                "T1": os.path.join(sc_data_dir, "WT-T1-TiTa_LegContact.csv"),
                "T2": os.path.join(sc_data_dir, "WT-T2-TiTa_LegContact.csv"),
                "T3": os.path.join(sc_data_dir, "WT-T3-TiTa_LegContact.csv"),
            },
            tt_joints=("L-fTT", "L-mTT", "L-hTT"),
            plane_axis=("R-mBC", "L-mBC"),
            reference_axis=("R-mBC", "R-hBC"),
            origin_keypoint="R-mBC",
            origin_frame="moc",
            trial_types=("Landing", "Flying"),
            tau=0.71,
            axis_average_anchor="moc",
            file_name="WT_TiTa_projected_TT_MOC_to_SLC_or_fallback_endpoint",
            point_size=8,
            alpha=0.6,
            show_trajectories=True,
            show_points=True,
            show_aep=False,
            show_vep=False,
            extreme_point_size=24,
            connector_linewidth=0.15,
            connector_alpha=0.25,
            plot_radial_displacement=True,
            radial_circle_diameter=3,
            radial_linewidth=0.5,
            radial_alpha=0.35,
            save_csv=True
        )
        os.chdir(figures_dir)

    # ============================================================
    # KIR Figure 3 workflow
    # ============================================================
    RUN_KIR_FIGURE_3 = False

    if RUN_KIR_FIGURE_3:
        n_perm = 20000
        kir_figure_dir = use_output_folder(figures_dir, "KIR_Figure_3")

        kir_group_keys = [
            "WT_T2_TTa",
            "CSS-0039_T2_TiTa",
            "CSS-0048_T2_TiTa",
            "G106_T2_TTa",
            "G107_T2_TTa",
            "G108_T2_TTa",
            "G114_T2_TTa",
            "G115_T2_TTa",
            "G116_T2_TTa",
            "G117_T2_TTa",
            "G118_T2_TTa",
            "G119_T2_TTa",
        ]
        kir_colors = sns.color_palette("viridis", n_colors=len(kir_group_keys))

        # 3A and 3B are requested as two panels with at most seven groups per
        # panel. Existing plotting helpers save one panel per call, so this
        # usage code saves panel 1 and panel 2 as separate files.
        use_output_folder(kir_figure_dir, "3A_landing_probability")
        for panel_idx, start_idx in enumerate(range(0, len(kir_group_keys), 7), start=1):
            panel_keys = kir_group_keys[start_idx:start_idx + 7]
            plotter.plot_LP_summary_from_groups(
                groups=[groups[key] for key in panel_keys],
                file_name=f"Figure_3A_KIR_landing_probability_panel{panel_idx}",
                colors=kir_colors[start_idx:start_idx + len(panel_keys)]
            )

        use_output_folder(kir_figure_dir, "3B_landing_latency_KM")
        for panel_idx, start_idx in enumerate(range(0, len(kir_group_keys), 7), start=1):
            panel_keys = kir_group_keys[start_idx:start_idx + 7]
            plotter.plot_KM_curve_from_groups(
                groups=[groups[key] for key in panel_keys],
                file_name=f"Figure_3B_KIR_landing_latency_KM_panel{panel_idx}",
                colors=kir_colors[start_idx:start_idx + len(panel_keys)],
                opto=False
            )

        use_output_folder(kir_figure_dir, "stats")
        save_unpaired_lp_ll_stats(
            group_pairs=[("WT_T2_TTa", key) for key in kir_group_keys if key != "WT_T2_TTa"],
            out_prefix="Figure_3_KIR_WT_T2_TTa_vs_KIR",
            n_perm=n_perm
        )
        os.chdir(figures_dir)

    # ============================================================
    # CsChrimson Figure 4 workflow
    # ============================================================
    RUN_CSCHRIMSON_FIGURE_4 = False

    if RUN_CSCHRIMSON_FIGURE_4:
        n_perm = 20000
        chr_figure_dir = use_output_folder(figures_dir, "CsChrimson_Figure_4")

        chr_low_keys = [
            "ADxChr-400uW",
            "IavxChr-400uW",
            "HP2xChr-400uW",
            "TaCSxCHR-400uW",
            "AllCSxChr-400uW",
        ]
        chr_med_keys = [
            "IAVxCHR-4mW",
            "HP2xCHR-4mW",
            "TaBriLexAR-4mW",
            "TaCSxCHR-4mW",
            "CSS0048xCHR-4mW",
            "BiCSxCHR-4mW",
            "BiCS-HaltxCHR-4mW",
        ]
        chr_high_keys = [
            "IAVxCHR-12mW",
            "HP2xChr-12mW",
            "TaBriLexAR-12mW",
            "TaCSxCHR-12mW",
            "CSS0048xCHR-12mW",
            "BiCS-HaltWgxCHR-12mW",
            "BICSxCHR-12mW",
            "BICSHALTxCHR-12mW",
            "CSS0021xCHR-12mW",
        ]
        chr_intensity_blocks = {
            "low": {
                "lp_label": "4A",
                "ll_label": "4D",
                "keys": chr_low_keys,
            },
            "medium": {
                "lp_label": "4B",
                "ll_label": "4E",
                "keys": chr_med_keys,
            },
            "high": {
                "lp_label": "4C",
                "ll_label": "4F",
                "keys": chr_high_keys,
            },
        }

        chrimson_on_ll_by_key = {}

        def run_chrimson_intensity_block(intensity_name, block_info):
            group_keys = block_info["keys"]
            use_output_folder(chr_figure_dir, f"{block_info['lp_label']}_{intensity_name}_landing_probability")
            ll_data = []
            for group_key in group_keys:
                group_ll_data = plotter.plot_chrimson_LP(
                    group_info=groups[group_key],
                    color="red",
                    threshold=0.71
                )
                ll_data.append(group_ll_data)
                chrimson_on_ll_by_key[group_key] = group_ll_data
            use_output_folder(chr_figure_dir, f"{block_info['ll_label']}_{intensity_name}_landing_latency_KM")
            plotter.plot_kmc_and_unpaired_rmst_perm(
                data_list=ll_data,
                file_name=f"Figure_{block_info['ll_label']}_CsChrimson_{intensity_name}_ON_landing_latency_KM",
                tau=0.71,
                n_perm=n_perm,
                random_state=0
            )

        for intensity_name, block_info in chr_intensity_blocks.items():
            run_chrimson_intensity_block(intensity_name, block_info)
            # pass
        chr_lp_change_keys = [
            "ADxChr-400uW",
            "IavxChr-400uW",
            "IAVxCHR-4mW",
            "IAVxCHR-12mW",
            "HP2xChr-400uW",
            "HP2xCHR-4mW",
            "HP2xChr-12mW",
            "AllCSxChr-400uW",
            "BiCSxCHR-4mW",
            "BICSxCHR-12mW",
            "BiCS-HaltxCHR-4mW",
            "BICSHALTxCHR-12mW",
            "BiCS-HaltWgxCHR-12mW",
            "CSS0048xCHR-4mW",
            "CSS0048xCHR-12mW",
            "CSS0021xCHR-12mW",
            "TaCSxCHR-400uW",
            "TaCSxCHR-4mW",
            "TaCSxCHR-12mW",
            "TaBriLexAR-4mW",
            "TaBriLexAR-12mW",
        ]
        use_output_folder(chr_figure_dir, "4A_4C_combined_LP_change_summary")
        plotter.plot_chrimson_LP_change_summary(
            groups={group_key: groups[group_key] for group_key in chr_lp_change_keys},
            file_name="Figure_4A_4C_CsChrimson_combined_LP_change_ON_minus_OFF",
            threshold=0.71,
            n_perm=n_perm,
            random_state=0
        )

        chrimson_selected_km_keys = [
            "ADxChr-400uW",
            "AllCSxChr-400uW",
            "CSS0048xCHR-12mW",
            "TaBriLexAR-4mW",
            "CSS0021xCHR-12mW",
        ]
        chrimson_selected_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

        use_output_folder(chr_figure_dir, "4_selected_groups_ON_OFF_landing_latency_KM")
        for group_key, group_color in zip(chrimson_selected_km_keys, chrimson_selected_colors):
            plotter.plot_chrimson_on_off_latency_km(
                group_info=groups[group_key],
                file_name=f"Figure_4_{group_key}_ON_OFF_landing_latency_KM",
                color=group_color,
                tau=0.71
            )

        use_output_folder(chr_figure_dir, "4_selected_groups_ON_OFF_angle_traces")
        for group_key, group_color in zip(chrimson_selected_km_keys, chrimson_selected_colors):
            plotter.plot_chrimson_on_off_leg_wing_angle_traces(
                group_info=groups[group_key],
                angles=[
                    ["R-mCT", "R-mFT", "R-mTT"],  # R-m FT angle
                    ["wing", "wing", "wing"],  # wing angle
                ],
                file_name=f"Figure_4_{group_key}_ON_OFF_RmFT_wing_angle_traces",
                start=-0.5,
                end=3,
                color=group_color,
                show_sem=True
            )

        chr_angles = [
            ["R-mCT", "R-mFT", "R-mTT"],
            ["wing", "wing", "wing"],
        ]
        chr_angle_blocks = {
            "4G_low_angle_traces": chr_low_keys,
            "4H_medium_angle_traces": chr_med_keys,
            "4I_high_angle_traces": chr_high_keys,
        }
        for block_name, group_keys in chr_angle_blocks.items():
            use_output_folder(chr_figure_dir, block_name)
            for group_key in group_keys:
                plotter.plot_group_angle_trace_opto(
                    group_info=groups[group_key],
                    angles=chr_angles,
                    start=-0.5,
                    end=3,
                    colors="red",
                    chrimson=True
                )
        os.chdir(figures_dir)

    # ============================================================
    # GtACR Figure 5 workflow
    # ============================================================
    RUN_GTACR_FIGURE_5 = False

    if RUN_GTACR_FIGURE_5:
        n_perm = 20000
        gtacr_figure_dir = use_output_folder(figures_dir, "GtACR_Figure_5")

        gtacr_keys = [
            "WT_Green",
            "LexA_Br",
            "MTGal4",
            "IavxGTACR",
            "CSS048xGTACR",
            "CSS021xGTACR",
        ]
        gtacr_colors = {
            "WT_Green": "black",
            "LexA_Br": "green",
            "MTGal4": "blue",
            "IavxGTACR": "brown",
            "CSS048xGTACR": "red",
            "CSS021xGTACR": "orange",
        }

        use_output_folder(gtacr_figure_dir, "5A_landing_probability_ON_OFF")
        for group_key in gtacr_keys:
            plotter.plot_LP_summary_light_from_group(
                group_info=groups[group_key],
                file_name=f"Figure_5A_{group_key}_LP_ON_OFF",
                color = "green"
            )

        use_output_folder(gtacr_figure_dir, "5A_combined_LP_change_summary")
        plotter.plot_gtacr_LP_change_summary(
            groups={group_key: groups[group_key] for group_key in gtacr_keys},
            file_name="Figure_5A_GtACR_combined_LP_change_ON_minus_OFF",
            n_perm=n_perm,
            random_state=0
        )

        use_output_folder(gtacr_figure_dir, "5B_landing_latency_KM_ON_OFF")
        for group_key in gtacr_keys:
            plotter.plot_KM_curve_from_groups(
                groups=[groups[group_key]],
                file_name=f"Figure_5B_{group_key}_LL_KM_ON_OFF",
                colors=["black", "green"],
                opto=True
            )

        use_output_folder(gtacr_figure_dir, "stats")
        save_paired_opto_lp_ll_stats(
            group_keys=gtacr_keys,
            out_prefix="Figure_5_GtACR",
            chr_data=False,
            n_perm=n_perm
        )
        os.chdir(figures_dir)

    # ============================================================
    # Figure 6 workflow: AD/AN CsChrimson and MTGal4/AN GtACR
    # ============================================================
    RUN_AN_FIGURE_6 = False

    if RUN_AN_FIGURE_6:
        n_perm = 20000
        figure_6_dir = use_output_folder(figures_dir, "Figure_6")

        fig6_chr_keys = [
            "ADxChr-400uW",
            "ANxCHR-400uW",
            "ANxChr-4mW",
            "ANxCHR-12mW",
        ]
        fig6_chr_colors = {
            "ADxChr-400uW": "#8c8c8c",
            "ANxCHR-400uW": "#F4A3A3",
            "ANxChr-4mW": "#D73027",
            "ANxCHR-12mW": "#7F0000",
        }
        fig6_gtacr_keys = ["MTGal4", "ANxGTACR"]

        # 6A: CsChrimson AD/AN fly-wise LP change summary.
        use_output_folder(figure_6_dir, "6A_AD_AN_CsChrimson_LP_change_summary")
        plotter.plot_chrimson_LP_change_summary(
            groups={group_key: groups[group_key] for group_key in fig6_chr_keys},
            file_name="Figure_6A_AD_AN_CsChrimson_LP_change_ON_minus_OFF",
            threshold=0.71,
            n_perm=n_perm,
            random_state=0
        )

        # 6B: CsChrimson AD/AN individual ON/OFF LP plots.
        use_output_folder(figure_6_dir, "6B_AD_AN_CsChrimson_individual_LP")
        fig6_chr_ll_data = []
        for group_key in fig6_chr_keys:
            fig6_chr_ll_data.append(
                plotter.plot_chrimson_LP(
                    group_info=groups[group_key],
                    color=fig6_chr_colors[group_key],
                    threshold=0.71
                )
            )

        # 6C: CsChrimson AD/AN ON landing latency inverted KM curves.
        use_output_folder(figure_6_dir, "6C_AD_AN_CsChrimson_landing_latency_KM")
        plotter.plot_kmc_and_unpaired_rmst_perm(
            data_list=fig6_chr_ll_data,
            file_name="Figure_6C_AD_AN_CsChrimson_ON_landing_latency_KM",
            tau=0.71,
            n_perm=n_perm,
            random_state=0,
            colors=[fig6_chr_colors[group_key] for group_key in fig6_chr_keys],
            invert_curve=True
        )

        # 6D: CsChrimson AD/AN light-triggered leg and wing angle changes in one plot.
        use_output_folder(figure_6_dir, "6D_AD_AN_CsChrimson_angle_change")
        plotter.plot_selected_chrimson_angle_traces(
            groups={group_key: groups[group_key] for group_key in fig6_chr_keys},
            angles=[
                ["R-mCT", "R-mFT", "R-mTT"],  # leg FT angle
                ["wing", "wing", "wing"],  # wing angle
            ],
            file_name="Figure_6D_AD_AN_CsChrimson_ON_leg_and_wing_angle_traces",
            start=-0.5,
            end=3,
            condition="ON",
            colors=[fig6_chr_colors[group_key] for group_key in fig6_chr_keys],
            show_sem=True
        )

        # 6E: GtACR MTGal4/AN fly-wise LP change summary.
        use_output_folder(figure_6_dir, "6E_MTGal4_AN_GtACR_LP_change_summary")
        plotter.plot_gtacr_LP_change_summary(
            groups={group_key: groups[group_key] for group_key in fig6_gtacr_keys},
            file_name="Figure_6E_MTGal4_AN_GtACR_LP_change_ON_minus_OFF",
            n_perm=n_perm,
            random_state=0,
            color="green",
            box_color="#B7E1B0"
        )

        # 6F: GtACR MTGal4/AN individual ON/OFF LP plots.
        use_output_folder(figure_6_dir, "6F_MTGal4_AN_GtACR_individual_LP")
        for group_key in fig6_gtacr_keys:
            plotter.plot_LP_summary_light_from_group(
                group_info=groups[group_key],
                file_name=f"Figure_6F_{group_key}_LP_ON_OFF",
                color="green"
            )

        # 6G: GtACR MTGal4/AN individual ON/OFF landing latency curves.
        use_output_folder(figure_6_dir, "6G_MTGal4_AN_GtACR_individual_latency_KM")
        for group_key in fig6_gtacr_keys:
            plotter.plot_KM_curve_from_groups(
                groups=[groups[group_key]],
                file_name=f"Figure_6G_{group_key}_LL_KM_ON_OFF",
                colors=["black", "green"],
                opto=True
            )

        use_output_folder(figure_6_dir, "stats")
        save_paired_opto_lp_ll_stats(
            group_keys=fig6_chr_keys,
            out_prefix="Figure_6_AD_AN_CsChrimson",
            chr_data=True,
            n_perm=n_perm
        )
        save_paired_opto_lp_ll_stats(
            group_keys=fig6_gtacr_keys,
            out_prefix="Figure_6_MTGal4_AN_GtACR",
            chr_data=False,
            n_perm=n_perm
        )
        os.chdir(figures_dir)
