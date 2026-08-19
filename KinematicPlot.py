import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.signal import find_peaks
from scipy.stats import linregress, ttest_rel
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kinematic_object import Group, Trial, Point
import kinematic_utilities as ku
import plot_common as pc
import trial_helpers as th
import plot_geometry as pg
import plot_landing as pl
import plot_optogenetics as po
import plot_secondary_contact as psc
import plot_angles as pa


class PlotCreator:
    def __init__(self):
        self.calculator = ku.SimpleCalculation()
        self.analyzer = ku.GroupDataAnalyzer()

        self.key_point_pairs = [
            ["L-wing", "L-wing-hinge"],
            ["R-wing", "R-wing-hinge"],
            ["abdomen-tip"],
            ["platform-tip"],
            ["L-platform-tip"],
            ["R-platform-tip"],
            ["platform-axis"],
            ["R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT"],
            ["R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT"],
            ["R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT"],
            ["L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT"],
            ["L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT"],
            ["L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"]
        ]

        self.bodyparts = [
            "R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT",
            "R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT",
            "R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT",
            "L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT",
            "L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT",
            "L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"
        ]

    def formatting(self, ax, xticks=None, yticks=None, xlabel=None, ylabel=None,
                   yticklabel=None, xticklabel=None, ylabel_size=10, xlabel_size=10,
                   spine_width=3, tick_width=3):
        return pc.format_axes(
            ax,
            xticks=xticks,
            yticks=yticks,
            xlabel=xlabel,
            ylabel=ylabel,
            ylabel_size=ylabel_size,
            xlabel_size=xlabel_size,
            spine_width=spine_width,
            tick_width=tick_width
        )
    
    def plot_TT_MOC_to_SLC_endpoint_projected_combined(
            self,
            group_info,
            sc_csv_paths,
            tt_joints=("L-fTT", "L-mTT", "L-hTT"),
            plane_axis=("R-mBC", "L-mBC"),
            origin_keypoint="R-mBC",
            trial_types=("Landing", "Flying"),
            tau=0.71,
            axis_average_frames=100,
            axis_average_anchor="moc",
            file_name="TT_MOC_to_SLC_endpoint_projected_combined",
            colors=None,
            target_fps=250,
            trajectory_average_mode="absolute_time",
            normalized_average_points=200,
            trial_color="0.55",
            trial_linewidth=0.25,
            trial_alpha=0.35,
            fly_linewidth=1.4,
            fly_alpha=0.95,
            radial_circle_diameter=None,
            radial_coordinate_mode="displacement_origin",
            n_perm=20000,
            random_state=0,
            radial_stats_file_name=None,
            apply_tracking_qc=False,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False,
            save_csv=True
    ):
        return pg.plot_TT_MOC_to_SLC_endpoint_projected_combined(**locals())
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
        return pl.plot_LP_summary_from_groups(**locals())
    def plot_LP_summary_light_from_group(self, group_info, file_name, color):
        return pl.plot_LP_summary_light_from_group(**locals())
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
        return pl.plot_KM_curve_from_groups(**locals())
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
        return pa.plot_selected_chrimson_angle_traces(**locals())
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
        return pa.plot_wt_contact_group_angle_traces(**locals())
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
        # Keep the public notebook API on PlotCreator while delegating the
        # implementation to the angle-plotting module.
        return pa.flight_postural_change(**locals())
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
        return psc.plot_manual_sc_inverted_km_from_csv(**locals())
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
        return pl.plot_it_ot_landing_probability_and_latency(**locals())
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
        return psc.compare_manual_sc_rmst_across_contact_groups(**locals())
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
        return psc.plot_flywise_first_sc_probability_by_contact_group(**locals())
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
        return psc.plot_valid_sc_count_vs_landing_latency(**locals())
    def plot_left_TT_path_efficiency_grouped_stripplots(
            self,
            group_info,
            behavior_sources,
            file_name="left_TT_path_efficiency_grouped_stripplots",
            legs=("L-f", "L-m", "L-h"),
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="fixed",
            trajectory_window_s=0.10,
            min_frames=3,
            min_path_length=1e-6,
            sc_csv_path=None,
            colors=None,
            save_csv=True,
            n_perm=20000,
            apply_tracking_qc=False,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False
    ):
        return pg.plot_left_TT_path_efficiency_grouped_stripplots(**locals())
    def plot_chrimson_LP_metadata(
            self,
            group_info,
            color="red",
            tau=0.71,
            light_on_frame=750,
            min_trial_num=8
    ):
        return po.plot_chrimson_LP_metadata(**locals())
    def get_chrimson_metadata_on_ll_data(
            self,
            group_info,
            tau=0.71,
            light_on_frame=750,
            min_trial_num=8
    ):
        return po.get_chrimson_metadata_on_ll_data(
            group_info=group_info,
            tau=tau,
            light_on_frame=light_on_frame,
            min_trial_num=min_trial_num
        )
    def plot_gtacr_LP_change_summary(
            self,
            groups,
            file_name="GtACR_LP_change_summary",
            n_perm=20000,
            random_state=0,
            color="#0B6E2E",
            box_color="#B7E1B0"
    ):
        return po.plot_gtacr_LP_change_summary(**locals())
    def plot_kmc_and_unpaired_rmst_perm(self,
            data_list,
            file_name,
            tau=0.71,
            n_perm=20000,
            random_state=0,
            colors=None,
            invert_curve=False,
    ):
        return po.plot_kmc_and_unpaired_rmst_perm(**locals())
    def plot_TT_summary_metrics_vs_LL(
            self,
            group_info,
            leg="L-h",
            trial_types=("Landing", "Flying"),
            tau=0.71,
            trajectory_window_mode="mol_adjusted",
            trajectory_window_s=0.10,
            min_frames=3,
            min_path_length=1e-6,
            sc_csv_path=None,
            file_name="TT_summary_metrics_vs_LL",
            save_csv=True,
            n_perm=20000,
            random_state=0,
            apply_tracking_qc=False,
            min_cameras=2,
            max_interp_gap_s=0.02,
            min_valid_fraction=0.7,
            error_max=50,
            score_min=0.8,
            require_score=False
    ):
        return pg.plot_TT_summary_metrics_vs_LL(**locals())

