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


class PlotCreator:
    def __init__(self, platform_offset, platform_height, radius=0, fps=0):
        self.calculator = ku.SimpleCalculation()
        self.detector = ku.DetectCharacteristics(radius, fps)
        self.analyzer = ku.GroupDataAnalyzer(platform_offset=platform_offset, radius=radius, FPS=fps)
        self.manipulator = ku.FileManipulation()

        self.platform_offset = platform_offset
        self.radius = radius
        self.fps = fps
        self.platform_height = platform_height

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

        if not isinstance(ax, (list, tuple, np.ndarray)):
            axes = [ax]
        else:
            axes = ax.flatten() if isinstance(ax, np.ndarray) else ax

        for a in axes:
            a.spines["left"].set_linewidth(spine_width)
            a.spines["bottom"].set_linewidth(spine_width)

            a.tick_params(
                axis='both',
                width=tick_width,
                length=6
            )

            if xticks is not None:
                a.set_xticks(xticks)
            if yticks is not None:
                a.set_yticks(yticks)

            if xlabel is not None:
                a.set_xlabel(xlabel, fontsize=xlabel_size)
            if ylabel is not None:
                a.set_ylabel(ylabel, fontsize=ylabel_size)

    def centered_shades(self, color, n_shades=5, spread=0.6):
        import matplotlib.colors as mcolors

        if n_shades % 2 == 0:
            raise ValueError("n_shades should be odd to center on base color.")

        base_rgb = np.array(mcolors.to_rgb(color))
        factors = np.linspace(-spread, spread, n_shades)

        shades = []
        for f in factors:
            if f < 0:
                new_color = base_rgb * (1 + f)
            else:
                new_color = base_rgb + (1 - base_rgb) * f
            shades.append(tuple(new_color))

        return shades

    def _get_trial_meta(self, group_info, index):
        return group_info.trial_metadata[f"F{index[0]}T{index[1]}"]

    def _get_trial_obj(self, group_info, index):
        return group_info.fly_kinematic_data[f"F{index[0]}T{index[1]}"]

    def _ensure_trials_loaded(self, group_info, trial_types=None):
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        if trial_types is None:
            trial_types = ["Landing", "Flying", "NF", "NA"]

        group_info.read_kinematic_data(trial_types=trial_types)

    def plot_motion_vector_with_plane(self, kinematic_data, frame):
        line_points, plane_points, verts, cylinder_top, cylinder_bottom, direction, perp_vector1, perp_vector2 = (
            self.calculator.calculate_platform_surfaces(
                trial_info=kinematic_data,
                platform_offset=self.platform_offset,
                platform_height=self.platform_height,
                radius=self.radius
            )
        )

        coords = self.detector.ReadCoordsAll(kinematic_data, frame)

        center_points = self.calculator.ReadAndTranspose("platform-tip", kinematic_data)
        platform_ctr_pts_traces = np.array(center_points[200:250])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for g, group in enumerate(self.key_point_pairs):
            for i in range(len(group) - 1):
                p1 = coords[group[i]]
                p2 = coords[group[i + 1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        linewidth=5, marker='o', color=self.colors[g])

        side_surface = Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='none')
        ax.add_collection3d(side_surface)
        ax.plot_trisurf(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], color='cyan', alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        axis_limit = 1.5
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])

        plt.gca().set_aspect('equal')
        ax.view_init(elev=0, azim=180)
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        plt.savefig("Kinematic.pdf")
        plt.show(block=True)

    def plot_leg_angle_reaction(self, group_info, index, ax, joint, color):
        self._ensure_trials_loaded(group_info, trial_types=["Landing", "Flying"])

        self.angles = [
            ["R-fBC", "R-fCT", "R-fFT"],
            ["R-mBC", "R-mCT", "R-mFT"],
            ["R-hBC", "R-hCT", "R-hFT"],
            ["R-fCT", "R-fFT", "R-fTT"],
            ["R-mCT", "R-mFT", "R-mTT"],
            ["R-hCT", "R-hFT", "R-hTT"],
            ["R-fFT", "R-fTT", "R-fLT"],
            ["R-mFT", "R-mTT", "R-mLT"],
            ["R-hFT", "R-hTT", "R-hLT"]
        ]

        start = -0.2
        end = 0.7
        threshold = 0.71

        group_data = self.analyzer.Calculate_angle_traces(group_info, index, self.angles, threshold, start, end)
        frames = np.arange(int(start * 250), int(end * 250)) / 250

        # Calculate_angle_traces now returns a dict:
        # {joint_name: [trace, trace, ...]}. Keep the original angle-trace
        # calculation unchanged and only adapt this older plotting wrapper to
        # the current return format.
        trials = group_data.get(joint, [])

        if len(trials) == 0:
            print(f"No trace found for {group_info.group_name}-{joint}")
            return

        avg = np.nanmean(np.array(trials), axis=0)
        std = np.nanstd(np.array(trials), axis=0)

        sns.lineplot(x=frames, y=avg, color=color, linestyle="solid", linewidth=2,
                     label=f"{group_info.group_name}-{joint} (n = {len(trials)})", ax=ax)
        ax.fill_between(frames, avg - std, avg + std, color=color, alpha=0.2)
        ax.axvline(0, color="black", linestyle="dashed", label="MOC")

    def plot_angle_traces(self, groups, filename):
        start = -0.2
        stop = 0.7
        color = ["blue", "red", "green"]
        joints_to_plot = ["R-fFT", "R-mFT", "R-hFT"]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for g, group_info in enumerate(groups):
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()
                group_info.filter_nan_fly()
            ind = group_info.get_targeted_trials(("Landing", "Flying"))
            # S_ind, F_ind = group_info.get_SF_index()
            # dim_c = self.centered_shades(color[g], 5)[3]
            dark_c = self.centered_shades(color[g], 5)[0]

            self.plot_leg_angle_reaction(group_info, ind, ax, joints_to_plot[g], dark_c)
            # self.plot_leg_angle_reaction(group_info, F_ind, ax, joints_to_plot[g], dim_c)

        self.formatting(ax, [start, 0, stop], [0, 90, 180],
                        xlabel="seconds (s)", ylabel="T2-R-FT joint angle")
        sns.despine(trim=True)
        plt.legend()
        plt.savefig(f"{filename}.pdf")
        plt.show()

    def plot_group_angle_trace_opto(self, group_info:Group, angles,
                                    start=-0.2, end=0.7,
                                    threshold=0.71,
                                    colors=None,
                                    show_std=True,
                                    chrimson=False):
        if colors is None:
            colors = {
                "OFF": "blue",
                "ON": "red"
            }
        else:
            temp = colors
            colors = [self.centered_shades(colors, 11)[2],
                      self.centered_shades(colors, 11)[9]]

        # prepare opto metadata and load kinematic trials
        if len(group_info.trial_metadata) == 0:
            if chrimson:
                group_info.initialize_Chr_manual_data()
            else:
                group_info.initialize_manual_data()
        print("Reading kinematic data")
        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying"])


        ON_index, OFF_index = group_info.get_ON_OFF_index()

        frames = np.arange(int(start * 250), int(end * 250)) / 250

        fig, ax = plt.subplots(2, 1, figsize=(5, 8))

        stat_rows = []

        c = 0
        for condition, index_to_iterate in zip(["OFF", "ON"], [OFF_index, ON_index]):
            if len(index_to_iterate) == 0:
                print(f"No {condition} trials found for {group_info.group_name}")
                continue

            group_data = self.analyzer.Calculate_angle_traces(
                group_info=group_info,
                index_to_iterate=index_to_iterate,
                angles=angles,
                threshold=threshold,
                start=start,
                end=end,
                chrimson=chrimson
            )

            for a, angle_def in enumerate(angles):
                joint_name = angle_def[1]
                trials = group_data[joint_name]
                ax[a].set_ylabel(f"{joint_name} angle (°)")
                # ax.set_ylabel(f"{joint_name} angle (°)")

                if len(trials) == 0:
                    print(f"No valid {condition} trace found for {group_info.group_name}-{joint_name}")
                    continue

                trials = np.asarray(trials)
                avg = np.nanmean(trials, axis=0)
                sem = np.nanstd(trials, axis=0) / np.sqrt(len(trials))

                # if multiple angles are plotted, use solid/dashed to separate angles
                if a == 0:
                    linestyle = "solid"
                elif a == 1:
                    linestyle = "dotted"
                elif a == 2:
                    linestyle = "dashdot"
                else:
                    linestyle = "dotted"

                sns.lineplot(
                    x=frames,
                    y=avg,
                    ax=ax[a],
                    # ax=ax,
                    color=colors[c],
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{condition} (n={len(trials)})",
                )

                if show_std:
                    ax[a].fill_between(frames, avg - sem, avg + sem, color=colors[c], alpha=0.15)
                    # ax.fill_between(frames, avg - sem, avg + sem, color=colors[c], alpha=0.15)
                stat_rows.append({
                    "Group": group_info.group_name,
                    "Condition": condition,
                    "Joint": joint_name,
                    "n_trials": len(trials),
                    "start": start,
                    "end": end
                })
            c += 1
        # ax.axvline(0, color="black", linestyle="dashed", linewidth=2)
        ax[0].axvline(0, color="black", linestyle="dashed", linewidth=2)
        ax[1].axvline(0, color="black", linestyle="dashed", linewidth=2)

        ax[0].axvline(2, color="black", linestyle="dashed", linewidth=2)
        ax[1].axvline(2, color="black", linestyle="dashed", linewidth=2)
        self.formatting(
            ax,
            xticklabel=[str(start), "ON\n0", "OFF\n2", str(end)],
            xticks=[start, 0, 2, end],
            # xticks=[start, 0, end],
            yticks=[0, 90, 180],
            xlabel="Time (s)",
        )

        sns.despine(trim=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{group_info.group_name}-angle.pdf")
        # plt.show()
        plt.close()

        # pd.DataFrame(stat_rows).to_csv(f"{file_name}-stat.csv", index=False)

    def plot_selected_chrimson_angle_traces(
            self,
            groups,
            angles=None,
            file_name="selected_CsChrimson_angle_traces",
            start=-0.5,
            end=3,
            condition="ON",
            colors=None,
            show_sem=True
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
                ["wing", "wing", "wing"],
            ]
        if len(angles) < 2:
            raise ValueError("Provide at least two angle definitions: leg angle and wing angle.")

        if colors is None:
            colors = sns.color_palette("tab10", len(group_items))

        condition = str(condition).upper()
        frames = np.arange(int(start * 250), int(end * 250)) / 250
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
        stat_rows = []

        for group_idx, (group_label, group_info) in enumerate(group_items):
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_Chr_manual_data()

            group_info.filter_opto_data()
            group_info.read_kinematic_data(["Landing", "Flying"])
            on_index, off_index = group_info.get_ON_OFF_index()
            index_to_iterate = on_index if condition == "ON" else off_index

            if len(index_to_iterate) == 0:
                print(f"No {condition} trials found for {group_info.group_name}")
                continue

            group_data = self.analyzer.Calculate_angle_traces(
                group_info=group_info,
                index_to_iterate=index_to_iterate,
                angles=angles,
                start=start,
                end=end,
                chrimson=True
            )

            color = colors[group_idx % len(colors)]
            for angle_idx, angle_def in enumerate(angles[:2]):
                ax = axes[angle_idx]
                joint_name = angle_def[1]
                traces = group_data.get(joint_name, [])
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
                    "n_trials": len(traces),
                    "start": start,
                    "end": end,
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
        axes[0].legend(handles=style_handles, frameon=True, fontsize=8, loc="upper right")
        sns.despine(trim=True)
        plt.tight_layout()

        stat_df = pd.DataFrame(stat_rows)
        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
            stat_df.to_csv(f"{file_name}_summary.csv", index=False)
        plt.close(fig)
        return fig, axes, stat_df

    def plot_chrimson_on_off_latency_km(
            self,
            group_info,
            file_name="CsChrimson_ON_OFF_landing_latency_KM",
            color="#D73027",
            tau=0.71,
            linestyles=None,
            save_csv=True
    ):
        """
        Plot one CsChrimson group's inverted landing-latency KM curves for OFF
        and ON trials. ON and OFF share the same color and are separated by
        line style so the light condition is the visual comparison.

        CsChrimson metadata does not provide usable landing latency for this
        analysis. Latency is therefore recalculated from wing-folding detection,
        matching the existing plot_chrimson_LP logic.
        """
        if linestyles is None:
            linestyles = {
                "OFF": "solid",
                "ON": "dashed",
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_Chr_manual_data()

        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying", "NF", "NA"])
        on_index, off_index = group_info.get_ON_OFF_index()

        wing_angle_def = [["L-wing", "L-wing-hinge", "R-wing"]]
        rows = []
        for condition, indexes in (("OFF", off_index), ("ON", on_index)):
            for index in sorted(indexes, key=lambda item: (item[0], item[1])):
                key = group_info._trial_key(index[0], index[1])
                if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                    continue

                trial_info = group_info.fly_kinematic_data[key]
                meta = group_info.trial_metadata[key]

                # Preserve the established CsChrimson latency convention:
                # detect landing from wing folding in the post-light window.
                wing_angle = self.calculator.Calculate_joint_angle(
                    trial_info,
                    wing_angle_def
                )["L-wing-hinge"][750:1250]
                mol = self.detector.detect_landing(wing_angle)

                if mol == -1 or (mol / 250) > tau:
                    latency = tau
                    event = 0
                else:
                    latency = mol / 250
                    event = 1

                rows.append({
                    "Group_Name": group_info.group_name,
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Condition": condition,
                    "Light": condition,
                    "TrialType": meta["TrialType"],
                    "Latency": latency,
                    "Event": event,
                    "Detected_MOL_Frame_From_Light_Window": mol,
                    "Threshold_s": tau,
                })

        ll_df = pd.DataFrame(rows)
        if ll_df.empty:
            raise ValueError(f"No wing-detected landing-latency rows found for {group_info.group_name}.")

        km_rows = []
        stat_rows = []
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        kmf = KaplanMeierFitter()

        for condition in ("OFF", "ON"):
            sub = ll_df[ll_df["Condition"] == condition].copy()
            if sub.empty:
                continue

            label = f"{condition}"
            kmf.fit(
                sub["Latency"],
                event_observed=sub["Event"],
                label=label
            )
            survival = kmf.survival_function_[label]
            landing_probability = 1 - survival
            ax.step(
                survival.index.values,
                landing_probability.values,
                where="post",
                color=color,
                linestyle=linestyles.get(condition, "solid"),
                linewidth=2.6,
                label=label
            )

            for time_s, probability in zip(survival.index.values, landing_probability.values):
                km_rows.append({
                    "Group_Name": group_info.group_name,
                    "Condition": condition,
                    "Time_s": time_s,
                    "Landing_Probability": probability,
                })

            stat_rows.append({
                "Group_Name": group_info.group_name,
                "Condition": condition,
                "n_trials": int(len(sub)),
                "n_events": int(sub["Event"].sum()),
                "n_censored": int(len(sub) - sub["Event"].sum()),
                "event_fraction": float(sub["Event"].mean()),
                "tau_s": tau,
                })

        off_df = ll_df[ll_df["Condition"] == "OFF"]
        on_df = ll_df[ll_df["Condition"] == "ON"]
        logrank_df = pd.DataFrame()
        if not off_df.empty and not on_df.empty:
            result = logrank_test(
                off_df["Latency"],
                on_df["Latency"],
                event_observed_A=off_df["Event"],
                event_observed_B=on_df["Event"]
            )
            logrank_df = pd.DataFrame([{
                "Group_Name": group_info.group_name,
                "Comparison": "OFF_vs_ON",
                "test": "logrank",
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "n_OFF": int(len(off_df)),
                "n_ON": int(len(on_df)),
            }])

        ax.set_title(f"{group_info.group_name}: ON/OFF landing latency")
        ax.set_xlim(0, tau)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=True, title="Light", loc="lower right")
        self.formatting(
            ax,
            xticks=[0, tau / 2, tau],
            yticks=[0, 0.5, 1],
            xlabel="Time (s)",
            ylabel="Landing probability"
        )
        sns.despine(trim=True)
        plt.tight_layout()

        if save_csv and file_name is not None:
            ll_df.to_csv(f"{file_name}_wing_detected_latency_data.csv", index=False)
            pd.DataFrame(km_rows).to_csv(f"{file_name}_curve.csv", index=False)
            pd.DataFrame(stat_rows).to_csv(f"{file_name}_summary.csv", index=False)
            if not logrank_df.empty:
                logrank_df.to_csv(f"{file_name}_logrank.csv", index=False)
        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig, ax, ll_df, logrank_df

    def plot_chrimson_on_off_leg_wing_angle_traces(
            self,
            group_info,
            angles=None,
            file_name="CsChrimson_ON_OFF_RmFT_wing_angle_traces",
            start=-0.5,
            end=3,
            color="#D73027",
            condition_linestyles=None,
            show_sem=True,
            save_csv=True
    ):
        """
        Plot one CsChrimson group's ON/OFF R-mFT and wing angle traces.
        The two panels separate angle type; ON and OFF use the same color with
        different line styles to emphasize the light condition.
        """
        if angles is None:
            angles = [
                ["R-mCT", "R-mFT", "R-mTT"],
                ["wing", "wing", "wing"],
            ]
        if len(angles) < 2:
            raise ValueError("Provide leg and wing angle definitions.")

        if condition_linestyles is None:
            condition_linestyles = {
                "OFF": "solid",
                "ON": "dashed",
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_Chr_manual_data()

        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying"])
        on_index, off_index = group_info.get_ON_OFF_index()
        condition_indexes = {
            "OFF": off_index,
            "ON": on_index,
        }

        frames = np.arange(int(start * 250), int(end * 250)) / 250
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.6), sharex=True)
        summary_rows = []

        for condition, indexes in condition_indexes.items():
            if len(indexes) == 0:
                print(f"No {condition} trials found for {group_info.group_name}")
                continue

            group_data = self.analyzer.Calculate_angle_traces(
                group_info=group_info,
                index_to_iterate=indexes,
                angles=angles,
                start=start,
                end=end,
                chrimson=True
            )

            for angle_idx, angle_def in enumerate(angles[:2]):
                joint_name = angle_def[1]
                traces = group_data.get(joint_name, [])
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

                ax = axes[angle_idx]
                x = frames[:len(mean_trace)]
                ax.plot(
                    x,
                    mean_trace,
                    color=color,
                    linestyle=condition_linestyles.get(condition, "solid"),
                    linewidth=2.4,
                    label=condition
                )
                if show_sem:
                    ax.fill_between(
                        x,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=color,
                        alpha=0.14 if condition == "ON" else 0.09,
                        linewidth=0
                    )

                summary_rows.append({
                    "Group": group_info.group_name,
                    "Condition": condition,
                    "Joint": joint_name,
                    "Trace_Type": "R-mFT angle" if angle_idx == 0 else "Wing angle",
                    "n_trials": len(traces),
                    "start": start,
                    "end": end,
                })

        for axis_idx, ax in enumerate(axes):
            ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
            ax.axvline(2, color="black", linestyle="--", linewidth=1.0)
            ax.set_xlim(start, end)
            ax.set_ylabel("R-mFT angle (deg)" if axis_idx == 0 else "Wing angle (deg)")
            self.formatting(
                ax,
                xticks=[start, 0, 2, end],
                xlabel="Time from light ON (s)" if axis_idx == 1 else None
            )

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), frameon=True, title="Light", loc="upper right")
        axes[0].set_title(f"{group_info.group_name}: ON/OFF R-mFT and wing angle traces")
        sns.despine(trim=True)
        plt.tight_layout()

        summary_df = pd.DataFrame(summary_rows)
        if save_csv and file_name is not None:
            summary_df.to_csv(f"{file_name}_summary.csv", index=False)
        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig, axes, summary_df

    def plot_TT_trajectories_moc_origin_projected_plane(
            self,
            group_info,
            tt_joints=("L-fTT", "L-mTT", "L-hTT"),
            plane_axis=("R-mBC", "L-mBC"),
            reference_axis=("R-mBC", "R-hBC"),
            trial_indexes=None,
            file_name="TT_trajectories_MOC_origin_projected_plane",
            trial_types=("Landing", "Flying"),
            axis_average_frames=100,
            axis_average_anchor="moc",
            target_fps=250,
            average_mode="nan_pad",
            normalized_average_points=200,
            column_labels=None,
            colors=None,
            linewidth=0.3,
            alpha=0.2,
            show_start_end=False,
            show_average=True,
            average_linewidth=2.4,
            average_alpha=1.0,
            min_frames=3,
            save_csv=False
    ):
        """
        Plot projected TT trajectories from MOC to MOL.

        group_info can be one Group, a list/tuple of Groups, or a dict such as
        {"T1": groups["WT_T1_TTa"], "T2": ..., "T3": ...}. Multiple groups are
        plotted as columns, giving a len(tt_joints) x n_groups grid.

        average_mode:
        - "nan_pad": resample to target_fps and pad shorter trials with NaN.
        - "time_normalized": resample every trial to normalized MOC-to-MOL time.
        """
        if axis_average_anchor not in {"moc", "mol", "moc_to_mol"}:
            raise ValueError("axis_average_anchor must be 'moc', 'mol', or 'moc_to_mol'.")
        if average_mode not in {"nan_pad", "time_normalized"}:
            raise ValueError("average_mode must be 'nan_pad' or 'time_normalized'.")
        if axis_average_frames < 1:
            raise ValueError("axis_average_frames must be >= 1.")
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0.")
        if normalized_average_points < min_frames:
            raise ValueError("normalized_average_points must be >= min_frames.")

        if isinstance(tt_joints, str):
            tt_joints = (tt_joints,)
        else:
            tt_joints = tuple(tt_joints)
        if len(tt_joints) == 0:
            raise ValueError("tt_joints must contain at least one joint name.")
        if len(plane_axis) != 2 or len(reference_axis) != 2:
            raise ValueError("plane_axis and reference_axis must each contain two keypoint names.")

        if isinstance(group_info, dict):
            group_items = list(group_info.items())
        elif isinstance(group_info, (list, tuple)):
            group_items = [(group.group_name, group) for group in group_info]
        else:
            group_items = [(group_info.group_name, group_info)]

        if column_labels is not None:
            if len(column_labels) != len(group_items):
                raise ValueError("column_labels length must match number of groups.")
            group_items = [(label, item[1]) for label, item in zip(column_labels, group_items)]

        if trial_indexes is not None and not isinstance(trial_indexes, dict):
            if isinstance(trial_indexes, tuple) and len(trial_indexes) == 2:
                trial_indexes = [trial_indexes]
            trial_indexes = [tuple(index) for index in trial_indexes]

        if colors is None:
            colors = sns.color_palette("tab10", len(tt_joints))
        joint_colors = {joint: colors[i % len(colors)] for i, joint in enumerate(tt_joints)}

        required_points = set(tt_joints) | set(plane_axis) | set(reference_axis)
        rows = []
        skipped_rows = []
        projected_rows = []

        def average_slice(total_frames, moc, mol):
            if axis_average_anchor == "moc":
                start = min(moc - axis_average_frames, total_frames)
                stop = moc
            elif axis_average_anchor == "mol":
                stop = min(mol + 1, total_frames)
                start = max(stop - axis_average_frames, 0)
            else:
                start = moc
                stop = min(mol + 1, total_frames)
            return None if stop <= start else slice(start, stop)

        def read_axis(point_a, point_b, trial_info, avg_slice):
            coords_a = self.calculator.ReadAndTranspose(point_a, trial_info).astype(float)
            coords_b = self.calculator.ReadAndTranspose(point_b, trial_info).astype(float)
            mean_a = np.nanmean(coords_a[avg_slice], axis=0)
            mean_b = np.nanmean(coords_b[avg_slice], axis=0)
            return mean_a, mean_b - mean_a

        def unit(vector, name):
            vector = np.asarray(vector, dtype=float)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError(f"{name} has near-zero length.")
            return vector / norm

        for group_label, current_group in group_items:
            if len(current_group.trial_metadata) == 0:
                current_group.initialize_manual_data()
            if trial_indexes is None:
                current_group.filter_nan_fly()
            current_group.read_kinematic_data(list(trial_types))

            if isinstance(trial_indexes, dict):
                indexes = trial_indexes.get(group_label, trial_indexes.get(current_group.group_name, []))
                indexes = [tuple(index) for index in indexes]
            elif trial_indexes is None:
                indexes = current_group.get_targeted_trials(list(trial_types))
            else:
                indexes = trial_indexes

            for index in indexes:
                key = current_group._trial_key(index[0], index[1])
                if key not in current_group.fly_kinematic_data:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "kinematic trial was not loaded/found"})
                    continue

                trial_info = current_group.fly_kinematic_data[key]
                missing = [point for point in required_points if point not in trial_info.trial_data]
                if missing:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"missing required points: {missing}"})
                    continue

                moc = trial_info.moc
                mol = trial_info.mol
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(mol) or pd.isna(fps):
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing MOC/MOL/fps"})
                    continue
                moc = int(moc)
                mol = int(mol)
                if moc < 0 or mol <= moc or moc >= trial_info.total_frames_number:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"invalid MOC/MOL: MOC={moc}, MOL={mol}"})
                    continue
                mol = min(mol, trial_info.total_frames_number - 1)

                avg_slice = average_slice(trial_info.total_frames_number, moc, mol)
                if avg_slice is None:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "empty axis averaging window"})
                    continue

                try:
                    plane_origin, plane_vector = read_axis(plane_axis[0], plane_axis[1], trial_info, avg_slice)
                    _, reference_vector = read_axis(reference_axis[0], reference_axis[1], trial_info, avg_slice)
                    plane_normal = unit(plane_vector, "plane_axis")
                    reference_vector = unit(reference_vector, "reference_axis")
                    basis_x = reference_vector - np.dot(reference_vector, plane_normal) * plane_normal
                    basis_x = unit(basis_x, "reference_axis projected onto plane")
                    basis_y = unit(np.cross(plane_normal, basis_x), "projected y-axis")
                except ValueError as exc:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": str(exc)})
                    continue

                frames = np.arange(moc, mol + 1, dtype=int)
                if len(frames) < min_frames:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"fewer than {min_frames} frames from MOC to MOL"})
                    continue

                for joint in tt_joints:
                    xyz = self.calculator.ReadAndTranspose(joint, trial_info).astype(float)
                    coords = []
                    for frame in frames:
                        point = xyz[frame]
                        if not np.all(np.isfinite(point)):
                            coords.append([np.nan, np.nan])
                            continue
                        point_on_plane = point - np.dot(point - plane_origin, plane_normal) * plane_normal
                        relative = point_on_plane - plane_origin
                        coords.append([float(np.dot(relative, basis_x)), float(np.dot(relative, basis_y))])

                    coords = np.asarray(coords, dtype=float)
                    if len(coords) == 0 or not np.all(np.isfinite(coords[0])):
                        continue
                    coords = coords - coords[0]
                    finite = np.all(np.isfinite(coords), axis=1)
                    if np.sum(finite) < min_frames:
                        continue

                    time_s = (frames - moc) / fps
                    projected_rows.append({
                        "Group_Label": group_label,
                        "Group_Name": current_group.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Joint": joint,
                        "Color": joint_colors[joint],
                        "Coords": coords,
                        "Finite": finite,
                        "Time_s": time_s,
                    })

                    for i, frame in enumerate(frames):
                        rows.append({
                            "Group_Label": group_label,
                            "Group_Name": current_group.group_name,
                            "Fly#": index[0],
                            "Trial#": index[1],
                            "Index": str(index),
                            "Joint": joint,
                            "Frame": int(frame),
                            "Time_From_MOC_s": time_s[i],
                            "Projected_X_MOC_origin": coords[i, 0],
                            "Projected_Y_MOC_origin": coords[i, 1],
                            "Plane_Axis_A": plane_axis[0],
                            "Plane_Axis_B": plane_axis[1],
                            "Reference_Axis_A": reference_axis[0],
                            "Reference_Axis_B": reference_axis[1],
                            "Axis_Average_Anchor": axis_average_anchor,
                            "Axis_Average_Start_Frame": avg_slice.start,
                            "Axis_Average_End_Frame": avg_slice.stop - 1,
                            "Target_FPS_For_Average": target_fps,
                            "Average_Mode": average_mode,
                        })

        trajectory_df = pd.DataFrame(rows)
        skipped_df = pd.DataFrame(skipped_rows)
        if trajectory_df.empty:
            raise ValueError(f"No valid TT trajectories were available for plotting. Skipped: {skipped_rows}")

        fig, axes = plt.subplots(
            len(tt_joints),
            len(group_items),
            figsize=(4.2 * len(group_items), max(3.2, 3.0 * len(tt_joints))),
            squeeze=False
        )
        average_rows = []

        def compute_average(joint_rows):
            prepared = []
            max_times = []
            for row in joint_rows:
                coords = row["Coords"]
                time_s = np.asarray(row["Time_s"], dtype=float)
                valid = row["Finite"] & np.isfinite(time_s)
                if np.sum(valid) < min_frames:
                    continue
                valid_time = time_s[valid]
                valid_coords = coords[valid]
                order = np.argsort(valid_time)
                valid_time = valid_time[order]
                valid_coords = valid_coords[order]
                unique_time, unique_idx = np.unique(valid_time, return_index=True)
                valid_coords = valid_coords[unique_idx]
                if len(unique_time) < min_frames:
                    continue
                prepared.append((unique_time, valid_coords))
                max_times.append(float(unique_time[-1]))
            if not prepared:
                return None

            if average_mode == "time_normalized":
                average_time = np.linspace(0, 1, normalized_average_points)
                xs = [np.interp(average_time, t / t[-1], c[:, 0]) for t, c in prepared]
                ys = [np.interp(average_time, t / t[-1], c[:, 1]) for t, c in prepared]
                n_contributing = np.full(len(average_time), len(prepared), dtype=int)
                return average_time, np.nanmean(xs, axis=0), np.nanmean(ys, axis=0), n_contributing, 1.0

            average_end = max(max_times)
            average_time = np.arange(0, average_end + (0.5 / target_fps), 1 / target_fps)
            average_time = average_time[average_time <= average_end]
            if len(average_time) < min_frames:
                return None
            xs = []
            ys = []
            for valid_time, valid_coords in prepared:
                in_range = average_time <= valid_time[-1]
                x_values = np.full(len(average_time), np.nan, dtype=float)
                y_values = np.full(len(average_time), np.nan, dtype=float)
                x_values[in_range] = np.interp(average_time[in_range], valid_time, valid_coords[:, 0])
                y_values[in_range] = np.interp(average_time[in_range], valid_time, valid_coords[:, 1])
                xs.append(x_values)
                ys.append(y_values)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            n_contributing = np.sum(np.isfinite(xs), axis=0)
            return average_time, np.nanmean(xs, axis=0), np.nanmean(ys, axis=0), n_contributing, average_end

        for row_i, joint in enumerate(tt_joints):
            for col_i, (group_label, _) in enumerate(group_items):
                ax = axes[row_i, col_i]
                joint_rows = [row for row in projected_rows if row["Joint"] == joint and row["Group_Label"] == group_label]
                for row in joint_rows:
                    coords = row["Coords"]
                    finite = row["Finite"]
                    ax.plot(coords[finite, 0], coords[finite, 1], color=row["Color"], linewidth=linewidth, alpha=alpha)
                    if show_start_end:
                        first = coords[finite][0]
                        last = coords[finite][-1]
                        ax.scatter(first[0], first[1], color=row["Color"], marker="o", s=20, edgecolor="black", linewidth=0.4)
                        ax.scatter(last[0], last[1], color=row["Color"], marker="s", s=20, edgecolor="black", linewidth=0.4)

                if show_average and joint_rows:
                    average = compute_average(joint_rows)
                    if average is not None:
                        average_time, mean_x, mean_y, n_contributing, average_end = average
                        valid_mean = np.isfinite(mean_x) & np.isfinite(mean_y)
                        if np.sum(valid_mean) >= min_frames:
                            color = joint_colors[joint]
                            ax.plot(mean_x[valid_mean], mean_y[valid_mean], color=color, linewidth=average_linewidth, alpha=average_alpha, zorder=5)
                            ax.scatter(mean_x[valid_mean][0], mean_y[valid_mean][0], color=color, marker="o", s=42, edgecolor="black", linewidth=0.6, zorder=6)
                            ax.scatter(mean_x[valid_mean][-1], mean_y[valid_mean][-1], color=color, marker="s", s=42, edgecolor="black", linewidth=0.6, zorder=6)

                            for t_value, x_value, y_value, n_value in zip(average_time, mean_x, mean_y, n_contributing):
                                average_rows.append({
                                    "Group_Label": group_label,
                                    "Joint": joint,
                                    "Average_Time_Value": t_value,
                                    "Average_Time_Unit": "normalized_progress" if average_mode == "time_normalized" else "seconds_from_MOC",
                                    "Average_Projected_X_MOC_origin": x_value,
                                    "Average_Projected_Y_MOC_origin": y_value,
                                    "n_trials": len(joint_rows),
                                    "n_trials_contributing": int(n_value),
                                    "Target_FPS": target_fps,
                                    "Average_Mode": average_mode,
                                    "Average_Window_End": average_end,
                                })

                ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
                ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{group_label}: {joint}")
                ax.text(
                    0.98,
                    0.96,
                    f"n={len(joint_rows)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9
                )
                if col_i == 0:
                    ax.set_ylabel("Projected Y")
                if row_i == len(tt_joints) - 1:
                    ax.set_xlabel("Projected X")

        average_df = pd.DataFrame(average_rows)
        x_values = trajectory_df["Projected_X_MOC_origin"].to_numpy(dtype=float)
        y_values = trajectory_df["Projected_Y_MOC_origin"].to_numpy(dtype=float)
        if not average_df.empty:
            x_values = np.concatenate([x_values, average_df["Average_Projected_X_MOC_origin"].to_numpy(dtype=float)])
            y_values = np.concatenate([y_values, average_df["Average_Projected_Y_MOC_origin"].to_numpy(dtype=float)])

        finite_xy = np.isfinite(x_values) & np.isfinite(y_values)
        if np.any(finite_xy):
            x_min, x_max = np.nanmin(x_values[finite_xy]), np.nanmax(x_values[finite_xy])
            y_min, y_max = np.nanmin(y_values[finite_xy]), np.nanmax(y_values[finite_xy])
            x_pad = max((x_max - x_min) * 0.08, 0.05)
            y_pad = max((y_max - y_min) * 0.08, 0.05)
            x_lim = (x_min - x_pad, x_max + x_pad)
            y_lim = (y_min - y_pad, y_max + y_pad)

            from matplotlib.ticker import MaxNLocator
            x_locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            y_locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            for ax in axes.flatten():
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.xaxis.set_major_locator(x_locator)
                ax.yaxis.set_major_locator(y_locator)

        fig.suptitle(
            f"TT trajectories projected using {plane_axis[0]}->{plane_axis[1]} "
            f"and {reference_axis[0]}->{reference_axis[1]}"
        )
        sns.despine()
        plt.tight_layout()

        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
            if save_csv:
                trajectory_df.to_csv(f"{file_name}_projected_coordinates.csv", index=False)
                average_df.to_csv(f"{file_name}_average_projected_coordinates.csv", index=False)
                skipped_df.to_csv(f"{file_name}_skipped_trials.csv", index=False)
        plt.close(fig)
        return fig, axes, trajectory_df, average_df, skipped_df

    def plot_TT_MOC_to_SLC_endpoint_projected_scatter(
            self,
            group_info,
            sc_csv_paths,
            tt_joints=("L-fTT", "L-mTT", "L-hTT"),
            plane_axis=("R-mBC", "L-mBC"),
            reference_axis=("R-mBC", "R-hBC"),
            origin_keypoint="R-mBC",
            origin_frame="moc",
            trial_types=("Landing", "Flying"),
            tau=0.71,
            axis_average_frames=100,
            axis_average_anchor="moc",
            file_name="TT_MOC_to_SLC_endpoint_projected_scatter",
            colors=None,
            point_size=26,
            alpha=0.55,
            show_trajectories=True,
            show_points=True,
            show_aep=False,
            show_vep=False,
            extreme_point_size=32,
            connector_linewidth=0.25,
            connector_alpha=0.35,
            plot_radial_displacement=True,
            radial_circle_diameter=1.0,
            radial_linewidth=0.5,
            radial_alpha=0.35,
            radial_file_name=None,
            n_perm=20000,
            random_state=0,
            radial_stats_file_name=None,
            save_csv=True
    ):
        """
        Plot projected TT positions at MOC and at a leg-specific endpoint.

        Endpoint rule:
        - success: MOC -> valid SLC if present within MOC->MOL; otherwise MOC->MOL
        - failed: MOC -> valid SLC if present within MOC->MOC+tau; otherwise MOC->MOC+tau

        Optional extreme points:
        - AEP: the minimum projected X coordinate (leftmost point) in the trace
        - VEP: the minimum projected Y coordinate (lowest point) in the trace

        Radial displacement statistics are computed on fly-mean displacement
        vectors. For each joint and pair of contact groups, the primary test
        shuffles group labels and compares the Euclidean distance between the
        two group mean vectors.

        Coordinates are projected onto the plane whose normal is plane_axis.
        Projected X is calculated from the platform-tip motion direction fitted
        over frames 200-250: the platform motion is projected onto the plane,
        then crossed with the plane normal to define the reference X direction.
        The selected origin_keypoint is projected and subtracted so coordinates
        are relative to that anatomical/reference point rather than to TT at MOC.
        """
        if axis_average_anchor not in {"moc", "mol", "moc_to_endpoint"}:
            raise ValueError("axis_average_anchor must be 'moc', 'mol', or 'moc_to_endpoint'.")
        if origin_frame not in {"moc", "endpoint", "axis_average"}:
            raise ValueError("origin_frame must be 'moc', 'endpoint', or 'axis_average'.")
        if axis_average_frames < 1:
            raise ValueError("axis_average_frames must be >= 1.")
        if len(plane_axis) != 2 or len(reference_axis) != 2:
            raise ValueError("plane_axis and reference_axis must each contain two keypoint names.")
        if not any((show_trajectories, show_points, show_aep, show_vep)):
            raise ValueError(
                "At least one of show_trajectories, show_points, show_aep, or show_vep must be True."
            )
        if radial_circle_diameter <= 0:
            raise ValueError("radial_circle_diameter must be > 0.")
        if n_perm < 1:
            raise ValueError("n_perm must be >= 1.")

        if isinstance(tt_joints, str):
            tt_joints = (tt_joints,)
        else:
            tt_joints = tuple(tt_joints)
        if len(tt_joints) == 0:
            raise ValueError("tt_joints must contain at least one TT joint.")

        if isinstance(group_info, dict):
            group_items = list(group_info.items())
        elif isinstance(group_info, (list, tuple)):
            group_items = [(group.group_name, group) for group in group_info]
        else:
            group_items = [(group_info.group_name, group_info)]

        if not isinstance(sc_csv_paths, dict):
            if len(group_items) != 1:
                raise ValueError("sc_csv_paths must be a dict when plotting multiple groups.")
            sc_csv_paths = {group_items[0][0]: sc_csv_paths}

        if colors is None:
            colors = {
                "L-fTT": "#1f77b4",
                "L-mTT": "#d62728",
                "L-hTT": "#2ca02c",
            }
        joint_colors = {joint: colors.get(joint, colors.get(joint.replace("TT", ""), "black"))
                        if isinstance(colors, dict) else colors[i % len(colors)]
                        for i, joint in enumerate(tt_joints)}

        required_points = set(tt_joints) | set(plane_axis) | {origin_keypoint, "platform-tip"}
        rows = []
        trajectory_rows = []
        skipped_rows = []

        def unit(vector, name):
            vector = np.asarray(vector, dtype=float)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError(f"{name} has near-zero length.")
            return vector / norm

        def average_slice(total_frames, moc, endpoint):
            if axis_average_anchor == "moc":
                start = max(moc - axis_average_frames, 0)
                stop = moc
            elif axis_average_anchor == "mol":
                stop = min(endpoint + 1, total_frames)
                start = max(stop - axis_average_frames, 0)
            else:
                start = moc
                stop = min(endpoint + 1, total_frames)
            return None if stop <= start else slice(start, stop)

        def read_axis(point_a, point_b, trial_info, avg_slice):
            coords_a = self.calculator.ReadAndTranspose(point_a, trial_info).astype(float)
            coords_b = self.calculator.ReadAndTranspose(point_b, trial_info).astype(float)
            mean_a = np.nanmean(coords_a[avg_slice], axis=0)
            mean_b = np.nanmean(coords_b[avg_slice], axis=0)
            return mean_a, mean_b - mean_a

        def platform_motion_axis(trial_info, start_frame=200, stop_frame=250):
            platform_xyz = self.calculator.ReadAndTranspose("platform-tip", trial_info).astype(float)
            start = max(int(start_frame), 0)
            stop = min(int(stop_frame) + 1, len(platform_xyz))
            if stop - start < 2:
                raise ValueError("platform-tip motion window has fewer than 2 frames.")

            coords = platform_xyz[start:stop]
            coords = coords[np.all(np.isfinite(coords), axis=1)]
            if len(coords) < 2:
                raise ValueError("platform-tip motion window has fewer than 2 finite coordinates.")

            centered = coords - np.nanmean(coords, axis=0)
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            if singular_values[0] < 1e-8:
                raise ValueError("platform-tip motion window has near-zero movement.")
            motion = vh[0]

            net_motion = coords[-1] - coords[0]
            if np.dot(motion, net_motion) < 0:
                motion = -motion
            return motion, start, stop - 1

        def project_point(point, origin_3d, plane_normal, basis_x, basis_y):
            point = np.asarray(point, dtype=float)
            if not np.all(np.isfinite(point)):
                return np.nan, np.nan
            point_on_plane = point - np.dot(point - origin_3d, plane_normal) * plane_normal
            relative = point_on_plane - origin_3d
            return float(np.dot(relative, basis_x)), float(np.dot(relative, basis_y))

        def classify_trial(group, meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group.latency_threshold:
                return "Success"
            return "Failed"

        def lookup_sc_path(group_label, group):
            return sc_csv_paths.get(group_label, sc_csv_paths.get(group.group_name))

        for row_i, (group_label, current_group) in enumerate(group_items):
            sc_path = lookup_sc_path(group_label, current_group)
            if sc_path is None:
                raise ValueError(f"No SC CSV path provided for group '{group_label}'.")

            sc_df = pd.read_csv(sc_path)
            required_sc_columns = {"Index"} | {joint.replace("TT", "") for joint in tt_joints}
            missing_sc_columns = required_sc_columns.difference(sc_df.columns)
            if missing_sc_columns:
                raise ValueError(f"{group_label} SC CSV is missing columns: {sorted(missing_sc_columns)}")
            sc_lookup = {
                tuple(self.calculator.parse_index_cell(sc_row["Index"])): sc_row
                for _, sc_row in sc_df.iterrows()
            }

            if len(current_group.trial_metadata) == 0:
                current_group.initialize_manual_data()
                current_group.filter_nan_fly()
            current_group.read_kinematic_data(list(trial_types))

            for index in current_group.get_targeted_trials(list(trial_types)):
                index_tuple = tuple(index)
                key = current_group._trial_key(index[0], index[1])
                if key not in current_group.fly_kinematic_data or key not in current_group.trial_metadata:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing kinematic data or metadata"})
                    continue

                trial_info = current_group.fly_kinematic_data[key]
                meta = current_group.trial_metadata[key]
                missing_points = [point for point in required_points if point not in trial_info.trial_data]
                if missing_points:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"missing required points: {missing_points}"})
                    continue

                moc = trial_info.moc
                mol = trial_info.mol
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(fps):
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "missing MOC/fps"})
                    continue
                moc = int(moc)
                if moc < 0 or moc >= trial_info.total_frames_number:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"invalid MOC: {moc}"})
                    continue

                outcome = classify_trial(current_group, meta)
                if outcome == "Success":
                    if pd.isna(mol) or mol <= moc:
                        skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": f"success trial missing valid MOL: MOL={mol}"})
                        continue
                    valid_end = int(min(mol, trial_info.total_frames_number - 1))
                    fallback_rule = "MOL_no_valid_SLC"
                else:
                    valid_end = int(min(moc + tau * fps, trial_info.total_frames_number - 1))
                    fallback_rule = "MOC_plus_tau_no_valid_SLC"

                if valid_end <= moc:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "endpoint window is empty"})
                    continue

                avg_slice = average_slice(trial_info.total_frames_number, moc, valid_end)
                if avg_slice is None:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "empty axis averaging window"})
                    continue

                try:
                    plane_origin, plane_vector = read_axis(plane_axis[0], plane_axis[1], trial_info, avg_slice)
                    plane_normal = unit(plane_vector, "plane_axis")
                    platform_motion, motion_start, motion_stop = platform_motion_axis(trial_info)
                    platform_motion_on_plane = (
                            platform_motion
                            - np.dot(platform_motion, plane_normal) * plane_normal
                    )
                    basis_y = unit(platform_motion_on_plane, "platform-tip motion projected onto plane")
                    basis_x = unit(np.cross(basis_y, plane_normal), "platform-motion-derived projected x-axis")
                except ValueError as exc:
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": str(exc)})
                    continue

                origin_xyz = self.calculator.ReadAndTranspose(origin_keypoint, trial_info).astype(float)
                if origin_frame == "moc":
                    origin_3d = origin_xyz[moc]
                elif origin_frame == "endpoint":
                    origin_3d = origin_xyz[valid_end]
                else:
                    origin_3d = np.nanmean(origin_xyz[avg_slice], axis=0)
                origin_x, origin_y = project_point(origin_3d, plane_origin, plane_normal, basis_x, basis_y)
                if not np.isfinite(origin_x) or not np.isfinite(origin_y):
                    skipped_rows.append({"Group_Label": group_label, "Index": str(index), "Reason": "invalid projected origin"})
                    continue

                sc_row = sc_lookup.get(index_tuple)
                for joint in tt_joints:
                    leg = joint.replace("TT", "")
                    endpoint_frame = valid_end
                    endpoint_rule = fallback_rule
                    slc_frame = np.nan
                    slc_valid = False

                    if sc_row is not None:
                        slc_valid, candidate_slc_frame = self.calculator.validate_sc_frame_window(
                            sc_row[leg],
                            moc,
                            valid_end
                        )
                        if slc_valid:
                            endpoint_frame = int(min(candidate_slc_frame, trial_info.total_frames_number - 1))
                            endpoint_rule = "SLC"
                            slc_frame = int(candidate_slc_frame)

                    xyz = self.calculator.ReadAndTranspose(joint, trial_info).astype(float)
                    projected_trace = []
                    for frame in range(moc, endpoint_frame + 1):
                        x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                        if not np.isfinite(x) or not np.isfinite(y):
                            continue
                        projected_x = x - origin_x
                        projected_y = y - origin_y
                        projected_trace.append((frame, projected_x, projected_y))

                        if show_trajectories:
                            trajectory_rows.append({
                                "Group_Label": group_label,
                                "Group_Name": current_group.group_name,
                                "Index": str(index),
                                "Fly#": index[0],
                                "Trial#": index[1],
                                "Outcome": outcome,
                                "TrialType": meta["TrialType"],
                                "Joint": joint,
                                "Leg": leg,
                                "Frame": int(frame),
                                "Time_From_MOC_s": (frame - moc) / fps,
                                "Projected_X": projected_x,
                                "Projected_Y": projected_y,
                                "Endpoint_Frame": int(endpoint_frame),
                                "Endpoint_Rule": endpoint_rule,
                                "SLC_Frame": slc_frame,
                                "SLC_Valid_For_Window": slc_valid,
                                "Reference_X_Source": "platform_tip_motion_best_fit_200_250",
                                "Platform_Motion_Start_Frame": motion_start,
                                "Platform_Motion_End_Frame": motion_stop,
                            })

                    def append_point_row(point_type, frame, x, y, marker):
                        rows.append({
                            "Group_Label": group_label,
                            "Group_Name": current_group.group_name,
                            "Index": str(index),
                            "Fly#": index[0],
                            "Trial#": index[1],
                            "Outcome": outcome,
                            "TrialType": meta["TrialType"],
                            "Joint": joint,
                            "Leg": leg,
                            "Point_Type": point_type,
                            "Marker": marker,
                            "Frame": int(frame),
                            "Projected_X": x,
                            "Projected_Y": y,
                            "Origin_Keypoint": origin_keypoint,
                            "Origin_Frame_Mode": origin_frame,
                            "Endpoint_Frame": int(endpoint_frame),
                            "Endpoint_Rule": endpoint_rule,
                            "SLC_Frame": slc_frame,
                            "SLC_Valid_For_Window": slc_valid,
                            "Plane_Axis_A": plane_axis[0],
                            "Plane_Axis_B": plane_axis[1],
                            "Reference_X_Source": "platform_tip_motion_best_fit_200_250",
                            "Platform_Motion_Start_Frame": motion_start,
                            "Platform_Motion_End_Frame": motion_stop,
                            "Axis_Average_Anchor": axis_average_anchor,
                            "Axis_Average_Start_Frame": avg_slice.start,
                            "Axis_Average_End_Frame": avg_slice.stop - 1,
                        })

                    for point_type, frame, marker in (
                            ("MOC", moc, "o"),
                            ("Endpoint", endpoint_frame, "D"),
                    ):
                        x, y = project_point(xyz[frame], plane_origin, plane_normal, basis_x, basis_y)
                        if not np.isfinite(x) or not np.isfinite(y):
                            continue
                        append_point_row(point_type, frame, x - origin_x, y - origin_y, marker)

                    if projected_trace:
                        if show_aep:
                            aep_frame, aep_x, aep_y = min(projected_trace, key=lambda value: value[1])
                            append_point_row("AEP", aep_frame, aep_x, aep_y, "<")
                        if show_vep:
                            vep_frame, vep_x, vep_y = min(projected_trace, key=lambda value: value[2])
                            append_point_row("VEP", vep_frame, vep_x, vep_y, "v")

        point_df = pd.DataFrame(rows)
        trajectory_df = pd.DataFrame(trajectory_rows)
        skipped_df = pd.DataFrame(skipped_rows)
        if point_df.empty:
            raise ValueError(f"No valid projected TT endpoint points were available. Skipped: {skipped_rows}")

        radial_rows = []
        radial_group_cols = ["Group_Label", "Index", "Fly#", "Trial#", "Joint", "Leg"]
        for group_keys, sub in point_df.groupby(radial_group_cols):
            start = sub[sub["Point_Type"] == "MOC"]
            end = sub[sub["Point_Type"] == "Endpoint"]
            if start.empty or end.empty:
                continue
            start = start.iloc[0]
            end = end.iloc[0]
            dx = float(end["Projected_X"] - start["Projected_X"])
            dy = float(end["Projected_Y"] - start["Projected_Y"])
            radial_rows.append({
                "Group_Label": group_keys[0],
                "Index": group_keys[1],
                "Fly#": group_keys[2],
                "Trial#": group_keys[3],
                "Joint": group_keys[4],
                "Leg": group_keys[5],
                "Group_Name": end["Group_Name"],
                "Outcome": end["Outcome"],
                "TrialType": end["TrialType"],
                "Displacement_X": dx,
                "Displacement_Y": dy,
                "Displacement_Magnitude": float(np.hypot(dx, dy)),
                "Displacement_Angle_Deg": float(np.degrees(np.arctan2(dy, dx))),
                "Endpoint_Frame": end["Endpoint_Frame"],
                "Endpoint_Rule": end["Endpoint_Rule"],
                "SLC_Frame": end["SLC_Frame"],
                "SLC_Valid_For_Window": end["SLC_Valid_For_Window"],
            })
        radial_df = pd.DataFrame(radial_rows)

        radial_stats_df = pd.DataFrame()
        if not radial_df.empty:
            rng = np.random.default_rng(random_state)
            fly_vector_df = (
                radial_df
                .groupby(["Group_Label", "Joint", "Leg", "Fly#"], as_index=False)
                .agg(
                    Mean_Displacement_X=("Displacement_X", "mean"),
                    Mean_Displacement_Y=("Displacement_Y", "mean"),
                    n_trials=("Index", "nunique")
                )
            )
            fly_vector_df["Mean_Displacement_Magnitude"] = np.hypot(
                fly_vector_df["Mean_Displacement_X"],
                fly_vector_df["Mean_Displacement_Y"]
            )

            def vector_permutation_test(vectors_a, vectors_b):
                vectors_a = np.asarray(vectors_a, dtype=float)
                vectors_b = np.asarray(vectors_b, dtype=float)
                valid_a = np.all(np.isfinite(vectors_a), axis=1)
                valid_b = np.all(np.isfinite(vectors_b), axis=1)
                vectors_a = vectors_a[valid_a]
                vectors_b = vectors_b[valid_b]
                if len(vectors_a) == 0 or len(vectors_b) == 0:
                    return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                mean_a = np.mean(vectors_a, axis=0)
                mean_b = np.mean(vectors_b, axis=0)
                observed_dx = float(mean_b[0] - mean_a[0])
                observed_dy = float(mean_b[1] - mean_a[1])
                observed_distance = float(np.hypot(observed_dx, observed_dy))

                pooled = np.vstack([vectors_a, vectors_b])
                n_a = len(vectors_a)
                perm_stats = np.empty(n_perm, dtype=float)
                for perm_i in range(n_perm):
                    permuted = pooled[rng.permutation(len(pooled))]
                    perm_mean_a = np.mean(permuted[:n_a], axis=0)
                    perm_mean_b = np.mean(permuted[n_a:], axis=0)
                    perm_stats[perm_i] = np.hypot(
                        perm_mean_b[0] - perm_mean_a[0],
                        perm_mean_b[1] - perm_mean_a[1]
                    )

                p_value = (np.sum(perm_stats >= observed_distance) + 1) / (n_perm + 1)
                return (
                    observed_distance,
                    float(p_value),
                    observed_dx,
                    observed_dy,
                    float(mean_a[0]),
                    float(mean_a[1]),
                    float(mean_b[0]),
                    float(mean_b[1]),
                )

            stat_rows = []
            group_labels = [label for label, _ in group_items]
            for joint in tt_joints:
                joint_fly_df = fly_vector_df[fly_vector_df["Joint"] == joint]
                for group_a, group_b in itertools.combinations(group_labels, 2):
                    data_a = joint_fly_df[joint_fly_df["Group_Label"] == group_a]
                    data_b = joint_fly_df[joint_fly_df["Group_Label"] == group_b]
                    vectors_a = data_a[["Mean_Displacement_X", "Mean_Displacement_Y"]].to_numpy(dtype=float)
                    vectors_b = data_b[["Mean_Displacement_X", "Mean_Displacement_Y"]].to_numpy(dtype=float)
                    (
                        observed_vector_distance,
                        vector_p,
                        observed_dx,
                        observed_dy,
                        group_a_mean_x,
                        group_a_mean_y,
                        group_b_mean_x,
                        group_b_mean_y,
                    ) = vector_permutation_test(vectors_a, vectors_b)

                    def unpaired_or_nan(column):
                        values_a = data_a[column].to_numpy(dtype=float)
                        values_b = data_b[column].to_numpy(dtype=float)
                        values_a = values_a[np.isfinite(values_a)]
                        values_b = values_b[np.isfinite(values_b)]
                        if len(values_a) == 0 or len(values_b) == 0:
                            return np.nan, np.nan
                        return self.calculator._permutation_test_unpaired(
                            values_a,
                            values_b,
                            n_perm=n_perm,
                            rng=rng
                        )

                    x_diff, x_p = unpaired_or_nan("Mean_Displacement_X")
                    y_diff, y_p = unpaired_or_nan("Mean_Displacement_Y")
                    magnitude_diff, magnitude_p = unpaired_or_nan("Mean_Displacement_Magnitude")

                    stat_rows.append({
                        "Joint": joint,
                        "Leg": joint.replace("TT", ""),
                        "Group_A": group_a,
                        "Group_B": group_b,
                        "Test": "fly_mean_vector_label_shuffle",
                        "Primary_Statistic": "distance_between_group_mean_vectors",
                        "Observed_Vector_Distance": observed_vector_distance,
                        "Vector_Permutation_P": vector_p,
                        "Observed_Delta_X_GroupB_minus_GroupA": observed_dx,
                        "Observed_Delta_Y_GroupB_minus_GroupA": observed_dy,
                        "Group_A_Mean_X": group_a_mean_x,
                        "Group_A_Mean_Y": group_a_mean_y,
                        "Group_B_Mean_X": group_b_mean_x,
                        "Group_B_Mean_Y": group_b_mean_y,
                        "Secondary_X_Mean_Diff_GroupB_minus_GroupA": x_diff,
                        "Secondary_X_Permutation_P": x_p,
                        "Secondary_Y_Mean_Diff_GroupB_minus_GroupA": y_diff,
                        "Secondary_Y_Permutation_P": y_p,
                        "Secondary_Magnitude_Mean_Diff_GroupB_minus_GroupA": magnitude_diff,
                        "Secondary_Magnitude_Permutation_P": magnitude_p,
                        "Group_A_n_flies": int(data_a["Fly#"].nunique()),
                        "Group_B_n_flies": int(data_b["Fly#"].nunique()),
                        "Group_A_n_trials": int(data_a["n_trials"].sum()) if not data_a.empty else 0,
                        "Group_B_n_trials": int(data_b["n_trials"].sum()) if not data_b.empty else 0,
                        "n_perm": n_perm,
                        "random_state": random_state,
                    })
            radial_stats_df = pd.DataFrame(stat_rows)

        fig, axes = plt.subplots(
            len(group_items),
            1,
            figsize=(8.5, max(4.0, 4.0 * len(group_items))),
            sharex=True,
            sharey=True,
            squeeze=False
        )
        axes = axes[:, 0]

        for row_i, (group_label, _) in enumerate(group_items):
            ax = axes[row_i]
            group_df = point_df[point_df["Group_Label"] == group_label]
            for joint in tt_joints:
                joint_df = group_df[group_df["Joint"] == joint]
                color = joint_colors[joint]
                if show_trajectories and not trajectory_df.empty:
                    joint_traj_df = trajectory_df[
                        (trajectory_df["Group_Label"] == group_label)
                        & (trajectory_df["Joint"] == joint)
                    ]
                    for _, trial_traj in joint_traj_df.groupby(["Fly#", "Trial#"]):
                        trial_traj = trial_traj.sort_values("Frame")
                        ax.plot(
                            trial_traj["Projected_X"],
                            trial_traj["Projected_Y"],
                            color=color,
                            linewidth=connector_linewidth,
                            alpha=connector_alpha,
                            zorder=1
                        )
                if show_points:
                    for point_type, marker, label_suffix in (
                            ("MOC", "o", "MOC"),
                            ("Endpoint", "D", "endpoint"),
                    ):
                        sub = joint_df[joint_df["Point_Type"] == point_type]
                        if sub.empty:
                            continue
                        ax.scatter(
                            sub["Projected_X"],
                            sub["Projected_Y"],
                            s=point_size,
                            marker=marker,
                            color=color,
                            alpha=alpha,
                            edgecolors="black" if point_type == "Endpoint" else "none",
                            linewidths=0.35,
                            zorder=2,
                            label=f"{joint} {label_suffix}",
                        )
                for enabled, point_type, marker, label_suffix in (
                        (show_aep, "AEP", "<", "AEP"),
                        (show_vep, "VEP", "v", "VEP"),
                ):
                    if not enabled:
                        continue
                    sub = joint_df[joint_df["Point_Type"] == point_type]
                    if sub.empty:
                        continue
                    ax.scatter(
                        sub["Projected_X"],
                        sub["Projected_Y"],
                        s=extreme_point_size,
                        marker=marker,
                        color=color,
                        alpha=alpha,
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=3,
                        label=f"{joint} {label_suffix}",
                    )
            ax.axhline(0, color="0.82", linewidth=0.8, zorder=0)
            ax.axvline(0, color="0.82", linewidth=0.8, zorder=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{group_label}: TT position at MOC and endpoint")
            ax.set_ylabel(f"Projected Y from {origin_keypoint}")
            n_trials = group_df[["Fly#", "Trial#"]].drop_duplicates().shape[0]
            n_flies = group_df["Fly#"].nunique()
            ax.text(
                0.98,
                0.96,
                f"trials={n_trials}, flies={n_flies}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9
            )

        axes[-1].set_xlabel(f"Projected X from {origin_keypoint}")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), frameon=True, fontsize=8, loc="best", ncol=2)

        x_values = point_df["Projected_X"].to_numpy(dtype=float)
        y_values = point_df["Projected_Y"].to_numpy(dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if np.any(finite):
            x_min, x_max = np.nanmin(x_values[finite]), np.nanmax(x_values[finite])
            y_min, y_max = np.nanmin(y_values[finite]), np.nanmax(y_values[finite])
            x_pad = max((x_max - x_min) * 0.08, 0.05)
            y_pad = max((y_max - y_min) * 0.08, 0.05)
            for ax in axes:
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

        fig.suptitle(
            f"Projected TT MOC and endpoint positions using {plane_axis[0]}->{plane_axis[1]} normal "
            "and platform-tip motion-derived X"
        )
        sns.despine()
        plt.tight_layout()

        if file_name is not None:
            fig.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
            if save_csv:
                point_df.to_csv(f"{file_name}_projected_points.csv", index=False)
                trajectory_df.to_csv(f"{file_name}_projected_trajectories.csv", index=False)
                skipped_df.to_csv(f"{file_name}_skipped_trials.csv", index=False)
        plt.close(fig)

        if plot_radial_displacement and not radial_df.empty:
            radial_fig, radial_axes = plt.subplots(
                len(group_items),
                len(tt_joints),
                figsize=(4.2 * len(tt_joints), max(3.6, 3.6 * len(group_items))),
                sharex=True,
                sharey=True,
                squeeze=False
            )
            radial_radius = radial_circle_diameter / 2
            for row_i, (group_label, _) in enumerate(group_items):
                for col_i, joint in enumerate(tt_joints):
                    ax = radial_axes[row_i, col_i]
                    color = joint_colors[joint]
                    circle = plt.Circle(
                        (0, 0),
                        radial_radius,
                        fill=False,
                        edgecolor="0.65",
                        linewidth=0.9,
                        zorder=0
                    )
                    ax.add_patch(circle)
                    ax.axhline(0, color="0.86", linewidth=0.7, zorder=0)
                    ax.axvline(0, color="0.86", linewidth=0.7, zorder=0)

                    sub = radial_df[
                        (radial_df["Group_Label"] == group_label)
                        & (radial_df["Joint"] == joint)
                    ]
                    for _, row in sub.iterrows():
                        ax.plot(
                            [0, row["Displacement_X"]],
                            [0, row["Displacement_Y"]],
                            color=color,
                            linewidth=radial_linewidth,
                            alpha=radial_alpha,
                            zorder=1
                        )
                        ax.scatter(
                            row["Displacement_X"],
                            row["Displacement_Y"],
                            color=color,
                            s=10,
                            alpha=min(radial_alpha + 0.15, 1.0),
                            edgecolors="none",
                            zorder=2
                        )

                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlim(-radial_radius, radial_radius)
                    ax.set_ylim(-radial_radius, radial_radius)
                    if row_i == 0:
                        ax.set_title(joint)
                    if col_i == 0:
                        ax.set_ylabel(group_label)
                    n_trials = sub[["Fly#", "Trial#"]].drop_duplicates().shape[0]
                    n_flies = sub["Fly#"].nunique()
                    ax.text(
                        0.96,
                        0.94,
                        f"trials={n_trials}, flies={n_flies}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8
                    )

            radial_fig.suptitle("Projected TT endpoint displacement from MOC")
            sns.despine()
            radial_fig.tight_layout()

            if file_name is not None:
                radial_output = radial_file_name or f"{file_name}_radial_displacement"
                radial_fig.savefig(f"{radial_output}.pdf", dpi=300, bbox_inches="tight")
                if save_csv:
                    radial_df.to_csv(f"{radial_output}_data.csv", index=False)
                    radial_stats_output = radial_stats_file_name or f"{radial_output}_fly_mean_vector_stats"
                    radial_stats_df.to_csv(f"{radial_stats_output}.csv", index=False)
            plt.close(radial_fig)
        elif save_csv and file_name is not None:
            radial_output = radial_file_name or f"{file_name}_radial_displacement"
            radial_df.to_csv(f"{radial_output}_data.csv", index=False)
            radial_stats_output = radial_stats_file_name or f"{radial_output}_fly_mean_vector_stats"
            radial_stats_df.to_csv(f"{radial_stats_output}.csv", index=False)

        return fig, axes, point_df, trajectory_df, skipped_df

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
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import seaborn as sns

        combined_df = pd.concat(data_to_plot, ignore_index=True)

        fig, ax = plt.subplots(figsize=(len(data_to_plot) * 2, 8))

        group_names = combined_df["Group_Name"].unique()
        x_positions = np.arange(len(group_names))

        if colors is None:
            colors = sns.color_palette("tab20", len(data_to_plot))

        if markers is None:
            markers = ["o"] * len(data_to_plot)

        def get_style(style_input, i, label, default):
            if style_input is None:
                return default
            if isinstance(style_input, dict):
                return style_input.get(label, default)
            return style_input[i % len(style_input)]

        def soften_color(color, softness):
            """Blend a color toward white while retaining its group identity."""
            rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
            softness = float(np.clip(softness, 0, 1))
            return tuple(rgb + (1.0 - rgb) * softness)

        group_colors = []
        for group in group_names:
            matching_index = next(
                (
                    i for i, data in enumerate(data_to_plot)
                    if not data.empty and data["Group_Name"].iloc[0] == group
                ),
                0
            )
            group_colors.append(get_style(colors, matching_index, group, "black"))

        # ==========================================================
        # Boxplot shifted slightly LEFT
        # ==========================================================
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

        for i, patch in enumerate(bp["boxes"]):
            group_color = group_colors[i]
            if box_color is None:
                face_color = soften_color(group_color, box_softness)
            elif isinstance(box_color, dict):
                face_color = box_color.get(group_names[i], soften_color(group_color, box_softness))
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

        for i, group_color in enumerate(group_colors):
            for line in bp["whiskers"][2 * i:2 * i + 2]:
                line.set(color=group_color, linewidth=2)
            for line in bp["caps"][2 * i:2 * i + 2]:
                line.set(color=group_color, linewidth=2)

        # ==========================================================
        # Stripplot shifted slightly RIGHT
        # ==========================================================
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

            point_color = get_style(colors, i, group, "black")
            marker = get_style(markers, i, group, "o")

            ax.scatter(
                xvals,
                yvals,
                alpha=0.5,
                s=100,
                marker=marker,
                color=point_color,
                zorder=10
            )

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

        for group_info in groups:
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()
                group_info.filter_nan_fly()
            data_to_plot.append(group_info.get_LP_df())

        self.plot_LP_summary(
            data_to_plot=data_to_plot,
            file_name=file_name,
            colors=colors,
            markers=markers,
            box_color=box_color,
            box_width=box_width,
            box_softness=box_softness
        )

    def plot_LP_summary_light(self, combined_df, file_name, color):
        combined_df = combined_df.copy()
        combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

        # keep only flies that have both OFF and ON
        fly_counts = combined_df["Fly#"].value_counts()
        paired_flies = fly_counts[fly_counts == 2].index

        combined_df = combined_df[combined_df["Fly#"].isin(paired_flies)].copy()
        combined_df["Group_Name"] = pd.Categorical(combined_df["Group_Name"], categories=["OFF", "ON"], ordered=True)
        combined_df = combined_df.sort_values(by=["Fly#", "Group_Name"])

        paired_df = combined_df.pivot(index="Fly#", columns="Group_Name", values="LandingProb")
        paired_df = paired_df.dropna(subset=["OFF", "ON"])
        # paired_df.to_csv(f"{file_name}-paired_values.csv")

        fig, ax = plt.subplots(figsize=(4, 7))

        # ------------------------------------------------------------
        # exact box positions with matplotlib
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # paired lines centered at category positions
        # ------------------------------------------------------------
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

        # mean line centered
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
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_opto_data()

        combined_df = group_info.get_paired_LP_df()
        self.plot_LP_summary_light(combined_df, file_name, color)

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
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from lifelines import KaplanMeierFitter

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        kmf = KaplanMeierFitter()

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

        def get_style(style_input, i, label, default):
            if style_input is None:
                return default
            if isinstance(style_input, dict):
                return style_input.get(label, default)
            return style_input[i % len(style_input)]

        stat_out = []

        for i, d in enumerate(data_to_plot):
            if d is None or len(d) == 0:
                continue

            label = d["Group_Name"].iloc[0]

            kmf.fit(
                d["Latency"],
                event_observed=d["Event"],
                label=label
            )

            surv_df = kmf.survival_function_
            time = surv_df.index.values
            survival_prob = surv_df[label].values
            landing_prob = 1 - survival_prob

            line_color = get_style(colors, i, label, "black")
            line_style = get_style(linestyles, i, label, "solid")
            marker = get_style(markers, i, label, None)

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

            stat_out.append({
                "Group": label,
                "n": len(d),
                "event_num": int(np.sum(d["Event"])),
                "censored_num": int(len(d) - np.sum(d["Event"]))
            })

        pd.DataFrame(stat_out).to_csv(f"{file_name}-KM_stat.csv", index=False)

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

        self.plot_KM_curve(
            data_to_plot=data_to_plot,
            file_name=file_name,
            colors=colors,
            linestyles=linestyles,
            markers=markers,
            opto=opto,
            marker_every=marker_every
        )

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

        def collect_resampled_traces(group_info, leg, joint_type):
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()

            group_info.filter_nan_fly()
            group_info.read_kinematic_data(list(trial_types))

            angle_def = angle_definition(leg, joint_type)
            joint_name = angle_def[1]
            traces = []

            for index in group_info.get_targeted_trials(list(trial_types)):
                key = group_info._trial_key(index[0], index[1])
                if key not in group_info.fly_kinematic_data:
                    continue

                trial_info = group_info.fly_kinematic_data[key]
                moc = trial_info.moc
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(fps):
                    continue

                start_frame = int(round(moc + start_s * fps))
                end_frame = int(round(moc + end_s * fps))

                # Skip incomplete windows so every averaged trace represents
                # the same MOC-centered interval.
                if start_frame < 0 or end_frame >= trial_info.total_frames_number:
                    continue
                if any(point not in trial_info.trial_data for point in angle_def):
                    continue

                angle_trace = self.calculator.Calculate_joint_angle(trial_info, [angle_def])[joint_name]
                source_frames = np.arange(start_frame, end_frame + 1)
                source_time = (source_frames - moc) / fps
                source_trace = np.asarray(angle_trace[start_frame:end_frame + 1], dtype=float)

                valid = np.isfinite(source_time) & np.isfinite(source_trace)
                if np.sum(valid) < 2:
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

            return np.asarray(traces, dtype=float)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(column_labels),
            figsize=(5.2 * len(column_labels), 7.2),
            sharex=True,
            sharey=True,
            squeeze=False
        )

        summary_rows = []
        plotted_angle_values = []
        for col, column_label in enumerate(column_labels):
            contact_groups = groups_by_column[column_label]

            for row, (joint_type, ylabel) in enumerate(row_defs):
                ax = axes[row, col]

                for contact_group, group_info in contact_groups.items():
                    leg = contact_leg_map[contact_group]
                    traces = collect_resampled_traces(group_info, leg, joint_type)
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
                    })

                    if n_trials == 0:
                        continue

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
        if save_csv and file_name is not None:
            summary_df.to_csv(f"{file_name}_summary.csv", index=False)

        plt.close()
        return fig, axes, summary_df

    def _trial_indexes_from_labeled_ll_file(
            self,
            ll_file_path,
            behavior_label=None,
            selection_mode="numeric",
            fly_column="Fly#"
    ):
        """
        Return (fly, trial) indexes from an LL workbook used as a trial subset.

        selection_mode="numeric" selects cells that still contain LL values.
        selection_mode="marker" selects cells whose text equals behavior_label.
        """
        if selection_mode not in {"numeric", "marker"}:
            raise ValueError("selection_mode must be 'numeric' or 'marker'.")

        ll_df = pd.read_excel(ll_file_path)
        if fly_column not in ll_df.columns:
            raise ValueError(f"{ll_file_path} does not contain a '{fly_column}' column.")

        trial_indexes = []
        for _, row in ll_df.iterrows():
            fly = row[fly_column]
            if pd.isna(fly):
                continue

            for column in ll_df.columns:
                if column == fly_column:
                    continue

                value = row[column]
                if selection_mode == "numeric":
                    selected = pd.to_numeric(value, errors="coerce")
                    if pd.isna(selected):
                        continue
                else:
                    if behavior_label is None:
                        raise ValueError("behavior_label is required when selection_mode='marker'.")
                    if str(value).strip().upper() != str(behavior_label).strip().upper():
                        continue

                trial_text = str(column).replace("Trial_", "")
                try:
                    trial = int(trial_text)
                    trial_indexes.append((int(fly), trial))
                except ValueError:
                    continue

        return trial_indexes

    def _trial_sets_from_behavior_sources(self, behavior_sources):
        """
        Build named trial-index sets from LL-label workbooks.

        behavior_sources may be either:
        {"IT": "path/to/file.xlsx"}
        or:
        {"IT": {"path": "...", "selection_mode": "numeric"}}
        """
        trial_sets = {}
        for behavior_label, source in behavior_sources.items():
            if isinstance(source, dict):
                path = source["path"]
                selection_mode = source.get("selection_mode", "numeric")
                marker_label = source.get("marker_label", behavior_label)
            else:
                path = source
                selection_mode = "numeric"
                marker_label = behavior_label

            trial_sets[behavior_label] = self._trial_indexes_from_labeled_ll_file(
                ll_file_path=path,
                behavior_label=marker_label,
                selection_mode=selection_mode
            )

        return trial_sets

    def _resampled_angle_traces_for_indexes(
            self,
            group_info,
            trial_indexes,
            angle_def,
            start_s=-0.2,
            end_s=0.71,
            target_fps=250,
            trial_types=("Landing", "Flying")
    ):
        """Collect MOC-aligned angle traces and resample them to target_fps."""
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()

        group_info.read_kinematic_data(list(trial_types))
        target_n = int(round((end_s - start_s) * target_fps)) + 1
        target_time = np.linspace(start_s, end_s, target_n)
        joint_name = angle_def[1]
        traces = []

        for index in trial_indexes:
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            moc = trial_info.moc
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue

            start_frame = int(round(moc + start_s * fps))
            end_frame = int(round(moc + end_s * fps))
            if start_frame < 0 or end_frame >= trial_info.total_frames_number:
                continue
            if any(point not in trial_info.trial_data for point in angle_def):
                continue

            angle_trace = self.calculator.Calculate_joint_angle(trial_info, [angle_def])[joint_name]
            source_frames = np.arange(start_frame, end_frame + 1)
            source_time = (source_frames - moc) / fps
            source_trace = np.asarray(angle_trace[start_frame:end_frame + 1], dtype=float)
            valid = np.isfinite(source_time) & np.isfinite(source_trace)
            if np.sum(valid) < 2:
                continue

            traces.append(np.interp(
                target_time,
                source_time[valid],
                source_trace[valid],
                left=np.nan,
                right=np.nan
            ))

        return target_time, np.asarray(traces, dtype=float)

    def plot_angle_traces_by_trial_sets(
            self,
            group_info,
            angle_defs,
            trial_sets=None,
            behavior_sources=None,
            trial_types=("Landing", "Flying"),
            start_s=-0.2,
            end_s=0.71,
            target_fps=250,
            colors=None,
            show_sem=True,
            file_name="angle_traces_by_trial_sets",
            save_csv=True
    ):
        """
        Plot MOC-aligned average angle traces for named trial-index groups.

        Pass trial_sets={"Success": [(fly, trial), ...], "Failed": [...]} for
        arbitrary subsets, or behavior_sources for IT/OT LL-label workbooks.
        """
        if trial_sets is None:
            if behavior_sources is None:
                raise ValueError("Provide either trial_sets or behavior_sources.")
            trial_sets = self._trial_sets_from_behavior_sources(behavior_sources)

        if colors is None:
            colors = {
                "IT": "#8FD694",
                "OT": "#C7A0E8",
                "Success": "tab:blue",
                "Failed": "tab:red",
            }

        fig, axes = plt.subplots(
            len(angle_defs),
            1,
            figsize=(6.0, 3.2 * len(angle_defs)),
            sharex=True,
            sharey=True,
            squeeze=False
        )
        axes = axes[:, 0]
        summary_rows = []
        plotted_angle_values = []

        for row, angle_def in enumerate(angle_defs):
            ax = axes[row]
            joint_name = angle_def[1]

            for label, indexes in trial_sets.items():
                target_time, traces = self._resampled_angle_traces_for_indexes(
                    group_info=group_info,
                    trial_indexes=indexes,
                    angle_def=angle_def,
                    start_s=start_s,
                    end_s=end_s,
                    target_fps=target_fps,
                    trial_types=trial_types
                )
                n_trials = int(traces.shape[0]) if traces.ndim == 2 else 0
                summary_rows.append({
                    "Group_Name": group_info.group_name,
                    "Condition": label,
                    "Joint": joint_name,
                    "Angle_Definition": "-".join(angle_def),
                    "n_trials": n_trials,
                    "start_s": start_s,
                    "end_s": end_s,
                    "target_fps": target_fps,
                })
                if n_trials == 0:
                    continue

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

                color = colors.get(label, "black")
                ax.plot(
                    target_time,
                    mean_trace,
                    color=color,
                    linewidth=2.4,
                    label=f"{label} (n={n_trials})"
                )
                if show_sem:
                    ax.fill_between(
                        target_time,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=color,
                        alpha=0.20,
                        linewidth=0
                    )

            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(joint_name)
            ax.set_ylabel("Angle (deg)")
            self.formatting(
                ax,
                xticks=[start_s, 0, end_s],
                xlabel="Time from MOC (s)" if row == len(angle_defs) - 1 else "",
                ylabel="Angle (deg)"
            )
            if row == 0:
                ax.legend(frameon=False, fontsize=8)

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
                for ax in axes:
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)

        sns.despine(trim=True)
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        summary_df = pd.DataFrame(summary_rows)
        if save_csv and file_name is not None:
            summary_df.to_csv(f"{file_name}_summary.csv", index=False)

        return fig, axes, summary_df

    def compare_KM_two_groups(self, df1, df2, file_name):
        result = logrank_test(
            df1["Latency"], df2["Latency"],
            event_observed_A=df1["Event"],
            event_observed_B=df2["Event"]
        )

        out = pd.DataFrame({
            "Group_1": [df1["Group_Name"].iloc[0]],
            "Group_2": [df2["Group_Name"].iloc[0]],
            "test_statistic": [result.test_statistic],
            "p_value": [result.p_value]
        })
        out.to_csv(f"{file_name}-logrank.csv", index=False)

        return result

    def prepare_group_for_kinematic_summary(self, group_info):
        """
        Prepare a normal group for kinematic-dependent summary analysis.
        """
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(["Landing", "Flying"])

    def prepare_opto_group_for_kinematic_summary(self, group_info):
        """
        Prepare an opto group for kinematic-dependent summary analysis.
        """
        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_opto_data()
        group_info.read_kinematic_data(["Landing", "Flying"])


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
        if colors is None:
            colors = {
                "L-f": "tab:blue",
                "L-m": "tab:orange",
                "L-h": "tab:green",
                "R-f": "tab:grey",
                "R-m": "tab:brown",
                "R-h": "tab:red"
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(list(trial_types))

        sc_df = pd.read_csv(sc_csv_path)
        required_columns = {"Index", *legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

        sc_lookup = {}
        for _, row in sc_df.iterrows():
            index = self.calculator.parse_index_cell(row["Index"])
            sc_lookup[index] = row

        def classify_trial(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group_info.latency_threshold:
                return "Success"
            return "Failed"

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

            sc_row = sc_lookup.get(tuple(index))
            outcome = classify_trial(meta)

            for leg in legs:
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

        stat_df = pd.DataFrame(stat_rows)
        if save_csv and file_name is not None:
            stat_df.to_csv(f"{file_name}_stats.csv", index=False)

        return combined_fig, combined_ax, km_df, stat_df

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
            save_csv=True
    ):
        """
        Plot IT/OT fly-wise landing probability and landing-latency inverted KM.

        Fly-wise LP is # successful behavior trials / # total behavior trials.
        The trial-level permutation test shuffles success/fail outcomes while
        preserving the IT/OT sample sizes, then recalculates mean difference.
        """
        rng = np.random.default_rng(random_state)

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

        if isinstance(behavior_sources, (str, os.PathLike)):
            behavior_sources = {
                label: {
                    "path": behavior_sources,
                    "selection_mode": "marker",
                    "marker_label": label,
                }
                for label in behavior_labels
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()

        group_info.read_kinematic_data(list(trial_types))

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

        behavior_trial_sets = self._trial_sets_from_behavior_sources(behavior_sources)
        behavior_by_index = {}
        for behavior_label, indexes in behavior_trial_sets.items():
            for index in indexes:
                behavior_by_index[tuple(index)] = behavior_label

        def is_success(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            return (
                    meta["TrialType"] == "Landing"
                    and not pd.isna(ll)
                    and (ll / fps) <= group_info.latency_threshold
            )

        def latency_event(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if is_success(meta):
                return min(ll / fps, tau), 1
            return tau, 0

        angle_def = [f"{contacted_leg}CT", f"{contacted_leg}FT", f"{contacted_leg}TT"]
        target_n = int(round((angle_end_s - angle_start_s) * target_fps)) + 1
        target_time = np.linspace(angle_start_s, angle_end_s, target_n)

        trial_rows = []
        angle_trace_rows = []
        angular_velocity_rows = []
        for index in group_info.get_targeted_trials(list(trial_types)):
            index_tuple = tuple(index)
            behavior_label = behavior_by_index.get(index_tuple)
            if behavior_label not in behavior_labels:
                continue

            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.trial_metadata:
                continue

            meta = group_info.trial_metadata[key]
            duration, event = latency_event(meta)
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

            if key not in group_info.fly_kinematic_data:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            moc = trial_info.moc
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue
            if any(point not in trial_info.trial_data for point in angle_def):
                continue

            start_frame = int(round(moc + angle_start_s * fps))
            end_frame = int(round(moc + angle_end_s * fps))
            if start_frame < 0 or end_frame >= trial_info.total_frames_number or end_frame <= start_frame:
                continue

            angle_trace = self.calculator.Calculate_joint_angle(trial_info, [angle_def])[angle_def[1]]
            source_frames = np.arange(start_frame, end_frame + 1)
            source_time = (source_frames - moc) / fps
            source_trace = np.asarray(angle_trace[start_frame:end_frame + 1], dtype=float)
            valid = np.isfinite(source_time) & np.isfinite(source_trace)
            if np.sum(valid) < min_angle_frames:
                continue

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
                })

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
            })

        trial_df = pd.DataFrame(trial_rows)
        if trial_df.empty:
            raise ValueError("No IT/OT-labeled Landing/Flying trials were found.")
        angle_trace_df = pd.DataFrame(angle_trace_rows)
        angular_velocity_df = pd.DataFrame(angular_velocity_rows)

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

        if save_csv and file_name is not None:
            trial_df.to_csv(f"{file_name}_trial_data.csv", index=False)
            fly_lp_df.to_csv(f"{file_name}_fly_landing_probability.csv", index=False)
            stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)
            if not angle_trace_df.empty:
                angle_trace_df.to_csv(f"{file_name}_FT_angle_traces.csv", index=False)
            if not angular_velocity_df.empty:
                angular_velocity_df.to_csv(f"{file_name}_FT_angular_velocity.csv", index=False)

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

        fig_angle, ax_angle = plt.subplots(figsize=(5.6, 4.0))
        plotted_angle_values = []
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
                ax_angle.plot(
                    time_values,
                    mean_trace,
                    color=colors.get(label, "0.5"),
                    linewidth=2.4,
                    label=f"{display} (n={traces.shape[0]})"
                )
                ax_angle.fill_between(
                    time_values,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    color=colors.get(label, "0.5"),
                    alpha=0.20,
                    linewidth=0
                )
        ax_angle.axvline(0, color="black", linestyle="--", linewidth=1)
        ax_angle.set_xlabel("Time from MOC (s)")
        ax_angle.set_ylabel(f"{contacted_leg} FT angle (deg)")
        ax_angle.set_title(f"{group_info.group_name}: IT/OT {contacted_leg} FT angle trace")
        ax_angle.set_xlim(angle_start_s, angle_end_s)
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

        fig_vel, ax_vel = plt.subplots(figsize=(4.8, 4.2))
        velocity_stat_df = pd.DataFrame()
        if not angular_velocity_df.empty:
            vel_positions = np.arange(len(behavior_labels), dtype=float)
            box_positions = vel_positions - 0.16
            point_positions = vel_positions + 0.12
            for i, label in enumerate(behavior_labels):
                sub = angular_velocity_df[angular_velocity_df["Behavior_Label"] == label]
                values = sub["Mean_Angular_Velocity_deg_s"].astype(float).dropna().to_numpy()
                if len(values) > 0:
                    ax_vel.boxplot(
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
                    x = point_positions[i] + rng.uniform(-0.045, 0.045, size=len(values))
                    ax_vel.scatter(
                        x,
                        values,
                        s=42,
                        color=colors.get(label, "0.5"),
                        alpha=0.78,
                        edgecolor="black",
                        linewidth=0.4,
                    )

            if len(behavior_labels) >= 2:
                values_a = (
                    angular_velocity_df[angular_velocity_df["Behavior_Label"] == behavior_labels[0]]
                    ["Mean_Angular_Velocity_deg_s"].astype(float).dropna().to_numpy()
                )
                values_b = (
                    angular_velocity_df[angular_velocity_df["Behavior_Label"] == behavior_labels[1]]
                    ["Mean_Angular_Velocity_deg_s"].astype(float).dropna().to_numpy()
                )
                vel_p = np.nan
                vel_observed = np.nan
                if len(values_a) > 0 and len(values_b) > 0:
                    vel_observed, vel_p = self.calculator._permutation_test_unpaired(
                        values_a,
                        values_b,
                        n_perm=n_perm,
                        rng=rng
                    )
                velocity_stat_df = pd.DataFrame([{
                    "Group_A": behavior_labels[0],
                    "Group_B": behavior_labels[1],
                    "n_A": len(values_a),
                    "n_B": len(values_b),
                    "mean_A": np.nan if len(values_a) == 0 else float(np.mean(values_a)),
                    "mean_B": np.nan if len(values_b) == 0 else float(np.mean(values_b)),
                    "mean_diff_B_minus_A": vel_observed,
                    "permutation_p": vel_p,
                    "n_perm": n_perm,
                }])
                if save_csv and file_name is not None:
                    velocity_stat_df.to_csv(f"{file_name}_FT_angular_velocity_stats.csv", index=False)
                y_top = np.nanmax(angular_velocity_df["Mean_Angular_Velocity_deg_s"].astype(float))
                y_range = max(y_top - np.nanmin(angular_velocity_df["Mean_Angular_Velocity_deg_s"].astype(float)), 1)
                y_bracket = y_top + y_range * 0.08
                ax_vel.plot(
                    [point_positions[0], point_positions[0], point_positions[1], point_positions[1]],
                    [y_bracket, y_bracket + y_range * 0.04, y_bracket + y_range * 0.04, y_bracket],
                    color="black",
                    linewidth=1
                )
                ax_vel.text(np.mean(point_positions), y_bracket + y_range * 0.045, significance_label(vel_p),
                            ha="center", va="bottom", fontsize=12)
                ax_vel.set_ylim(top=y_bracket + y_range * 0.18)

        ax_vel.set_xticks(np.arange(len(behavior_labels)))
        ax_vel.set_xticklabels([behavior_display_names.get(label, label) for label in behavior_labels])
        ax_vel.set_ylabel(f"{contacted_leg} FT angular velocity (deg/s)")
        ax_vel.set_title(f"{group_info.group_name}: IT/OT {contacted_leg} FT angular velocity")
        sns.despine()
        plt.tight_layout()
        if file_name is not None:
            fig_vel.savefig(f"{file_name}_FT_angular_velocity.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig_vel)

        return (
            fig_lp, ax_lp, fig_km, ax_km, fig_angle, ax_angle, fig_vel, ax_vel,
            fly_lp_df, trial_df, stat_df, km_stat_df, logrank_df,
            angle_trace_df, angular_velocity_df, velocity_stat_df
        )

    def plot_manual_sc_inverted_km_by_behavior(
            self,
            group_info,
            sc_csv_path,
            behavior_sources,
            file_name="manual_SC_inverted_KM_by_behavior",
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            behavior_labels=("IT", "OT"),
            behavior_display_names=None,
            colors=None,
            save_csv=True
    ):
        """
        Plot inverted KM curves for manually labeled SC timing by IT/OT behavior.

        SC validation is the same as plot_manual_sc_inverted_km_from_csv.
        behavior_sources follows _trial_sets_from_behavior_sources.
        """
        if colors is None:
            colors = {
                "L-f": "tab:blue",
                "L-m": "tab:orange",
                "L-h": "tab:green",
                "R-f": "tab:grey",
                "R-m": "tab:brown",
                "R-h": "tab:red"
            }

        if behavior_display_names is None:
            behavior_display_names = {
                "IT": "Inward touch",
                "OT": "Outward touch",
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()

        group_info.filter_nan_fly()
        group_info.read_kinematic_data(list(trial_types))

        behavior_trial_sets = self._trial_sets_from_behavior_sources(behavior_sources)
        behavior_by_index = {}
        for behavior_label, indexes in behavior_trial_sets.items():
            for index in indexes:
                behavior_by_index[tuple(index)] = behavior_label

        sc_df = pd.read_csv(sc_csv_path)
        required_columns = {"Index", *legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

        sc_lookup = {}
        for _, row in sc_df.iterrows():
            index = self.calculator.parse_index_cell(row["Index"])
            sc_lookup[tuple(index)] = row

        rows = []
        for index in group_info.get_targeted_trials(list(trial_types)):
            index_tuple = tuple(index)
            behavior_label = behavior_by_index.get(index_tuple)
            if behavior_label is None:
                continue

            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue

            sc_row = sc_lookup.get(index_tuple)
            for leg in legs:
                raw_sc = np.nan if sc_row is None or leg not in sc_row else sc_row[leg]
                sc_result = self.calculator.validate_sc_timing(raw_sc, moc, mol, fps, threshold)
                rows.append({
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Behavior_Label": behavior_label,
                    "Behavior_Display": behavior_display_names.get(behavior_label, behavior_label),
                    "Leg": leg,
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
            raise ValueError("No behavior-labeled trials were available for manual SC KM plotting.")

        if save_csv and file_name is not None:
            km_df.to_csv(f"{file_name}_trial_leg_events.csv", index=False)

        def plot_km_panel(ax, data, title, ylabel=True):
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

        fig, axes = plt.subplots(
            len(behavior_labels),
            1,
            figsize=(7, 3.7 * len(behavior_labels)),
            sharex=True,
            sharey=True,
            squeeze=False
        )
        axes = axes[:, 0]

        stat_rows = []
        for ax, behavior_label in zip(axes, behavior_labels):
            display_name = behavior_display_names.get(behavior_label, behavior_label)
            stat_rows.extend(
                plot_km_panel(
                    ax,
                    km_df[km_df["Behavior_Label"] == behavior_label],
                    f"{group_info.group_name}: {display_name}"
                )
            )

        axes[-1].set_xlabel("SC timing after MOC (s)")
        sns.despine()
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        stat_df = pd.DataFrame(stat_rows)
        if save_csv and file_name is not None:
            stat_df.to_csv(f"{file_name}_stats.csv", index=False)

        return fig, axes, km_df, stat_df

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
        import itertools

        rng = np.random.default_rng(random_state)
        for contact_group in contact_groups:
            if contact_group not in group_infos:
                raise ValueError(f"Missing group_infos entry for contact group: {contact_group}")
            if contact_group not in sc_csv_paths:
                raise ValueError(f"Missing sc_csv_paths entry for contact group: {contact_group}")

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

        for contact_group, group_legs in legs_by_group.items():
            if len(group_legs) == 0:
                raise ValueError(f"No legs were selected for contact group: {contact_group}")

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

    def plot_sc_order_heatmaps_by_contact_group(
            self,
            group_infos,
            sc_csv_paths,
            file_name="SC_order_heatmaps_by_contact_group",
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            colors=None,
            sort_trials=False,
            save_csv=True
    ):
        """
        Plot SC contact-order heatmaps as one 1x3 figure.

        This is the Figure 2E wrapper around the same validation rule used by
        plot_sc_order_heatmap_from_csv, but arranged as one column per contact
        group so the panels can be placed directly into a slide.
        """
        from matplotlib.patches import Rectangle

        if colors is None:
            colors = {
                "L-f": "tab:blue",
                "L-m": "tab:orange",
                "L-h": "tab:green",
            }

        order_labels = ["First", "Second", "Third"]
        prepared = {}
        all_cell_rows = []
        all_trial_rows = []

        for contact_group in contact_groups:
            group_info = group_infos[contact_group]
            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()

            group_info.filter_nan_fly()
            group_info.read_kinematic_data(list(trial_types))

            sc_df = pd.read_csv(sc_csv_paths[contact_group])
            required_columns = {"Index", *legs}
            missing_columns = required_columns.difference(sc_df.columns)
            if missing_columns:
                raise ValueError(
                    f"{contact_group} SC CSV is missing required columns: {sorted(missing_columns)}"
                )

            trial_rows = []
            cell_rows = []
            for _, sc_row in sc_df.iterrows():
                index = self.calculator.parse_index_cell(sc_row["Index"])
                key = group_info._trial_key(index[0], index[1])
                if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                    continue

                trial_info = group_info.fly_kinematic_data[key]
                meta = group_info.trial_metadata[key]
                moc = trial_info.moc
                mol = trial_info.mol
                fps = trial_info.fps
                if pd.isna(moc) or pd.isna(fps):
                    continue

                ll = meta["LL"]
                outcome = "Success" if (
                        meta["TrialType"] == "Landing"
                        and not pd.isna(ll)
                        and (ll / fps) <= group_info.latency_threshold
                ) else "Failed"

                trial_id = f"F{index[0]}T{index[1]}"
                trial_rows.append({
                    "Contact_Group": contact_group,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Trial_ID": trial_id,
                    "Outcome": outcome,
                    "TrialType": meta["TrialType"],
                    "MOC_frame": moc,
                    "MOL_frame": mol,
                })

                valid_contacts = []
                for leg in legs:
                    sc_result = self.calculator.validate_sc_timing(sc_row[leg], moc, mol, fps, threshold)
                    if sc_result["is_valid"]:
                        valid_contacts.append((leg, sc_result["sc_time_s"], sc_result["sc_frame"]))

                valid_contacts = sorted(valid_contacts, key=lambda item: (item[1], legs.index(item[0])))
                for order_idx, (leg, sc_time_s, sc_frame) in enumerate(valid_contacts[:len(order_labels)]):
                    cell_rows.append({
                        "Contact_Group": contact_group,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Trial_ID": trial_id,
                        "Outcome": outcome,
                        "TrialType": meta["TrialType"],
                        "Order": order_labels[order_idx],
                        "Order_Rank": order_idx + 1,
                        "Leg": leg,
                        "Raw_SC_frame": sc_frame,
                        "SC_time_after_MOC_s": sc_time_s,
                    })

            trial_df = pd.DataFrame(trial_rows).drop_duplicates(subset=["Index"])
            cell_df = pd.DataFrame(cell_rows)
            if trial_df.empty:
                raise ValueError(f"No matching trials were found for {contact_group} heatmap plotting.")

            if sort_trials:
                if not cell_df.empty:
                    first_contact = (
                        cell_df[cell_df["Order_Rank"] == 1]
                        .set_index("Index")["SC_time_after_MOC_s"]
                        .rename("First_SC_time_after_MOC_s")
                    )
                    trial_df = trial_df.join(first_contact, on="Index")
                trial_df = trial_df.sort_values(
                    ["Outcome", "Fly#", "Trial#", "First_SC_time_after_MOC_s"],
                    na_position="last"
                )
            else:
                trial_df = trial_df.sort_values(["Fly#", "Trial#"])

            trial_df = trial_df.reset_index(drop=True)
            trial_df["Heatmap_Row"] = np.arange(len(trial_df))
            row_lookup = trial_df.set_index("Index")["Heatmap_Row"]
            if not cell_df.empty:
                cell_df["Heatmap_Row"] = cell_df["Index"].map(row_lookup)

            prepared[contact_group] = (trial_df, cell_df)
            all_trial_rows.append(trial_df)
            all_cell_rows.append(cell_df)

        max_trials = max(len(prepared[group][0]) for group in contact_groups)
        fig_height = min(max(3.2, 0.045 * max_trials), 6.0)
        fig, axes = plt.subplots(
            1,
            len(contact_groups),
            figsize=(5.0 * len(contact_groups), fig_height),
            squeeze=False
        )
        axes = axes[0, :]

        for ax, contact_group in zip(axes, contact_groups):
            trial_df, cell_df = prepared[contact_group]
            for row_idx in range(len(trial_df)):
                for col_idx, _ in enumerate(order_labels):
                    ax.add_patch(
                        Rectangle(
                            (col_idx - 0.5, row_idx - 0.5),
                            1,
                            1,
                            facecolor="white",
                            edgecolor="0.88",
                            linewidth=0.6
                        )
                    )

            for _, row in cell_df.iterrows():
                ax.add_patch(
                    Rectangle(
                        (row["Order_Rank"] - 1.5, row["Heatmap_Row"] - 0.5),
                        1,
                        1,
                        facecolor=colors.get(row["Leg"], "black"),
                        edgecolor="white",
                        linewidth=0.8
                    )
                )

            n_trials = len(trial_df)
            n_y_ticks = min(5, n_trials)
            y_ticks = np.linspace(0, n_trials - 1, n_y_ticks, dtype=int)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels((y_ticks + 1).astype(str))
            ax.set_xlim(-0.5, len(order_labels) - 0.5)
            ax.set_ylim(n_trials - 0.5, -0.5)
            ax.set_title(f"{contact_group} (n={n_trials})")
            ax.set_ylabel("Trial index" if ax is axes[0] else "")
            ax.set_xticks(range(len(order_labels)))
            ax.set_xticklabels(order_labels)
            ax.set_xlabel("SC contact order")

        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=colors.get(leg, "black"), edgecolor="none", label=leg)
            for leg in legs
        ]
        axes[-1].legend(
            handles=legend_handles,
            frameon=False,
            title="Leg",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5)
        )

        sns.despine(left=False, bottom=False)
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")

        trial_out = pd.concat(all_trial_rows, ignore_index=True)
        cell_out = pd.concat(all_cell_rows, ignore_index=True) if all_cell_rows else pd.DataFrame()
        if save_csv and file_name is not None:
            trial_out.to_csv(f"{file_name}_trials.csv", index=False)
            cell_out.to_csv(f"{file_name}_cells.csv", index=False)

        plt.close()
        return fig, axes, cell_out, trial_out

    def plot_flywise_sc_contact_order_probability_stripplot(
            self,
            group_infos,
            sc_csv_paths,
            file_name="flywise_SC_contact_order_probability",
            contact_groups=("T1", "T2", "T3"),
            legs=("L-f", "L-m", "L-h"),
            threshold=0.71,
            trial_types=("Landing", "Flying"),
            colors=None,
            save_csv=True
    ):
        """
        Plot fly-wise probabilities of each leg being first, second, third, or no SC.

        For each fly, probabilities are computed separately for each leg across
        that fly's valid trials. The output figure has one row per contact group.
        """
        if colors is None:
            colors = {
                "L-f": "tab:blue",
                "L-m": "tab:orange",
                "L-h": "tab:green",
            }
        summary_colors = dict(colors)
        summary_colors.setdefault("No SC", "0.45")

        for contact_group in contact_groups:
            if contact_group not in group_infos:
                raise ValueError(f"Missing group_infos entry for contact group: {contact_group}")
            if contact_group not in sc_csv_paths:
                raise ValueError(f"Missing sc_csv_paths entry for contact group: {contact_group}")

        order_labels = ["First", "Second", "Third"]
        category_order = ["First", "Second", "Third", "No SC contact"]
        trial_rows = []

        for contact_group in contact_groups:
            group_info = group_infos[contact_group]

            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()

            group_info.filter_nan_fly()
            group_info.read_kinematic_data(list(trial_types))

            sc_df = pd.read_csv(sc_csv_paths[contact_group])
            required_columns = {"Index", *legs}
            missing_columns = required_columns.difference(sc_df.columns)
            if missing_columns:
                raise ValueError(
                    f"{contact_group} SC CSV is missing required columns: {sorted(missing_columns)}"
            )

            for _, sc_row in sc_df.iterrows():
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

                leg_times = {}
                for leg in legs:
                    sc_result = self.calculator.validate_sc_timing(sc_row[leg], moc, mol, fps, threshold)
                    leg_times[leg] = sc_result["sc_time_s"]
                valid_legs = [
                    (leg, sc_time)
                    for leg, sc_time in leg_times.items()
                    if not pd.isna(sc_time)
                ]
                valid_legs = sorted(valid_legs, key=lambda item: (item[1], legs.index(item[0])))

                leg_order = {leg: "No SC contact" for leg in legs}
                for rank, (leg, _) in enumerate(valid_legs[:len(order_labels)]):
                    leg_order[leg] = order_labels[rank]

                for leg in legs:
                    trial_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "Order": leg_order[leg],
                        "SC_time_after_MOC_s": leg_times[leg],
                    })

        order_df = pd.DataFrame(trial_rows)
        if order_df.empty:
            raise ValueError("No matching SC rows were found for the requested contact groups.")

        count_df = (
            order_df
            .groupby(["Contact_Group", "Fly#", "Leg", "Order"])
            .size()
            .reset_index(name="Count")
        )
        total_df = (
            order_df
            .groupby(["Contact_Group", "Fly#", "Leg"])
            .size()
            .reset_index(name="Total")
        )

        all_index = pd.MultiIndex.from_product(
            [
                list(contact_groups),
                sorted(order_df["Fly#"].dropna().unique()),
                list(legs),
                category_order
            ],
            names=["Contact_Group", "Fly#", "Leg", "Order"]
        )

        prob_df = (
            count_df
            .set_index(["Contact_Group", "Fly#", "Leg", "Order"])
            .reindex(all_index, fill_value=0)
            .reset_index()
            .merge(total_df, on=["Contact_Group", "Fly#", "Leg"], how="left")
        )

        prob_df = prob_df.dropna(subset=["Total"]).copy()
        prob_df["Probability"] = prob_df["Count"] / prob_df["Total"]

        # Summary probabilities ignore contact order. For each fly, the
        # denominator is the number of valid trials in that contact group.
        valid_summary_rows = []
        for (contact_group, fly_id, leg), leg_df in order_df.groupby(["Contact_Group", "Fly#", "Leg"]):
            valid_summary_rows.append({
                "Contact_Group": contact_group,
                "Fly#": fly_id,
                "Category": leg,
                "Probability": float(leg_df["SC_time_after_MOC_s"].notna().mean()),
                "n_trials": int(len(leg_df)),
            })

        trial_valid_df = (
            order_df
            .groupby(["Contact_Group", "Fly#", "Index"])["SC_time_after_MOC_s"]
            .apply(lambda values: values.notna().any())
            .reset_index(name="Has_Valid_SC")
        )
        for (contact_group, fly_id), fly_df in trial_valid_df.groupby(["Contact_Group", "Fly#"]):
            valid_summary_rows.append({
                "Contact_Group": contact_group,
                "Fly#": fly_id,
                "Category": "No SC",
                "Probability": float((~fly_df["Has_Valid_SC"]).mean()),
                "n_trials": int(len(fly_df)),
            })

        valid_summary_df = pd.DataFrame(valid_summary_rows)
        summary_category_order = [*legs, "No SC"]

        if save_csv and file_name is not None:
            order_df.to_csv(f"{file_name}_trial_leg_order.csv", index=False)
            prob_df.to_csv(f"{file_name}_fly_probability.csv", index=False)
            valid_summary_df.to_csv(f"{file_name}_fly_valid_SC_summary.csv", index=False)

        fig, axes = plt.subplots(
            len(contact_groups),
            1,
            figsize=(6.5, 3.0 * len(contact_groups)),
            sharex=True,
            sharey=True,
            squeeze=False
        )
        axes = axes.flatten()

        for ax, contact_group in zip(axes, contact_groups):
            sub = prob_df[prob_df["Contact_Group"] == contact_group].copy()
            sns.stripplot(
                data=sub,
                x="Order",
                y="Probability",
                hue="Leg",
                order=category_order,
                hue_order=list(legs),
                palette=colors,
                dodge=True,
                jitter=True,
                size=6,
                alpha=0.85,
                ax=ax
            )
            ax.set_title(f"{contact_group}: fly-wise leg contact order probability after MOC")
            ax.set_xlabel("")
            ax.set_ylabel("Probability")
            ax.set_ylim(-0.05, 1.05)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        axes[-1].set_xlabel("SC contact order")
        sns.despine()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            title="Leg",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        summary_fig, summary_ax = plt.subplots(figsize=(7.0, 4.2))
        sns.stripplot(
            data=valid_summary_df,
            x="Contact_Group",
            y="Probability",
            hue="Category",
            order=list(contact_groups),
            hue_order=summary_category_order,
            palette=summary_colors,
            dodge=True,
            jitter=True,
            size=6,
            alpha=0.85,
            ax=summary_ax
        )
        summary_ax.set_xlabel("Contact group")
        summary_ax.set_ylabel("Fly-wise probability")
        summary_ax.set_ylim(-0.05, 1.05)
        summary_ax.set_title("Fly-wise valid SC probability")
        summary_ax.legend(
            frameon=False,
            title="Category",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )
        sns.despine()
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        if file_name is not None:
            plt.savefig(f"{file_name}_valid_SC_summary.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return fig, axes, order_df, prob_df, summary_fig, summary_ax, valid_summary_df

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
        contributes to secondary contact probability only when it is the first
        valid SC among the selected legs in that trial. Ties are broken using
        the order of the legs argument, matching the contact-order probability
        plot.
        """
        import itertools

        if colors is None:
            colors = {
                "L-f": "tab:blue",
                "L-m": "tab:orange",
                "L-h": "tab:green",
            }

        for contact_group in contact_groups:
            if contact_group not in group_infos:
                raise ValueError(f"Missing group_infos entry for contact group: {contact_group}")
            if contact_group not in sc_csv_paths:
                raise ValueError(f"Missing sc_csv_paths entry for contact group: {contact_group}")

        trial_rows = []
        for contact_group in contact_groups:
            group_info = group_infos[contact_group]

            if len(group_info.trial_metadata) == 0:
                group_info.initialize_manual_data()

            group_info.filter_nan_fly()
            group_info.read_kinematic_data(list(trial_types))

            sc_df = pd.read_csv(sc_csv_paths[contact_group])
            required_columns = {"Index", *legs}
            missing_columns = required_columns.difference(sc_df.columns)
            if missing_columns:
                raise ValueError(
                    f"{contact_group} SC CSV is missing required columns: {sorted(missing_columns)}"
            )

            for _, sc_row in sc_df.iterrows():
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

                leg_times = {}
                for leg in legs:
                    sc_result = self.calculator.validate_sc_timing(sc_row[leg], moc, mol, fps, threshold)
                    leg_times[leg] = sc_result["sc_time_s"]
                valid_times = [
                    (leg, sc_time)
                    for leg, sc_time in leg_times.items()
                    if not pd.isna(sc_time)
                ]
                valid_times = sorted(valid_times, key=lambda item: (item[1], legs.index(item[0])))
                first_leg = valid_times[0][0] if valid_times else None

                for leg in legs:
                    is_first_sc = leg == first_leg
                    trial_rows.append({
                        "Contact_Group": contact_group,
                        "Group_Name": group_info.group_name,
                        "Index": str(index),
                        "Fly#": index[0],
                        "Trial#": index[1],
                        "Leg": leg,
                        "SC_time_after_MOC_s": leg_times[leg],
                        "First_SC": int(is_first_sc),
                    })

        trial_df = pd.DataFrame(trial_rows)
        if trial_df.empty:
            raise ValueError("No matching SC rows were found for the requested contact groups.")

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

        def run_permutation(group_a, group_b, comparison_type, group_a_label, group_b_label):
            values_a = (
                group_a["Secondary_Contact_Probability"]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            values_b = (
                group_b["Secondary_Contact_Probability"]
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

        if save_csv and file_name is not None:
            trial_df.to_csv(f"{file_name}_trial_first_sc.csv", index=False)
            prob_df.to_csv(f"{file_name}_fly_probability.csv", index=False)
            stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        sns.stripplot(
            data=prob_df,
            x="Contact_Group",
            y="Secondary_Contact_Probability",
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
        ax.set_ylabel("Secondary contact probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Fly-wise secondary contact probability")
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
            save_csv=True
    ):
        """
        Plot trial-wise number of valid SC events against raw landing latency.

        Success is defined as Landing with LL/fps <= group latency threshold.
        Failed includes flying trials and landing trials above threshold.
        """
        if colors is None:
            colors = {
                "Success": "tab:blue",
                "Failed": "tab:red",
            }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()

        group_info.read_kinematic_data(list(trial_types))

        sc_df = pd.read_csv(sc_csv_path)
        required_columns = {"Index", *legs}
        missing_columns = required_columns.difference(sc_df.columns)
        if missing_columns:
            raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")

        def classify_trial(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group_info.latency_threshold:
                return "Success"
            return "Failed"

        rows = []
        for _, sc_row in sc_df.iterrows():
            index = self.calculator.parse_index_cell(sc_row["Index"])
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            moc = trial_info.moc
            mol = trial_info.mol
            fps = trial_info.fps
            if pd.isna(moc) or pd.isna(fps):
                continue

            ll = meta["LL"]
            latency_s = np.nan
            if not pd.isna(ll) and ll != -1:
                latency_s = ll / meta["fps"]
            if pd.isna(latency_s):
                continue

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

            rows.append({
                "Group_Name": group_info.group_name,
                "Index": str(index),
                "Fly#": index[0],
                "Trial#": index[1],
                "Outcome": classify_trial(meta),
                "TrialType": meta["TrialType"],
                "LL_frame": ll,
                "Landing_Latency_s": latency_s,
                "Valid_SC_Count": valid_sc_count,
                "Valid_SC_Legs": ",".join(valid_legs),
            })

        count_df = pd.DataFrame(rows)
        if count_df.empty:
            raise ValueError("No matching SC rows with valid landing latency were found.")

        if save_csv and file_name is not None:
            count_df.to_csv(f"{file_name}_data.csv", index=False)

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        sns.stripplot(
            data=count_df,
            x="Valid_SC_Count",
            y="Landing_Latency_s",
            hue="Outcome",
            hue_order=["Success", "Failed"],
            palette=colors,
            jitter=0.08,
            dodge=False,
            size=5,
            alpha=0.78,
            ax=ax
        )
        ax.axhline(group_info.latency_threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("# valid leg contact events per trial")
        ax.set_ylabel("Landing latency (s)")
        ax.set_xticks(range(len(legs) + 1))
        ax.set_xlim(-0.5, len(legs) + 0.5)
        ax.set_title(f"{group_info.group_name}: valid leg contact count vs landing latency")
        ax.legend(
            frameon=False,
            title="Outcome",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )
        sns.despine()
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return fig, ax, count_df

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
            n_perm=20000
    ):
        """
        Plot left-leg TT path efficiency grouped by outcome and IT/OT behavior.

        Figure layout:
        - top panel: Success vs Failed within each leg
        - bottom panel: Inward touch vs Outward touch within each leg

        Window modes match plot_TT_summary_metrics_vs_LL:
        - fixed: MOC -> MOC + trajectory_window_s
        - mol_adjusted: success MOC -> MOL, failed MOC -> MOC + tau
        - SLC_adjusted: MOC -> leg-specific valid SC when present; otherwise
          success uses MOC -> MOL and failed uses MOC -> MOC + tau
        """
        if trajectory_window_mode not in {"fixed", "mol_adjusted", "SLC_adjusted"}:
            raise ValueError("trajectory_window_mode must be 'fixed', 'mol_adjusted', or 'SLC_adjusted'.")

        if isinstance(legs, str):
            legs = (legs,)
        else:
            legs = tuple(legs)

        if colors is None:
            colors = {
                "Success": "tab:blue",
                "Failed": "tab:red",
                "IT": "#8FD694",
                "OT": "#C7A0E8",
            }

        behavior_display_names = {
            "IT": "Inward touch",
            "OT": "Outward touch",
        }

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()

        group_info.read_kinematic_data(list(trial_types))

        sc_lookup = {}
        if trajectory_window_mode == "SLC_adjusted":
            if sc_csv_path is None:
                raise ValueError("sc_csv_path is required when trajectory_window_mode='SLC_adjusted'.")
            sc_df = pd.read_csv(sc_csv_path)
            required_columns = {"Index", *legs}
            missing_columns = required_columns.difference(sc_df.columns)
            if missing_columns:
                raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")
            for _, sc_row in sc_df.iterrows():
                index = self.calculator.parse_index_cell(sc_row["Index"])
                sc_lookup[tuple(index)] = sc_row

        behavior_trial_sets = self._trial_sets_from_behavior_sources(behavior_sources)
        behavior_by_index = {}
        for behavior_label, indexes in behavior_trial_sets.items():
            for index in indexes:
                behavior_by_index[tuple(index)] = behavior_label

        def classify_trial(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group_info.latency_threshold:
                return "Success"
            return "Failed"

        def latency_seconds(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if not pd.isna(ll) and ll != -1:
                return ll / fps, False, "metadata_LL"
            if meta["TrialType"] == "Flying":
                return tau, True, "flying_no_MOL_censored_at_tau"
            return np.nan, False, "missing_LL"

        def base_window_end(moc, mol, fps, total_frames, outcome):
            if trajectory_window_mode == "fixed":
                return (
                    int(min(moc + trajectory_window_s * fps, total_frames - 1)),
                    f"MOC_to_MOC_plus_{trajectory_window_s}s"
                )
            if outcome == "Success" and not pd.isna(mol) and mol > moc:
                return int(min(mol, total_frames - 1)), "MOC_to_MOL"
            return int(min(moc + tau * fps, total_frames - 1)), "MOC_to_MOC_plus_tau"

        def window_end_for_leg(index, leg, moc, mol, fps, total_frames, outcome):
            base_end, base_rule = base_window_end(moc, mol, fps, total_frames, outcome)
            if trajectory_window_mode != "SLC_adjusted":
                return base_end, base_rule, np.nan, False

            if outcome == "Success" and not pd.isna(mol) and mol > moc:
                valid_end = int(min(mol, total_frames - 1))
                no_sc_rule = "MOC_to_MOL_no_valid_SLC"
            elif outcome == "Success":
                valid_end = int(min(moc + tau * fps, total_frames - 1))
                no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC_missing_MOL"
            else:
                valid_end = int(min(moc + tau * fps, total_frames - 1))
                no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC"

            sc_row = sc_lookup.get(tuple(index))
            if sc_row is None or leg not in sc_row:
                return valid_end, no_sc_rule, np.nan, False

            is_valid, sc_frame = self.calculator.validate_sc_frame_window(
                sc_row[leg],
                moc,
                valid_end
            )
            if is_valid:
                return int(min(sc_frame, total_frames - 1)), "MOC_to_SLC", sc_frame, True
            return valid_end, no_sc_rule, np.nan, False

        def compute_path_efficiency(tt_xyz):
            tt_xyz = np.asarray(tt_xyz, dtype=float)
            valid = np.all(np.isfinite(tt_xyz), axis=1)
            tt_xyz = tt_xyz[valid]
            if len(tt_xyz) < min_frames:
                return np.nan, np.nan, np.nan

            steps = np.diff(tt_xyz, axis=0)
            path_length = np.sum(np.linalg.norm(steps, axis=1))
            displacement = np.linalg.norm(tt_xyz[-1] - tt_xyz[0])
            path_efficiency = np.nan
            if path_length > min_path_length:
                path_efficiency = displacement / path_length
            return path_efficiency, path_length, displacement

        records = []
        for index in group_info.get_targeted_trials(list(trial_types)):
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            fps = trial_info.fps
            moc = trial_info.moc
            mol = trial_info.mol
            if pd.isna(moc) or pd.isna(fps):
                continue

            ll_s, ll_censored, ll_source = latency_seconds(meta)
            if pd.isna(ll_s):
                continue

            moc_i = int(moc)
            outcome = classify_trial(meta)
            behavior_label = behavior_by_index.get(tuple(index))
            behavior_display = (
                behavior_display_names.get(behavior_label, behavior_label)
                if behavior_label is not None
                else np.nan
            )

            for leg in legs:
                tt_point = f"{leg}TT"
                if tt_point not in trial_info.trial_data:
                    continue

                end_frame, window_rule, slc_frame, slc_valid = window_end_for_leg(
                    index,
                    leg,
                    moc_i,
                    mol,
                    fps,
                    trial_info.total_frames_number,
                    outcome
                )
                if moc_i < 0 or end_frame <= moc_i:
                    continue

                tt_xyz = self.calculator.ReadAndTranspose(tt_point, trial_info)
                tt_segment = tt_xyz[moc_i:min(end_frame + 1, len(tt_xyz))]
                path_efficiency, path_length, displacement = compute_path_efficiency(tt_segment)
                if pd.isna(path_efficiency):
                    continue

                records.append({
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Leg": leg,
                    "Outcome": outcome,
                    "Behavior_Label": behavior_label,
                    "Behavior_Display": behavior_display,
                    "TrialType": meta["TrialType"],
                    "Landing_Latency_s": ll_s,
                    "LL_frame": meta["LL"],
                    "Landing_Latency_Censored": ll_censored,
                    "Landing_Latency_Source": ll_source,
                    "TT_Path_Efficiency": path_efficiency,
                    "TT_Path_Length": path_length,
                    "TT_Displacement": displacement,
                    "Trajectory_Window_Mode": trajectory_window_mode,
                    "Analysis_Window_Rule": window_rule,
                    "Analysis_Window_Start_Frame": moc_i,
                    "Analysis_Window_End_Frame": end_frame,
                    "SLC_Frame": slc_frame,
                    "SLC_Valid_For_Window": slc_valid,
                })

        path_df = pd.DataFrame(records)
        if path_df.empty:
            raise ValueError("No valid left-leg TT path efficiency rows were found.")

        def significance_label(p_value):
            if pd.isna(p_value):
                return "n.s."
            if p_value < 0.001:
                return "***"
            if p_value < 0.01:
                return "**"
            if p_value < 0.05:
                return "*"
            return "n.s."

        def run_unpaired_test(data, leg, group_col, group_a, group_b, comparison_type):
            sub = data[data["Leg"] == leg]
            values_a = sub[sub[group_col] == group_a]["TT_Path_Efficiency"].astype(float).dropna().to_numpy()
            values_b = sub[sub[group_col] == group_b]["TT_Path_Efficiency"].astype(float).dropna().to_numpy()
            row = {
                "Comparison_Type": comparison_type,
                "Leg": leg,
                "Group_A": group_a,
                "Group_B": group_b,
                "n_A": len(values_a),
                "n_B": len(values_b),
                "mean_A": np.nan if len(values_a) == 0 else float(np.mean(values_a)),
                "mean_B": np.nan if len(values_b) == 0 else float(np.mean(values_b)),
                "mean_diff_B_minus_A": np.nan,
                "permutation_p": np.nan,
                "significance": "n.s.",
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
            row["significance"] = significance_label(p_value)
            return row

        stat_rows = []
        for leg in legs:
            stat_rows.append(run_unpaired_test(
                path_df,
                leg,
                "Outcome",
                "Success",
                "Failed",
                "success_vs_failed"
            ))

        behavior_df = path_df.dropna(subset=["Behavior_Label"]).copy()
        behavior_keys = list(behavior_sources.keys())
        if len(behavior_keys) >= 2:
            for leg in legs:
                stat_rows.append(run_unpaired_test(
                    behavior_df,
                    leg,
                    "Behavior_Label",
                    behavior_keys[0],
                    behavior_keys[1],
                    "behavior_group"
                ))

        stat_df = pd.DataFrame(stat_rows)

        if save_csv and file_name is not None:
            path_df.to_csv(f"{file_name}_data.csv", index=False)
            stat_df.to_csv(f"{file_name}_permutation_stats.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.0), sharex=True, sharey=True)

        def add_bracket(ax, x1, x2, y, text):
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            h = y_range * 0.025
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
            ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=11)

        sns.stripplot(
            data=path_df,
            x="Leg",
            y="TT_Path_Efficiency",
            hue="Outcome",
            order=list(legs),
            hue_order=["Success", "Failed"],
            palette={key: colors[key] for key in ("Success", "Failed")},
            dodge=True,
            jitter=True,
            size=5,
            alpha=0.4,
            ax=axes[0]
        )
        axes[0].set_title("TT path efficiency by landing outcome")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("TT path efficiency")
        axes[0].set_ylim(-0.05, 1.05)
        for leg_i, leg in enumerate(legs):
            stat_match = stat_df[
                (stat_df["Comparison_Type"] == "success_vs_failed")
                & (stat_df["Leg"] == leg)
            ]
            if stat_match.empty:
                continue
            add_bracket(axes[0], leg_i - 0.18, leg_i + 0.18, 0.94, stat_match.iloc[0]["significance"])
        axes[0].legend(
            frameon=False,
            title="Outcome",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )

        sns.stripplot(
            data=behavior_df,
            x="Leg",
            y="TT_Path_Efficiency",
            hue="Behavior_Label",
            order=list(legs),
            hue_order=list(behavior_sources.keys()),
            palette={key: colors.get(key, "0.4") for key in behavior_sources.keys()},
            dodge=True,
            jitter=True,
            size=5,
            alpha=0.78,
            ax=axes[1]
        )
        axes[1].set_title("TT path efficiency by touch behavior")
        axes[1].set_xlabel("Left-side leg")
        axes[1].set_ylabel("TT path efficiency")
        axes[1].set_ylim(-0.05, 1.05)
        for leg_i, leg in enumerate(legs):
            stat_match = stat_df[
                (stat_df["Comparison_Type"] == "behavior_group")
                & (stat_df["Leg"] == leg)
            ]
            if stat_match.empty:
                continue
            add_bracket(axes[1], leg_i - 0.18, leg_i + 0.18, 0.94, stat_match.iloc[0]["significance"])

        handles, labels = axes[1].get_legend_handles_labels()
        labels = [behavior_display_names.get(label, label) for label in labels]
        axes[1].legend(
            handles,
            labels,
            frameon=False,
            title="Behavior",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )

        sns.despine()
        plt.tight_layout(rect=[0, 0, 0.84, 1])
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return fig, axes, path_df, stat_df

    def plot_chrimson_LP(self, group_info, color="red", threshold=0.71):
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

                wing_angle = self.calculator.Calculate_joint_angle(trial_info, angs)["L-wing-hinge"][750:1250]
                MOL = self.detector.detect_landing(wing_angle)
                if trial_info.LL == -1:
                    flying_num += 1
                    Fly_ON_LL.append(threshold)
                    Fly_ON_event.append(0)
                    Fly_ON_Idx.append(index[0])

                elif trial_info.LL == 1:
                    if MOL == -1 or (MOL / 250) > threshold:
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
            "Fly#": Fly_ON_Idx
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
            mean_color="black"
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

        def success_after_light_on(trial_info):
            wing_angle = self.calculator.Calculate_joint_angle(trial_info, angle_defs)["L-wing-hinge"][750:1250]
            mol = self.detector.detect_landing(wing_angle)
            if trial_info.LL == -1:
                return False
            if trial_info.LL == 1:
                return mol != -1 and (mol / 250) <= threshold
            return False

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
                on_rows.append({
                    "Fly#": index[0],
                    "Success": int(success_after_light_on(group_info.fly_kinematic_data[key])),
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

        def significance_label(p_value):
            if pd.isna(p_value):
                return ""
            if p_value < 1e-4:
                return "****"
            if p_value < 1e-3:
                return "***"
            if p_value < 1e-2:
                return "**"
            if p_value < 0.05:
                return "*"
            return "n.s."

        ytick_labels = []
        for plot_label in ordered_labels:
            stat_sub = stat_df[stat_df["Plot_Label"] == plot_label]
            if not stat_sub.empty:
                sig = significance_label(stat_sub.iloc[0]["p_value"])
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

        def significance_label(p_value):
            if pd.isna(p_value):
                return "n/a"
            if p_value < 1e-4:
                return "****"
            if p_value < 1e-3:
                return "***"
            if p_value < 1e-2:
                return "**"
            if p_value < 0.05:
                return "*"
            return "n.s."

        ytick_labels = []
        for group in group_order:
            stat_sub = stat_df[stat_df["Group"] == group]
            if stat_sub.empty:
                ytick_labels.append(group)
                continue
            n_flies = int(stat_sub.iloc[0]["n_paired_flies"])
            sig = significance_label(stat_sub.iloc[0]["p_value"])
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
        import itertools
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from lifelines import KaplanMeierFitter
        from lifelines.utils import restricted_mean_survival_time

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


    def plot_TT_summary_metrics_vs_LL(
            self,
            group_info,
            legs=("L-f", "L-m", "L-h"),
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
            random_state=0
    ):
        """
        Plot TT average speed, path efficiency, and path length vs LL.

        Statistics are calculated separately for each metric and leg. Each panel
        reports trial-level Spearman correlation and fly-average Spearman
        correlation. Permutation p-values are computed by shuffling the y-values
        within that panel and recalculating Spearman rho.
        """
        rng = np.random.default_rng(random_state)
        records = []
        metric_options = {
            "average_speed": {
                "column": "TT_Average_Speed",
                "label": "TT average speed (mm/s)",
                "title": "Average speed",
            },
            "path_efficiency": {
                "column": "TT_Path_Efficiency",
                "label": "TT path efficiency (displacement/path)",
                "title": "Path efficiency",
            },
            "path_length": {
                "column": "TT_Path_Length",
                "label": "TT path length (mm)",
                "title": "Path length",
            },
        }

        if trajectory_window_mode not in {"fixed", "mol_adjusted", "SLC_adjusted"}:
            raise ValueError("trajectory_window_mode must be 'fixed', 'mol_adjusted', or 'SLC_adjusted'.")

        if len(group_info.trial_metadata) == 0:
            group_info.initialize_manual_data()
            group_info.filter_nan_fly()

        group_info.read_kinematic_data(list(trial_types))
        trial_indexes = group_info.get_targeted_trials(list(trial_types))

        sc_lookup = {}
        if trajectory_window_mode == "SLC_adjusted":
            if sc_csv_path is None:
                raise ValueError("sc_csv_path is required when trajectory_window_mode='SLC_adjusted'.")
            sc_df = pd.read_csv(sc_csv_path)
            required_columns = {"Index", *legs}
            missing_columns = required_columns.difference(sc_df.columns)
            if missing_columns:
                raise ValueError(f"SC CSV is missing required columns: {sorted(missing_columns)}")
            for _, sc_row in sc_df.iterrows():
                index = self.calculator.parse_index_cell(sc_row["Index"])
                sc_lookup[tuple(index)] = sc_row

        def classify_trial(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if meta["TrialType"] == "Landing" and not pd.isna(ll) and ll != -1 and (ll / fps) <= group_info.latency_threshold:
                return "Success"
            return "Failed"

        def latency_seconds(meta):
            ll = meta["LL"]
            fps = meta["fps"]
            if not pd.isna(ll) and ll != -1:
                return ll / fps, False, "metadata_LL"
            if meta["TrialType"] == "Flying":
                return tau, True, "flying_no_MOL_censored_at_tau"
            return np.nan, False, "missing_LL"

        def base_window_end(moc, mol, fps, total_frames, outcome):
            if trajectory_window_mode == "fixed":
                return (
                    int(min(moc + trajectory_window_s * fps, total_frames - 1)),
                    f"MOC_to_MOC_plus_{trajectory_window_s}s"
                )

            if outcome == "Success" and not pd.isna(mol) and mol > moc:
                return int(min(mol, total_frames - 1)), "MOC_to_MOL"
            return int(min(moc + tau * fps, total_frames - 1)), "MOC_to_MOC_plus_tau"

        def window_end_for_leg(index, leg, moc, mol, fps, total_frames, outcome):
            base_end, base_rule = base_window_end(moc, mol, fps, total_frames, outcome)
            if trajectory_window_mode != "SLC_adjusted":
                return base_end, base_rule, np.nan, False

            if outcome == "Success" and not pd.isna(mol) and mol > moc:
                valid_end = int(min(mol, total_frames - 1))
                no_sc_rule = "MOC_to_MOL_no_valid_SLC"
            elif outcome == "Success":
                valid_end = int(min(moc + tau * fps, total_frames - 1))
                no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC_missing_MOL"
            else:
                valid_end = int(min(moc + tau * fps, total_frames - 1))
                no_sc_rule = "MOC_to_MOC_plus_tau_no_valid_SLC"

            sc_row = sc_lookup.get(tuple(index))
            if sc_row is None or leg not in sc_row:
                return valid_end, no_sc_rule, np.nan, False

            is_valid, sc_frame = self.calculator.validate_sc_frame_window(
                sc_row[leg],
                moc,
                valid_end
            )
            if is_valid:
                return int(min(sc_frame, total_frames - 1)), "MOC_to_SLC", sc_frame, True
            return valid_end, no_sc_rule, np.nan, False

        def compute_summary_metrics(tt_xyz, fps):
            tt_xyz = np.asarray(tt_xyz, dtype=float)
            valid = np.all(np.isfinite(tt_xyz), axis=1)
            tt_xyz = tt_xyz[valid]
            if len(tt_xyz) < min_frames:
                return np.nan, np.nan, np.nan, np.nan, np.nan

            steps = np.diff(tt_xyz, axis=0)
            path_length = np.sum(np.linalg.norm(steps, axis=1))
            displacement = np.linalg.norm(tt_xyz[-1] - tt_xyz[0])
            duration_s = (len(tt_xyz) - 1) / fps

            average_speed = np.nan
            if duration_s > 0:
                average_speed = path_length / duration_s

            path_efficiency = np.nan
            if path_length > min_path_length:
                path_efficiency = displacement / path_length

            return average_speed, path_efficiency, path_length, displacement, duration_s

        for index in trial_indexes:
            key = group_info._trial_key(index[0], index[1])
            if key not in group_info.fly_kinematic_data or key not in group_info.trial_metadata:
                continue

            trial_info = group_info.fly_kinematic_data[key]
            meta = group_info.trial_metadata[key]
            fps = trial_info.fps
            moc = trial_info.moc
            mol = trial_info.mol

            if pd.isna(moc) or pd.isna(fps):
                continue

            ll_s, ll_censored, ll_source = latency_seconds(meta)
            if pd.isna(ll_s):
                continue

            moc_i = int(moc)
            outcome = classify_trial(meta)
            for leg in legs:
                point_name = f"{leg}TT"
                if point_name not in trial_info.trial_data:
                    continue

                end_frame, window_rule, slc_frame, slc_valid = window_end_for_leg(
                    index,
                    leg,
                    moc_i,
                    mol,
                    fps,
                    trial_info.total_frames_number,
                    outcome
                )
                if moc_i < 0 or end_frame <= moc_i:
                    continue

                tt_xyz = self.calculator.ReadAndTranspose(point_name, trial_info)
                end = min(end_frame + 1, len(tt_xyz))
                tt_seg = tt_xyz[moc_i:end]

                average_speed, path_efficiency, path_length, displacement, duration_s = compute_summary_metrics(
                    tt_seg,
                    fps
                )
                if all(pd.isna(value) for value in (average_speed, path_efficiency, path_length)):
                    continue

                records.append({
                    "Group_Name": group_info.group_name,
                    "Index": str(index),
                    "Fly#": index[0],
                    "Trial#": index[1],
                    "Leg": leg,
                    "Joint": "TT",
                    "Outcome": outcome,
                    "TrialType": meta["TrialType"],
                    "Landing_Latency_s": ll_s,
                    "LL_frame": meta["LL"],
                    "Landing_Latency_Censored": ll_censored,
                    "Landing_Latency_Source": ll_source,
                    "TT_Average_Speed": average_speed,
                    "TT_Path_Efficiency": path_efficiency,
                    "TT_Path_Length": path_length,
                    "TT_Displacement": displacement,
                    "Window_Duration_s": duration_s,
                    "Trajectory_Window_Mode": trajectory_window_mode,
                    "Analysis_Window_Rule": window_rule,
                    "Analysis_Window_Start_Frame": moc_i,
                    "Analysis_Window_End_Frame": end_frame,
                    "SLC_Frame": slc_frame,
                    "SLC_Valid_For_Window": slc_valid,
                })

        metric_df = pd.DataFrame(records)
        if metric_df.empty:
            print("No valid TT summary metric data found.")
            return None, None, metric_df, pd.DataFrame()


        stat_rows = []
        for leg in legs:
            leg_df = metric_df[metric_df["Leg"] == leg]
            for metric_name, metric_info in metric_options.items():
                y_col = metric_info["column"]
                clean = leg_df[["Landing_Latency_s", y_col]].dropna()
                fly_average_clean = (
                    leg_df
                    .groupby("Fly#")[["Landing_Latency_s", y_col]]
                    .mean()
                    .dropna()
                    .reset_index()
                )

                stat_row = {
                    "Group_Name": group_info.group_name,
                    "Leg": leg,
                    "Metric": metric_name,
                    "Metric_Column": y_col,
                    "n": len(clean),
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "spearman_permutation_p": np.nan,
                    "fly_average_n": len(fly_average_clean),
                    "fly_average_spearman_rho": np.nan,
                    "fly_average_spearman_p": np.nan,
                    "fly_average_spearman_permutation_p": np.nan,
                    "linear_slope": np.nan,
                    "linear_intercept": np.nan,
                    "linear_r": np.nan,
                    "linear_p": np.nan,
                    "within_fly_permutation_slope": np.nan,
                    "within_fly_permutation_p": np.nan,
                    "n_perm": n_perm,
                    "Trajectory_Window_Mode": trajectory_window_mode,
                }

                if len(clean) >= 3 and clean["Landing_Latency_s"].nunique() >= 2 and clean[y_col].nunique() >= 2:
                    spearman_rho, spearman_p, spearman_perm_p = self.calculator.spearman_permutation_test(
                        clean["Landing_Latency_s"],
                        clean[y_col],
                        n_perm=n_perm,
                        rng=rng
                    )
                    linear = linregress(clean["Landing_Latency_s"], clean[y_col])

                    stat_row.update({
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "spearman_permutation_p": spearman_perm_p,
                        "linear_slope": float(linear.slope),
                        "linear_intercept": float(linear.intercept),
                        "linear_r": float(linear.rvalue),
                        "linear_p": float(linear.pvalue),
                    })

                if (
                        len(fly_average_clean) >= 3
                        and fly_average_clean["Landing_Latency_s"].nunique() >= 2
                        and fly_average_clean[y_col].nunique() >= 2
                ):
                    fly_rho, fly_p, fly_perm_p = self.calculator.spearman_permutation_test(
                        fly_average_clean["Landing_Latency_s"],
                        fly_average_clean[y_col],
                        n_perm=n_perm,
                        rng=rng
                    )
                    stat_row.update({
                        "fly_average_spearman_rho": fly_rho,
                        "fly_average_spearman_p": fly_p,
                        "fly_average_spearman_permutation_p": fly_perm_p,
                    })

                stat_rows.append(stat_row)

        stat_df = pd.DataFrame(stat_rows)

        if save_csv and file_name is not None:
            metric_df.to_csv(f"{file_name}_data.csv", index=False)
            stat_df.to_csv(f"{file_name}_trend_stats.csv", index=False)

        palette = {
            "Success": "tab:blue",
            "Failed": "tab:red",
        }

        def format_p_value(p_value):
            if pd.isna(p_value):
                return "p=NA"
            if p_value < 0.001:
                return "p<0.001"
            return f"p={p_value:.3f}"

        def format_rho_value(rho):
            if pd.isna(rho):
                return "rho=NA"
            return f"rho={rho:.2f}"

        fig, axes = plt.subplots(
            len(metric_options),
            len(legs),
            figsize=(4.2 * len(legs), 3.6 * len(metric_options)),
            sharex=True,
            squeeze=False
        )

        x_min = metric_df["Landing_Latency_s"].min()
        x_max = metric_df["Landing_Latency_s"].max()
        x_pad = max((x_max - x_min) * 0.05, 0.02)

        for row, (metric_name, metric_info) in enumerate(metric_options.items()):
            y_col = metric_info["column"]
            y_values = metric_df[y_col].to_numpy(dtype=float)
            y_values = y_values[np.isfinite(y_values)]
            y_min = np.nanmin(y_values)
            y_max = np.nanmax(y_values)
            y_pad = max((y_max - y_min) * 0.05, 0.02)

            for col, leg in enumerate(legs):
                ax = axes[row, col]
                leg_df = metric_df[metric_df["Leg"] == leg]
                sns.scatterplot(
                    data=leg_df,
                    x="Landing_Latency_s",
                    y=y_col,
                    hue="Outcome",
                    hue_order=["Success", "Failed"],
                    palette=palette,
                    s=45,
                    alpha=0.75,
                    ax=ax
                )
                fly_average_df = (
                    leg_df
                    .groupby("Fly#")[["Landing_Latency_s", y_col]]
                    .mean()
                    .dropna()
                    .reset_index()
                )
                if not fly_average_df.empty:
                    ax.scatter(
                        fly_average_df["Landing_Latency_s"],
                        fly_average_df[y_col],
                        color="black",
                        marker="D",
                        s=52,
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=5,
                        label="Fly average"
                    )
                    if (
                            len(fly_average_df) >= 3
                            and fly_average_df["Landing_Latency_s"].nunique() >= 2
                            and fly_average_df[y_col].nunique() >= 2
                    ):
                        fit = linregress(fly_average_df["Landing_Latency_s"], fly_average_df[y_col])
                        x_fit = np.linspace(
                            fly_average_df["Landing_Latency_s"].min(),
                            fly_average_df["Landing_Latency_s"].max(),
                            100
                        )
                        ax.plot(
                            x_fit,
                            fit.intercept + fit.slope * x_fit,
                            color="black",
                            linewidth=1.6,
                            alpha=0.85,
                            label="Fly-average linear fit"
                        )

                stat_match = stat_df[
                    (stat_df["Leg"] == leg)
                    & (stat_df["Metric"] == metric_name)
                ]
                if not stat_match.empty:
                    rho = stat_match.iloc[0]["spearman_rho"]
                    p_value = stat_match.iloc[0]["spearman_permutation_p"]
                    n_points = int(stat_match.iloc[0]["n"])
                    fly_rho = stat_match.iloc[0]["fly_average_spearman_rho"]
                    fly_p_value = stat_match.iloc[0]["fly_average_spearman_permutation_p"]
                    fly_n = int(stat_match.iloc[0]["fly_average_n"])
                    stat_label = (
                        f"trial n={n_points}, {format_rho_value(rho)}, perm {format_p_value(p_value)}\n"
                        f"fly n={fly_n}, {format_rho_value(fly_rho)}, perm {format_p_value(fly_p_value)}"
                    )
                else:
                    stat_label = "trial n=0, rho=NA, p=NA\nfly n=0, rho=NA, p=NA"

                ax.axvline(group_info.latency_threshold, color="black", linestyle="--", linewidth=1)
                ax.set_title(f"{leg} TT {metric_info['title']}\n{stat_label}")
                ax.set_xlabel("Landing latency (s)" if row == len(metric_options) - 1 else "")
                ax.set_ylabel(metric_info["label"] if col == 0 else "")
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
                if row == 0 and col == len(legs) - 1:
                    ax.legend(frameon=False, fontsize=8)
                else:
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()

        sns.despine()
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(f"{file_name}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return fig, axes, metric_df, stat_df
