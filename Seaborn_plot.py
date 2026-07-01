import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import numpy as np
from scipy.stats import t
from sklearn.utils import resample
from scipy.stats import kendalltau
from scipy.stats import ttest_ind, ttest_rel
import warnings
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
warnings.filterwarnings(action="ignore", category=RuntimeWarning)

def Bootstrapping_test(data1, data2):

    num_bootstrap_samples = 30000

    # Perform t-test on the original data

    original_median_diff = calculate_median_diff(data1, data2)
    original_mean_diff = calculate_mean_diff(data1, data2)

    # Bootstrap resampling
    bootstrap_mean_diffs = []
    bootstrap_median_diffs = []
    resample_data = np.concatenate((data1, data2))
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample1 = resample(resample_data, n_samples=len(data1))
        # print(bootstrap_sample1)
        bootstrap_sample2 = resample(resample_data, n_samples=len(data2))
        # print(bootstrap_sample2)
        # print(bootstrap_sample1)

        # Perform t-test on the bootstrap samples
        bootstrap_median_diff = calculate_median_diff(bootstrap_sample1, bootstrap_sample2)
        bootstrap_mean_diff = calculate_mean_diff(bootstrap_sample1, bootstrap_sample2)
        # print(bootstrap_mean_diff)

        bootstrap_mean_diffs.append(bootstrap_mean_diff)
        bootstrap_median_diffs.append(bootstrap_median_diff)

    k = 0
    for m in bootstrap_mean_diffs:
        # print(m)
        if abs(m) > abs(original_mean_diff):
            k += 1
    Mean_diff_p_value = (np.sum(np.abs(bootstrap_mean_diffs) >= np.abs(original_mean_diff))) / (num_bootstrap_samples)
    Median_diff_p_value = (np.sum(np.abs(bootstrap_median_diffs) >= np.abs(original_median_diff))) / (num_bootstrap_samples)

    # print(f"Difference in mean's P-value = {Mean_diff_p_value}")
    # print(f"Original mean diff = {original_mean_diff}")
    return Mean_diff_p_value
def calculate_mean_diff(data1, data2):
    return np.mean(data1) - np.mean(data2)
def calculate_median_diff(data1, data2):
    return np.median(data1) - np.median(data2)
def LPAcrossFlies(
        Data_to_plot,
        file_name,
        colors=None,
        marker_styles=None,
        box_width=0.25,
        box_softness=0.65
):
    global Color_blind_palette
    global markers
    global vio

    if colors is None:
        colors = Color_blind_palette
    if marker_styles is None:
        marker_styles = markers

    combined_df = pd.concat(Data_to_plot)
    plt.figure(figsize=(len(Data_to_plot) * 2, 10))
    ax = plt.gca()

    stat = dict()
    stat["Group"] = []
    stat["mean"] = []
    stat["std"] = []

    # ------------------------------------------------------------
    # stripplot (unchanged)
    # ------------------------------------------------------------
    for i, d in enumerate(Data_to_plot):
        stat["Group"].append(d["Group_Name"].iloc[0])
        stat["mean"].append(np.nanmean(d["LandingProb"]))
        stat["std"].append(np.nanstd(d["LandingProb"]))

        sns.stripplot(
            x="Group_Name",
            y="LandingProb",
            data=d,
            alpha=0.4,
            jitter=0.1,
            dodge=False,
            size=30,
            marker=marker_styles[i % len(marker_styles)],
            color=colors[i % len(colors)],
            zorder=10,
            ax=ax
        )

    pd.DataFrame(stat).to_csv(f"{file_name}-stat.csv")

    # ------------------------------------------------------------
    # boxplot (NEW, shifted left)
    # ------------------------------------------------------------
    group_order = combined_df["Group_Name"].unique()

    shift = 0.25

    def soften_color(color):
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
        softness = float(np.clip(box_softness, 0, 1))
        return tuple(rgb + (1.0 - rgb) * softness)

    for i, group in enumerate(group_order):
        vals = combined_df.loc[combined_df["Group_Name"] == group, "LandingProb"].dropna().values
        if len(vals) == 0:
            continue

        group_color = colors[i % len(colors)]
        ax.boxplot(
            [vals],
            positions=[i - shift],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor=soften_color(group_color),
                edgecolor=group_color,
                linewidth=2
            ),
            whiskerprops=dict(color=group_color, linewidth=2),
            capprops=dict(color=group_color, linewidth=2),
            medianprops=dict(color="black", linewidth=2),
        )

    # ------------------------------------------------------------
    # formatting (unchanged)
    # ------------------------------------------------------------
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    ax.set_ylim(-0.1, 1.1)

    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(group_order)

    plt.ylabel("Landing Probability", fontsize=25)
    plt.tick_params(axis="y", labelsize=25)
    plt.tick_params(axis="x", labelsize=25, rotation=45)
    plt.tick_params(width=3, length=10)

    plt.yticks([0, 0.5, 1])
    plt.xlim(-0.5, len(group_order) - 0.5)

    sns.despine(trim=True)
    plt.tight_layout()

    plt.savefig(f"{file_name}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
def LPAcrossLight(Data_to_plot, file_name, color):
    global Color_blind_palette
    global markers
    global samejoint
    global vio

    data_type = "LandingProb"
    combined_df = pd.concat(Data_to_plot, ignore_index=True)
    plt.figure(figsize=(len(Data_to_plot) * 3, 10))

    # Sort by Fly# to keep connection consistent
    combined_df = combined_df.sort_values(by=['Fly#', 'Group_Name'])
    # print(combined_df)

    g = sns.pointplot(data=combined_df, x='Group_Name', y=data_type, ci=None, dodge=True, color=color, join=False)

    # Connect same flies with lines
    for fly_id, group in combined_df.groupby('Fly#'):
        print(fly_id, group)
        plt.plot(group['Group_Name'], group[data_type], marker='o', markersize=20,  color="lightgrey", linewidth=5)

    group_stat = combined_df.groupby('Group_Name')['LandingProb'].agg(['mean', 'std', 'count', 'sem']).reset_index()
    # 3 group_stat['ci'] = 1.96 * group_stat['std'] / np.sqrt(group_stat['count'])
    pd.DataFrame(group_stat).to_csv(f"{file_name}-stat.csv")

    sns.pointplot(x='Group_Name', y='mean', data=group_stat, color=color, linestyles=" ", markers="s", errorbar=None, scale=2, zorder=10)
    plt.errorbar(x=group_stat['Group_Name'], y=group_stat['mean'], yerr=group_stat['sem'], fmt="none", color=color, capsize=10, zorder=10)

    mean_df = combined_df.groupby('Group_Name', as_index=False)[data_type].mean()
    plt.plot(mean_df['Group_Name'], mean_df[data_type], color=color, marker='o', markersize=20, linewidth=5, label='Mean')

    plt.title('Change in Landing Probability Across Groups')
    plt.xlabel('Group')
    g.spines['left'].set_linewidth(3)
    g.spines['bottom'].set_linewidth(3)
    # plt.legend(handles=legend_elements, loc='upper right', fontsize=20)
    # plt.ylabel("Mean Landing Latency", fontsize=25)
    plt.ylabel("Landing Probability", fontsize=25)
    plt.tick_params(axis="y", labelsize=25)
    plt.tick_params(axis="x", labelsize=25, rotation=45)
    plt.tick_params(width=3, length=10)
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.5, 1.5)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf")
    plt.close()
def ecdfPlot(Original_Data, filename):
    global labels
    global lines
    global latency_threshold
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    legend_handles = []  # List to store legend handles
    legend_labels = []
    # legend_labels = ["ON", "OFF"]
    # combined_df = pd.concat(Original_Data)
    stat = dict()
    stat["Group"] = []
    stat["mean"] = []
    stat["std"] = []
    stat["sem"] = []

    for i, d in enumerate(Original_Data):
        stat["Group"].append(d["Group_Name"][0])
        stat["mean"].append(np.nanmean(d["TrialLandingLatency"]))
        stat["std"].append(np.nanstd(d["TrialLandingLatency"]))
        stat["sem"].append(np.nanstd(d["TrialLandingLatency"]) / np.sqrt(len(d["TrialLandingLatency"])))
        sns.ecdfplot(d["TrialLandingLatency"], alpha=0.8, color=Color_blind_palette[i], linestyle=lines[i], legend=True, linewidth=3)
        legend_handles.append(Line2D([0], [0], color=Color_blind_palette[i], linestyle=lines[i], lw=3))
        # print(d["Group_Name"][0])
        legend_labels.append(d["Group_Name"].iloc[2])

    pd.DataFrame(stat).to_csv(f"{filename}-stat.csv")
    if FilterHighLatency:
        ax.set_xlim(-0.1, latency_threshold + 0.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, latency_threshold])
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, 1.5, 3])
        ax.set_yticks([0, 0.5, 1])
    plt.tick_params(axis="y", labelsize=25)
    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(width=3, length=10)
    sns.despine(trim=True)
    ax.spines["left"].set_linewidth(2)  # Top border
    ax.spines["bottom"].set_linewidth(2)
    ax.set_xlabel("Landing latency (s)", fontsize=25)
    ax.set_ylabel("Percentage", fontsize=25)

    ax.legend(legend_handles, legend_labels, fontsize=20, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf")
    # plt.show()
    return None
def ReadAndFilterData(GroupName, Flies_to_pick, Landing_Data_path):
    global Trial_num
    global Threshold
    Landing_Data = pd.read_excel(Landing_Data_path)
    Landing_Data = Landing_Data.iloc[Flies_to_pick[0] - 1:Flies_to_pick[1]]
    Valid_data_index = []
    for index, row in Landing_Data.iterrows():
        str_nan_count = 0
        for data in row.values:
            if isinstance(data, str) or pd.isna(data):
                str_nan_count += 1
        if Threshold:
            if str_nan_count < (len(row.values) / 2):
                Valid_data_index.append(index)
        else:
            Valid_data_index.append(index)
    Landing_Data = Landing_Data[Landing_Data.index.isin(Valid_data_index)]
    GroupNameCol = [GroupName] * len(Valid_data_index)
    Landing_Data["Group_Name"] = GroupNameCol
    return Landing_Data
def ReadAndFilterOptogeneticData(GroupName, Flies_to_pick, Landing_Data_paths):
    global Trial_num
    Landing_Data_LO = pd.read_excel(Landing_Data_paths[0])
    Landing_Data_NL = pd.read_excel(Landing_Data_paths[1])
    Landing_Data_LO = Landing_Data_LO.iloc[Flies_to_pick[0] - 1:Flies_to_pick[1]]
    Landing_Data_NL = Landing_Data_NL.iloc[Flies_to_pick[0] - 1:Flies_to_pick[1]]

    Valid_data_index_LO = []
    for index, row in Landing_Data_LO.iterrows():
        str_nan_count = 0
        for data in row.values:
            if isinstance(data, str) or pd.isna(data):
                str_nan_count += 1
        if str_nan_count < (len(row.values) / 2):
            Valid_data_index_LO.append(index)

    Valid_data_index_NL = []
    for index, row in Landing_Data_NL.iterrows():
        str_nan_count = 0
        for data in row.values:
            if isinstance(data, str) or pd.isna(data):
                str_nan_count += 1
        if str_nan_count < (len(row.values) / 2):
            Valid_data_index_NL.append(index)

    Valid_index = []
    if len(Valid_data_index_LO) <= len(Valid_data_index_NL):
        Valid_index = Valid_data_index_LO
    else:
        Valid_index = Valid_data_index_NL

    Landing_Data_LO = Landing_Data_LO[Landing_Data_LO.index.isin(Valid_index)]
    GroupNameCol = [GroupName + " LO"] * len(Valid_index)
    Landing_Data_LO["Group_Name"] = GroupNameCol

    Landing_Data_NL = Landing_Data_NL[Landing_Data_NL.index.isin(Valid_index)]
    GroupNameCol = [GroupName + " NL"] * len(Valid_index)
    Landing_Data_NL["Group_Name"] = GroupNameCol

    return Landing_Data_LO, Landing_Data_NL
def CalculateOptogeneticLPLL(GroupName, Landing_Data, fps):
    global FilterHighLatency
    global trial_offset

    latency_threshold = 0.5
    LP_mLL_Data = dict()
    LP_mLL_Data["LandingProb"] = []
    LP_mLL_Data["MLandingLatency"] = []
    LP_mLL_Data["Fly#"] = []
    Trials = ["Trial_" + str(i + 1 + trial_offset) for i in range(Trial_num - trial_offset)]
    Landing_Data = Landing_Data[Trials]
    for index, row in Landing_Data.iterrows():
        if FilterHighLatency:
            Landing_latency = [l / fps for l in row if not isinstance(l, str) and l > 0 and l < latency_threshold * fps]
            Nan_data = [n for n in row if pd.isna(n) or isinstance(n, str) or n < -1]
            Flying = [f for f in row if not (isinstance(f, str) or pd.isna(f)) and (f == -1 or f >= latency_threshold * fps)]
        else:
            Landing_latency = [l / fps for l in row if not isinstance(l, str) and l > 0]
            Nan_data = [n for n in row if pd.isna(n) or isinstance(n, str)]
            Flying = [f for f in row if not (isinstance(f, str) or pd.isna(f)) and (f == -1)]
        if len(Nan_data) + len(Flying) + len(Landing_latency) != Trial_num - trial_offset:
            print(f"Error while filtering data")
            print(f"Index: {index}")
            print(f"# of Nan: {len(Nan_data)}\t{Nan_data}")
            print(f"# of Flying: {len(Flying)}\t{Flying}")
            print(f"# of Landing: {len(Landing_latency)}\t{Landing_latency}")
        if len(Flying) + len(Landing_latency) != 0:
            LP_mLL_Data["Fly#"].append(index + 1)
            LP_mLL_Data["LandingProb"].append(len(Landing_latency) / (len(Flying) + len(Landing_latency)))
            LP_mLL_Data["MLandingLatency"].append(np.mean(Landing_latency))
    LP_mLL_Data["Group_Name"] = [GroupName] * len(LP_mLL_Data["Fly#"])
    LP_mLL_Data = pd.DataFrame(LP_mLL_Data)
    return LP_mLL_Data
def ReadData(GroupName, Flies_to_pick, Landing_Data_path):
    global Trial_num
    Landing_Data = pd.read_excel(Landing_Data_path)
    Landing_Data = Landing_Data.iloc[Flies_to_pick[0] - 1:Flies_to_pick[1]]

    GroupNameCol = [GroupName] * (Flies_to_pick[1] - Flies_to_pick[0] + 1)
    Landing_Data["Group_Name"] = GroupNameCol
    return Landing_Data
def CalculateLPAndmLLAcrossFlies(GroupName, Landing_Data, fps):
    global FilterHighLatency
    global trial_offset
    global latency_threshold
    LP_mLL_Data = dict()
    LP_mLL_Data["LandingProb"] = []
    LP_mLL_Data["MLandingLatency"] = []
    LP_mLL_Data["Fly#"] = []
    Trials = ["Trial_" + str(i + 1 + trial_offset) for i in range(Trial_num - trial_offset)]
    Landing_Data = Landing_Data[Trials]
    for index, row in Landing_Data.iterrows():
        if FilterHighLatency:
            Landing_latency = [l / fps for l in row if not isinstance(l, str) and l > 0 and l < latency_threshold * fps]
            Nan_data = [n for n in row if pd.isna(n) or isinstance(n, str) or n < -1]
            Flying = [f for f in row if not (isinstance(f, str) or pd.isna(f)) and (f == -1 or f >= latency_threshold * fps)]
        else:
            Landing_latency = [l / fps for l in row if not isinstance(l, str) and l > 0]
            Nan_data = [n for n in row if pd.isna(n) or isinstance(n, str)]
            Flying = [f for f in row if not (isinstance(f, str) or pd.isna(f)) and (f == -1)]
        if len(Nan_data) + len(Flying) + len(Landing_latency) != Trial_num - trial_offset:
            print(f"Error while filtering data")
            print(f"Index: {index}")
            print(f"# of Nan: {len(Nan_data)}\t{Nan_data}")
            print(f"# of Flying: {len(Flying)}\t{Flying}")
            print(f"# of Landing: {len(Landing_latency)}\t{Landing_latency}")
        if len(Flying) + len(Landing_latency) != 0:
            LP_mLL_Data["Fly#"].append(index + 1)
            LP_mLL_Data["LandingProb"].append(len(Landing_latency) / (len(Flying) + len(Landing_latency)))
            LP_mLL_Data["MLandingLatency"].append(np.mean(Landing_latency))
    LP_mLL_Data["Group_Name"] = [GroupName] * len(LP_mLL_Data["Fly#"])
    LP_mLL_Data = pd.DataFrame(LP_mLL_Data)
    return LP_mLL_Data
def GetTrial_Landing_Data(LandingData, group_name, fps):
    landing_data = []
    global trial_offset
    global FilterHighLatency
    global latency_threshold
    Trials = ["Trial_" + str(i + 1 + trial_offset) for i in range(Trial_num - trial_offset)]
    LandingData = LandingData[Trials]

    for index, row in LandingData.iterrows():
        t = 0
        # print(row)
        for data in row.values:
            if FilterHighLatency:
                if not (isinstance(data, str) or pd.isna(data) or float(data) == -1 or data > latency_threshold * fps):
                    # print(group_name, index, t + trial_offset)
                    landing_data.append(data/fps)
                    t += 1
            else:
                if not (isinstance(data, str) or pd.isna(data) or float(data) == -1):
                    # print(group_name, index, t + trial_offset)
                    landing_data.append(data/fps)
                    t += 1
        # print(t)
    landing_data = pd.DataFrame(
        {
            "TrialLandingLatency": landing_data,
            "Group_Name": [group_name] * len(landing_data)
        }
    )
    # print(group_name, len(landing_data))
    return landing_data
def ReadLandingData(FileName, GroupName, FPS, FirstFly, LastFly):
    FilterdData = ReadAndFilterData(GroupName, [FirstFly, LastFly], FileName)
    LPData = CalculateLPAndmLLAcrossFlies(GroupName, FilterdData, FPS)
    LLData = GetTrial_Landing_Data(FilterdData, GroupName, FPS)
    return LPData, LLData
def ReadOptogeneticData(ONFile, OFFFile, GroupName, FPS, FirstFly, LastFly):
    ONFiltered, OFFFiltered = ReadAndFilterOptogeneticData(GroupName, [FirstFly, LastFly], [ONFile, OFFFile])
    ONLPData = CalculateLPAndmLLAcrossFlies(GroupName + "-ON", ONFiltered, FPS)
    ONLLData = GetTrial_Landing_Data(ONFiltered, GroupName + "-ON", FPS)
    OFFLPData = CalculateLPAndmLLAcrossFlies(GroupName + "-OFF", OFFFiltered, FPS)
    OFFLLData = GetTrial_Landing_Data(OFFFiltered, GroupName + "-OFF", FPS)
    return ONLPData, ONLLData, OFFLPData, OFFLLData
def compare_from_excel(file_a, file_b, label_a="Group_A", label_b="Group_B",
                       fps=300, tau=0.71, n_perm=20000, random_state=0):
    """
    Convenience wrapper:
    load two Excel files and run both landing probability and landing latency tests.
    """
    df_a = pd.read_excel(file_a)
    df_b = pd.read_excel(file_b)

    lp_summary, lp_fly = compare_landing_probability_unpaired(
        df_a, df_b,
        label_a=label_a,
        label_b=label_b,
        n_perm=n_perm,
        random_state=random_state
    )

    ll_summary, ll_fly = compare_landing_latency_unpaired(
        df_a, df_b,
        label_a=label_a,
        label_b=label_b,
        fps=fps,
        tau=tau,
        n_perm=n_perm,
        random_state=random_state
    )

    return {
        "lp_summary": lp_summary,
        "lp_fly_table": lp_fly,
        "ll_summary": ll_summary,
        "ll_fly_table": ll_fly,
    }
def compare_landing_latency_unpaired(df_a, df_b, label_a="Group_A", label_b="Group_B",
                                     fps=300, tau=0.71, n_perm=20000, random_state=0):
    """
    Unpaired permutation test on per-fly RMST for landing latency.
    """
    fly_a = compute_fly_landing_rmst(df_a, fps=fps, tau=tau)
    fly_b = compute_fly_landing_rmst(df_b, fps=fps, tau=tau)

    if len(fly_a) == 0 or len(fly_b) == 0:
        raise ValueError("One or both groups have no valid flies for landing latency comparison.")

    x = fly_a["RMST"].values
    y = fly_b["RMST"].values

    observed, p_value, _ = _permutation_test_unpaired(
        x, y, n_perm=n_perm, random_state=random_state
    )

    summary = pd.DataFrame([{
        "comparison_type": "landing_latency_unpaired_rmst",
        "group_a": label_a,
        "group_b": label_b,
        "n_fly_a": len(fly_a),
        "n_fly_b": len(fly_b),
        "mean_rmst_a": np.mean(x),
        "mean_rmst_b": np.mean(y),
        "estimate_b_minus_a": observed,
        "permutation_p": p_value,
        "fps": fps,
        "tau": tau,
    }])

    fly_table = pd.concat([
        fly_a.assign(Group=label_a),
        fly_b.assign(Group=label_b)
    ], ignore_index=True)

    return summary, fly_table
def compare_landing_probability_unpaired(df_a, df_b, label_a="Group_A", label_b="Group_B",
                                         n_perm=20000, random_state=0):
    """
    Unpaired permutation test on per-fly landing probability.
    """
    fly_a = compute_fly_landing_probability(df_a)
    fly_b = compute_fly_landing_probability(df_b)

    if len(fly_a) == 0 or len(fly_b) == 0:
        raise ValueError("One or both groups have no valid flies for LP comparison.")

    x = fly_a["LandingProb"].values
    y = fly_b["LandingProb"].values

    observed, p_value, _ = _permutation_test_unpaired(
        x, y, n_perm=n_perm, random_state=random_state
    )

    summary = pd.DataFrame([{
        "comparison_type": "landing_probability_unpaired",
        "group_a": label_a,
        "group_b": label_b,
        "n_fly_a": len(fly_a),
        "n_fly_b": len(fly_b),
        "mean_lp_a": np.mean(x),
        "mean_lp_b": np.mean(y),
        "estimate_b_minus_a": observed,
        "permutation_p": p_value,
    }])

    fly_table = pd.concat([
        fly_a.assign(Group=label_a),
        fly_b.assign(Group=label_b)
    ], ignore_index=True)

    return summary, fly_table
def compute_fly_landing_rmst(df, fps=300, tau=0.71, value_unit="frames"):
    clean_df, trial_cols = _extract_trial_matrix(df)

    rows = []
    kmf = KaplanMeierFitter()

    for _, row in clean_df.iterrows():
        fly = row["Fly#"]
        vals = row[trial_cols].values

        latencies = []
        events = []

        for v in vals:
            if pd.isna(v):
                continue

            if v == -1:
                latencies.append(tau)
                events.append(0)
                continue

            lat = float(v) / fps

            if lat > tau:
                latencies.append(tau)
                events.append(0)
            else:
                latencies.append(lat)
                events.append(1)

        if len(latencies) == 0:
            continue

        kmf.fit(latencies, event_observed=events)
        rmst = float(restricted_mean_survival_time(kmf, t=tau))

        rows.append({
            "Fly#": fly,
            "RMST": rmst,
            "n_valid_trials": len(latencies),
            "n_events": int(np.sum(events)),
            "event_rate": float(np.mean(events)),
        })

    return pd.DataFrame(rows)
def compute_fly_landing_probability(df, fps=300, tau=0.71):
    """
    Compute one landing probability value per fly.

    Included trials:
        numeric latency values and -1
    Excluded trials:
        N/A, NF, NotFlying, blanks

    Landing = numeric latency >= 0
    Flying  = -1
    """
    clean_df, trial_cols = _extract_trial_matrix(df)

    rows = []
    for _, row in clean_df.iterrows():
        fly = row["Fly#"]
        vals = row[trial_cols].values

        valid = [v for v in vals if not pd.isna(v)]
        if len(valid) == 0:
            continue

        landing_num = sum(v >= 0 and (v / fps) < tau for v in valid)
        total_num = len(valid)

        lp = landing_num / total_num

        rows.append({
            "Fly#": fly,
            "LandingProb": lp,
            "n_valid_trials": total_num,
            "n_landing_trials": landing_num,
            "n_flying_trials": sum(v == -1 for v in valid),
        })
    # print(rows)
    return pd.DataFrame(rows)
def _permutation_test_unpaired(x, y, n_perm=20000, random_state=0):
    rng = np.random.default_rng(random_state)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed = np.mean(y) - np.mean(x)

    pooled = np.concatenate([x, y])
    n_x = len(x)

    perm_stats = []
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        x_perm = perm[:n_x]
        y_perm = perm[n_x:]
        perm_stats.append(np.mean(y_perm) - np.mean(x_perm))

    perm_stats = np.asarray(perm_stats)

    p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, p_value, perm_stats
def _clean_trial_value(val):
    """
    Convert one trial entry into a standardized code.

    Returns:
        np.nan -> excluded trial (N/A, NF, NotFlying, blank)
        -1.0   -> flying / censored trial
        float  -> numeric landing latency in frames
    """
    if pd.isna(val):
        return np.nan

    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"n/a", "na", "nf", "notflying", ""}:
            return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan

    try:
        return float(val)
    except Exception:
        return np.nan
def _extract_trial_matrix(df, exclude_first_n=3, max_nan_trials=10):
    """
    Keep Fly# and selected Trial_* columns, clean values,
    and exclude flies with too many NaN trials.

    exclude_first_n:
        number of earliest trials to exclude (e.g., 3)

    max_nan_trials:
        maximum allowed NaN trials per fly
        flies with > max_nan_trials will be excluded
    """
    # collect trial columns
    trial_cols = [c for c in df.columns if str(c).startswith("Trial_")]

    if "Fly#" not in df.columns:
        # Older landing-latency sheets in this script identify flies by row
        # position. Preserve that convention for KM/stat helpers.
        df = df.copy()
        df["Fly#"] = np.arange(1, len(df) + 1)
    if len(trial_cols) == 0:
        raise ValueError("No trial columns found.")

    # ------------------------------------------------------------
    # sort trial columns numerically
    # ------------------------------------------------------------
    trial_cols = sorted(trial_cols, key=lambda x: int(x.split("_")[1]))

    # ------------------------------------------------------------
    # exclude first N trials
    # ------------------------------------------------------------
    if exclude_first_n > 0:
        trial_cols = trial_cols[exclude_first_n:]

    # ------------------------------------------------------------
    # extract + clean
    # ------------------------------------------------------------
    out = df[["Fly#"] + trial_cols].copy()

    for col in trial_cols:
        out[col] = out[col].apply(_clean_trial_value)

    # ------------------------------------------------------------
    # exclude flies with too many NaNs
    # ------------------------------------------------------------
    nan_counts = out[trial_cols].isna().sum(axis=1)
    mask = nan_counts <= max_nan_trials

    excluded_flies = out.loc[~mask, "Fly#"].tolist()

    if len(excluded_flies) > 0:
        print(f"Excluding {len(excluded_flies)} flies with > {max_nan_trials} NaN trials: {excluded_flies}")

    out = out.loc[mask].reset_index(drop=True)

    return out, trial_cols
def build_landing_latency_survival_df(df, group_name, fps=250, tau=0.71, value_unit="frames", exclude_first_n=3):
    """
    Convert Excel-style landing latency sheet into KM-ready trial-level dataframe.

    Numeric value <= tau: Event = 1
    Numeric value > tau: Event = 0, censored at tau
    -1: Event = 0, censored at tau
    N/A / NF / blank: excluded
    """
    clean_df, trial_cols = _extract_trial_matrix(df, exclude_first_n=exclude_first_n)

    rows = []

    for _, row in clean_df.iterrows():
        fly = row["Fly#"]

        for col in trial_cols:
            v = row[col]

            if pd.isna(v):
                continue

            trial_num = int(str(col).split("_")[1])

            if v == -1:
                latency = tau
                event = 0
            else:
                if value_unit == "frames":
                    latency_sec = float(v) / fps
                else:
                    latency_sec = float(v)

                if latency_sec > tau:
                    latency = tau
                    event = 0
                else:
                    latency = latency_sec
                    event = 1

            rows.append({
                "Fly#": fly,
                "Trial#": trial_num,
                "Latency": latency,
                "Event": event,
                "Group_Name": group_name
            })

    return pd.DataFrame(rows)
def plot_landing_latency_km_from_excel_dfs(
    data_to_plot,
    file_name,
    fps=250,
    tau=0.71,
    value_unit="frames",
    exclude_first_n=3,
    colors=None,
    show_ci=False
):
    """
    Plot KM curves for landing latency.

    data_to_plot:
        [(df1, "Group 1"), (df2, "Group 2"), ...]
    """
    if colors is None:
        colors = ["red", "green", "blue", "red", "green"]
    lines = ["solid"] * len(data_to_plot)
    fig, ax = plt.subplots(figsize=(7, 7))
    kmf = KaplanMeierFitter()

    all_survival_df = []

    for i, (df, group_name) in enumerate(data_to_plot):
        surv_df = build_landing_latency_survival_df(
            df=df,
            group_name=group_name,
            fps=fps,
            tau=tau,
            value_unit=value_unit,
            exclude_first_n=exclude_first_n
        )

        if len(surv_df) == 0:
            print(f"No valid KM data for {group_name}")
            continue

        all_survival_df.append(surv_df)

        kmf.fit(
            durations=surv_df["Latency"],
            event_observed=surv_df["Event"],
            label=f"{group_name} (n={len(surv_df['Fly#'])})"
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=show_ci,
            color=colors[i % len(colors)],
            linewidth=3,
            linestyle=lines[i % len(lines)]
        )

    ax.set_xlim(0, tau)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Landing latency (s)", fontsize=22)
    ax.set_ylabel("Probability of no landing", fontsize=22)

    ax.set_xticks([0, tau])
    ax.set_yticks([0, 0.5, 1])

    ax.tick_params(axis="x", labelsize=18, width=3, length=8)
    ax.tick_params(axis="y", labelsize=18, width=3, length=8)

    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)

    sns.despine(trim=True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(f"{file_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")

    if len(all_survival_df) > 0:
        combined_surv_df = pd.concat(all_survival_df, ignore_index=True)
        combined_surv_df.to_csv(f"{file_name}-KM-data.csv", index=False)

    plt.show()
    plt.close()

def run_pairwise_landing_stats(
    group_dfs,
    out_prefix="PairwiseLandingStats",
    fps=300,
    tau=0.71,
    n_perm=20000,
    random_state=0
):
    """
    Run pairwise unpaired permutation tests for:
    1. landing probability
    2. landing latency RMST

    group_dfs:
        dictionary like:
        {
            "T2-TiTa-SL": T2TTa_SL,
            "T3-TiTa-SL": T3TTa_SL,
            ...
        }
    """
    import itertools
    lp_results = []
    ll_results = []
    lp_fly_tables = []
    ll_fly_tables = []

    for label_a, label_b in itertools.combinations(group_dfs.keys(), 2):
        df_a = group_dfs[label_a]
        df_b = group_dfs[label_b]

        # ----------------------------
        # Landing probability
        # ----------------------------
        lp_summary, lp_fly = compare_landing_probability_unpaired(
            df_a,
            df_b,
            label_a=label_a,
            label_b=label_b,
            n_perm=n_perm,
            random_state=random_state
        )

        lp_summary["comparison"] = f"{label_a} vs {label_b}"
        lp_results.append(lp_summary)

        lp_fly["comparison"] = f"{label_a} vs {label_b}"
        lp_fly_tables.append(lp_fly)

        # ----------------------------
        # Landing latency / RMST
        # ----------------------------
        ll_summary, ll_fly = compare_landing_latency_unpaired(
            df_a,
            df_b,
            label_a=label_a,
            label_b=label_b,
            fps=fps,
            tau=tau,
            n_perm=n_perm,
            random_state=random_state
        )

        ll_summary["comparison"] = f"{label_a} vs {label_b}"
        ll_results.append(ll_summary)

        ll_fly["comparison"] = f"{label_a} vs {label_b}"
        ll_fly_tables.append(ll_fly)

    lp_summary_all = pd.concat(lp_results, ignore_index=True)
    ll_summary_all = pd.concat(ll_results, ignore_index=True)

    lp_fly_all = pd.concat(lp_fly_tables, ignore_index=True)
    ll_fly_all = pd.concat(ll_fly_tables, ignore_index=True)

    lp_summary_all.to_csv(f"{out_prefix}-LandingProbability-pairwise-summary.csv", index=False)
    ll_summary_all.to_csv(f"{out_prefix}-LandingLatency-pairwise-summary.csv", index=False)

    lp_fly_all.to_csv(f"{out_prefix}-LandingProbability-pairwise-fly-values.csv", index=False)
    ll_fly_all.to_csv(f"{out_prefix}-LandingLatency-pairwise-fly-values.csv", index=False)

    return {
        "lp_summary": lp_summary_all,
        "ll_summary": ll_summary_all,
        "lp_fly_values": lp_fly_all,
        "ll_fly_values": ll_fly_all,
    }

FilterHighLatency = True
OPTO = True
Trial_num = 20
vio = False
Threshold = True
trial_offset = 3
DataFolder = r"C:\Users\agrawal-admin\Desktop\LandingDataSummary"

# latency_threshold = 0.5
latency_threshold = 0.71

if __name__ == "__main__":
    T1CTF_path = os.path.join(DataFolder, r"6LegsLP\T1-CxTr-LL_new.xlsx")
    T1CTF_LP, T1CTF_LL = ReadLandingData(T1CTF_path, r"WT-T1-CxTr", 250, 1, 15)

    T2CTF_path = os.path.join(DataFolder, r"6LegsLP\T2-CxTrLP.xlsx")
    T2CTF_LP, T2CTF_LL = ReadLandingData(T2CTF_path, r"WT-T2-CxTr", 250, 1, 18)

    T1CTF_SL_path = os.path.join(DataFolder, r"Necessity\T1RightIntact_CTF_LL_All.xlsx")
    T1CTF_SL_LP, T1CTF_SL_LL = ReadLandingData(T1CTF_SL_path, r"WT-SL-T1-CxTr", 300, 1, 21)

    T2CTF_SL_path = os.path.join(DataFolder, r"Necessity\T2RightIntact_CTF_LL_All.xlsx")
    T2CTF_SL_LP, T2CTF_SL_LL = ReadLandingData(T2CTF_SL_path, r"WT-SL-T2-CxTr", 300, 1, 20)

    T3CTF_path = os.path.join(DataFolder, r"6LegsLP\T3-CxTrLP.xlsx")
    T3CTF_LP, T3CTF_LL = ReadLandingData(T3CTF_path, r"WT-T3-CxTr", 250, 1, 17)

    T3CTF_SL_path = os.path.join(DataFolder, r"Necessity\T3RightIntact_CTF_LL_All.xlsx")
    T3CTF_SL_LP, T3CTF_SL_LL = ReadLandingData(T3CTF_SL_path, r"WT-SL-T3-CxTr", 300, 1, 20)



    T1TTa_path = os.path.join(DataFolder, r"6LegsLP\T1-TiTaLP.xlsx")
    T1TTa_LP200FPS, T1TTa_LL200FPS = ReadLandingData(T1TTa_path, r"WT-T1-TiTa", 200, 1, 12)
    T1TTa_LP250FPS, T1TTa_LL250FPS = ReadLandingData(T1TTa_path, r"WT-T1-TiTa", 250, 13, 15)
    T1TTa_LP = pd.concat([T1TTa_LP200FPS, T1TTa_LP250FPS])
    T1TTa_LL = pd.concat([T1TTa_LL200FPS, T1TTa_LL250FPS])


    T2TTa_path = os.path.join(DataFolder, r"6LegsLP\T2-TiTaLP.xlsx")
    T2TTa_LP, T2TTa_LL = ReadLandingData(T2TTa_path, r"WT-T2-TiTa", 200, 1, 15)

    T2TTa_SL_path = os.path.join(DataFolder, r"Necessity\T2RightIntact_TTa_All.xlsx")
    T2TTa_SL_LP, T2TTa_SL_LL = ReadLandingData(T2TTa_SL_path, r"WT-SL-T2-TiTa", 300, 1, 21)

    T2TTa_inward_path = os.path.join(DataFolder, r"Others\WT-T2-TiTa_new_OT_filtered.xlsx")
    T2TTa_inward_LP, T2TTa_inward_LL = ReadLandingData(T2TTa_inward_path, r"WT-IN", 200, 1, 15)

    T2TTa_outward_path = os.path.join(DataFolder, r"Others\WT-T2-TiTa_new_IT_filtered.xlsx")
    T2TTa_outward_LP, T2TTa_outward_LL = ReadLandingData(T2TTa_outward_path, r"WT-OUT", 200, 1, 15)

    T3TTa_path = os.path.join(DataFolder, r"6LegsLP\T3-TiTaLP.xlsx")
    T3TTa_LP200FPS, T3TTa_LL200FPS = ReadLandingData(T3TTa_path, r"WT-T3-TiTa", 200, 1, 15)
    T3TTa_LP250FPS, T3TTa_LL250FPS = ReadLandingData(T3TTa_path, r"WT-T3-TiTa", 250, 16, 20)
    T3TTa_LP = pd.concat([T3TTa_LP200FPS, T3TTa_LP250FPS])
    T3TTa_LL = pd.concat([T3TTa_LL200FPS, T3TTa_LL250FPS])

    T3TTa_SL_path = os.path.join(DataFolder, r"Necessity\T3RightIntact_TTa_All.xlsx")
    T3TTa_SL_LP, T3TTa_SL_LL = ReadLandingData(T3TTa_SL_path, r"WT-SL-T3-TiTa", 300, 1, 17)

    T2_TTa_Ab_path = os.path.join(DataFolder, r"CONTROL\WT-Abdomen-ALL.xlsx")
    T2_TTa_Ab_LP, T2_TTa_Ab_LL = ReadLandingData(T2_TTa_Ab_path, r"WT Abdomen", 250, 1, 11)

    T3Cut_path = os.path.join(DataFolder, r"CONTROL\WT-Ab-LegCutOff-ALL.xlsx")
    T3Cut_LP, T3Cut_LL = ReadLandingData(T3Cut_path, r"WT Ab T3CutOff", 250, 1, 10)

    NoContact_path = os.path.join(DataFolder, r"CONTROL\IntactFly_Control.xlsx")
    NoContact_LP, NoContact_LL = ReadLandingData(NoContact_path, r"No contact", 300, 1, 10)

    CSS39_path = os.path.join(DataFolder, r"KIR\CSS-0039_T2-TiTa-ALL.xlsx")
    CSS39_LP, CSS39_LL = ReadLandingData(CSS39_path, r"CSS39", 250, 1, 15)

    CSS48_path = os.path.join(DataFolder, r"KIR\CSS-0048_T2-TiTa-ALL.xlsx")
    CSS48_LP, CSS48_LL = ReadLandingData(CSS48_path, r"CSS48", 250, 1, 17)

    HP1_path = os.path.join(DataFolder, r"KIR\G106-HP1_T2-TiTa-ALL.xlsx")
    HP1_LP, HP1_LL = ReadLandingData(HP1_path, r"HP-1", 250, 1, 16)

    HP2_path = os.path.join(DataFolder, r"KIR\G107-HP2_T2-TiTa-ALL.xlsx")
    HP2_LP, HP2_LL = ReadLandingData(HP2_path, r"HP-2", 250, 1, 14)

    HP2_Gal4_path = r"C:\Users\agrawal-admin\Desktop\LandingDataSummary\CONTROL\G107-Gal4_T2-TiTa.xlsx"
    HP2_Gal4_LP, HP2_Gal4_LL = ReadLandingData(HP2_Gal4_path, r"HP-2-Gal4", 250, 1, 14)

    HP3_path = os.path.join(DataFolder, r"KIR\G108-HP3_T2-TiTa-ALL.xlsx")
    HP3_LP, HP3_LL = ReadLandingData(HP3_path, r"HP-3", 250, 1, 17)


    ClFl_path = os.path.join(DataFolder, r"KIR\G114-ClFl_T2-TiTa-ALL.xlsx")
    ClFl_LP, ClFl_LL = ReadLandingData(ClFl_path, r"CL-FL", 250, 1, 16)

    ClFl_Gal4_path = r"C:\Users\agrawal-admin\Desktop\LandingDataSummary\CONTROL\G114-Gal4_T2-TiTa.xlsx"
    ClFl_Gal4_LP, ClFl_Gal4_LL = ReadLandingData(ClFl_Gal4_path, r"CL-FL-Gal4", 250, 1, 15)

    ClEx_path = os.path.join(DataFolder, r"KIR\G116-ClEx_T2-TiTa-ALL.xlsx")
    ClEx_LP, ClEx_LL = ReadLandingData(ClEx_path, r"CL-EX", 250, 1, 18)

    HkFl_path = os.path.join(DataFolder, r"KIR\G117-HkFl_T2-TiTa-ALL.xlsx")
    HkFl_LP, HkFl_LL = ReadLandingData(HkFl_path, r"HK-FL", 250, 1, 18)

    HkEx_path = os.path.join(DataFolder, r"KIR\G118-HkEx_T2-TiTa-ALL.xlsx")
    HkEx_LP, HkEx_LL = ReadLandingData(HkEx_path, r"HK-EX", 250, 1, 15)


    Club_path = os.path.join(DataFolder, r"KIR\G119-Club_T2-TiTa-ALL.xlsx")
    Club_LP, Club_LL = ReadLandingData(Club_path, r"Club", 250, 1, 18)

    Iav_path = os.path.join(DataFolder, r"KIR\G115-Iav_T2-TiTa-ALL.xlsx")
    Iav_LP, Iav_LL = ReadLandingData(Iav_path, r"Iav", 250, 1, 16)



    # lines = ["dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "solid", "solid", "solid"]
    lines = ["solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid","solid","solid","solid"]
    markers = ["o", "o", "o", ">", ">", "o", "o", "o", "o", "o", "o", "o", "o"]

    # Choose a colormap (e.g., viridis, plasma, coolwarm, etc.)
    cmap = cm.get_cmap('viridis', 12)

    # Generate list of colors
    colors = [cmap(i) for i in range(12)]
    Color_blind_palette = ["blue", "green", "orange", "dodgerblue", "lawngreen", "orange"]
    Color_blind_palette = ["blue", "peru", "sandybrown", "olive", "darkkhaki", "gold", "darkgreen", "seagreen", "mediumseagreen", "mediumaquamarine", "lightseagreen", "teal"]

    LP_data_type = "LandingProb"
    LL_data_type = "TrialLandingLatency"

    Color_blind_palette = ["red", "green", "blue", "red", "green", "orange"]
    # Color_blind_palette = ["indigo", "deepskyblue", "orangered"]
    LPAcrossFlies([T1CTF_SL_LP, T2TTa_SL_LP, T2CTF_SL_LP, T3TTa_SL_LP, T3CTF_SL_LP], "Ablation")
    LPAcrossFlies([T2_TTa_Ab_LP, T3Cut_LP, NoContact_LP], "NoContact")
    # ------------------------------------------------------------
    # KIR experiment usage
    # ------------------------------------------------------------

    kir_groups = [
        {
            "label": "CSS39",
            "path": CSS39_path,
            "lp": CSS39_LP,
        },
        {
            "label": "CSS48",
            "path": CSS48_path,
            "lp": CSS48_LP,
        },
        {
            "label": "HP-1",
            "path": HP1_path,
            "lp": HP1_LP,
        },
        {
            "label": "HP-2",
            "path": HP2_path,
            "lp": HP2_LP,
        },
        {
            "label": "HP-2-Gal4",
            "path": HP2_Gal4_path,
            "lp": HP2_Gal4_LP,
        },
        {
            "label": "HP-3",
            "path": HP3_path,
            "lp": HP3_LP,
        },
        {
            "label": "CL-FL",
            "path": ClFl_path,
            "lp": ClFl_LP,
        },
        {
            "label": "CL-FL-Gal4",
            "path": ClFl_Gal4_path,
            "lp": ClFl_Gal4_LP,
        },
        {
            "label": "CL-EX",
            "path": ClEx_path,
            "lp": ClEx_LP,
        },
        {
            "label": "HK-FL",
            "path": HkFl_path,
            "lp": HkFl_LP,
        },
        {
            "label": "HK-EX",
            "path": HkEx_path,
            "lp": HkEx_LP,
        },
        {
            "label": "Club",
            "path": Club_path,
            "lp": Club_LP,
        },
        {
            "label": "Iav",
            "path": Iav_path,
            "lp": Iav_LP,
        },
    ]

    kir_panels = [
        kir_groups[:7],
        kir_groups[7:],
    ]

    kir_output_dir = "KIR_Seaborn_Figures"
    os.makedirs(kir_output_dir, exist_ok=True)
    os.chdir(kir_output_dir)

    for panel_idx, panel_groups in enumerate(kir_panels, start=1):
        panel_colors = sns.color_palette("viridis", n_colors=len(panel_groups))
        Color_blind_palette = panel_colors
        markers = ["o"] * len(panel_groups)

        LPAcrossFlies(
            [group["lp"] for group in panel_groups],
            f"KIR_LP_panel{panel_idx}"
        )

        plot_landing_latency_km_from_excel_dfs(
            data_to_plot=[
                (pd.read_excel(group["path"]), group["label"])
                for group in panel_groups
            ],
            file_name=f"KIR_LL_KM_panel{panel_idx}",
            fps=250,
            tau=0.71,
            value_unit="frames",
            exclude_first_n=3,
            colors=panel_colors,
            show_ci=False
        )
