"""Shared trial lookup and landing-outcome helpers."""

import pandas as pd


def trial_key(index):
    """Return the standard metadata/kinematic dictionary key for an index."""
    return f"F{index[0]}T{index[1]}"


def get_trial_metadata(group_info, index):
    return group_info.trial_metadata[trial_key(index)]


def get_trial_object(group_info, index):
    return group_info.fly_kinematic_data[trial_key(index)]


def ensure_trials_loaded(group_info, trial_types=None):
    """Initialize metadata when needed and load the requested trial types."""
    if len(group_info.trial_metadata) == 0:
        group_info.initialize_manual_data()

    if trial_types is None:
        trial_types = ["Landing", "Flying", "NF", "NA"]

    group_info.read_kinematic_data(trial_types=trial_types)


def is_successful_landing(meta, latency_threshold, allow_missing_sentinel=False):
    """Return whether metadata describes a landing within the threshold."""
    ll = meta["LL"]
    fps = meta["fps"]
    return (
        meta["TrialType"] == "Landing"
        and not pd.isna(ll)
        and (allow_missing_sentinel or ll != -1)
        and (ll / fps) <= latency_threshold
    )


def classify_landing(meta, latency_threshold):
    return "Success" if is_successful_landing(meta, latency_threshold) else "Failed"


def inverted_km_latency_event(
        meta,
        latency_threshold,
        tau,
        allow_missing_sentinel=False
):
    """Return the duration and event indicator used by inverted landing KM plots."""
    if is_successful_landing(
            meta,
            latency_threshold,
            allow_missing_sentinel=allow_missing_sentinel
    ):
        return min(meta["LL"] / meta["fps"], tau), 1
    return tau, 0


def landing_latency_seconds(meta, tau):
    """Return latency seconds, censoring status, and the latency source label."""
    ll = meta["LL"]
    fps = meta["fps"]
    if not pd.isna(ll) and ll != -1:
        return ll / fps, False, "metadata_LL"
    if meta["TrialType"] == "Flying":
        return tau, True, "flying_no_MOL_censored_at_tau"
    return float("nan"), False, "missing_LL"


def trial_indexes_from_labeled_ll_file(
        ll_file_path,
        behavior_label=None,
        selection_mode="numeric",
        fly_column="Fly#"
):
    """Return ``(fly, trial)`` indexes selected from a labeled LL workbook."""
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
                if pd.isna(pd.to_numeric(value, errors="coerce")):
                    continue
            else:
                if behavior_label is None:
                    raise ValueError("behavior_label is required when selection_mode='marker'.")
                if str(value).strip().upper() != str(behavior_label).strip().upper():
                    continue

            trial_text = str(column).replace("Trial_", "")
            try:
                trial_indexes.append((int(fly), int(trial_text)))
            except ValueError:
                continue

    return trial_indexes


def trial_sets_from_behavior_sources(behavior_sources):
    """Build named trial-index sets from LL-label workbook definitions."""
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

        trial_sets[behavior_label] = trial_indexes_from_labeled_ll_file(
            ll_file_path=path,
            behavior_label=marker_label,
            selection_mode=selection_mode
        )
    return trial_sets
