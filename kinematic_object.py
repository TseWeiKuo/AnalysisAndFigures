import os
import re
import warnings
import numpy as np
import pandas as pd
from natsort import natsorted

warnings.filterwarnings(action="ignore", category=FutureWarning)


# ------------------------------------------------------------
# Helpers for reading and grouping 3D csv files
# ------------------------------------------------------------

def group_files_by_fly(file_names):
    """
    Old filename parser.
    Keep this for backward compatibility in case some older data folders still use
    the old format.
    """
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}\.\d+.*?_Fly_(\d+)_Trial_(\d+)_")

    unique_combinations = {}
    grouped_files = {}
    current_fly_number = 1

    for file in file_names:
        match = pattern.search(file)
        if match:
            date = match.group(1)
            fly_number = match.group(2)
            trial_number = match.group(3)
            unique_fly_id = (date, fly_number)

            if unique_fly_id not in unique_combinations:
                unique_combinations[unique_fly_id] = current_fly_number
                current_fly_number += 1

            grouped_files[f"F{unique_combinations[unique_fly_id]}T{trial_number}"] = file

    return grouped_files


def group_files_by_fly_new(file_names):
    """
    New filename parser.

    Supported examples:
    1. 2025-10-20-13-42-48.38_..._Fly_5_Trial_1_.csv
    2. 2025-1012-1312-43.06_..._Fly1_Trial1_.csv
    """
    pattern = re.compile(
        r"(?P<date>\d{4}-(?:\d{2}-\d{2}|\d{4}))-"
        r"(?P<time>\d{2}-\d{2}-\d{2}|\d{4}(?:-\d{2})?)"
        r"(?:\.\d+)?"
        r".*?"
        r"_Fly_?(?P<fly>\d+)"
        r"_Trial_?(?P<trial>\d+)_"
    )

    unique_combinations = {}
    grouped_files = {}
    current_fly_number = 1

    for file_path in file_names:
        file_name = os.path.basename(file_path)
        match = pattern.search(file_name)

        if not match:
            print(f"Skipped (pattern not matched): {file_name}")
            continue

        date = match.group("date")
        fly_number = match.group("fly")
        trial_number = match.group("trial")
        unique_fly_id = (date, fly_number)

        if unique_fly_id not in unique_combinations:
            unique_combinations[unique_fly_id] = current_fly_number
            current_fly_number += 1

        grouped_files[f"F{unique_combinations[unique_fly_id]}T{trial_number}"] = file_path

    return grouped_files


def parse_new_names(file_names):
    """
    Parser for reorganized kinematic CSV names generated from the video
    manifest naming convention.

    Supported examples:
    - F001_T001_condNone_fps200_20250130_153141_.csv
    - F001_T001_condOFF_fps250_20250830_105626_.csv

    The metadata fly and trial IDs are taken directly from F### and T###.
    """
    pattern = re.compile(
        r"^F(?P<fly>\d+)_T(?P<trial>\d+)"
        r"(?:_cond(?P<condition>[^_]+))?"
        r"(?:_fps(?P<fps>[^_]+))?"
        r"_(?P<date>\d{8})_(?P<time>\d{6})_?\.csv$",
        re.IGNORECASE
    )

    grouped_files = {}
    skipped_files = []

    for file_path in file_names:
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if not match:
            skipped_files.append(file_name)
            continue

        fly_number = int(match.group("fly"))
        trial_number = int(match.group("trial"))
        grouped_files[f"F{fly_number}T{trial_number}"] = file_path

    return grouped_files


def Get3D_path(source_folder, required=True):
    """
    Read all csv files in a folder and group them into keys like F1T1, F1T2...
    """
    if source_folder in (None, "", "NoPath"):
        if not required:
            return {}
        raise FileNotFoundError("Kinematic data folder was not provided.")
    if not os.path.isdir(source_folder):
        if not required:
            return {}
        raise FileNotFoundError(f"Kinematic data folder does not exist: {source_folder}")

    all_files = []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    all_files = natsorted(all_files)
    grouped_data_path = parse_new_names(all_files)
    if len(grouped_data_path) == 0:
        grouped_data_path = group_files_by_fly_new(all_files)

    if len(all_files) == 0:
        if not required:
            return {}
        raise FileNotFoundError(f"No kinematic CSV files found under: {source_folder}")
    if len(grouped_data_path) == 0:
        if not required:
            return {}
        raise ValueError(
            f"Kinematic CSV files were found, but none matched the expected Fly/Trial filename pattern: {source_folder}"
        )

    return grouped_data_path


def validate_kinematic_metadata_file_mapping(group_info, save_csv_path=None):
    """
    Check whether kinematic CSV paths align with metadata fly/trial indexes.

    Validation rules:
    - Metadata fly number must match the fly number encoded in the parent fly
      folder, e.g. ANxGTACR-Max-F11.
    - Metadata trial number must match the Trial_# encoded in the CSV filename.
    - Filename Fly_# is recorded as within-session fly number and is not treated
      as an error.

    Returns a DataFrame with one row per kinematic CSV mapping, plus one row for
    each metadata trial that lacks a matching kinematic CSV.
    """
    if len(group_info.trial_metadata) == 0:
        if "CHR" in group_info.group_name or "Chr" in group_info.group_name:
            group_info.initialize_Chr_manual_data()
        else:
            group_info.initialize_manual_data()

    # Use already grouped paths when available. This avoids reading CSV
    # contents and avoids re-parsing a dict through Get3D_path.
    source_paths = getattr(group_info, "fly_kinematic_data_path", None)
    if isinstance(source_paths, dict):
        grouped_paths = source_paths
    else:
        source_folder = getattr(group_info, "kinematic_data_path", source_paths)
        grouped_paths = Get3D_path(source_folder, required=False)
    metadata_keys = set(group_info.trial_metadata.keys())
    mapped_metadata_keys = set()
    rows = []

    key_pattern = re.compile(r"^F(?P<fly>\d+)T(?P<trial>\d+)$")
    folder_fly_pattern = re.compile(r"(?:^|[-_])F(?P<fly>\d+)(?:$|[-_])", re.IGNORECASE)
    filename_trial_pattern = re.compile(
        r"_Fly_?(?P<filename_fly>\d+)_Trial_?(?P<trial>\d+)_",
        re.IGNORECASE
    )

    for key, csv_path in grouped_paths.items():
        key_match = key_pattern.match(key)
        metadata_fly = int(key_match.group("fly")) if key_match else np.nan
        metadata_trial = int(key_match.group("trial")) if key_match else np.nan

        path_parts = os.path.normpath(csv_path).split(os.sep)
        fly_folder = ""
        folder_fly = np.nan
        for part in reversed(path_parts[:-1]):
            folder_match = folder_fly_pattern.search(part)
            if folder_match:
                fly_folder = part
                folder_fly = int(folder_match.group("fly"))
                break

        filename = os.path.basename(csv_path)
        filename_match = filename_trial_pattern.search(filename)
        filename_fly = int(filename_match.group("filename_fly")) if filename_match else np.nan
        filename_trial = int(filename_match.group("trial")) if filename_match else np.nan

        fly_matches_folder = (
            not pd.isna(metadata_fly)
            and not pd.isna(folder_fly)
            and int(metadata_fly) == int(folder_fly)
        )
        trial_matches_filename = (
            not pd.isna(metadata_trial)
            and not pd.isna(filename_trial)
            and int(metadata_trial) == int(filename_trial)
        )

        statuses = []
        if key not in metadata_keys:
            statuses.append("CSV_WITHOUT_METADATA")
        else:
            mapped_metadata_keys.add(key)

        if pd.isna(folder_fly):
            statuses.append("MISSING_FOLDER_FLY")
        elif not fly_matches_folder:
            statuses.append("FLY_FOLDER_MISMATCH")

        if pd.isna(filename_trial):
            statuses.append("MISSING_FILENAME_TRIAL")
        elif not trial_matches_filename:
            statuses.append("TRIAL_FILENAME_MISMATCH")

        if not statuses:
            statuses.append("OK")

        rows.append({
            "Group_Name": group_info.group_name,
            "Key": key,
            "Metadata_Fly": metadata_fly,
            "Metadata_Trial": metadata_trial,
            "Fly_Folder": fly_folder,
            "Folder_Fly": folder_fly,
            "Filename_Fly_Within_Session": filename_fly,
            "Filename_Trial": filename_trial,
            "Metadata_Fly_Matches_Folder": fly_matches_folder,
            "Metadata_Trial_Matches_Filename": trial_matches_filename,
            "Status": ";".join(statuses),
            "CSV_Path": csv_path,
        })

    missing_csv_keys = sorted(metadata_keys - mapped_metadata_keys)
    for key in missing_csv_keys:
        key_match = key_pattern.match(key)
        rows.append({
            "Group_Name": group_info.group_name,
            "Key": key,
            "Metadata_Fly": int(key_match.group("fly")) if key_match else np.nan,
            "Metadata_Trial": int(key_match.group("trial")) if key_match else np.nan,
            "Fly_Folder": "",
            "Folder_Fly": np.nan,
            "Filename_Fly_Within_Session": np.nan,
            "Filename_Trial": np.nan,
            "Metadata_Fly_Matches_Folder": False,
            "Metadata_Trial_Matches_Filename": False,
            "Status": "MISSING_CSV_FOR_METADATA",
            "CSV_Path": "",
        })

    result_df = pd.DataFrame(rows)
    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False)
    return result_df


# ------------------------------------------------------------
# Data objects
# ------------------------------------------------------------

class Point:
    def __init__(self, name="point", x=None, y=None, z=None, cam_count=None, error=None):
        self.name = name
        self.x_coord = x
        self.y_coord = y
        self.z_coord = z
        self.camera_count = cam_count
        self.error = error


class Trial:
    """
    Trial object only stores information for actual kinematic analysis.
    This means metadata-only analysis can be done first without constructing Trial.
    """

    def __init__(self, fly_number=0, trial_number=0, fps=0, total_frames_number=0,
                 landing_latency=0, moc=0, mol=0, group_name="NoName",
                 trial_data_path="NoPath", joints=None):

        self.fly_number = fly_number
        self.trial_number = trial_number
        self.fps = fps
        self.total_frames_number = total_frames_number
        self.group_name = group_name

        # Manual labels / timing
        self.LL = landing_latency
        self.moc = moc
        self.mol = mol

        # Optional fields used in some analyses
        self.L_stable_FT_angle = np.nan
        self.R_stable_FT_angle = np.nan

        self.joints = joints
        self.data_path = trial_data_path
        self.trial_data = self.read_trial_data(trial_data_path)

    def read_trial_data(self, data_path):
        kine_data = pd.read_csv(data_path)
        data = dict()

        for j in self.joints:
            data[j] = Point(
                name=j,
                x=kine_data[f"{j}_x"],
                y=kine_data[f"{j}_y"],
                z=kine_data[f"{j}_z"],
                cam_count=kine_data[f"{j}_ncams"],
                error=kine_data[f"{j}_error"]
            )

        return data

    def get_point(self, point):
        return np.asarray([self.trial_data[point].x_coord,
                           self.trial_data[point].y_coord,
                           self.trial_data[point].z_coord])
class Group:
    """
    Group object now has two layers:

    1. trial_metadata:
       metadata-only information used for LP, LL, good fly filtering,
       opto grouping, etc.

    2. fly_kinematic_data:
       actual Trial objects, only loaded when kinematic analysis is needed.
    """

    def __init__(self, moc_data_path="NoPath", mol_data_path="NoPath", ll_data_path="NoPath",
                 fly_kinematic_data_path="NoPath", group_name="NoName", joints=None,
                 segments=None, total_fly_number=0, fps=None,
                 trial_num=20, video_duration=7, trials_offset=0,
                 latency_threshold=0.71):

        # Basic paths
        self.fly_kinematic_data_path = Get3D_path(fly_kinematic_data_path, required=False)
        self.moc_data_path = moc_data_path
        self.mol_data_path = mol_data_path
        self.ll_data_path = ll_data_path

        # Meta info
        self.group_name = group_name
        self.joints = joints
        self.segment = segments
        self.total_fly_number = total_fly_number
        self.fps = fps
        self.video_duration = video_duration
        self.latency_threshold = latency_threshold

        # Trial control
        self.trial_num = trial_num
        self.trial_offset = trials_offset
        self.trials_index = [f"Trial_{i + 1}" for i in range(trial_num)]

        # Manual data
        self.ll_data = self.read_manual_data(self.ll_data_path)
        self.mol_data = self.read_manual_data(self.mol_data_path)
        self.moc_data = self.read_manual_data(self.moc_data_path)

        # Trial classification
        self.landing_trial_index = []
        self.flying_trial_index = []
        self.not_flying_trial_index = []
        self.NA_trial_index = []
        self.good_fly_index = list(range(1, self.total_fly_number + 1))
        self.ON_index = []
        self.OFF_index = []

        # Storage
        self.trial_metadata = dict()
        self.fly_kinematic_data = dict()
        self.predicted_data = dict()

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------

    def _trial_key(self, fly, trial):
        return f"F{fly}T{trial}"

    def _trial_exists(self, fly, trial):
        return self._trial_key(fly, trial) in self.fly_kinematic_data_path

    def _safe_int(self, val):
        """
        Convert numeric values to int, but keep NaN / strings as-is.
        """
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            return val
        return int(val)

    def _get_opto_label_from_path(self, path):
        if "_LO_" in path or "ON" in path:
            return "ON"
        if "_NL_" in path or "OFF" in path:
            return "OFF"
        return None

    def _classify_trial(self, val, max_latency):
        """
        Keep your current logic:
        - str -> NF
        - nan -> NA
        - 0 to threshold -> Landing
        - -1 or beyond threshold -> Flying
        """
        if isinstance(val, str):
            return "NF"
        if pd.isna(val):
            return "NA"
        if 0 <= val <= max_latency:
            return "Landing"
        if val == -1 or val > max_latency:
            return "Flying"
        return "Unknown"

    # ------------------------------------------------------------
    # Manual data reading
    # ------------------------------------------------------------

    def read_manual_data(self, data_path):
        if data_path != "NoPath":
            return pd.read_excel(data_path)[self.trials_index].iloc[:self.total_fly_number]
        else:
            return None

    # ------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------

    def initialize_manual_data(self, require_kinematics=False):
        """
        Standard LL/MOC/MOL initialization.

        By default this builds metadata from manual sheets even when matching
        kinematic CSVs are absent. Set require_kinematics=True for workflows
        that need every manual trial to have a CSV path.
        """
        if self.ll_data is None:
            raise ValueError(f"LL data is required to initialize group {self.group_name}.")
        if self.fps is None or len(self.fps) < self.total_fly_number:
            raise ValueError(
                f"FPS list for {self.group_name} must contain at least {self.total_fly_number} values."
            )

        self.landing_trial_index = []
        self.flying_trial_index = []
        self.not_flying_trial_index = []
        self.NA_trial_index = []
        self.trial_metadata = dict()
        missing_trials = []

        for i in range(self.total_fly_number):
            max_latency = self.fps[i] * self.latency_threshold

            for t in range(self.trial_num):
                fly = i + 1
                trial = t + 1
                key = self._trial_key(fly, trial)

                path = None
                light = None
                if len(self.fly_kinematic_data_path) > 0 and key in self.fly_kinematic_data_path:
                    path = self.fly_kinematic_data_path[key]
                    light = self._get_opto_label_from_path(path)
                else:
                    missing_trials.append(key)

                ll_val = self.ll_data.iloc[i, t] if self.ll_data is not None else np.nan
                moc_val = self.moc_data.iloc[i, t] if self.moc_data is not None else np.nan
                mol_val = self.mol_data.iloc[i, t] if self.mol_data is not None else np.nan
                trial_type = self._classify_trial(ll_val, max_latency)
                if trial_type == "Unknown":
                    raise ValueError(
                        f"Cannot classify LL value for {self.group_name} {key}: {ll_val}"
                    )

                self.trial_metadata[key] = {
                    "Fly#": fly,
                    "Trial#": trial,
                    "LL": ll_val,
                    "MOC": moc_val,
                    "MOL": mol_val,
                    "fps": self.fps[i],
                    "TrialType": trial_type,
                    "Path": path,
                    "Light": light
                }

                idx = (fly, trial)
                if trial_type == "Landing":
                    self.landing_trial_index.append(idx)
                elif trial_type == "Flying":
                    self.flying_trial_index.append(idx)
                elif trial_type == "NF":
                    self.not_flying_trial_index.append(idx)
                elif trial_type == "NA":
                    self.NA_trial_index.append(idx)

        if require_kinematics and missing_trials:
            preview = ", ".join(missing_trials[:10])
            more = "" if len(missing_trials) <= 10 else f" ... and {len(missing_trials) - 10} more"
            raise FileNotFoundError(
                f"Missing kinematic CSVs for group {self.group_name}: {preview}{more}"
            )

    def initialize_Chr_manual_data(self, require_kinematics=False):
        """
        Special initialization for Chr data where LL is often coded differently.
        Metadata can be initialized without kinematic CSVs unless
        require_kinematics=True.
        """
        if self.ll_data is None:
            raise ValueError(f"LL data is required to initialize Chr group {self.group_name}.")
        if self.fps is None or len(self.fps) < self.total_fly_number:
            raise ValueError(
                f"FPS list for {self.group_name} must contain at least {self.total_fly_number} values."
            )

        self.landing_trial_index = []
        self.flying_trial_index = []
        self.not_flying_trial_index = []
        self.NA_trial_index = []
        self.trial_metadata = dict()
        missing_trials = []

        for i in range(self.total_fly_number):
            for t in range(self.trial_num):
                fly = i + 1
                trial = t + 1
                key = self._trial_key(fly, trial)

                path = None
                light = None
                if key in self.fly_kinematic_data_path:
                    path = self.fly_kinematic_data_path[key]
                    light = self._get_opto_label_from_path(path)
                else:
                    missing_trials.append(key)

                ll_val = self.ll_data.iloc[i, t]

                if ll_val == 1:
                    trial_type = "Landing"
                    mol_val = 1
                elif isinstance(ll_val, str) and ll_val == "NF":
                    trial_type = "NF"
                    mol_val = 100
                elif ll_val == -1:
                    trial_type = "Flying"
                    mol_val = -1
                elif pd.isna(ll_val):
                    trial_type = "NA"
                    mol_val = 100
                else:
                    trial_type = "Unknown"
                    mol_val = np.nan

                self.trial_metadata[key] = {
                    "Fly#": fly,
                    "Trial#": trial,
                    "LL": ll_val,
                    "MOC": np.nan,
                    "MOL": mol_val,
                    "fps": self.fps[i],
                    "TrialType": trial_type,
                    "Path": path,
                    "Light": light
                }

                idx = (fly, trial)
                if trial_type == "Landing":
                    self.landing_trial_index.append(idx)
                elif trial_type == "Flying":
                    self.flying_trial_index.append(idx)
                elif trial_type == "NF":
                    self.not_flying_trial_index.append(idx)
                elif trial_type == "NA":
                    self.NA_trial_index.append(idx)

        if require_kinematics and missing_trials:
            preview = ", ".join(missing_trials[:10])
            more = "" if len(missing_trials) <= 10 else f" ... and {len(missing_trials) - 10} more"
            raise FileNotFoundError(
                f"Missing kinematic CSVs for Chr group {self.group_name}: {preview}{more}"
            )

    # ------------------------------------------------------------
    # Kinematic loading
    # ------------------------------------------------------------

    def read_kinematic_data(self, trial_types=None):
        """
        Create Trial objects only for the trial types you actually need.
        """
        if len(self.trial_metadata) == 0:
            raise ValueError(
                "No trial metadata has been initialized. Call initialize_manual_data() "
                "or initialize_Chr_manual_data() before read_kinematic_data()."
            )

        if trial_types is None:
            trial_types = ["Landing", "Flying", "NF", "NA"]

        for key, meta in self.trial_metadata.items():
            if meta["TrialType"] not in trial_types:
                continue

            if key in self.fly_kinematic_data:
                continue

            fly = meta["Fly#"]
            trial = meta["Trial#"]
            trial_path = meta.get("Path")
            if trial_path in (None, "", "NoPath"):
                raise FileNotFoundError(
                    f"No kinematic CSV path is available for {self.group_name} {key}. "
                    "Metadata-only initialization is allowed, but kinematic analysis "
                    "requires matching CSV files."
                )

            self.fly_kinematic_data[key] = Trial(
                fly_number=fly,
                trial_number=trial,
                fps=meta["fps"],
                total_frames_number=meta["fps"] * self.video_duration,
                landing_latency=self._safe_int(meta["LL"]),
                moc=self._safe_int(meta["MOC"]),
                mol=self._safe_int(meta["MOL"]),
                group_name=self.group_name + "-" + meta["TrialType"],
                trial_data_path=trial_path,
                joints=self.joints
            )

    # ------------------------------------------------------------
    # Trial selection
    # ------------------------------------------------------------

    def get_targeted_trials(self, trial_types):
        merge_trial = []

        if "NA" in trial_types:
            merge_trial = sorted(merge_trial + [index for index in self.NA_trial_index if index[1] > self.trial_offset])
        if "NF" in trial_types:
            merge_trial = sorted(merge_trial + [index for index in self.not_flying_trial_index if index[1] > self.trial_offset])
        if "Landing" in trial_types:
            merge_trial = sorted(merge_trial + [index for index in self.landing_trial_index if index[1] > self.trial_offset])
        if "Flying" in trial_types:
            merge_trial = sorted(merge_trial + [index for index in self.flying_trial_index if index[1] > self.trial_offset])

        merge_trial = [trial for trial in merge_trial if trial[0] in self.good_fly_index]
        return merge_trial

    # ------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------

    def filter_nan_fly(self):
        """
        Keep flies that have at least half of the total trials as Landing/Flying.
        This now uses metadata only.
        """
        good_data_threshold = self.trial_num / 2
        self.good_fly_index = []

        for f in range(self.total_fly_number):
            fly_num = f + 1
            good_data_num = 0

            for key, meta in self.trial_metadata.items():
                if meta["Fly#"] == fly_num and meta["TrialType"] in ["Landing", "Flying"]:
                    good_data_num += 1

            if good_data_num >= good_data_threshold:
                self.good_fly_index.append(fly_num)

    def filter_opto_data(self, min_trial_num=8):
        """
        Keep flies that have enough valid ON and OFF trials.
        This now uses metadata only.
        """
        self.ON_index = []
        self.OFF_index = []

        for key, meta in self.trial_metadata.items():
            idx = (meta["Fly#"], meta["Trial#"])

            if meta["Light"] == "ON":
                self.ON_index.append(idx)
            if meta["Light"] == "OFF":
                self.OFF_index.append(idx)

        self.good_fly_index = []

        for f in range(self.total_fly_number):
            fly_num = f + 1
            good_on_num = 0
            good_off_num = 0

            for key, meta in self.trial_metadata.items():
                if meta["Fly#"] != fly_num:
                    continue
                if meta["TrialType"] not in ["Landing", "Flying"]:
                    continue

                if meta["Light"] == "ON":
                    good_on_num += 1
                if meta["Light"] == "OFF":
                    good_off_num += 1

            if good_on_num >= min_trial_num and good_off_num >= min_trial_num:
                self.good_fly_index.append(fly_num)

    # ------------------------------------------------------------
    # Summary data
    # ------------------------------------------------------------

    def get_LP(self):
        """
        Return landing probability per fly.
        """
        LP = []

        for f in range(self.total_fly_number):
            fly_num = f + 1

            if fly_num not in self.good_fly_index:
                continue

            total = 0
            land = 0

            for ind in self.get_targeted_trials(["Landing", "Flying"]):
                if ind[0] != fly_num:
                    continue

                key = self._trial_key(ind[0], ind[1])
                meta = self.trial_metadata[key]

                total += 1

                if meta["TrialType"] == "Landing":
                    if (meta["LL"] / meta["fps"]) <= self.latency_threshold:
                        land += 1

            if total > 0:
                LP.append(land / total)

        return LP

    def get_LP_df(self, group_name=None):
        """
        Return LP as DataFrame.
        This is convenient for seaborn plotting.
        """
        if group_name is None:
            group_name = self.group_name

        out = dict()
        out["Fly#"] = []
        out["LandingProb"] = []
        out["Group_Name"] = []

        for f in range(self.total_fly_number):
            fly_num = f + 1

            if fly_num not in self.good_fly_index:
                continue

            total = 0
            land = 0

            for ind in self.get_targeted_trials(["Landing", "Flying"]):
                if ind[0] != fly_num:
                    continue

                key = self._trial_key(ind[0], ind[1])
                meta = self.trial_metadata[key]
                total += 1

                if meta["TrialType"] == "Landing":
                    if (meta["LL"] / meta["fps"]) <= self.latency_threshold:
                        land += 1

            if total > 0:
                out["Fly#"].append(fly_num)
                out["LandingProb"].append(land / total)
                out["Group_Name"].append(group_name)

        return pd.DataFrame(out)

    def get_paired_LP_df(self):
        """
        Return LP dataframe for ON/OFF paired plot.
        """
        ON_rows = []
        OFF_rows = []

        for f in range(self.total_fly_number):
            fly_num = f + 1

            if fly_num not in self.good_fly_index:
                continue

            on_land = 0
            on_total = 0
            off_land = 0
            off_total = 0

            for ind in self.get_targeted_trials(["Landing", "Flying"]):
                if ind[0] != fly_num:
                    continue

                key = self._trial_key(ind[0], ind[1])
                meta = self.trial_metadata[key]

                if meta["Light"] == "ON":
                    on_total += 1
                    if meta["TrialType"] == "Landing" and (meta["LL"] / meta["fps"]) <= self.latency_threshold:
                        on_land += 1

                if meta["Light"] == "OFF":
                    off_total += 1
                    if meta["TrialType"] == "Landing" and (meta["LL"] / meta["fps"]) <= self.latency_threshold:
                        off_land += 1

            if on_total > 0:
                ON_rows.append((fly_num, on_land / on_total, "ON"))
            if off_total > 0:
                OFF_rows.append((fly_num, off_land / off_total, "OFF"))

        on_df = pd.DataFrame(ON_rows, columns=["Fly#", "LandingProb", "Group_Name"])
        off_df = pd.DataFrame(OFF_rows, columns=["Fly#", "LandingProb", "Group_Name"])
        combined_df = pd.concat([on_df, off_df], ignore_index=True)

        return combined_df

    def get_LL(self, return_df=False):
        """
        Return latency data in KMC-ready format.

        For each Landing/Flying trial:
        - Landing within threshold -> Event = 1, Latency = actual latency
        - Flying / censored -> Event = 0, Latency = threshold
        """
        durations = []
        events = []
        rows = []
        MLL = []

        for f in range(self.total_fly_number):
            fly_num = f + 1

            if fly_num not in self.good_fly_index:
                continue

            fly_land = []

            for ind in self.get_targeted_trials(["Landing", "Flying"]):
                if ind[0] != fly_num:
                    continue

                key = self._trial_key(ind[0], ind[1])
                meta = self.trial_metadata[key]
                ll = meta["LL"]
                fps = meta["fps"]
                trial_type = meta["TrialType"]

                if trial_type == "Landing" and not pd.isna(ll) and (ll / fps) <= self.latency_threshold:
                    latency = ll / fps
                    event = 1
                    fly_land.append(latency)
                else:
                    latency = self.latency_threshold
                    event = 0

                durations.append(latency)
                events.append(event)

                rows.append({
                    "Fly#": meta["Fly#"],
                    "Trial#": meta["Trial#"],
                    "Latency": latency,
                    "Event": event,
                    "Group_Name": self.group_name,
                    "TrialType": trial_type,
                    "Light": meta["Light"]
                })

            if len(fly_land) > 0:
                MLL.append((fly_num, np.mean(fly_land)))

        if return_df:
            return pd.DataFrame(rows)

        return MLL, durations, events

    def get_ON_OFF_index(self):
        ON = []
        OFF = []

        for ind in self.get_targeted_trials(["Landing", "Flying"]):
            key = self._trial_key(ind[0], ind[1])
            meta = self.trial_metadata[key]

            if meta["Light"] == "ON":
                ON.append(ind)
            if meta["Light"] == "OFF":
                OFF.append(ind)

        return ON, OFF

    def get_SF_index(self):
        """
        Return success / failed trial indices.
        """
        Success = []
        Failed = []

        for ind in self.get_targeted_trials(["Landing", "Flying"]):
            key = self._trial_key(ind[0], ind[1])
            meta = self.trial_metadata[key]
            ll = meta["LL"]
            fps = meta["fps"]

            if ll == -1:
                Failed.append(ind)
            elif (ll / fps) > self.latency_threshold:
                Failed.append(ind)
            else:
                Success.append(ind)

        return Success, Failed
