import os
import re
import warnings
import numpy as np
import pandas as pd
from natsort import natsorted

warnings.filterwarnings(action="ignore", category=FutureWarning)


class KinematicObjectValidationError(ValueError):
    """Raised when group configuration or source data are inconsistent."""


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

    grouped_files = {}

    for file_path in file_names:
        file_name = os.path.basename(file_path)
        match = pattern.search(file_name)

        if not match:
            raise KinematicObjectValidationError(
                f"Kinematic CSV filename does not match expected Fly/Trial pattern: {file_name}"
            )

        date = match.group("date")
        fly_number = match.group("fly")
        trial_number = match.group("trial")
        key = f"F{int(fly_number)}T{int(trial_number)}"

        if key in grouped_files:
            raise KinematicObjectValidationError(
                "Duplicate fly/trial key while grouping kinematic files. "
                f"Key {key} appears for multiple files, including {grouped_files[key]} and {file_path}. "
                f"The parser preserves source Fly IDs; rename files or split dates before analysis. Date={date}"
            )

        grouped_files[key] = file_path

    return grouped_files


def Get3D_path(source_folder):
    """
    Read all csv files in a folder and group them into keys like F1T1, F1T2...
    """
    if source_folder in (None, "", "NoPath"):
        raise KinematicObjectValidationError(
            "fly_kinematic_data_path must point to a real folder; got no path."
        )
    if not os.path.isdir(source_folder):
        raise KinematicObjectValidationError(
            f"fly_kinematic_data_path does not exist or is not a folder: {source_folder}"
        )

    all_files = []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    all_files = natsorted(all_files)
    if len(all_files) == 0:
        raise KinematicObjectValidationError(
            f"No .csv kinematic files found under: {source_folder}"
        )

    grouped_data_path = group_files_by_fly_new(all_files)
    if len(grouped_data_path) == 0:
        raise KinematicObjectValidationError(
            f"CSV files were found, but none matched the expected Fly/Trial filename pattern in: {source_folder}"
        )
    return grouped_data_path


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
        if data_path in (None, "", "NoPath"):
            raise KinematicObjectValidationError(
                f"Trial data path is invalid for F{self.fly_number}T{self.trial_number}: {data_path}"
            )
        if not os.path.isfile(data_path):
            raise KinematicObjectValidationError(
                f"Trial data file does not exist for F{self.fly_number}T{self.trial_number}: {data_path}"
            )
        if self.joints is None:
            raise KinematicObjectValidationError(
                f"joints must be provided before reading trial data for F{self.fly_number}T{self.trial_number}."
            )

        kine_data = pd.read_csv(data_path)
        data = dict()

        for j in self.joints:
            missing_columns = [
                col for col in (f"{j}_x", f"{j}_y", f"{j}_z", f"{j}_ncams", f"{j}_error")
                if col not in kine_data.columns
            ]
            if missing_columns:
                raise KinematicObjectValidationError(
                    f"Trial CSV is missing columns for joint {j}: {missing_columns}. File: {data_path}"
                )
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

        if fps is None:
            raise KinematicObjectValidationError("fps must be provided as one value per fly.")
        if len(fps) != total_fly_number:
            raise KinematicObjectValidationError(
                f"fps length ({len(fps)}) must equal total_fly_number ({total_fly_number}) "
                f"for group {group_name}."
            )
        if total_fly_number <= 0:
            raise KinematicObjectValidationError(
                f"total_fly_number must be positive for group {group_name}; got {total_fly_number}."
            )
        if trial_num <= 0:
            raise KinematicObjectValidationError(
                f"trial_num must be positive for group {group_name}; got {trial_num}."
            )

        # Basic paths
        self.fly_kinematic_data_path = Get3D_path(fly_kinematic_data_path)
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
        if path in (None, ""):
            raise KinematicObjectValidationError(
                f"Cannot infer optogenetic label from empty path in group {self.group_name}."
            )

        tokens = re.split(r"[^A-Za-z0-9]+", os.path.basename(path))
        parent_tokens = re.split(r"[^A-Za-z0-9]+", os.path.dirname(path))
        all_tokens = {token.upper() for token in tokens + parent_tokens if token}

        if "LO" in all_tokens or "ON" in all_tokens:
            return "ON"
        if "NL" in all_tokens or "OFF" in all_tokens:
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
        if data_path == "NoPath":
            return None

        if data_path in (None, ""):
            raise KinematicObjectValidationError(
                f"Manual data path for group {self.group_name} is empty."
            )
        if not os.path.isfile(data_path):
            raise KinematicObjectValidationError(
                f"Manual data file does not exist for group {self.group_name}: {data_path}"
            )

        manual_df = pd.read_excel(data_path)
        missing_columns = [col for col in self.trials_index if col not in manual_df.columns]
        if missing_columns:
            raise KinematicObjectValidationError(
                f"Manual data file for group {self.group_name} is missing trial columns: "
                f"{missing_columns}. Expected columns like Trial_1 ... Trial_{self.trial_num}."
            )
        if len(manual_df) < self.total_fly_number:
            raise KinematicObjectValidationError(
                f"Manual data file for group {self.group_name} has {len(manual_df)} rows, "
                f"but total_fly_number is {self.total_fly_number}: {data_path}"
            )

        return manual_df[self.trials_index].iloc[:self.total_fly_number]

    # ------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------

    def initialize_manual_data(self):
        """
        Standard LL/MOC/MOL initialization.

        This strict variant raises on missing trials rather than silently
        truncating a fly.
        """
        if self.ll_data is None:
            raise KinematicObjectValidationError(
                f"ll_data_path is required for standard metadata initialization in group {self.group_name}."
            )

        self.landing_trial_index = []
        self.flying_trial_index = []
        self.not_flying_trial_index = []
        self.NA_trial_index = []
        self.trial_metadata = dict()

        for i in range(self.total_fly_number):
            max_latency = self.fps[i] * self.latency_threshold

            for t in range(self.trial_num):
                fly = i + 1
                trial = t + 1
                key = self._trial_key(fly, trial)

                if len(self.fly_kinematic_data_path) > 0:
                    if key not in self.fly_kinematic_data_path:
                        raise KinematicObjectValidationError(
                            f"Missing kinematic CSV for group {self.group_name}, trial key {key}. "
                            "Strict initialization does not silently drop later trials."
                        )
                    else:
                        path = self.fly_kinematic_data_path[key]
                        light = self._get_opto_label_from_path(path)
                else:
                    path = None
                    light = None


                ll_val = self.ll_data.iloc[i, t] if self.ll_data is not None else np.nan
                moc_val = self.moc_data.iloc[i, t] if self.moc_data is not None else np.nan
                mol_val = self.mol_data.iloc[i, t] if self.mol_data is not None else np.nan
                trial_type = self._classify_trial(ll_val, max_latency)
                if trial_type == "Unknown":
                    raise KinematicObjectValidationError(
                        f"Cannot classify LL value for group {self.group_name}, {key}: {ll_val}"
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
                    "Light": light,
                    "LL_Is_Binary": False
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

    def initialize_Chr_manual_data(self):
        """
        Special initialization for Chr data where LL is often coded differently.
        """
        if self.ll_data is None:
            raise KinematicObjectValidationError(
                f"ll_data_path is required for Chr metadata initialization in group {self.group_name}."
            )

        self.landing_trial_index = []
        self.flying_trial_index = []
        self.not_flying_trial_index = []
        self.NA_trial_index = []
        self.trial_metadata = dict()

        for i in range(self.total_fly_number):
            for t in range(self.trial_num):
                fly = i + 1
                trial = t + 1
                key = self._trial_key(fly, trial)

                if key not in self.fly_kinematic_data_path:
                    raise KinematicObjectValidationError(
                        f"Missing kinematic CSV for Chr group {self.group_name}, trial key {key}. "
                        "Strict initialization does not silently drop later trials."
                    )

                ll_val = self.ll_data.iloc[i, t]
                path = self.fly_kinematic_data_path[key]
                light = self._get_opto_label_from_path(path)

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
                    raise KinematicObjectValidationError(
                        f"Cannot classify Chr LL value for group {self.group_name}, {key}: {ll_val}"
                    )

                self.trial_metadata[key] = {
                    "Fly#": fly,
                    "Trial#": trial,
                    "LL": ll_val,
                    "MOC": np.nan,
                    "MOL": mol_val,
                    "fps": self.fps[i],
                    "TrialType": trial_type,
                    "Path": path,
                    "Light": light,
                    "LL_Is_Binary": True
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

    # ------------------------------------------------------------
    # Kinematic loading
    # ------------------------------------------------------------

    def read_kinematic_data(self, trial_types=None):
        """
        Create Trial objects only for the trial types you actually need.
        """
        if len(self.trial_metadata) == 0:
            raise KinematicObjectValidationError(
                f"No trial metadata initialized for group {self.group_name}. "
                "Call initialize_manual_data() or initialize_Chr_manual_data() before read_kinematic_data()."
            )

        if trial_types is None:
            trial_types = ["Landing", "Flying", "NF", "NA"]

        for key, meta in self.trial_metadata.items():
            if meta["TrialType"] not in trial_types:
                continue

            if key in self.fly_kinematic_data:
                continue

            if meta["Path"] in (None, "", "NoPath"):
                raise KinematicObjectValidationError(
                    f"Cannot load kinematic data for group {self.group_name}, {key}: invalid path {meta['Path']}."
                )

            fly = meta["Fly#"]
            trial = meta["Trial#"]

            self.fly_kinematic_data[key] = Trial(
                fly_number=fly,
                trial_number=trial,
                fps=meta["fps"],
                total_frames_number=meta["fps"] * self.video_duration,
                landing_latency=self._safe_int(meta["LL"]),
                moc=self._safe_int(meta["MOC"]),
                mol=self._safe_int(meta["MOL"]),
                group_name=self.group_name + "-" + meta["TrialType"],
                trial_data_path=meta["Path"],
                joints=self.joints
            )

    # ------------------------------------------------------------
    # Trial selection
    # ------------------------------------------------------------

    def get_targeted_trials(self, trial_types):
        if len(self.trial_metadata) == 0:
            raise KinematicObjectValidationError(
                f"No trial metadata initialized for group {self.group_name}; cannot select targeted trials."
            )

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
        if len(self.trial_metadata) == 0:
            raise KinematicObjectValidationError(
                f"No trial metadata initialized for group {self.group_name}; cannot filter flies."
            )

        usable_trial_num = self.trial_num - self.trial_offset
        if usable_trial_num <= 0:
            raise KinematicObjectValidationError(
                f"trial_offset ({self.trial_offset}) must be smaller than trial_num ({self.trial_num}) "
                f"for group {self.group_name}."
            )

        good_data_threshold = usable_trial_num / 2
        self.good_fly_index = []

        for f in range(self.total_fly_number):
            fly_num = f + 1
            good_data_num = 0

            for key, meta in self.trial_metadata.items():
                if (
                    meta["Fly#"] == fly_num
                    and meta["Trial#"] > self.trial_offset
                    and meta["TrialType"] in ["Landing", "Flying"]
                ):
                    good_data_num += 1

            if good_data_num >= good_data_threshold:
                self.good_fly_index.append(fly_num)

    def filter_opto_data(self, min_trial_num=8):
        """
        Keep flies that have enough valid ON and OFF trials.
        This now uses metadata only.
        """
        if len(self.trial_metadata) == 0:
            raise KinematicObjectValidationError(
                f"No trial metadata initialized for group {self.group_name}; cannot filter opto data."
            )

        self.ON_index = []
        self.OFF_index = []

        for key, meta in self.trial_metadata.items():
            idx = (meta["Fly#"], meta["Trial#"])
            if meta["Trial#"] <= self.trial_offset:
                continue

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
                if meta["Trial#"] <= self.trial_offset:
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

        if any(meta.get("LL_Is_Binary", False) for meta in self.trial_metadata.values()):
            raise KinematicObjectValidationError(
                f"Group {self.group_name} contains binary Chr LL labels. "
                "get_LL() requires frame-based latency values and would otherwise convert 1/FPS into a false latency."
            )

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
        if any(meta.get("LL_Is_Binary", False) for meta in self.trial_metadata.values()):
            raise KinematicObjectValidationError(
                f"Group {self.group_name} contains binary Chr LL labels. "
                "get_SF_index() requires frame-based latency values."
            )

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
