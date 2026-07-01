import os
import re
import shutil
from collections import OrderedDict


# ============================================================
# User-editable settings
# ============================================================

# Folder containing the videos for one group. The script scans recursively.
GROUP_DATA_FOLDER = r"D:\DataFolder\HCS+_UASKir2.1eGFP\G119-Club_T2-TiTa"
# False = preview only. True = create F### folders inside GROUP_DATA_FOLDER
# and move/rename videos into the corresponding fly folder.
APPLY_CHANGES = True

# Write a CSV manifest during dry run and before applying changes.
WRITE_MANIFEST = True
MANIFEST_FILE_NAME = "video_reorganization_manifest.csv"

# Group-level metadata. Keep filenames short. These labels can be used to build
# the proposed output folder if USE_METADATA_OUTPUT_PATH = True.
EXPERIMENT = "KIR"
GROUP_NAME = "G119-Club_T2-TiTa"
GENOTYPE = "Club neuron"
EFFECTOR = "KIR"
TARGET = "ML-R-ti-ta"
MANIPULATION = "None"
INTENSITY = "None"

# Proposed folder and filename settings. The proposed output structure is:
# If USE_METADATA_OUTPUT_PATH = False:
#   GROUP_DATA_FOLDER / F### / filename.mp4
# If USE_METADATA_OUTPUT_PATH = True:
#   dirname(dirname(GROUP_DATA_FOLDER)) / EXPERIMENT / GROUP_NAME / F### / filename.mp4
USE_METADATA_OUTPUT_PATH = False
FLY_DIGITS = 3
TRIAL_DIGITS = 3
INCLUDE_CONDITION_IN_FILENAME = True
INCLUDE_FPS_IN_FILENAME = True
SKIP_ALREADY_ORGANIZED_FOLDERS = True

# Set this to True for optogenetic folders where every video should contain a
# light condition token such as ON/OFF/LO/NL. Set False for WT/non-light data.
REQUIRE_CONDITION_IN_FILENAME = EXPERIMENT.upper() in {"OPTO", "OPTOGENETICS"}

# If True, non-light videos are named with condNone. If False, the condition
# token is omitted when no condition is present.
INCLUDE_NONE_CONDITION_TOKEN = True

# Move messy per-fly metadata files together with videos from the same original
# Fly_# folder. These files keep their original names by default.
INCLUDE_FLY_METADATA_FILES = True
FLY_METADATA_EXTENSIONS = {".csv"}
RENAME_FLY_METADATA_FILES = False

# Tokens in old filenames can be mapped to the condition label used in new names.
# For your optogenetic files, LO is treated as ON and NL as OFF by default.
CONDITION_MAP = {
    "ON": "ON",
    "OFF": "OFF",
    "LO": "ON",
    "NL": "OFF",
}


# ============================================================
# Parsing rules
# ============================================================

VIDEO_EXTENSIONS = {".mp4"}

DATE_TIME_PATTERNS = [
    # 2025-02-27-12-14-48.24
    re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-"
        r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})(?:\.\d+)?"
    ),
    # 2025-0830-1056-26.69
    re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{2})(?P<day>\d{2})-"
        r"(?P<hour>\d{2})(?P<minute>\d{2})-(?P<second>\d{2})(?:\.\d+)?"
    ),
    # 2025-1103-1411
    re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{2})(?P<day>\d{2})-"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?![-\d])"
    ),
]

FLY_PATTERN = re.compile(r"(?:^|_)Fly_?(?P<fly>\d+)(?:_|$)", re.IGNORECASE)
TRIAL_PATTERN = re.compile(r"(?:^|_)Trial_?(?P<trial>\d+)(?:_|$)", re.IGNORECASE)
CAM_PATTERN = re.compile(r"(?:^|_)Cam(?P<cam>\d+)(?:\.[^.]+)?$", re.IGNORECASE)
CONDITION_PATTERN = re.compile(r"(?:^|[-_])(?P<condition>ON|OFF|LO|NL)(?:[-_]|$)", re.IGNORECASE)
FOLDER_FLY_PATTERN = re.compile(r"^Fly_?(?P<fly>\d+)$", re.IGNORECASE)
DATE_FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ORGANIZED_FLY_FOLDER_PATTERN = re.compile(r"^F\d+$", re.IGNORECASE)


def natural_key(text):
    parts = re.split(r"(\d+)", str(text))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_date_time(file_name):
    for pattern in DATE_TIME_PATTERNS:
        match = pattern.search(file_name)
        if match:
            date = f"{match.group('year')}{match.group('month')}{match.group('day')}"
            second = match.groupdict().get("second", "00")
            time = f"{match.group('hour')}{match.group('minute')}{second}"
            sort_key = (date, time)
            return date, time, sort_key
    return "UnknownDate", "UnknownTime", ("99999999", "999999")


def find_first_int(pattern, text):
    match = pattern.search(text)
    if not match:
        return None
    group_name = next(iter(match.groupdict()))
    return int(match.group(group_name))


def find_condition(file_name):
    match = CONDITION_PATTERN.search(file_name)
    if not match:
        return "None"
    raw_condition = match.group("condition").upper()
    return CONDITION_MAP.get(raw_condition, raw_condition)


def read_video_fps(path):
    """
    Read FPS from video metadata only. This should be fast and does not decode
    frames. Returns None if OpenCV is unavailable or the metadata cannot be read.
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if fps is None or fps <= 0:
        return None
    return float(fps)


def find_folder_fly(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts[:-1]):
        match = FOLDER_FLY_PATTERN.match(part)
        if match:
            return int(match.group("fly"))
    return None


def find_folder_date(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts[:-1]):
        if DATE_FOLDER_PATTERN.match(part):
            return part.replace("-", "")
    return None


def iter_video_files(root):
    for current_root, _, files in os.walk(root):
        if SKIP_ALREADY_ORGANIZED_FOLDERS:
            folder_name = os.path.basename(os.path.normpath(current_root))
            if ORGANIZED_FLY_FOLDER_PATTERN.match(folder_name):
                continue
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                yield os.path.join(current_root, file_name)


def find_source_fly_folder(path):
    parts = os.path.normpath(path).split(os.sep)
    for idx in range(len(parts) - 2, -1, -1):
        if FOLDER_FLY_PATTERN.match(parts[idx]):
            return os.sep.join(parts[:idx + 1])
    return None


def iter_fly_metadata_files(records):
    if not INCLUDE_FLY_METADATA_FILES:
        return

    seen_fly_folders = set()
    for row in records:
        fly_folder = find_source_fly_folder(row["path"])
        if fly_folder is None or fly_folder in seen_fly_folders:
            continue
        seen_fly_folders.add(fly_folder)

        for file_name in os.listdir(fly_folder):
            path = os.path.join(fly_folder, file_name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(file_name)[1].lower()
            if ext in FLY_METADATA_EXTENSIONS:
                yield fly_folder, path


def parse_video_path(path):
    file_name = os.path.basename(path)
    date, time, sort_key = find_date_time(file_name)
    filename_fly = find_first_int(FLY_PATTERN, file_name)
    folder_fly = find_folder_fly(path)
    folder_date = find_folder_date(path)
    trial = find_first_int(TRIAL_PATTERN, file_name)
    cam = find_first_int(CAM_PATTERN, file_name)
    condition = find_condition(file_name)
    fps = read_video_fps(path)

    original_fly = folder_fly if folder_fly is not None else filename_fly

    return {
        "path": path,
        "file_name": file_name,
        "date": date,
        "folder_date": folder_date,
        "time": time,
        "sort_key": sort_key,
        "folder_fly": folder_fly,
        "filename_fly": filename_fly,
        "original_fly": original_fly,
        "trial": trial,
        "cam": cam,
        "condition": condition,
        "fps": fps,
    }


def build_fly_remap(records):
    """
    Treat each unique (date, original fly ID) as one biological fly and assign
    new fly IDs in chronological order.
    """
    unique_flies = OrderedDict()
    sorted_records = sorted(
        records,
        key=lambda row: (
            row["sort_key"],
            row["original_fly"] if row["original_fly"] is not None else 999999,
        )
    )

    for row in sorted_records:
        key = (row["date"], row["original_fly"])
        if row["original_fly"] is None:
            key = (row["date"], row["file_name"])
        if key not in unique_flies:
            unique_flies[key] = len(unique_flies) + 1

    return unique_flies


def proposed_file_name(row, new_fly_id):
    fly_token = f"F{new_fly_id:0{FLY_DIGITS}d}"

    if row["trial"] is None:
        trial_token = "Tunknown"
    else:
        trial_token = f"T{row['trial']:0{TRIAL_DIGITS}d}"

    if row["cam"] is None:
        cam_token = "CamUnknown"
    else:
        cam_token = f"Cam{row['cam']}"

    tokens = [fly_token, trial_token]
    if INCLUDE_CONDITION_IN_FILENAME and (row["condition"] != "None" or INCLUDE_NONE_CONDITION_TOKEN):
        tokens.append(f"cond{row['condition']}")
    if INCLUDE_FPS_IN_FILENAME:
        if row["fps"] is None:
            tokens.append("fpsUnknown")
        else:
            fps_value = int(round(row["fps"]))
            tokens.append(f"fps{fps_value}")
    tokens.extend([row["date"], row["time"], cam_token])
    return "_".join(tokens) + ".mp4"


def proposed_output_path(row, new_fly_id):
    new_name = proposed_file_name(row, new_fly_id)
    fly_folder = f"F{new_fly_id:0{FLY_DIGITS}d}"
    if USE_METADATA_OUTPUT_PATH:
        experiment_root = os.path.dirname(os.path.dirname(GROUP_DATA_FOLDER))
        return os.path.join(experiment_root, EXPERIMENT, GROUP_NAME, fly_folder, new_name)
    return os.path.join(GROUP_DATA_FOLDER, fly_folder, new_name)


def build_plan(records, fly_remap):
    plan = []
    for row in records:
        remap_key = (row["date"], row["original_fly"])
        if row["original_fly"] is None:
            remap_key = (row["date"], row["file_name"])
        new_fly_id = fly_remap[remap_key]
        out_path = proposed_output_path(row, new_fly_id)
        plan.append({
            "item_type": "video",
            "row": row,
            "new_fly_id": new_fly_id,
            "source": row["path"],
            "destination": out_path,
        })

    for fly_folder, metadata_path in iter_fly_metadata_files(records):
        sample_rows = [
            row for row in records
            if find_source_fly_folder(row["path"]) == fly_folder
        ]
        if not sample_rows:
            continue

        sample_row = sample_rows[0]
        remap_key = (sample_row["date"], sample_row["original_fly"])
        if sample_row["original_fly"] is None:
            remap_key = (sample_row["date"], sample_row["file_name"])
        new_fly_id = fly_remap[remap_key]
        fly_token = f"F{new_fly_id:0{FLY_DIGITS}d}"

        if RENAME_FLY_METADATA_FILES:
            ext = os.path.splitext(metadata_path)[1]
            metadata_name = f"{fly_token}_metadata_{sample_row['date']}_{sample_row['time']}{ext}"
        else:
            metadata_name = os.path.basename(metadata_path)

        if USE_METADATA_OUTPUT_PATH:
            experiment_root = os.path.dirname(os.path.dirname(GROUP_DATA_FOLDER))
            destination = os.path.join(experiment_root, EXPERIMENT, GROUP_NAME, fly_token, metadata_name)
        else:
            destination = os.path.join(GROUP_DATA_FOLDER, fly_token, metadata_name)

        metadata_row = dict(sample_row)
        metadata_row["file_name"] = os.path.basename(metadata_path)
        metadata_row["trial"] = None
        metadata_row["cam"] = None
        metadata_row["condition"] = "Metadata"
        metadata_row["fps"] = None

        plan.append({
            "item_type": "fly_metadata",
            "row": metadata_row,
            "new_fly_id": new_fly_id,
            "source": metadata_path,
            "destination": destination,
        })
    return plan


def validate_plan(plan):
    errors = []
    destinations = {}
    logical_video_keys = {}

    for item in plan:
        row = item["row"]
        source = os.path.abspath(item["source"])
        destination = os.path.abspath(item["destination"])

        required_parse_fields = {
            "date": row["date"],
            "time": row["time"],
            "original_fly": row["original_fly"],
        }
        if item["item_type"] == "video":
            required_parse_fields.update({
                "trial": row["trial"],
                "cam": row["cam"],
            })
            if REQUIRE_CONDITION_IN_FILENAME:
                required_parse_fields["condition"] = row["condition"]
        for field_name, value in required_parse_fields.items():
            if value in (None, "UnknownDate", "UnknownTime", "None"):
                errors.append(
                    f"Unrecognized or missing {field_name} for file: {source}. "
                    "No files will be moved until all required fields parse correctly."
                )

        if item["item_type"] == "video":
            if (
                row["folder_date"] is not None
                and row["date"] not in ("UnknownDate", None)
                and row["folder_date"] != row["date"]
            ):
                errors.append(
                    f"Date mismatch for file: {source}. "
                    f"Folder date={row['folder_date']}, filename date={row['date']}."
                )

            if (
                row["folder_fly"] is not None
                and row["filename_fly"] is not None
                and row["folder_fly"] != row["filename_fly"]
            ):
                errors.append(
                    f"Fly mismatch for file: {source}. "
                    f"Folder fly={row['folder_fly']}, filename fly={row['filename_fly']}."
                )

            logical_key = (
                item["new_fly_id"],
                row["trial"],
                row["cam"],
                row["condition"],
            )
            if logical_key in logical_video_keys:
                errors.append(
                    "Duplicate logical video mapping:\n"
                    f"  {logical_video_keys[logical_key]}\n"
                    f"  {source}\n"
                    f"  -> new_fly=F{item['new_fly_id']:0{FLY_DIGITS}d}, "
                    f"trial={row['trial']}, cam={row['cam']}, condition={row['condition']}"
                )
            else:
                logical_video_keys[logical_key] = source

        if source == destination:
            errors.append(f"Source and destination are identical: {source}")

        if not os.path.exists(source):
            errors.append(f"Source does not exist: {source}")

        if destination in destinations:
            errors.append(
                "Duplicate destination:\n"
                f"  {destinations[destination]}\n"
                f"  {source}\n"
                f"  -> {destination}"
            )
        else:
            destinations[destination] = source

        if os.path.exists(destination) and source != destination:
            errors.append(f"Destination already exists: {destination}")

    return errors


def print_mapping_audit(plan, fly_remap):
    video_items = [item for item in plan if item["item_type"] == "video"]
    metadata_items = [item for item in plan if item["item_type"] == "fly_metadata"]

    print("\nMapping audit:")
    print(f"  Video files: {len(video_items)}")
    print(f"  Fly metadata files: {len(metadata_items)}")
    print(f"  Remapped flies: {len(fly_remap)}")

    fly_summary = OrderedDict()
    for item in video_items:
        row = item["row"]
        fly_token = f"F{item['new_fly_id']:0{FLY_DIGITS}d}"
        if fly_token not in fly_summary:
            fly_summary[fly_token] = {
                "source_keys": set(),
                "trials": set(),
                "cams": set(),
                "conditions": set(),
                "date_folder_mismatches": 0,
                "fly_mismatches": 0,
            }

        fly_summary[fly_token]["source_keys"].add((row["date"], row["original_fly"]))
        fly_summary[fly_token]["trials"].add(row["trial"])
        fly_summary[fly_token]["cams"].add(row["cam"])
        fly_summary[fly_token]["conditions"].add(row["condition"])

        if row["folder_date"] is not None and row["folder_date"] != row["date"]:
            fly_summary[fly_token]["date_folder_mismatches"] += 1
        if (
            row["folder_fly"] is not None
            and row["filename_fly"] is not None
            and row["folder_fly"] != row["filename_fly"]
        ):
            fly_summary[fly_token]["fly_mismatches"] += 1

    for fly_token, summary in fly_summary.items():
        trials = sorted([value for value in summary["trials"] if value is not None])
        cams = sorted([value for value in summary["cams"] if value is not None])
        conditions = sorted([value for value in summary["conditions"] if value is not None])
        source_keys = sorted(summary["source_keys"])
        trial_text = "none"
        if trials:
            trial_text = f"{trials[0]}-{trials[-1]} (n={len(trials)})"
        print(
            f"  {fly_token}: source={source_keys}, trials={trial_text}, "
            f"cams={cams}, conditions={conditions}, "
            f"date_mismatch={summary['date_folder_mismatches']}, "
            f"fly_mismatch={summary['fly_mismatches']}"
        )


def write_manifest(plan, manifest_path):
    import csv

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "group_name",
                "genotype",
                "effector",
                "target",
                "manipulation",
                "intensity",
                "source",
                "destination",
                "item_type",
                "date",
                "folder_date",
                "time",
                "folder_fly",
                "filename_fly",
                "new_fly",
                "trial",
                "cam",
                "condition",
                "fps",
            ]
        )
        writer.writeheader()
        for item in plan:
            row = item["row"]
            writer.writerow({
                "experiment": EXPERIMENT,
                "group_name": GROUP_NAME,
                "genotype": GENOTYPE,
                "effector": EFFECTOR,
                "target": TARGET,
                "manipulation": MANIPULATION,
                "intensity": INTENSITY,
                "source": item["source"],
                "destination": item["destination"],
                "item_type": item["item_type"],
                "date": row["date"],
                "folder_date": row["folder_date"],
                "time": row["time"],
                "folder_fly": row["folder_fly"],
                "filename_fly": row["filename_fly"],
                "new_fly": f"F{item['new_fly_id']:0{FLY_DIGITS}d}",
                "trial": row["trial"],
                "cam": row["cam"],
                "condition": row["condition"],
                "fps": row["fps"],
            })


def apply_plan(plan):
    errors = validate_plan(plan)
    if errors:
        print("Preflight failed. No files were moved.")
        for error in errors:
            print(f"ERROR: {error}")
        return False

    for item in plan:
        destination_folder = os.path.dirname(item["destination"])
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(item["source"], item["destination"])
        print(f"Moved: {item['source']} -> {item['destination']}")

    return True


def proposed_group_folder():
    if USE_METADATA_OUTPUT_PATH:
        experiment_root = os.path.dirname(os.path.dirname(GROUP_DATA_FOLDER))
        return os.path.join(experiment_root, EXPERIMENT, GROUP_NAME)
    return GROUP_DATA_FOLDER


def manifest_output_path():
    return os.path.join(proposed_group_folder(), MANIFEST_FILE_NAME)


def main():
    records = [parse_video_path(path) for path in iter_video_files(GROUP_DATA_FOLDER)]
    records = sorted(
        records,
        key=lambda row: (
            row["sort_key"],
            row["original_fly"] if row["original_fly"] is not None else 999999,
            row["trial"] if row["trial"] is not None else 999999,
            row["cam"] if row["cam"] is not None else 999999,
            row["file_name"],
        )
    )

    if len(records) == 0:
        print(f"No MP4 files found under: {GROUP_DATA_FOLDER}")
        return

    fly_remap = build_fly_remap(records)

    plan = build_plan(records, fly_remap)

    if APPLY_CHANGES:
        print("APPLY_CHANGES=True - files will be moved/renamed after preflight checks.\n")
    else:
        print("DRY RUN ONLY - no files or folders will be modified.\n")
    print("Group metadata:")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"  Group name: {GROUP_NAME}")
    print(f"  Genotype: {GENOTYPE}")
    print(f"  Effector: {EFFECTOR}")
    print(f"  Target: {TARGET}")
    print(f"  Manipulation: {MANIPULATION}")
    print(f"  Intensity: {INTENSITY}")
    print(f"  Source folder: {GROUP_DATA_FOLDER}")
    print(f"  Proposed organized folder: {proposed_group_folder()}\\F###\n")

    print("Fly remap:")
    for (date, original_fly), new_fly_id in fly_remap.items():
        print(f"  date={date}, original_fly={original_fly} -> F{new_fly_id:0{FLY_DIGITS}d}")

    print("\nOriginal -> proposed:")
    for item in plan:
        row = item["row"]
        new_fly_id = item["new_fly_id"]

        print("\nORIGINAL:")
        print(f"  {item['source']}")
        print("PARSED:")
        print(
            f"  item_type={item['item_type']}, "
            f"  date={row['date']}, folder_date={row['folder_date']}, time={row['time']}, "
            f"folder_fly={row['folder_fly']}, filename_fly={row['filename_fly']}, "
            f"new_fly=F{new_fly_id:0{FLY_DIGITS}d}, "
            f"trial={row['trial']}, cam={row['cam']}, "
            f"condition={row['condition']}, fps={row['fps']}"
        )
        print("PROPOSED:")
        print(f"  {item['destination']}")

    print(f"\nSummary: {len(records)} MP4 files, {len(fly_remap)} remapped flies.")
    print_mapping_audit(plan, fly_remap)

    errors = validate_plan(plan)
    if errors:
        print("\nPreflight warnings/errors:")
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print("\nPreflight passed: no duplicate destinations or existing destination files found.")

    if WRITE_MANIFEST:
        manifest_path = manifest_output_path()
        if APPLY_CHANGES:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            write_manifest(plan, manifest_path)
            print(f"Wrote manifest: {manifest_path}")
        else:
            if errors:
                print(f"Manifest not written because preflight found errors: {manifest_path}")
            else:
                os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                write_manifest(plan, manifest_path)
                print(f"Wrote dry-run manifest: {manifest_path}")

    if APPLY_CHANGES:
        apply_plan(plan)


if __name__ == "__main__":
    main()
