import os
import re
import csv
import shutil
from collections import defaultdict

import pandas as pd


# ============================================================
# User-editable settings
# ============================================================
group_name = "BICSxCHR-12mW"
# Folder containing organized video group folders. The script only reads
# video_reorganization_manifest.csv files under this folder.
VIDEO_DATA_ROOT = os.path.join(r"D:\DataFolder\OPTO", group_name)

# Folder containing one kinematic group. The script scans recursively for CSVs
# that are directly inside pose-3d folders.
KINEMATIC_DATA_FOLDER = os.path.join(r"D:\TibiaTarsusPlatformODLight-Wayne-2024-10-19\Network-07-04\OPTO", group_name)

# False = preview only. True = create F###/pose-3d and F###/pose-2d-filtered
# folders under KINEMATIC_DATA_FOLDER and move/rename matched files.
APPLY_CHANGES = True

VIDEO_MANIFEST_FILE_NAME = "video_reorganization_manifest.csv"
KINEMATIC_MANIFEST_FILE_NAME = "kinematic_reorganization_manifest.csv"

POSE_3D_FOLDER_NAME = "pose-3d"
POSE_2D_FILTERED_FOLDER_NAME = "pose-2d-filtered"
FLY_DIGITS = 3

# LO/NL are mapped to ON/OFF so old kinematic filenames match the video
# manifest condition labels.
CONDITION_MAP = {
    "ON": "ON",
    "OFF": "OFF",
    "LO": "ON",
    "NL": "OFF",
}


# ============================================================
# Parsing rules
# ============================================================

DATE_TIME_PATTERNS = [
    # 2025-10-20-13-42-48.38
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
GROUP_FLY_FOLDER_PATTERN = re.compile(r"(?P<group>.+?)[-_]F(?P<fly>\d+)$", re.IGNORECASE)
ORGANIZED_FLY_PATTERN = re.compile(r"^F(?P<fly>\d+)$", re.IGNORECASE)
VIDEO_DESTINATION_PATTERN = re.compile(
    r"(?P<fly>F\d+)_T(?P<trial>\d+)"
    r"(?:_cond(?P<condition>[^_]+))?"
    r"(?:_fps(?P<fps>[^_]+))?"
    r"_(?P<date>\d{8})_(?P<time>\d{6})"
    r"_Cam(?P<cam>\d+)\.[^.]+$",
    re.IGNORECASE,
)
KINEMATIC_3D_EXTENSIONS = {".csv"}
KINEMATIC_2D_EXTENSIONS = {".h5", ".hdf5"}


def find_date_time(file_name):
    for pattern in DATE_TIME_PATTERNS:
        match = pattern.search(file_name)
        if match:
            date = f"{match.group('year')}{match.group('month')}{match.group('day')}"
            second = match.groupdict().get("second", "00")
            time = f"{match.group('hour')}{match.group('minute')}{second}"
            return date, time
    return "UnknownDate", "UnknownTime"


def first_int(pattern, text):
    match = pattern.search(text)
    if not match:
        return None
    group_name = next(iter(match.groupdict()))
    return int(match.group(group_name))


def find_condition(file_name):
    match = CONDITION_PATTERN.search(file_name)
    if not match:
        return "None"
    raw = match.group("condition").upper()
    return CONDITION_MAP.get(raw, raw)


def is_directly_in_folder(path, folder_name):
    return os.path.basename(os.path.dirname(path)).lower() == folder_name.lower()


def find_kinematic_folder_fly(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts[:-1]):
        organized = ORGANIZED_FLY_PATTERN.match(part)
        if organized:
            return int(organized.group("fly")), part

        match = GROUP_FLY_FOLDER_PATTERN.match(part)
        if match:
            return int(match.group("fly")), part

    return None, None


def iter_video_manifests(root):
    for current_root, _, files in os.walk(root):
        if VIDEO_MANIFEST_FILE_NAME in files:
            yield os.path.join(current_root, VIDEO_MANIFEST_FILE_NAME)


def iter_kinematic_files(root):
    for current_root, _, files in os.walk(root):
        for file_name in files:
            path = os.path.join(current_root, file_name)
            ext = os.path.splitext(file_name)[1].lower()
            if ext in KINEMATIC_3D_EXTENSIONS and is_directly_in_folder(path, POSE_3D_FOLDER_NAME):
                yield "pose_3d", path
            if ext in KINEMATIC_2D_EXTENSIONS and is_directly_in_folder(path, POSE_2D_FILTERED_FOLDER_NAME):
                yield "pose_2d_filtered", path


def parse_video_destination(destination):
    file_name = os.path.basename(str(destination))
    match = VIDEO_DESTINATION_PATTERN.search(file_name)
    if not match:
        return None
    condition = match.group("condition") or "None"
    cam_token_start = file_name.rfind("_Cam", 0, match.start("cam"))
    if cam_token_start == -1:
        destination_stem_without_cam = file_name[:match.start("cam")]
    else:
        # Keep the separator before Cam so output follows old kinematic naming:
        # F001_T001_condNone_fps200_20250130_153141_.csv
        destination_stem_without_cam = file_name[:cam_token_start + 1]
    return {
        "new_fly": match.group("fly").upper(),
        "trial": int(match.group("trial")),
        "condition": condition,
        "fps": match.group("fps"),
        "date": match.group("date"),
        "time": match.group("time"),
        "cam": int(match.group("cam")),
        "destination_file_name": file_name,
        "destination_stem_without_cam": destination_stem_without_cam,
    }


def load_video_manifest_index(video_root):
    records = []
    errors = []

    def clean_manifest_value(value):
        if pd.isna(value):
            return ""
        return str(value)

    for manifest_path in iter_video_manifests(video_root):
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception as exc:
            errors.append(f"Could not read manifest {manifest_path}: {exc}")
            continue

        if "destination" not in manifest.columns:
            errors.append(f"Manifest is missing destination column: {manifest_path}")
            continue

        if "item_type" in manifest.columns:
            manifest = manifest[manifest["item_type"].astype(str).str.lower() == "video"].copy()

        for row_idx, row in manifest.iterrows():
            parsed = parse_video_destination(row["destination"])
            if parsed is None:
                errors.append(
                    f"Could not parse video destination filename in {manifest_path}, row {row_idx}: "
                    f"{row['destination']}"
                )
                continue

            record = {
                "manifest_path": manifest_path,
                "manifest_row": row_idx,
                "experiment": clean_manifest_value(row.get("experiment", "")),
                "group_name": clean_manifest_value(row.get("group_name", "")),
                "genotype": clean_manifest_value(row.get("genotype", "")),
                "effector": clean_manifest_value(row.get("effector", "")),
                "target": clean_manifest_value(row.get("target", "")),
                "manipulation": clean_manifest_value(row.get("manipulation", "")),
                "intensity": clean_manifest_value(row.get("intensity", "")),
                **parsed,
            }
            records.append(record)

    # Ignore Cam# by de-duplicating all camera views of the same video trial.
    by_key = defaultdict(list)
    for record in records:
        key = (
            record["date"],
            record["time"],
            record["trial"],
            record["condition"],
            record["new_fly"],
        )
        by_key[key].append(record)

    deduped = {}
    duplicate_errors = []
    for key, matching_records in by_key.items():
        fps_values = {clean_manifest_value(record["fps"]) for record in matching_records}
        group_values = {
            (
                record["experiment"],
                record["group_name"],
                record["genotype"],
                record["effector"],
                record["target"],
                record["manipulation"],
                record["intensity"],
            )
            for record in matching_records
        }
        if len(fps_values) > 1 or len(group_values) > 1:
            duplicate_errors.append(
                "Video manifest rows disagree after ignoring Cam#:\n"
                f"  key={key}\n"
                f"  fps_values={sorted(fps_values)}\n"
                f"  group_values={sorted(group_values)}"
            )
            continue
        deduped[key] = matching_records[0]

    errors.extend(duplicate_errors)
    cam_index = defaultdict(list)
    for record in records:
        cam_key = (
            record["date"],
            record["time"],
            record["trial"],
            record["condition"],
            record["new_fly"],
            record["cam"],
        )
        cam_index[cam_key].append(record)

    for key, matching_records in cam_index.items():
        if len(matching_records) > 1:
            errors.append(
                "Duplicate video manifest rows for exact Cam# key:\n"
                f"  key={key}\n"
                f"  rows={[record['manifest_row'] for record in matching_records]}"
            )

    cam_deduped = {
        key: matching_records[0]
        for key, matching_records in cam_index.items()
        if len(matching_records) == 1
    }

    return deduped, cam_deduped, errors


def parse_kinematic_path(path):
    file_name = os.path.basename(path)
    date, time = find_date_time(file_name)
    folder_fly, fly_folder_name = find_kinematic_folder_fly(path)
    filename_fly = first_int(FLY_PATTERN, file_name)
    trial = first_int(TRIAL_PATTERN, file_name)
    cam = first_int(CAM_PATTERN, file_name)
    condition = find_condition(file_name)
    folder_fly_token = None if folder_fly is None else f"F{folder_fly:0{FLY_DIGITS}d}"

    return {
        "path": path,
        "file_name": file_name,
        "date": date,
        "time": time,
        "folder_fly": folder_fly,
        "folder_fly_token": folder_fly_token,
        "fly_folder_name": fly_folder_name,
        "filename_fly": filename_fly,
        "trial": trial,
        "cam": cam,
        "condition": condition,
    }


def proposed_3d_file_name(video_record):
    # Match the organized video filename exactly, except remove Cam# and change
    # extension. The retained stem intentionally includes the underscore before
    # Cam#, e.g. F001_T001_condNone_fps200_20250130_153141_.csv
    return video_record["destination_stem_without_cam"] + ".csv"


def proposed_2d_file_name(video_record, old_path):
    ext = os.path.splitext(old_path)[1].lower()
    video_stem = os.path.splitext(video_record["destination_file_name"])[0]
    return video_stem + ext


def proposed_output_path(video_record, data_type, old_path):
    if data_type == "pose_3d":
        output_folder = POSE_3D_FOLDER_NAME
        new_name = proposed_3d_file_name(video_record)
    elif data_type == "pose_2d_filtered":
        output_folder = POSE_2D_FILTERED_FOLDER_NAME
        new_name = proposed_2d_file_name(video_record, old_path)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return os.path.join(
        KINEMATIC_DATA_FOLDER,
        video_record["new_fly"],
        output_folder,
        new_name,
    )


def match_kinematic_to_video(kin_row, video_index, video_cam_index, require_cam):
    if kin_row["trial"] is None:
        return None, "missing trial in kinematic filename"
    if kin_row["date"] in {"UnknownDate", None} or kin_row["time"] in {"UnknownTime", None}:
        return None, "missing date/time in kinematic filename"
    if kin_row["folder_fly_token"] is None:
        return None, "missing metadata fly number in kinematic folder"
    if require_cam and kin_row["cam"] is None:
        return None, "missing Cam# in 2D filtered filename"

    def same_timestamp(video_key):
        same_date = video_key[0] == kin_row["date"]
        if not same_date:
            return False
        video_time = str(video_key[1])
        kin_time = str(kin_row["time"])
        if video_time == kin_time:
            return True
        # Some older video files only encode HHMM. They are parsed as HHMM00.
        # Treat those as minute-level timestamps and match any kinematic file
        # acquired within the same minute.
        if video_time.endswith("00") and video_time[:4] == kin_time[:4]:
            return True
        return False

    if require_cam:
        exact_key = (
            kin_row["date"],
            kin_row["time"],
            kin_row["trial"],
            kin_row["condition"],
            kin_row["folder_fly_token"],
            kin_row["cam"],
        )
        if exact_key in video_cam_index:
            return video_cam_index[exact_key], None
    else:
        exact_key = (
            kin_row["date"],
            kin_row["time"],
            kin_row["trial"],
            kin_row["condition"],
            kin_row["folder_fly_token"],
        )
        if exact_key in video_index:
            return video_index[exact_key], None

    # If condition is absent in one side, try condition-agnostic matching only
    # when it produces exactly one candidate.
    candidates = [
        record for key, record in (video_cam_index.items() if require_cam else video_index.items())
        if same_timestamp(key)
        and key[2] == kin_row["trial"]
        and key[4] == kin_row["folder_fly_token"]
        and (not require_cam or key[5] == kin_row["cam"])
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, f"ambiguous video manifest match for {kin_row}; candidates={len(candidates)}"

    return None, f"no video manifest match for {kin_row}"


def build_plan(video_index, video_cam_index):
    plan = []
    errors = []
    for data_type, path in iter_kinematic_files(KINEMATIC_DATA_FOLDER):
        kin_row = parse_kinematic_path(path)
        require_cam = data_type == "pose_2d_filtered"
        video_record, error = match_kinematic_to_video(
            kin_row,
            video_index,
            video_cam_index,
            require_cam=require_cam
        )
        if error is not None:
            errors.append(f"{path}: {error}")
            continue

        plan.append({
            "data_type": data_type,
            "source": path,
            "destination": proposed_output_path(video_record, data_type, path),
            "kinematic": kin_row,
            "video": video_record,
        })

    return plan, errors


def validate_plan(plan):
    errors = []
    destination_seen = {}
    source_seen = set()
    logical_seen = {}

    def timestamps_match(kin_date, kin_time, video_date, video_time):
        kin_time = str(kin_time)
        video_time = str(video_time)
        if kin_date != video_date:
            return False
        if kin_time == video_time:
            return True
        return video_time.endswith("00") and video_time[:4] == kin_time[:4]

    for item in plan:
        source = os.path.abspath(item["source"])
        destination = os.path.abspath(item["destination"])
        kin = item["kinematic"]
        video = item["video"]
        data_type = item["data_type"]

        expected_folder = POSE_3D_FOLDER_NAME if data_type == "pose_3d" else POSE_2D_FILTERED_FOLDER_NAME
        if not is_directly_in_folder(source, expected_folder):
            errors.append(f"{data_type} file is not directly inside {expected_folder}: {source}")
        if not os.path.exists(source):
            errors.append(f"Source does not exist: {source}")
        if source == destination:
            errors.append(f"Source and destination are identical: {source}")
        if os.path.exists(destination) and source != destination:
            errors.append(f"Destination already exists: {destination}")
        if source in source_seen:
            errors.append(f"Duplicate source in plan: {source}")
        source_seen.add(source)

        if destination in destination_seen:
            errors.append(
                "Duplicate destination:\n"
                f"  {destination_seen[destination]}\n"
                f"  {source}\n"
                f"  -> {destination}"
            )
        destination_seen[destination] = source

        logical_key = (
            data_type,
            video["new_fly"],
            video["trial"],
            video["condition"],
            video["date"],
            video["time"],
            video["cam"] if data_type == "pose_2d_filtered" else "no_cam",
        )
        if logical_key in logical_seen:
            errors.append(
                "Duplicate logical kinematic mapping:\n"
                f"  {logical_seen[logical_key]}\n"
                f"  {source}\n"
                f"  -> {logical_key}"
            )
        logical_seen[logical_key] = source

        if kin["folder_fly_token"] != video["new_fly"]:
            errors.append(
                f"Fly mismatch after manifest match: {source}. "
                f"Kinematic folder fly={kin['folder_fly_token']}, video new_fly={video['new_fly']}."
            )
        if not timestamps_match(kin["date"], kin["time"], video["date"], video["time"]):
            errors.append(
                f"Timestamp mismatch after manifest match: {source}. "
                f"Kinematic timestamp={kin['date']}_{kin['time']}, "
                f"video timestamp={video['date']}_{video['time']}."
            )
        if data_type == "pose_2d_filtered" and kin["cam"] != video["cam"]:
            errors.append(
                f"Cam mismatch after manifest match: {source}. "
                f"Kinematic Cam{kin['cam']}, video Cam{video['cam']}."
            )

    return errors


def write_manifest(plan, manifest_path):
    fieldnames = [
        "experiment",
        "group_name",
        "genotype",
        "effector",
        "target",
        "manipulation",
        "intensity",
        "source",
        "destination",
        "data_type",
        "old_file_name",
        "new_file_name",
        "new_fly",
        "folder_fly",
        "filename_fly",
        "trial",
        "cam",
        "condition",
        "date",
        "time",
        "fps",
        "video_manifest_path",
        "video_manifest_row",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in plan:
            kin = item["kinematic"]
            video = item["video"]
            writer.writerow({
                "experiment": video["experiment"],
                "group_name": video["group_name"],
                "genotype": video["genotype"],
                "effector": video["effector"],
                "target": video["target"],
                "manipulation": video["manipulation"],
                "intensity": video["intensity"],
                "source": item["source"],
                "destination": item["destination"],
                "data_type": item["data_type"],
                "old_file_name": kin["file_name"],
                "new_file_name": os.path.basename(item["destination"]),
                "new_fly": video["new_fly"],
                "folder_fly": kin["folder_fly"],
                "filename_fly": kin["filename_fly"],
                "trial": video["trial"],
                "cam": video["cam"] if item["data_type"] == "pose_2d_filtered" else "",
                "condition": video["condition"],
                "date": video["date"],
                "time": video["time"],
                "fps": video["fps"],
                "video_manifest_path": video["manifest_path"],
                "video_manifest_row": video["manifest_row"],
            })


def apply_plan(plan):
    for item in plan:
        os.makedirs(os.path.dirname(item["destination"]), exist_ok=True)
        shutil.move(item["source"], item["destination"])


def print_preview(plan, match_errors, video_errors):
    print(f"\nAPPLY_CHANGES = {APPLY_CHANGES}")
    print(f"VIDEO_DATA_ROOT = {VIDEO_DATA_ROOT}")
    print(f"KINEMATIC_DATA_FOLDER = {KINEMATIC_DATA_FOLDER}")
    type_counts = defaultdict(int)
    for item in plan:
        type_counts[item["data_type"]] += 1
    print(f"\nPlanned kinematic file moves: {len(plan)}")
    for data_type, count in sorted(type_counts.items()):
        print(f"  {data_type}: {count}")
    print(f"Unmatched kinematic files: {len(match_errors)}")
    print(f"Video manifest parse errors: {len(video_errors)}")

    by_fly = defaultdict(int)
    for item in plan:
        by_fly[item["video"]["new_fly"]] += 1
    print("\nPlanned files by fly:")
    for fly, count in sorted(by_fly.items()):
        print(f"  {fly}: {count} CSV files")

    print("\nPreview:")
    for item in plan:
        kin = item["kinematic"]
        video = item["video"]
        print("\nORIGINAL:")
        print(f"  {item['source']}")
        print(f"DATA TYPE: {item['data_type']}")
        print("MATCHED VIDEO MANIFEST:")
        print(
            f"  group={video['group_name']}, new_fly={video['new_fly']}, "
            f"trial={video['trial']}, condition={video['condition']}, "
            f"cam={video['cam']}, fps={video['fps']}, timestamp={video['date']}_{video['time']}"
        )
        print("KINEMATIC PARSED:")
        print(
            f"  folder_fly={kin['folder_fly_token']}, filename_fly={kin['filename_fly']}, "
            f"trial={kin['trial']}, cam={kin['cam']}, condition={kin['condition']}, "
            f"timestamp={kin['date']}_{kin['time']}"
        )
        print("PROPOSED:")
        print(f"  {item['destination']}")


def main():
    video_index, video_cam_index, video_errors = load_video_manifest_index(VIDEO_DATA_ROOT)
    plan, match_errors = build_plan(video_index, video_cam_index)
    validation_errors = validate_plan(plan)

    print_preview(plan, match_errors, video_errors)

    all_errors = video_errors + match_errors + validation_errors
    if all_errors:
        print("\nPreflight warnings/errors:")
        for error in all_errors:
            print(f"ERROR: {error}")
        print("\nNo files were moved.")
        return

    manifest_path = os.path.join(KINEMATIC_DATA_FOLDER, KINEMATIC_MANIFEST_FILE_NAME)
    write_manifest(plan, manifest_path)
    print(f"\nKinematic manifest written: {manifest_path}")

    if not APPLY_CHANGES:
        print("\nDry run only. Set APPLY_CHANGES = True to move and rename files.")
        return

    apply_plan(plan)
    print("\nDone. Kinematic CSV files were moved and renamed.")


if __name__ == "__main__":
    main()
