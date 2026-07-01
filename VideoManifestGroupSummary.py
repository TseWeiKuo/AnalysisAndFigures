import os
import csv
import pandas as pd


# ============================================================
# User-editable settings
# ============================================================

# Root folder that contains many organized group subfolders.
ROOT_FOLDER = r"D:\DataFolder"

# Manifest filename created by Data_organization.py.
MANIFEST_FILE_NAME = "video_reorganization_manifest.csv"

# Output CSV path. If this is a simple filename, it is written in ROOT_FOLDER.
OUTPUT_CSV = "video_group_summary.csv"


# ============================================================
# Summary settings
# ============================================================

GROUP_COLUMNS = [
    "experiment",
    "group_name",
    "genotype",
    "effector",
    "target",
    "manipulation",
    "intensity",
]


def find_manifest_files(root_folder, manifest_file_name):
    for current_root, _, files in os.walk(root_folder):
        if manifest_file_name in files:
            yield os.path.join(current_root, manifest_file_name)


def summarize_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    if df.empty:
        return []

    missing = [column for column in GROUP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {missing}")

    if "new_fly" not in df.columns:
        raise ValueError(f"{manifest_path} is missing required column: new_fly")

    if "item_type" in df.columns:
        video_df = df[df["item_type"].astype(str).str.lower() == "video"].copy()
        if video_df.empty:
            video_df = df.copy()
    else:
        video_df = df.copy()

    row = {}
    warnings = []
    for column in GROUP_COLUMNS:
        values = (
            video_df[column]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        values = values[values != ""].unique().tolist()
        if len(values) == 0:
            row[column] = ""
        else:
            row[column] = values[0]
        if len(values) > 1:
            warnings.append(f"{column}: {values}")

    row.update({
        "total_fly_n": int(video_df["new_fly"].dropna().nunique()),
        "total_video_n": int(len(video_df)),
        "manifest_path": manifest_path,
        "group_folder": os.path.dirname(manifest_path),
        "metadata_warnings": "; ".join(warnings),
    })

    return [row]


def write_summary(rows, output_csv):
    fieldnames = GROUP_COLUMNS + [
        "total_fly_n",
        "total_video_n",
        "manifest_path",
        "group_folder",
        "metadata_warnings",
    ]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    output_csv = OUTPUT_CSV
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(ROOT_FOLDER, output_csv)

    manifest_paths = sorted(find_manifest_files(ROOT_FOLDER, MANIFEST_FILE_NAME))
    if not manifest_paths:
        print(f"No {MANIFEST_FILE_NAME} files found under: {ROOT_FOLDER}")
        return

    summary_rows = []
    errors = []
    for manifest_path in manifest_paths:
        try:
            summary_rows.extend(summarize_manifest(manifest_path))
        except Exception as exc:
            errors.append((manifest_path, str(exc)))

    if summary_rows:
        write_summary(summary_rows, output_csv)
        print(f"Wrote group summary: {output_csv}")
        print(f"Manifest files summarized: {len(manifest_paths) - len(errors)}")
        print(f"Summary rows: {len(summary_rows)}")
    else:
        print("No valid summary rows were produced.")

    if errors:
        print("\nManifest files with errors:")
        for manifest_path, message in errors:
            print(f"  {manifest_path}")
            print(f"    {message}")


if __name__ == "__main__":
    main()
