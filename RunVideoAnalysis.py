import time
import subprocess
import os
import shutil
import re
from natsort import natsorted


FLY_FOLDER_PATTERN = re.compile(r"^(?:F|Fly_?)(?P<fly_num>\d+)$", re.IGNORECASE)


def getVideoPaths(video_folder, filetype):
    video_paths = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                # Get the full path of the input file
                input_file_path = os.path.join(root, file)
                video_paths.append(input_file_path)

    video_paths = natsorted(video_paths)
    return video_paths


def get_fly_number(file_path, data_folder):
    """
    Read the fly ID from the first folder below data_folder.

    Expected input layout:
        data_folder/F001/video.mp4
        data_folder/Fly_1/video.mp4
    """
    relative_path = os.path.relpath(file_path, data_folder)
    path_parts = relative_path.split(os.sep)
    if len(path_parts) < 2:
        raise ValueError(
            f"Video is not inside a fly folder below Data_folder: {file_path}"
        )

    source_fly_folder = path_parts[0]
    match = FLY_FOLDER_PATTERN.fullmatch(source_fly_folder)
    if match is None:
        raise ValueError(
            "Expected the first folder below Data_folder to be named F###, "
            f"F#, Fly_#, or Fly#. Found '{source_fly_folder}' for: {file_path}"
        )

    return int(match.group("fly_num"))



Anipose_path = r"D:\TibiaTarsusPlatformODLight-Wayne-2024-10-19\Network-01-18-Test"
DLC_analyzed_data_folder = r"C:\Users\agrawal-admin\DLCData\Network-01-18-2026"
DLC_config_path = r"C:\Users\agrawal-admin\Desktop\TibiaTarsusPlatformODLight-Wayne-2024-10-19\network\config.yaml"
Data_folder = r"D:\DataFolder\WT-LP\T2-TiTa"
RUN_DLC = True
video_paths = getVideoPaths(Data_folder, ".mp4")

# Data_folder is the group folder:
#     <data root>/<experiment>/<group>/<fly folder>/<video>.mp4
group_name = os.path.basename(os.path.normpath(Data_folder))
experiment = os.path.basename(os.path.dirname(os.path.normpath(Data_folder)))

destinations = {}
grouped_files = {}
Anipose_Data_Folders = {}

# Iterate through the file paths
for file_path in video_paths:
    fly_folder_number = get_fly_number(file_path, Data_folder)
    fly_token = f"F{fly_folder_number:03d}"
    unique_key = f"{group_name}\\{fly_token}"

    # Create the experiment folder if it doesn't exist
    experiment_folder = os.path.join(DLC_analyzed_data_folder, experiment)
    if RUN_DLC and not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)
        print(f"Created experiment folder: {experiment_folder}")

    group_folder = os.path.join(experiment_folder, group_name)
    if RUN_DLC and not os.path.exists(group_folder):
        os.makedirs(group_folder, exist_ok=True)
        print(f"Created group folder: {group_folder}")

    # Create the group + fly folder
    fly_folder_name = fly_token
    fly_folder = os.path.join(group_folder, fly_folder_name)

    if unique_key not in destinations:
        destinations[unique_key] = fly_folder

    if RUN_DLC and not os.path.exists(fly_folder):
        os.makedirs(fly_folder, exist_ok=True)
        print(f"Created folder: {fly_folder}")

    # Add the file to the corresponding Fly_N group
    if unique_key not in grouped_files:
        grouped_files[unique_key] = []
    grouped_files[unique_key].append(file_path)


    Anipose_Experiment_folder = os.path.join(Anipose_path, experiment)
    if RUN_DLC and not os.path.exists(Anipose_Experiment_folder):
        os.makedirs(Anipose_Experiment_folder, exist_ok=True)

    Anipose_group_folder = os.path.join(Anipose_Experiment_folder, group_name)
    if RUN_DLC and not os.path.exists(Anipose_group_folder):
        os.makedirs(Anipose_group_folder, exist_ok=True)


    Anipose_fly_folder = os.path.join(Anipose_group_folder, os.path.join(fly_folder_name, r"pose-2d"))
    if unique_key not in Anipose_Data_Folders:
        Anipose_Data_Folders[unique_key] = Anipose_fly_folder

    if RUN_DLC and not os.path.exists(Anipose_fly_folder):
        os.makedirs(Anipose_fly_folder, exist_ok=True)


for k in destinations.keys():
    print("\nDeepLabCut analysis batch")
    print(f"Fly key: {k}")
    print("Videos:")
    for video_path in grouped_files[k]:
        print(f"  {video_path}")
    print(f"DLC output folder: {destinations[k]}")
    print(f"Anipose pose-2d folder: {Anipose_Data_Folders[k]}")

if not RUN_DLC:
    print("\nDry run complete. Set RUN_DLC = True to run DeepLabCut.")
else:
    import deeplabcut
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(tf.config.list_physical_devices("GPU"))
    start_time = time.perf_counter()
    try:
        for k in destinations.keys():
            print(grouped_files[k])
            print(destinations[k])
            deeplabcut.analyze_videos(
                DLC_config_path,
                grouped_files[k],
                save_as_csv=False,
                destfolder=destinations[k],
                videotype="mp4",
            )
    except KeyboardInterrupt:
        print("Video analysis interrupted")

    for k in destinations.keys():
        for root, dirs, files in os.walk(destinations[k]):
            for file in files:
                if file.endswith(".h5"):
                    # Get the full path of the input file
                    input_file_path = os.path.join(root, file)
                    print(input_file_path)
                    output_file_path = os.path.join(Anipose_Data_Folders[k], file)
                    new_filename = output_file_path[:output_file_path.find("Cam") + 4] + ".h5"
                    new_destination_path = os.path.join(Anipose_Data_Folders[k], new_filename)
                    print(new_destination_path)

                    if os.path.isfile(new_destination_path):
                        print("File exists.")
                    else:
                        shutil.copy(input_file_path, new_destination_path)
                        print("File does not exist.")

    # Run 'anipose filter' command in the specified directory
    subprocess.run(["anipose", "filter"], cwd=Anipose_path)

# print(f"Total analysis time {time.perf_counter() - start_time}")
