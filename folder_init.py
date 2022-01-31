import os
import shutil

target_folder = ["sites", "output_data"]

if __name__ == "__main__":
    root = os.getcwd()
    data_folder = os.path.join(os.path.dirname(root), "Floors/data")
    for dir_name in target_folder:
        dir_path = os.path.join(data_folder, dir_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)