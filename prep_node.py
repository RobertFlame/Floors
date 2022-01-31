import os
import json
from const import *

settings_id = "preprocess"
settings_node_id = "preprocess node"
settings_file = "settings.json"

if __name__ == "__main__":
    input_dir_name = ""
    output_dir_name = ""
    building_id = ""
    with open(settings_file, 'r') as f_in:
        all_settings = json.load(f_in)
        settings = all_settings[settings_id]
        input_dir_name = settings["raw input dir"]
        output_dir_name = settings["node output dir"]

    root = os.getcwd()
    for building_id in building_ids:
        print (f"building_id: {building_id}")
        train_path = os.path.abspath(os.path.join(root, input_dir_name, building_id))
        for floor_id in os.listdir(train_path): # loop through each floor
            floor_path = os.path.abspath(os.path.join(train_path, floor_id))
            breakpoint_info_list = []
            for filename in os.listdir(floor_path): # loop through all files in each floor
                file_path = os.path.abspath(os.path.join(floor_path, filename))
                with open(file_path, 'rb') as f_in:
                    for line in f_in:
                        line_decoded = line.decode("utf-8")
                        if ('TYPE_WAYPOINT' in line_decoded):
                            waypoint_info = line_decoded.split('\t')
                            breakpoint_info_list.append([waypoint_info[2], waypoint_info[3].strip(), waypoint_info[0]])
            breakpoint_list = sorted(breakpoint_info_list, key=lambda x:int(x[2]))

            output_folder = os.path.abspath(os.path.join(output_dir_name, building_id))
            if not (os.path.exists(output_folder)):
                os.mkdir(output_folder)
            output_file = os.path.abspath(os.path.join(output_folder, "{}_WiFi.txt".format(floor_id)))

            with open(file_path, 'rb') as f_in, open(output_file, "a") as f_out:
                f_out.write("Start: \n") # first line
                for i in range(len(breakpoint_list)): # second line
                    f_out.write("{},{} ".format(breakpoint_list[i][0], breakpoint_list[i][1]))
                f_out.write("\n")
                for i in range(len(breakpoint_list)): # third line
                    f_out.write("{} ".format(breakpoint_list[i][2]))
                f_out.write("\n")

            current_time = 0
            first_time = True
            for filename in os.listdir(floor_path):
                with open(file_path, 'rb') as f_in, open(output_file, "a") as f_out:
                    for line in f_in:
                        line_decoded = line.decode("utf-8")
                        if ('TYPE_WIFI' in line_decoded):
                            wifi_info = line_decoded.split('\t')
                            if int(wifi_info[0])!=current_time:
                                if first_time:
                                    first_time = False
                                else:
                                    f_out.write("\n")
                                f_out.write("{} ".format(wifi_info[0]))
                                current_time = int(wifi_info[0])
                            f_out.write("{},{} ".format(wifi_info[2], wifi_info[4]))
