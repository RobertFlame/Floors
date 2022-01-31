import os
import pickle5 as pickle
import json
from const import *

for building_id in building_ids:
    raw_path = os.path.abspath(os.path.join(os.getcwd(), "./data/output_data/{}/raw_data").format(building_id))
    obs_path = os.path.join(raw_path, "_observations_prune_{}.pkl".format(building_id))
    out_file_floors = os.path.abspath(os.path.join(os.getcwd(), "./data/output_data/{}/raw_data/floors.txt".format(building_id)))

    with open(obs_path, 'rb') as f_in1, open(out_file_floors, "w") as f_out:
        series_obs = pickle.load(f_in1)
        series_obs = {series_obs[i][0]:series_obs[i][1]for i in range(len(series_obs))}
        f_out.write(str(series_obs))