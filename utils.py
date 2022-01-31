import pickle5 as pickle
import json
from const import *


def load_file_comparison(filename, observation_ids, observation_sign, file_type, prune):
    max_id = 0
    if observation_ids:
        max_id = max(max_id, int(observation_ids[-1][0][1:]))
    mac_all = []
    with open(filename, "r") as f_in:
        while True:
            line = f_in.readline().rstrip(" \n")
            if not line:
                break
            if file_type == "path":
                if line.startswith("Start:"):
                    device_id = os.path.basename(filename)[:2]
                    try:
                        breakpoints = [[float(coor) for coor in item.split(
                            ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                        timestamps = [
                            int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    except ValueError:
                        break
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = os.path.basename(filename)
                coor, rssi_pairs = line.split(" ", 1)
                ground_truth = [float(item) for item in coor.split(",")]
            elif file_type == "new":
                wifi_json = json.loads(line)
                timestamp = wifi_json['sysTimeMs']
                rssi_pairs = ''
                for item in wifi_json['data']:
                    rssi_pairs += str(item['bssid'].replace(':','')) + ',' + str(item['rssi']) + ' '
                rssi_pairs = rssi_pairs.strip(' ')
                ground_truth = [None,None]
                device_id = None

            rssi_dict = {}
            for rssi_pair in rssi_pairs.split(" "):
                mac = rssi_pair.split(",")[0]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                if prune and is_virtual_mac(mac):
                    continue
                try:
                    rssi_dict[mac] = float(rssi)
                except ValueError:
                    continue
                if mac not in mac_all:
                    mac_all.append(mac)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids, mac_all


def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))


def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series


def rssi2weight(offset, rssi):
    return offset + rssi


def is_virtual_mac(mac_addr):
    mac_addr = mac_addr.replace(":", "").upper()
    first_hex = int(mac_addr[0:2], 16)
    return first_hex & 0x02 != 0


def interpolate_point(timestamp, timestamps, breakpoints):
    if timestamp <= timestamps[0]:
        print("timestamp too small: {} <= {}".format(timestamp, timestamps[0]))
        return breakpoints[0]
    if timestamp >= timestamps[-1]:
        print("timestamp too large: {} >= {}".format(
            timestamp, timestamps[-1]))
        return breakpoints[-1]

    for idx in range(len(timestamps)-1):
        if timestamps[idx] <= timestamp <= timestamps[idx+1]:
            return [breakpoints[idx][coor_id] + (timestamp - timestamps[idx]) /
                    (timestamps[idx+1] - timestamps[idx]) *
                    (breakpoints[idx+1][coor_id] - breakpoints[idx][coor_id])
                    for coor_id in [0, 1]]