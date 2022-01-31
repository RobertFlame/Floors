import os
import pickle

ROOT_FOLDER = os.getcwd()

def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))


def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series

def load_original(site):
    observation_ids = file_to_series(os.path.join(site, "original", "_observations.pkl"))
    ap_ids = file_to_series(os.path.join(site,"original","_APs.pkl"))
    return observation_ids, ap_ids

def load_test(site):
    observation_ids = file_to_series(os.path.join(site, "test", "_observations_test.pkl"))
    ap_ids = file_to_series(os.path.join(site, "test", "_APs_test.pkl"))
    return observation_ids, ap_ids

def output_file(obj, site, filename):
    series_to_file(obj, os.path.join(site, filename))

def load_file(site, filename):
    return file_to_series(os.path.join(site, filename))