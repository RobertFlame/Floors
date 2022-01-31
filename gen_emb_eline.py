import random
import tensorflow as tf
from const import *
from graph import Graph
from eline import E_LINE
from paths import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES']='0'


def clean_main(site_id, order, rep_size, negative_ratio):
    # step 1: preprocess, generate network edgelist for further processing
    site_folder = os.path.abspath(os.path.join(ROOT_FOLDER, output_dir_graph, site_id))
    site_embedding_folder = os.path.join(site_folder, "embedding")
    observation_ids = file_to_series(os.path.join(site_folder, "raw_data", "_observations_{}.pkl".format(site_id)))
    ap_ids = file_to_series(os.path.join(site_folder, "raw_data", "_APs_{}.pkl".format(site_id)))
    pruned_observation_ids = file_to_series(os.path.join(site_folder, "raw_data", "_observations_prune_{}.pkl".format(site_id)))
    prunde_ap_ids = file_to_series(os.path.join(site_folder, "raw_data", "_APs_prune_{}.pkl".format(site_id)))

    # edgelist_file = gen_partial_edgelist_file(site_id, selected_ids, percent)
    edgelist_file = os.path.join(site_folder, "raw_data", "{}.edgelist".format(site_id))
    if not os.path.isfile(edgelist_file):
        generate_graph_file(edgelist_file, observation_ids, ap_ids)

    # step 2: E-LINE
    embedding_file = os.path.join(site_embedding_folder, "dim_{}.txt".format(rep_size))
    if not os.path.isfile(embedding_file):
        g = Graph()
        g.read_edgelist(filename=edgelist_file, weighted=True)
        model = E_LINE(g, epoch=100, rep_size=rep_size, order=order, negative_ratio=negative_ratio)
        model.save_embeddings(embedding_file)


if __name__ == "__main__":
    order = 2
    rep_size = 8
    negative_ratio = 4

    with tf.device('/GPU:0'):
        for idx0 in range(len(building_ids)):
            site_folder = os.path.abspath(os.path.join(ROOT_FOLDER, output_dir_graph, building_ids[idx0]))
            clean_main(building_ids[idx0], order, rep_size, negative_ratio)