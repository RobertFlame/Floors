import os
settings_file = "settings.json"
building_ids = [
                '5cd969be39e2fc0b4afe732d',
                ]

rss_threshold = -90
#==================================================================
# prerpcess_node/graph
settings_id = "preprocess"
input_dir_node = "./data/sites"
output_dir_node = "./data/sites"
input_dir_graph = output_dir_node
output_dir_graph = './data/output_data'
target_dir_graph = "raw_data"

#===================================================================
# gen_emb_eline.py
default_config = dict(
    emb_dim = 8,
    ref_num_per_floor = 4
)

root_base = os.getcwd()
edgelist_file = os.path.abspath(os.path.join(root_base, "./data/output_data/{}/raw_data/{}.edgelist"))

out_file_embedding = os.path.abspath(os.path.join(root_base, "./data/output_data/{}/embedding/dim_{}.txt"))
out_file_report = os.path.abspath(os.path.join(root_base, "./data/result/report_{}_emb_{}.txt"))

out_file_floors = os.path.abspath(os.path.join(os.getcwd(), "./data/output_data/{}/raw_data/floors.txt"))