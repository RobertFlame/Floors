from hierarchical_centroid import *
from multiprocess import Pool
from const import *

dimension_of_vector = default_config['emb_dim']
num_reference_points_each_floor = default_config['ref_num_per_floor']


def main_hierarchical_average(reference_matrix, out_file):
    global ground_truth_value_list
    cluster_list = clustering_method_hierarchical_average_new(dataset, reference_matrix)
    pred_list = generate_pred_list_hierarchical_average(cluster_list, dataset, reference_matrix, ground_truth_value_list)
    print("reference matrix size is", reference_matrix.size)
    print("reference matrix is:\n", reference_matrix)
    print("pred list is:\n", pred_list)
    print("len of pred list is", len(pred_list))
    accuracy, report = assess_hierarchical_average(pred_list, ground_truth_list, ground_truth_value_list)
    print ("Out file: {}".format(out_file))
    with open(out_file, 'a') as f_out:
        f_out.write('\n'+str(report))
    return accuracy, pred_list


# Hierarchical test starts here ===================================================================
repeat_times = 10
acc_list = []
total_pred_list = []
total_truth_list = []


def util(i):
    global embedding_filename, ground_truth_filename, dataset, reference_matrix, ground_truth_value_list, ground_truth_list
    file_path = ''
    for building_id in building_ids:
        embedding_filename = out_file_embedding.format(building_id, dimension_of_vector)
        if not os.path.exists(embedding_filename):
            continue
        ground_truth_filename = out_file_floors.format(building_id)
        out_file_undirected = out_file_report.format(building_id, dimension_of_vector)
        dataset, desired_num_of_clusters, ground_truth_list, ground_truth_value_list, num_of_samples_in_each_floor = \
            get_dataset_and_truth_list_two(file_path, dimension_of_vector, embedding_filename, ground_truth_filename)
        print(len(dataset))
        print(len(ground_truth_list))

        dataset = np.array(dataset)
        reference_matrix = generate_reference_matrix(ground_truth_list, ground_truth_value_list,
                                                     num_reference_points_each_floor)
        main_hierarchical_average(reference_matrix, out_file_undirected)


t_start = time.time()


if __name__ == '__main__':
    p = Pool(repeat_times)
    list_of_pred_list = p.map(util, range(repeat_times))
    p.close()
    p.join()

t_end = time.time()
print("Total time is", (t_end - t_start)/60, "min")