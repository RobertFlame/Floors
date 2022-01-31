import numpy as np
from prettytable import PrettyTable
from hierarchical_average import *


def get_initial_center(dataset, dimension_of_vector, ground_truth_list, ground_truth_value_list):
    separation_matrix = []

    for i in range(len(ground_truth_value_list)):
        temp_separation_list = []
        separation_matrix.append(temp_separation_list)

    for i in range(len(ground_truth_list)):
        index_in_ground_truth_value_list = ground_truth_value_list.index(ground_truth_list[i])
        separation_matrix[index_in_ground_truth_value_list].append(i)

    initial_center_index = []

    for i in range(len(ground_truth_value_list)):
        random_center_index = random.choice(separation_matrix[i])
        initial_center_index.append(random_center_index)

    initial_center_dataset = np.zeros((len(ground_truth_value_list), dimension_of_vector))

    for i in range(len(ground_truth_value_list)):
        initial_center_dataset[i] = dataset[initial_center_index[i]]

    print(initial_center_index)
    for i in range(len(ground_truth_value_list)):
        print(ground_truth_list[initial_center_index[i]])
    print(initial_center_dataset)

    return initial_center_dataset


def get_initial_center_reference_points(dataset, dimension_of_vector, reference_points_index_list):
    initial_center_dataset = np.zeros((len(reference_points_index_list), dimension_of_vector))

    for i in range(len(reference_points_index_list)):
        initial_center_dataset[i] = dataset[reference_points_index_list[i]]

    print(initial_center_dataset)
    return initial_center_dataset


def kmeans_method(dataset, desired_num_of_clusters):
    kmeans = KMeans(n_clusters=desired_num_of_clusters, n_init=1000, max_iter=300000, tol=10 ** (-8)).fit(dataset)
    labels = kmeans.labels_
    return labels


def kmeans_method_initial_center(dataset, desired_num_of_clusters, initial_center_dataset):
    kmeans = KMeans(n_clusters=desired_num_of_clusters, init=initial_center_dataset, n_init=1000, max_iter=300000,
                    tol=10 ** (-8)).fit(dataset)
    labels = kmeans.labels_
    return labels


def kmedoids_method(dataset, desired_num_of_clusters):
    kmedoids = KMedoids(n_clusters=desired_num_of_clusters, metric='cosine', init='heuristic', max_iter=3000000).fit(
        dataset)
    labels = kmedoids.labels_
    return labels


def dbscan_method(dataset, epsilon):
    dbscan = DBSCAN(eps=epsilon).fit(dataset)
    labels = dbscan.labels_
    return labels


def hierarchical_clustering_method(dataset, desired_num_of_clusters):
    hierarchical = cluster.AgglomerativeClustering(n_clusters=desired_num_of_clusters).fit(dataset)
    labels = hierarchical.labels_
    return labels


def gaussian_mixture(dataset, desired_num_of_clusters):
    labels = GaussianMixture(n_components=desired_num_of_clusters).fit_predict(dataset)
    return labels


def generate_prediction_list(labels, num_of_clusters, ground_truth_list, ground_truth_value_list):
    counter_matrix = np.zeros([num_of_clusters, len(ground_truth_value_list)])
    for i in range(len(labels)):
        for j in range(len(ground_truth_value_list)):
            if ground_truth_list[i] == ground_truth_value_list[j]:
                counter_matrix[labels[i], j] += 1
                break

    # print(counter_matrix)

    selected_index = []
    for i in range(num_of_clusters):
        selected_index.append(np.argmax(counter_matrix[i, :]))

    print("The selected index for all clusters are:")
    print(selected_index)

    selected_floor = []
    for i in range(num_of_clusters):
        selected_floor.append(ground_truth_value_list[selected_index[i]])

    print("The selected floor for all clusters are:")
    print(selected_floor)

    prediction_list = []
    for i in range(len(labels)):
        prediction_list.append(selected_floor[labels[i]])

    for i in range(len(labels)):
        if labels[i] == -1:
            prediction_list[i] = 'Noise'

    # print("The pred list is", prediction_list)
    return prediction_list


def assess(labels, ground_truth_list, prediction_list, ground_truth_value_list):
    # 1. print cluster matrix
    all_cluster_labels = list(set(labels))
    actual_num_of_clusters = len(all_cluster_labels)
    cluster_matrix = np.zeros((len(ground_truth_value_list), actual_num_of_clusters))

    for i in range(len(labels)):
        true_value = ground_truth_list[i]
        pred_cluster = labels[i]
        cluster_matrix[ground_truth_value_list.index(true_value), all_cluster_labels.index(pred_cluster)] += 1

    y = PrettyTable()
    y.add_column('Left: Truth \ Top: Cluster', ground_truth_value_list)

    for i in range(len(all_cluster_labels)):
        y.add_column(str(all_cluster_labels[i]), cluster_matrix[:, i])

    print("The cluster matrix is:")
    print(y)

    # 2. print pred_matrix
    # each row is a cluster, each column is a truth label
    # true value on the left, predicted value on the top
    prediction_value_list = list(set(prediction_list))
    all_possible_value_list = []

    if "Noise" in prediction_value_list:
        pred_matrix = np.zeros([len(ground_truth_value_list), len(ground_truth_value_list) + 1])
        all_possible_value_list = list(ground_truth_value_list)
        all_possible_value_list.append("Noise")
    else:
        pred_matrix = np.zeros([len(ground_truth_value_list), len(ground_truth_value_list)])
        all_possible_value_list = ground_truth_value_list

    num_correct_pred = 0

    for i in range(len(labels)):
        pred_value = prediction_list[i]
        true_value = ground_truth_list[i]
        pred_matrix[ground_truth_value_list.index(true_value), all_possible_value_list.index(pred_value)] += 1
        if pred_value == true_value:
            num_correct_pred += 1

    x = PrettyTable()
    x.add_column('Left: Truth \ Top: Pred', ground_truth_value_list)

    for i in range(pred_matrix.shape[1]):
        x.add_column(all_possible_value_list[i], pred_matrix[:, i])

    print("The prediction matrix is:")
    print(x)

    # 3. Calculate accuracy
    num_noise_point = 0
    for i in range(len(labels)):
        if labels[i] == -1:
            num_noise_point += 1
    num_non_noise_point = len(prediction_list) - num_noise_point

    print("Total number of points is", len(prediction_list))
    print("Number of noise points is", num_noise_point)
    print("Number of non noise points is", num_non_noise_point)
    print("Number of correctly predicted points is", num_correct_pred)

    accuracy = num_correct_pred / (len(prediction_list) - num_noise_point)
    print("Accuracy calculated by myself is", accuracy)

    # 4. print classification report
    true_label_list = []
    pred_label_list = []

    for i in range(len(ground_truth_list)):
        if labels[i] != -1:
            true_label_list.append(ground_truth_value_list.index(ground_truth_list[i]))
            pred_label_list.append(ground_truth_value_list.index(prediction_list[i]))
        # else:
        #     true_label_list.append(ground_truth_value_list.index(ground_truth_list[i]))
        #     pred_label_list.append(len(ground_truth_value_list))

    target_names = ground_truth_value_list
    # target_names.append("'Noise'")
    # print(target_names)
    print("The classification report is:")
    print(classification_report(true_label_list, pred_label_list, target_names=target_names))
    return accuracy, num_non_noise_point


def generate_prediction_list_reference_points(labels, num_of_clusters, ground_truth_list,
                                              ground_truth_value_list, reference_points_index_list):
    counter_matrix = np.zeros([num_of_clusters, len(ground_truth_value_list)])
    for i in range(len(labels)):
        if i in reference_points_index_list:
            for j in range(len(ground_truth_value_list)):
                if ground_truth_list[i] == ground_truth_value_list[j]:
                    counter_matrix[labels[i], j] += 1
                    break

    # print(counter_matrix)

    selected_index = []
    for i in range(num_of_clusters):
        selected_index.append(np.argmax(counter_matrix[i, :]))

    print("The selected index for all clusters are:")
    print(selected_index)

    selected_floor = []
    for i in range(num_of_clusters):
        selected_floor.append(ground_truth_value_list[selected_index[i]])

    print("The selected floor for all clusters are:")
    print(selected_floor)

    prediction_list = []
    for i in range(len(labels)):
        prediction_list.append(selected_floor[labels[i]])

    for i in range(len(labels)):
        if labels[i] == -1:
            prediction_list[i] = 'Noise'

    # print("The pred list is", prediction_list)
    return prediction_list


def generate_prediction_list_reference_matrix(labels, dataset, reference_matrix, ground_truth_value_list):
    label_value_list = list(set(labels))

    if -1 in label_value_list:
        actual_num_valid_clusters = len(label_value_list) - 1
    else:
        actual_num_valid_clusters = len(label_value_list)

    pred_cluster_label_list = []

    for i in range(actual_num_valid_clusters):

        # avoid noise
        if label_value_list[i] == -1:
            continue

        # get the cluster with label_value_list[i]
        temp_list = get_index(labels, label_value_list[i])
        # initialization
        nearest_dis = calculate_distance_between_clusters(dataset, temp_list, reference_matrix[0])
        nearest_reference_cluster_index = 0

        # compare the dis between this cluster and all reference clusters
        for j in range(len(ground_truth_value_list)):
            temp_dis = calculate_distance_between_clusters(dataset, temp_list, reference_matrix[j])
            if temp_dis < nearest_dis:
                nearest_dis = temp_dis
                nearest_reference_cluster_index = j

        pred_cluster_label_list.append(ground_truth_value_list[nearest_reference_cluster_index])

    # for all clusters, we have a prediction cluster label list
    # construct a list with same length as labels
    pred_list = []
    for i in range(len(labels)):
        pred_list.append('Noise')

    for i in range(actual_num_valid_clusters):
        # get the cluster with label_value_list[i]
        temp_list = get_index(labels, label_value_list[i])
        for j in temp_list:
            pred_list[j] = pred_cluster_label_list[i]

    return pred_list



















