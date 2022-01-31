import numpy as np
import os
import random


def str_to_list_embedding(vector_string):
    element_list = vector_string.split()
    node_type = element_list[0][0]
    coordinate_list = []

    for i in range(1, len(element_list)):
        coordinate_list.append(float(element_list[i]))

    return node_type, coordinate_list


def generate_dataset_embedding(lines, dimension_of_vector):
    num_rows = 0
    for i in range(len(lines)):
        node_type, coordinate_list = str_to_list_embedding(lines[i])
        if node_type == "u":
            num_rows = num_rows + 1

    dataset = np.zeros((num_rows, dimension_of_vector))
    counter = 0
    for i in range(len(lines)):
        node_type, coordinate_list = str_to_list_embedding(lines[i])
        if node_type == "u":
            dataset[counter] = coordinate_list
            counter += 1
        if counter == num_rows:
            break

    return dataset


def generate_truth_label_list_embedding(truth_string):
    truth_string = truth_string.strip('\n')
    element_list = truth_string.split(', ')
    truth_label_list = []

    for i in range(len(element_list)):
        temp_label = element_list[i].split(': ')
        if temp_label[1].endswith('}'):
            label = temp_label[1].strip('}')
        else:
            label = temp_label[1]
        truth_label_list.append(label)

    return truth_label_list


def get_dataset_and_truth_list_two(file_path, dimension_of_vector, embedding_filename, ground_truth_filename):
    ground_truth_list = []
    ground_truth_value_list = []
    num_of_samples_in_each_floor = []
    dataset = np.zeros((0, 4))

    dataset_path = file_path + embedding_filename  #'/embedding_multi_2_64_64.txt'
    truth_label_path = os.path.abspath(file_path + ground_truth_filename)  #'/floors.txt'

    with open(dataset_path) as file:
        temp_data = file.readlines()
        del (temp_data[0])
        dataset = generate_dataset_embedding(temp_data, dimension_of_vector)

    with open(truth_label_path) as file:
        temp_data = file.readlines()
        truth_string = temp_data[0]
        ground_truth_list = generate_truth_label_list_embedding(truth_string)

    ground_truth_value_list = list(set(ground_truth_list))
    num_of_clusters = len(ground_truth_value_list)

    for i in range(len(ground_truth_value_list)):
        num_of_samples_in_each_floor.append(ground_truth_list.count(ground_truth_value_list[i]))

    print("len of ground truth list is", len(ground_truth_list))
    print("ground truth value list is", ground_truth_value_list)
    print("desired num of clusters is", num_of_clusters)

    return dataset, num_of_clusters, ground_truth_list, ground_truth_value_list, num_of_samples_in_each_floor


def get_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] == item]


def generate_reference_points(ground_truth_list, ground_truth_value_list, num_reference_points_each_floor):
    reference_points_index_list = []

    for i in range(len(ground_truth_value_list)):
        # get the indices of the elements in this floor
        temp_list = get_index(ground_truth_list, ground_truth_value_list[i])
        # get a specific number of points
        random_reference_points_list = random.sample(temp_list, num_reference_points_each_floor)
        # add these points to the reference points list
        reference_points_index_list.extend(random_reference_points_list)

    return reference_points_index_list


def generate_reference_matrix(ground_truth_list, ground_truth_value_list, num_reference_points_each_floor):
    reference_points_index_matrix = np.zeros((len(ground_truth_value_list), num_reference_points_each_floor))

    for i in range(len(ground_truth_value_list)):
        # get the indices of the elements in this floor
        temp_list = get_index(ground_truth_list, ground_truth_value_list[i])
        if len(temp_list) > num_reference_points_each_floor:
            random_reference_points_list = random.sample(temp_list, num_reference_points_each_floor)
            # add these points to the reference points matrix
            reference_points_index_matrix[i] = random_reference_points_list

    return reference_points_index_matrix


# @jit(nopython=True)
def calculate_distance_between_clusters(dataset, index_list_one, index_list_two):
    sum_pairwise_distance = 0

    for i in range(len(index_list_one)):
        for j in range(len(index_list_two)):
            index_one = int(index_list_one[i])
            index_two = int(index_list_two[j])
            # # 1
            # dist = [(a - b) ** 2 for a, b in zip(dataset[index_one], dataset[index_two])]
            # dist = math.sqrt(sum(dist))
            # sum_pairwise_distance += dist
            # # 2: 0.18
            # sum_pairwise_distance += distance.euclidean(dataset[index_one], dataset[index_two])
            # 3: 0.17
            dist = np.linalg.norm(dataset[index_one] - dataset[index_two])
            sum_pairwise_distance += dist

    distance_between_clusters = sum_pairwise_distance / (len(index_list_one) * len(index_list_two))

    # 4
    # X = dataset[index_list_one]
    # Y = dataset[index_list_two]
    # distance_between_clusters = sum(sum(pairwise.euclidean_distances(X, Y))) / (len(index_list_one) * len(index_list_two))

    return distance_between_clusters

