import numpy as np
import scipy
import matplotlib.pyplot as plt
import json
import os
import sys
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
import math
import random
import time
from basic_embedding import *
from cluster_method_embedding import *
# from scipy.spatial import distance
# from scipy.cluster import hierarchy
# from sklearn.metrics import pairwise_distances
# import numba as nb
# import parfor
# import multiprocessing
# from sklearn import metrics


class Cluster_centroid:
    def __init__(self, cluster_id, index_list, vector, ref_point_indicator=False, left=None, right=None):
        self.cluster_id = cluster_id
        self.index_list = index_list
        self.contain_reference_point = ref_point_indicator
        self.vector = vector
        self.left = left
        self.right = right
        self.is_leaf = self.left is None and self.right is None
        self.count = len(self.index_list)

    def check_reference_point(self, reference_matrix):
        for i in range(len(self.index_list)):
            if self.index_list[i] in reference_matrix:
                return True
        return False


# def generate_dis_matrix(dataset):
#     start = time.clock()
#     dis_matrix = pairwise_distances(dataset)
#     end = time.clock()
#     print("Time for dis matrix is:", end - start)
#     return dis_matrix


def generate_dis_matrix_centroid(dataset):
    start = time.clock()
    dis_matrix = pairwise_distances(dataset)
    end = time.clock()
    print("Time for dis matrix is:", end - start)
    return dis_matrix


def euler_dis(vec1: np, vec2):
    return np.linalg.norm(vec1 - vec2)


def get_dis_between_clusters_centroid(cluster_one, cluster_two):
    # all test: 150 points, 5 ref points

    # # 0.85, 0.04375148666666667 h
    # return distance.euclidean(cluster_one.vector, cluster_two.vector)

    # # 0.78, 0.05425135416666667 h
    # dist = [(a - b) ** 2 for a, b in zip(cluster_one.vector, cluster_two.vector)]
    # dist = math.sqrt(sum(dist))
    # return dist

    # 0.81, 0.047219791111111103 h
    return np.linalg.norm(cluster_one.vector - cluster_two.vector)

    # # 0.8, 0.04822814027777778 h
    # diff_vector = np.array(cluster_one.vector) - np.array(cluster_two.vector)
    # diff_vector = diff_vector ** 2
    # return math.sqrt(sum(diff_vector))

    # # 0.8, 0.054039783888888894 h
    # dist = 0
    # for i in range(len(cluster_one.vector)):
    #     dist += (cluster_one.vector[i] - cluster_two.vector[i]) ** 2
    # return math.sqrt(dist)


def merge_cluster_centroid(cluster_one, cluster_two, cluster_counter):
    new_index_list = []

    for i in range(len(cluster_one.index_list)):
        new_index_list.append(cluster_one.index_list[i])

    for j in range(len(cluster_two.index_list)):
        new_index_list.append(cluster_two.index_list[j])

    new_vec = (cluster_one.vector * cluster_one.count + cluster_two.vector * cluster_two.count) / (cluster_one.count + cluster_two.count)
    # new_vec = [(cluster_one.vector[i] * len(cluster_one.index_list)
    #             + cluster_two.vector[i] * len(cluster_two.index_list))
    #            / (len(cluster_one.index_list) + len(cluster_two.index_list))
    #            for i in range(len(cluster_one.vector))]
    new_ref_point_indicator = cluster_one.contain_reference_point or cluster_two.contain_reference_point

    new_cluster = Cluster_centroid(cluster_counter, new_index_list, new_vec, new_ref_point_indicator)

    return new_cluster


def clustering_method_hierarchical_centroid_dis_matrix(dataset, reference_matrix):
    # at the beginning, every point is a cluster
    cluster_counter = 0
    cluster_list = []
    for i in range(len(dataset)):
        temp_cluster = Cluster_centroid(cluster_counter, [i], dataset[i])
        temp_cluster.contain_reference_point = temp_cluster.check_reference_point(reference_matrix)
        cluster_list.append(temp_cluster)
        cluster_counter += 1

    # generate distance matrix
    dis_matrix = generate_dis_matrix_centroid(dataset)
    # set lower triangle and diagonal to inf
    for i in range(np.shape(dis_matrix)[0]):
        for j in range(np.shape(dis_matrix)[1]):
            if i >= j or (cluster_list[i].contain_reference_point and cluster_list[j].contain_reference_point):
                dis_matrix[i][j] = np.inf
    # np.fill_diagonal(dis_matrix, np.inf)

    start_while_loop = time.clock()

    while True:
        start = time.clock()
        print("\nNum of clusters:")
        print("Remaining:", len(cluster_list), ", Original:", len(dataset), ", Final:", reference_matrix.size)
        print("Rate of progress: {:.2%}".format((len(dataset) - len(cluster_list)) / (len(dataset) - reference_matrix.size)))

        # check whether each cluster has a reference point
        # if so, we are done, break the while loop
        # otherwise, we need to find the nearest 2 clusters and merge them
        each_cluster_has_a_reference_point = True
        for i in range(len(cluster_list)):
            if not cluster_list[i].contain_reference_point:
                each_cluster_has_a_reference_point = False

        if each_cluster_has_a_reference_point:
            break

        # check pairwise distance between all of the clusters and find the nearest among them
        (num_row, num_col) = np.shape(dis_matrix)
        line_index = np.argmin(dis_matrix)
        index_of_first_cluster_to_merge = int(line_index / num_col)
        index_of_second_cluster_to_merge = line_index % num_col
        # nearest_dis = dis_matrix[index_of_first_cluster_to_merge][index_of_second_cluster_to_merge]

        # merge the two clusters and refresh the cluster list
        new_cluster = merge_cluster_centroid(cluster_list[index_of_first_cluster_to_merge],
                                             cluster_list[index_of_second_cluster_to_merge],
                                             cluster_counter)
        cluster_counter += 1
        del cluster_list[index_of_second_cluster_to_merge], cluster_list[index_of_first_cluster_to_merge]
        cluster_list.append(new_cluster)

        # generate the new dis array between the new cluster and all other clusters
        new_dis_array = np.zeros((1, len(cluster_list) - 1))
        for i in range(len(cluster_list) - 1):
            new_dis_array[0][i] = get_dis_between_clusters_centroid(cluster_list[i], new_cluster)

        # update the distance matrix
        # delete 2 rows and 2 cols, delete the one with larger index first
        dis_matrix = np.delete(dis_matrix, index_of_second_cluster_to_merge, axis=0)
        dis_matrix = np.delete(dis_matrix, index_of_first_cluster_to_merge, axis=0)
        dis_matrix = np.delete(dis_matrix, index_of_second_cluster_to_merge, axis=1)
        dis_matrix = np.delete(dis_matrix, index_of_first_cluster_to_merge, axis=1)
        # add 1 new row and 1 new col
        dis_matrix = np.r_[dis_matrix, new_dis_array]
        new_dis_array = np.append(new_dis_array, 0)
        dis_matrix = np.c_[dis_matrix, new_dis_array.T]
        # set lower triangle and diagonal to inf
        for i in range(np.shape(dis_matrix)[0]):
            for j in range(np.shape(dis_matrix)[1]):
                if i >= j or (cluster_list[i].contain_reference_point and cluster_list[j].contain_reference_point):
                    dis_matrix[i][j] = np.inf

        end = time.clock()

        print("Running time of this loop is", end - start)
        print("Running time from start is", (end - start_while_loop) / 60, "min")
        print("Estimated remaining time is",
              (end - start) * (len(cluster_list) - reference_matrix.size) / 60, "min")

    end_while_loop = time.clock()
    print("\nTotal running time of while loop part is", (end_while_loop - start_while_loop) / 60, "min")

    # print the final cluster list
    print("\nThe final clusters are:")
    total_num_points = 0
    for i in range(len(cluster_list)):
        print("Cluster", i, "index list is", cluster_list[i].index_list)
    for i in range(len(cluster_list)):
        print("Cluster", i, "index list len is", len(cluster_list[i].index_list))
        total_num_points += len(cluster_list[i].index_list)
    print("Total num points is", total_num_points)

    return cluster_list


def clustering_method_hierarchical_centroid(dataset, reference_matrix):
    # # generate distance matrix
    # dis_matrix = generate_dis_matrix(dataset)
    # distance dictionary between clusters
    dis_dic = {}

    # at the beginning, every point is a cluster
    cluster_counter = 0
    cluster_list = []
    for i in range(len(dataset)):
        temp_cluster = Cluster_centroid(cluster_counter, [i], dataset[i])
        temp_cluster.contain_reference_point = temp_cluster.check_reference_point(reference_matrix)
        cluster_list.append(temp_cluster)
        cluster_counter += 1

    start_while_loop = time.clock()

    while True:
        start = time.clock()
        print("\nNum of clusters:")
        print("Remaining:", len(cluster_list), ", Original:", len(dataset), ", Final:", reference_matrix.size)
        print("Rate of progress: {:.2%}".format((len(dataset) - len(cluster_list)) / (len(dataset) - reference_matrix.size)))

        # check whether each cluster has a reference point
        # if so, we are done, break the while loop
        # otherwise, we need to find the nearest 2 clusters and merge them
        each_cluster_has_a_reference_point = True
        for i in range(len(cluster_list)):
            if not cluster_list[i].contain_reference_point:
                each_cluster_has_a_reference_point = False

        if each_cluster_has_a_reference_point:
            break

        # check pairwise distance between all of the clusters and find the nearest among them
        # initialization
        nearest_dis = math.inf
        index_of_first_cluster_to_merge = None
        index_of_second_cluster_to_merge = None

        # check all clusters in the cluster list, find the nearest pair of clusters
        for i in range(len(cluster_list) - 1):
            for j in range(i + 1, len(cluster_list)):
                # if two clusters both have a reference point, ignore
                # only consider one cluster has a reference point, or neither has a reference point
                if not (cluster_list[i].contain_reference_point and cluster_list[j].contain_reference_point):
                    d_key = (cluster_list[i].cluster_id, cluster_list[j].cluster_id)
                    if d_key not in dis_dic:
                        dis_dic[d_key] = euler_dis(cluster_list[i].vector, cluster_list[j].vector)
                        # dis_dic[d_key] = get_dis_between_clusters_centroid(cluster_list[i], cluster_list[j])

                    temp_dis = dis_dic[d_key]
                    if temp_dis < nearest_dis:
                        nearest_dis = temp_dis
                        index_of_first_cluster_to_merge = i
                        index_of_second_cluster_to_merge = j

        # merge the two clusters and refresh the cluster list
        new_cluster = merge_cluster_centroid(cluster_list[index_of_first_cluster_to_merge],
                                             cluster_list[index_of_second_cluster_to_merge],
                                             cluster_counter)
        cluster_counter += 1
        del cluster_list[index_of_second_cluster_to_merge], cluster_list[index_of_first_cluster_to_merge]
        cluster_list.append(new_cluster)

        end = time.clock()

        print("Running time of this loop is", end - start)
        print("Running time from start is", (end - start_while_loop) / 60, "min")
        print("Estimated remaining time is",
              (end - start) * (len(cluster_list) - reference_matrix.size) / 60, "min")

    end_while_loop = time.clock()
    print("\nTotal running time of while loop is", (end_while_loop - start_while_loop) / 60, "min")

    # print the final cluster list
    print("\nThe final clusters are:")
    total_num_points = 0
    for i in range(len(cluster_list)):
        print("Cluster", i, "index list is", cluster_list[i].index_list)
    for i in range(len(cluster_list)):
        print("Cluster", i, "index list len is", len(cluster_list[i].index_list))
        total_num_points += len(cluster_list[i].index_list)
    print("Total num points is", total_num_points)

    return cluster_list


def generate_pred_list_hierarchical_centroid(cluster_list, dataset, reference_matrix, ground_truth_value_list):
    pred_list = [None] * len(dataset)
    for i in range(len(cluster_list)):
        # firstly, find the reference point in this cluster
        ref_point_index = 0
        for j in cluster_list[i].index_list:
            if j in reference_matrix:
                ref_point_index = j
                break

        # secondly, the label of all points in this cluster is the same as ref point
        for k in cluster_list[i].index_list:
            # find the index of the reference point in reference matrix
            index_ref_point_in_ref_matrix = np.argwhere(reference_matrix == ref_point_index)
            # find which line it is in, and determine the ground truth
            pred_list[k] = ground_truth_value_list[index_ref_point_in_ref_matrix[0][0]]

    return pred_list


def assess_hierarchical_centroid(pred_list, ground_truth_list, ground_truth_value_list):
    # calculate accuracy by myself
    num_correct = 0
    for i in range(len(ground_truth_list)):
        if pred_list[i] == ground_truth_list[i]:
            num_correct += 1
    accuracy = num_correct / len(ground_truth_list)
    print("Accuracy calculated by myself is", accuracy)

    # print classification report
    true_label_list = []
    pred_label_list = []

    for i in range(len(ground_truth_list)):
        true_label_list.append(ground_truth_value_list.index(ground_truth_list[i]))
        pred_label_list.append(ground_truth_value_list.index(pred_list[i]))

    target_names = ground_truth_value_list
    print("The classification report is:")
    print(classification_report(true_label_list, pred_label_list, target_names=target_names))






















