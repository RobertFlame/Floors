from numba import prange
from cluster_method_embedding import *
from sklearn.metrics import pairwise_distances
import numba
from sklearn.metrics import classification_report


class Cluster_average:
    def __init__(self, cluster_id, index_list, ref_point_indicator=False, left=None, right=None):
        self.id = cluster_id
        self.index_list = index_list
        self.contain_reference_point = ref_point_indicator
        self.left = left
        self.right = right
        self.is_leaf = self.left is None and self.right is None
        self.count = len(self.index_list)

    def check_reference_point(self, reference_matrix):
        for i in range(len(self.index_list)):
            if self.index_list[i] in reference_matrix:
                return True
        return False


def cluster_average_deep_copy(old_cluster):
    return Cluster_average(old_cluster.id, old_cluster.index_list, old_cluster.contain_reference_point,
                           old_cluster.left, old_cluster.right)


def generate_dis_matrix_average(dataset):
    dis_matrix = pairwise_distances(dataset)
    return dis_matrix


def generate_util_dis_matrix_average(dataset):
    dis_matrix = pairwise_distances(dataset)
    util_dis_matrix = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            util_dis_matrix[i][j] = math.exp(-dis_matrix[i][j])
    return util_dis_matrix


# for simplicity, write cluster_one as c_one, cluster_two as c_two
def get_cluster_dis_average(cluster_one, cluster_two, dis_matrix, dis_dic):
    if (cluster_one.id, cluster_two.id) in dis_dic:
        return dis_dic[(cluster_one.id, cluster_two.id)]
    elif (cluster_two.id, cluster_one.id) in dis_dic:
        return dis_dic[(cluster_two.id, cluster_one.id)]
    else:
        if cluster_one.is_leaf and cluster_two.is_leaf:
            return dis_matrix[cluster_one.index_list[0]][cluster_two.index_list[0]]

        if cluster_one.is_leaf and not cluster_two.is_leaf:
            dis_one = get_cluster_dis_average(cluster_one, cluster_two.left, dis_matrix,
                                              dis_dic) * cluster_one.count * cluster_two.left.count
            dis_two = get_cluster_dis_average(cluster_one, cluster_two.right, dis_matrix,
                                              dis_dic) * cluster_one.count * cluster_two.right.count
            return (dis_one + dis_two) / (cluster_one.count * cluster_two.count)

        if not cluster_one.is_leaf and cluster_two.is_leaf:
            dis_one = get_cluster_dis_average(cluster_one.left, cluster_two, dis_matrix,
                                              dis_dic) * cluster_one.left.count * cluster_two.count
            dis_two = get_cluster_dis_average(cluster_one.right, cluster_two, dis_matrix,
                                              dis_dic) * cluster_one.right.count * cluster_two.count
            return (dis_one + dis_two) / (cluster_one.count * cluster_two.count)

        if not cluster_one.is_leaf and not cluster_two.is_leaf:
            dis_one = get_cluster_dis_average(cluster_one.left, cluster_two.left, dis_matrix,
                                              dis_dic) * cluster_one.left.count * cluster_two.left.count
            dis_two = get_cluster_dis_average(cluster_one.left, cluster_two.right, dis_matrix,
                                              dis_dic) * cluster_one.left.count * cluster_two.right.count
            dis_three = get_cluster_dis_average(cluster_one.right, cluster_two.left, dis_matrix,
                                                dis_dic) * cluster_one.right.count * cluster_two.left.count
            dis_four = get_cluster_dis_average(cluster_one.right, cluster_two.right, dis_matrix,
                                               dis_dic) * cluster_one.right.count * cluster_two.right.count
            return (dis_one + dis_two + dis_three + dis_four) / (cluster_one.count * cluster_two.count)


def get_dis_between_new_and_old(old_cluster_index, new_cluster_left_index, new_cluster_right_index,
                                dis_matrix, cluster_list):
    if old_cluster_index < new_cluster_left_index:
        dis_old_new_left = dis_matrix[old_cluster_index][new_cluster_left_index]
    else:
        dis_old_new_left = dis_matrix[new_cluster_left_index][old_cluster_index]

    if old_cluster_index < new_cluster_right_index:
        dis_old_new_right = dis_matrix[old_cluster_index][new_cluster_right_index]
    else:
        dis_old_new_right = dis_matrix[new_cluster_right_index][old_cluster_index]

    old_count = cluster_list[old_cluster_index].count
    new_left_count = cluster_list[new_cluster_left_index].count
    new_right_count = cluster_list[new_cluster_right_index].count
    sum_of_pairwise_dis = (dis_old_new_left * old_count * new_left_count +
                           dis_old_new_right * old_count * new_right_count)
    total_count = old_count * (new_left_count + new_right_count)

    return sum_of_pairwise_dis / total_count


def merge_cluster_average(cluster_one, cluster_two, cluster_counter):
    new_index_list = []

    for i in cluster_one.index_list:
        new_index_list.append(i)

    for j in cluster_two.index_list:
        if j not in new_index_list:
            new_index_list.append(j)

    new_ref_point_indicator = cluster_one.contain_reference_point or cluster_two.contain_reference_point
    left_cluster = cluster_average_deep_copy(cluster_one)
    right_cluster = cluster_average_deep_copy(cluster_two)
    new_cluster = Cluster_average(cluster_counter, new_index_list, new_ref_point_indicator, left_cluster, right_cluster)

    return new_cluster


def k2ij(k, n):
    """
    This function will convert the index in upper triangle back into the indices in the original matrix

    Originally, the upper triangular portion of a matrix, offset above the diagonal,
    is stored as a linear array

    If we have the index k of some element in the 1-D array, this function can transfer it back
    into the (i,j) indices of the original matrix

    For example, the linear array [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] is storage in the matrix below:

    0  a0  a1  a2  a3
    0   0  a4  a5  a6
    0   0   0  a7  a8
    0   0   0   0  a9
    0   0   0   0   0

    And we want to know the (i,j) index in the array corresponding to an offset in the linear matrix
    This function, k2ij(int k, int n) -> (int, int), would satisfy:

    k2ij(k=0, n=5) = (0, 1)
    k2ij(k=1, n=5) = (0, 2)
    k2ij(k=4, n=5) = (1, 2)
    k2ij(k=5, n=5) = (1, 3)

    """
    i = n - 2 - int(math.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
    return int(i), int(j)


def remove_ij(x, i, j):
    # Row i and column j divide the array into 4 quadrants
    y = x[:-1, :-1]
    y[:i, j:] = x[:i, j+1:]
    y[i:, :j] = x[i+1:, :j]
    y[i:, j:] = x[i+1:, j+1:]
    return y


def fast_triu_indices(dim, k=0):
    """
    A fast way of finding upper triangle indices
    """
    tmp_range = np.arange(dim-k)
    rows = np.repeat(tmp_range, (tmp_range+1)[::-1])

    cols = np.ones(rows.shape[0], dtype=np.int)
    inds = np.cumsum(tmp_range[1:][::-1]+1)

    np.put(cols, inds, np.arange(dim*-1+2+k, 1))
    cols[0] = k
    np.cumsum(cols, out=cols)
    return (rows, cols)


def util(dis_matrix):
    line_index = np.argmin(dis_matrix)
    return line_index


@numba.njit
def upper_min(m):
    x = np.inf
    min_r = None
    min_c = None
    for r in prange(0, m.shape[0] - 1):
        for c in prange(r + 1, m.shape[1]):
            if m[r, c] < x:
                x = m[r, c]
                min_r = r
                min_c = c

    return min_r, min_c


def clustering_method_hierarchical_average_new(dataset, reference_matrix):
    # at the beginning, every point is a cluster
    # initialize cluster counter and cluster list
    cluster_counter = 0
    cluster_list = []
    for i in range(len(dataset)):
        temp_cluster = Cluster_average(cluster_counter, [i])
        temp_cluster.contain_reference_point = temp_cluster.check_reference_point(reference_matrix)
        cluster_list.append(temp_cluster)
        cluster_counter += 1

    # generate distance matrix
    dis_matrix = generate_dis_matrix_average(dataset)
    num_dim = list(dis_matrix.shape)[0]

    # set lower triangle and diagonal to inf
    # if 2 cluster both has ref point, set dis to inf
    for i in range(num_dim):
        for j in range(num_dim):
            if i >= j or (cluster_list[i].contain_reference_point and cluster_list[j].contain_reference_point):
                dis_matrix[i][j] = np.inf
    # idx_triu = np.triu_indices(num_dim, 1)
    num_valid_clusters = len(cluster_list)

    while True:
        print("\nNum of clusters:")
        print("Remaining:", num_valid_clusters, ", Original:", len(dataset), ", Final:", reference_matrix.size)
        print("Rate of progress: {:.2%}".format((len(dataset) - num_valid_clusters) / (len(dataset) - reference_matrix.size)))

        # check whether each cluster has a reference point
        # if so, we are done, break the while loop
        # otherwise, we need to find the nearest 2 clusters and merge them
        each_cluster_has_a_reference_point = True
        for i in range(len(cluster_list)):
            if cluster_list[i] is None:
                continue
            if not cluster_list[i].contain_reference_point:
                each_cluster_has_a_reference_point = False

        if each_cluster_has_a_reference_point:
            break

        # check pairwise distance between all of the clusters and find the nearest among them
        # line_index = np.argpartition(dis_matrix.ravel(), 1)[:1]
        # index_2d = np.unravel_index(line_index, dis_matrix.shape)
        # idx_c1 = index_2d[0][0]
        # idx_c2 = index_2d[1][0]

        # t7 = time.clock()
        # indices_triu = fast_triu_indices(num_dim, 1)
        # t8 = time.clock()
        # upper_dis_matrix = dis_matrix[indices_triu]
        # line_index = np.argmin(upper_dis_matrix)
        # t9 = time.clock()
        # idx_c1, idx_c2 = k2ij(line_index, num_dim)
        # print("Time for ut", t8 - t7)
        # print("Time for argmin in ut", t9 - t8)

        # i = np.argwhere(dis_matrix == np.min(dis_matrix))
        # idx_c1 = i[0][0]
        # idx_c2 = i[0][1]

        # idx_c1, idx_c2 = divmod(dis_matrix.argmin(), dis_matrix.shape[1])

        # find the nearest 2 clusters c1 c2, here idx_c1 < idx_c2
        # line_index = np.argmin(dis_matrix)
        # idx_c1 = int(line_index / num_dim)
        # idx_c2 = line_index % num_dim

        idx_c1, idx_c2 = upper_min(dis_matrix)

        # merge the two clusters
        new_cluster = merge_cluster_average(cluster_list[idx_c1], cluster_list[idx_c2], cluster_counter)

        # generate the new dis array between the new cluster and all other clusters
        new_dis_array = np.zeros(num_dim)
        # print(type(new_dis_array))
        for i in range(len(cluster_list)):
            if cluster_list[i] is None:
                new_dis_array[i] = np.inf
            else:
                if i == idx_c1 or i == idx_c2:
                    new_dis_array[i] = np.inf
                if new_cluster.contain_reference_point and cluster_list[i].contain_reference_point:
                    new_dis_array[i] = np.inf
                new_dis_array[i] = get_dis_between_new_and_old(i, idx_c1, idx_c2, dis_matrix, cluster_list)

        # new_dis_array = torch.from_numpy(new_dis_array)

        # update the cluster list
        cluster_counter += 1
        num_valid_clusters -= 1
        cluster_list[idx_c1] = new_cluster
        cluster_list[idx_c2] = None

        # update the distance matrix
        # delete 2 rows and 2 cols, add 1 new row and 1 new col
        dis_matrix[idx_c1] = new_dis_array
        dis_matrix[:, idx_c1] = new_dis_array.T
        dis_matrix[idx_c1, 0:idx_c1] = np.inf
        dis_matrix[idx_c1::, idx_c1] = np.inf

        dis_matrix[idx_c2] = np.inf
        dis_matrix[:, idx_c2] = np.inf

    # generate the valid cluster list, del all None
    cluster_ptr = len(cluster_list) - 1
    while cluster_ptr >= 0:
        if cluster_list[cluster_ptr] is None:
            del cluster_list[cluster_ptr]
        cluster_ptr -= 1

    # print the final cluster list
    print("\nFinal num of clusters is", len(cluster_list))
    print("The final clusters are:")
    for i in range(len(cluster_list)):
        print("Cluster", i, "index list is", cluster_list[i].index_list)

    return cluster_list


def generate_pred_list_hierarchical_average(cluster_list, dataset, reference_matrix, ground_truth_value_list):
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


def assess_hierarchical_average(pred_list, ground_truth_list, ground_truth_value_list):
    print(f"pred_list length: {len(pred_list)}")
    print(f"ground_truth_list length: {len(ground_truth_list)}")
    min_len = min(len(pred_list), len(ground_truth_list))
    pred_list = pred_list[:min_len]
    ground_truth_list = ground_truth_list[:min_len]
    num_correct = 0
    for i in range(len(ground_truth_list)):
        if pred_list[i] == ground_truth_list[i]:
            num_correct += 1
    accuracy = num_correct / len(ground_truth_list)

    # print classification report
    true_label_list = []
    pred_label_list = []

    for i in range(len(ground_truth_list)):
        true_label_list.append(ground_truth_value_list.index(ground_truth_list[i]))
        pred_label_list.append(ground_truth_value_list.index(pred_list[i]))

    target_names = ground_truth_value_list
    print("The classification report is:")
    report = classification_report(true_label_list, pred_label_list, output_dict=True)
    print(report)
    return accuracy, report






















