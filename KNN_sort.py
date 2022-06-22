import numpy as np
from math import sqrt
import copy
import matplotlib.pyplot as plt
import pyvista as pv




# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    return sqrt(((row1[0] - row2[0] )**2 ) +((row1[1] - row2[1] )**2 ) +((row1[2] - row2[2] )**2))


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append(dist)

    ind = np.argmin(distances)
    neighbors = train[int(ind)]



    return neighbors

def get_sorted_arr(arr,index):

    dataset= copy.deepcopy(arr)
    neighbors = []  # this list aims to collect the neighbors data
    order_list = []  # this list aims to collect the order labels according to distances


    index_first =index  # first index (one of side minimum), for exemple index_first =10
    order_list.append(index_first)
    dataset_removed = copy.deepcopy \
        (dataset)  # this dataset will be the same of your original one but you will replace values during the process
    dataset_removed[order_list[0]] = [0 ,0 ,0]
    # neighbors1 = get_neighbors(train = dataset_removed, test_row = dataset[order_list[0]], num_neighbors=1,distances=[],neighbors=[])
    neighbors1 = get_neighbors(dataset_removed, dataset[order_list[0]] ,1)

    # To find the index of the dataset for the values of the neighbors


    cond1 = np.logical_and(dataset[: ,0 ]==neighbors1[0], dataset[: ,1 ]==neighbors1[1])
    cond2 = np.logical_and(cond1, dataset[: ,2 ]==neighbors1[2])
    index_pos = np.where(cond2)

    # add this index in the list
    order_list.append(index_pos[0][0])
    dataset_removed[order_list[1]] = [0 ,0 ,0]

    # make the same thing in the following loop for
    for index_range in range(1 ,len(dataset)):
        # print("Loop step : " ,index_range)
        dataset_removed[order_list[index_range]] = [0 ,0 ,0]
        # neighbors2 = get_neighbors(train = dataset_removed, test_row = dataset[order_list[index_range]], num_neighbors=1,neighbors = [],distances=[])
        neighbors2 = get_neighbors(dataset_removed, dataset[order_list[index_range]], 1)



        cond1 = np.logical_and(dataset[: ,0 ]==neighbors2[0], dataset[: ,1 ]==neighbors2[1])
        cond2 = np.logical_and(cond1, dataset[: ,2 ]==neighbors2[2])
        index_pos = np.where(cond2)

        if index_range == len(dataset ) -1:
            break

        # print("Index of the neighbor find : ", index_pos[0][0])
        order_list.append(index_pos[0][0])
        # print("Values of the neigbor find : " ,neighbors2[0])
        # print("_____ ______ _____ ______ ______ _____ _____ ______ ____ _____")

    # print(order_list)

    arr_list =list(dataset)
    arr_list2 = [arr_list[i] for i in order_list]
    # print(len(arr_list2))

    for i in range(len(arr_list2) - 1):
        dist = euclidean_distance(arr_list2[i], arr_list2[i + 1])
        # print(dist)
        if dist > 5:
            arr_list2[i] = [0, 0, 0]
            arr_list2[i + 1] = [0, 0, 0]

    arr_list3 = np.array(arr_list2)
    arr_list3 = arr_list3[~np.all(arr_list3 == 0, axis=1)]  # arr_list3 its you point cloud after sorting
    return arr_list3
# colors = [i for i in range(len(arr_list3))]
# plt.scatter(arr_list3[:, 0], arr_list3[:, 1], c=colors[:])
# plt.show()