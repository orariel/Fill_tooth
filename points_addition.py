import numpy as np
import pyvista as pv


def add_points_between_the_boundary(order_list, numOFpoints):
    arr_sorted = order_list
    points_to_add = []
    points_to_add = []
    for i in range(order_list.shape[0] - 1):
        line = pv.Line(pointa=arr_sorted[i], pointb=arr_sorted[i + 1], resolution=numOFpoints)
        points_to_add.append(arr_sorted[i])
        for j in range(numOFpoints):
            points_to_add.append(line.points[j])
        points_to_add.append(arr_sorted[i + 1])
    points_to_add = np.concatenate(points_to_add).reshape(len(points_to_add), 3)
    return points_to_add


def add_lines_in_the_boundary(points_to_add, numOFlines):
    max_index = np.argmax(points_to_add[:, 2], axis=0)
    lines_to_add = []
    half = int(points_to_add.shape[0] / 2)
    j = 0
    # for index in range(half - 1):
    for index in range(90):
        if (max_index + index < points_to_add.shape[0] - 1):
            line = pv.Line(pointa=points_to_add[max_index - index], pointb=points_to_add[max_index + index],
                           resolution=numOFlines)
            lines_to_add.append(line.points)

        if (max_index + index >= points_to_add.shape[0] - 1):
            line = pv.Line(pointa=points_to_add[j], pointb=points_to_add[max_index - index], resolution=numOFlines)
            lines_to_add.append(line.points)
            j += 1
    lines_to_add = np.concatenate(lines_to_add)
    all_points = np.concatenate((lines_to_add, points_to_add), axis=0)
    return all_points

