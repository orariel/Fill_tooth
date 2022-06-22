#<-----------------------------------------------imports---------------------------------->
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from points_addition import add_points_between_the_boundary,add_lines_in_the_boundary



tooth = pv.read("tooth_5jo.stl")
arr_sorted = np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/gvul.txt")
# arr_b = np.asarray(tooth_b.points)
# arr_sorted = get_sorted_arr(arr_b, 0)
pcd=pv.PolyData(arr_sorted)
p__=pv.Plotter()
p__.add_mesh(pcd)
p__.add_mesh(tooth,color="red")
p__.show()
pcd.plot(eye_dome_lighting=True)

arr = add_points_between_the_boundary(arr_sorted, 40)
pcd=pv.PolyData(arr)
pcd.plot(eye_dome_lighting=True)
arr=add_lines_in_the_boundary(arr,150)
pcd=pv.PolyData(arr)
pcd.plot(eye_dome_lighting=True)

