#<-----------------------------------------------imports---------------------------------->
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from points_addition import add_points_between_the_boundary,add_lines_in_the_boundary
from interpolation import get_curve_interpolation
import open3d as o3d
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from sklearn.neighbors import NearestNeighbors

#
# lower_part=pv.read("surface_2.stl").points
# s1=np.loadtxt("side1_2.txt")
# s2=np.loadtxt("side2_2.txt")
# p=pv.Plotter()
# p.enable_eye_dome_lighting()
# p.add_mesh(pv.PolyData(s1))
# p.add_mesh(pv.PolyData(s2))
# p.add_mesh(pv.PolyData(lower_part))
# p.show()
#
# temp = s1[s1[:, 2].argsort()]
# temp_2 = s2[s2[:, 2].argsort()]
#
# z_min_1 = temp[0, 2]
# z_min_2 = temp_2[0, 2]
# z_min=z_min_1
# if z_min_2<z_min_1:
#     z_min=z_min_2
# points_to_remove = []
# for j in range(lower_part.shape[0]):
#     if (lower_part[j, 2] > z_min):
#         points_to_remove.append(j)
# # points_to_remove=np.asarray(points_to_remove)
# source_arr = np.delete(lower_part, points_to_remove, axis=0)
#
# p=pv.Plotter()
# p.enable_eye_dome_lighting()
# p.add_mesh(pv.PolyData(s1))
# p.add_mesh(pv.PolyData(s2))
# p.add_mesh(pv.PolyData(source_arr),color="red")
# p.show()
p=pv.Plotter()
gum=pv.read("tooth_f_14.stl")
t_0=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_0.stl")
t_1=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_1.stl")
t_2=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_2.stl")
t_3=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_3.stl")
t_4=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_4.stl")
t_5=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_5.stl")
t_6=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_6.stl")
t_7=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_7.stl")
t_8=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_8.stl")
t_9=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_9.stl")
t_10=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_10.stl")
t_11=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_11.stl")
t_12=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_12.stl")
t_13=pv.read("C:/Users/orari/PycharmProjects/Fill_Tooth/complete_teeth/full_tooth_13.stl")

mesh=t_0.merge([t_0,t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8,t_9,t_10,t_11,t_12,t_13])

mesh.plot()
p.export_gltf("test.gltf")
p.show()

data_b=pv.read("tooth_f_10.stl").decimate(0.95)
data_b.save("d_tooth_f_10.stl")

data_b_2 = data_b.connectivity(largest=False)

data_b_2.plot()

# make the data

# x = 4 + np.random.normal(0, 2, 24)
# y = 4 + np.random.normal(0, 2, len(x))
# # size and color:
# sizes = np.random.uniform(15, 80, len(x))
# colors = np.random.uniform(15, 80, len(x))
# # plot
# fig, ax = plt.subplots()
# ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
#
# plt.show()