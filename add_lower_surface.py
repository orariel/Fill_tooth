# <----------------------------------imports----------------------------------------->
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from points_addition import add_lines_in_the_boundary
from interpolation import get_curve_interpolation
from filling_holes_final import fill_holes_poisson
# <----------------------------------imports---------------------------------------->
tooth_num=1
file_name="d_tooth_f_"+str(tooth_num)
toot_t = pv.read(file_name+".stl")
data_b=toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
data = get_curve_interpolation("d_tooth_f_"+str(tooth_num))
data = get_sorted_arr(data,0)
p=pv.Plotter()
p.add_mesh(toot_t,color="red")
p.add_mesh(pv.PolyData(data))
p.show()
# <----------------------------------create circles ----------------------------------------->
c_section=toot_t.project_points_to_plane()
bounds=c_section.bounds
r_x=bounds[3]-bounds[2]
r_y=bounds[1]-bounds[0]
z_level=5# the min hight of the initial b-box + offset
z_min=bounds[4]/2
abs_z=z_level-z_min
dif=abs_z/2
#first circle
# circle_1=pv.Circle(radius=r_y*0.40,resolution=40).translate((toot_t.center),inplace=False)
# circle_1=circle_1.translate((0,0,-z_min-2*dif),inplace=False)

# circle_p=np.asarray(circle_1.points)
# data=np.concatenate((circle_p,data),axis=0)
# #
# # #second circle
# # circle_2=pv.Circle(radius=r_y*0.4,resolution=40).translate((circle_1.center),inplace=False)
# # circle_2=circle_2.translate((0,0,-2*dif),inplace=False)
# # circle_p=np.asarray(circle_2.points)
# # data=np.concatenate((circle_p,data),axis=0)
# # pcd_test=pv.PolyData(data)
# # pcd_test.plot()
# # # #third circle
# # circle_3=pv.Circle(radius=r_y*0.4,resolution=40).translate((circle_2.center),inplace=False)
# # circle_3=circle_3.translate((0,0,-2*dif),inplace=False)
# # circle_p=np.asarray(circle_3.points)
# # data=np.concatenate((circle_p,data),axis=0)
# # pcd_test=pv.PolyData(data)
# # pcd_test.plot()
# #
# # # #Fourth circle
# # circle_4=pv.Circle(radius=r_y*0.35,resolution=40).translate(circle_3.center)
# # circle_4=circle_4.translate((0,0,-dif))
# # circle_p=np.asarray(circle_4.points)
# # data=np.concatenate((circle_p,data),axis=0)
# # pcd_test=pv.PolyData(data)
# # pcd_test.plot()
# # #
# # #Fourth circle
# # circle_5=pv.Circle(radius=r_y*0.30,resolution=40).translate(circle_4.center)
# # circle_5=circle_5.translate((0,0,-dif))
# # circle_p=np.asarray(circle_5.points)
# # data=np.concatenate((circle_p,data),axis=0)
# # pcd_test=pv.PolyData(data)
# # pcd_test.plot()
# # # # #
# # # #
# # # 5 circle
circle_l= pv.Sphere(radius=r_y*0.2,theta_resolution=20, phi_resolution=15).translate((toot_t.center),inplace=False)
circle_l =circle_l.project_points_to_plane()
circle_l=circle_l.translate((0,0,-4*dif),inplace=False)
#
# circle_p=np.asarray(circle_l.points)
# p=pv.Plotter()
# # p.add_mesh(circle_l)
# # p.add_mesh(toot_t)
# # p.show()
# data=np.concatenate((circle_p,data),axis=0)
# # pcd_test=pv.PolyData(data)
# # pcd_test.plot()
# #
# #
# # # # #
# #

# # # <-------------------------------------------------------------------------------->

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# # sort data to avoid plotting problems
# x, y, z = zip(*sorted(zip(x, y, z)))
#
x = np.array(x)
y = np.array(y)
z = np.array(z)

import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)

x_grid = np.linspace(min(x), max(x), len(x))
y_grid = np.linspace(min(y), max(y), len(y))
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
#

Z = spline(B1, B2)
fig = plt.figure(figsize=(15, 6))
ax = Axes3D(fig)
ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z, alpha=0.5)
ax.scatter3D(x, y, z, c='r')
plt.show()

# surface = pv.StructuredGrid(B1, B2, Z)
# toot_c = toot_t.clip_surface(surface)
# clipped = surface.clip_surface(toot_c)
# all_tooth = clipped.merge(toot_c)
# all_tooth = all_tooth.triangulate()
#
# # all_tooth.plot()
# pcd = pv.PolyData(np.asarray(surface.points))
# p = pv.Plotter()
# p.add_mesh(toot_c, color="red")
# p.add_mesh(clipped)
# p.show()
# # #
#
#
mesh = pv.PolyData(np.asarray(all_tooth.points), np.asarray(all_tooth.cells))
# mesh.save(str(tooth_num)+"_1_socket.stl")
# # # #
# # fill_holes_poisson(str(tooth_num)+"_1_socket.stl",tooth_num)