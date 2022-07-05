# <----------------------------------imports----------------------------------------->
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from points_addition import add_lines_in_the_boundary
from interpolation import get_curve_interpolation
tooth_num=6

data = get_curve_interpolation("tooth_"+str(tooth_num))

# <---------------------------------load files ----------------------------------------->
# toot_t=pv.read("tooth_3jo.stl").subdivide(1,"loop")
# toot_t.plot()
# data_b=toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
# arr_b=np.asarray(toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).points)
# <----------------------------------create circles ----------------------------------------->
# c_section=data_b.project_points_to_plane()
# bounds=c_section.bounds
# r_x=bounds[3]-bounds[2]
# r_y=bounds[1]-bounds[0]
# z_level=5# the min hight of the initial b-box + offset
# z_min=bounds[4]/2
# abs_z=z_level-z_min
# dif=abs_z/2
# #first circle
# circle_1=pv.Circle(radius=r_y*0.5,resolution=40).translate(data_b.center)
# circle_1=circle_1.translate((0,0,-z_min))
# circle_p=np.asarray(circle_1.points)
# data=np.concatenate((circle_p,arr_b),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
# #second circle
# circle_2=pv.Circle(radius=r_y*0.45,resolution=40).translate(circle_1.center)
# circle_2=circle_2.translate((0,0,-dif))
# circle_p=np.asarray(circle_2.points)
# data=np.concatenate((circle_p,data),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
# #third circle
# circle_3=pv.Circle(radius=r_y*0.40,resolution=40).translate(circle_2.center)
# circle_3=circle_3.translate((0,0,-dif))
# circle_p=np.asarray(circle_3.points)
# data=np.concatenate((circle_p,data),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
#
# # #Fourth circle
# circle_4=pv.Circle(radius=r_y*0.35,resolution=40).translate(circle_3.center)
# circle_4=circle_4.translate((0,0,-dif))
# circle_p=np.asarray(circle_4.points)
# data=np.concatenate((circle_p,data),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
# #
# #Fourth circle
# circle_5=pv.Circle(radius=r_y*0.30,resolution=40).translate(circle_4.center)
# circle_5=circle_5.translate((0,0,-dif))
# circle_p=np.asarray(circle_5.points)
# data=np.concatenate((circle_p,data),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
# #
#
# 5 circle
# circle_l= pv.Sphere(radius=r_y*0.2,theta_resolution=10, phi_resolution=10).translate(circle_1.center)
# circle_l =circle_l.project_points_to_plane()
# circle_l=circle_l.translate((0,0,-abs_z))
# circle_p=np.asarray(circle_l.points)
# data=np.concatenate((circle_p,data),axis=0)
# pcd_test=pv.PolyData(data)
# pcd_test.plot()
# <----------------------------------fix waves ----------------------------------------->
# circle_2= pv.Circle(radius=r_x*0.8).translate(toot_t.center)
# circle_2=circle_2.translate((0,0,-4))
# circle_p_2=np.asarray(circle_2.points)
# data=np.concatenate((circle_p_2,data),axis=0)
# <-------------------------------------------------------------------------------->
data = get_sorted_arr(data, 0)
data = add_lines_in_the_boundary(data, 1)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# sort data to avoid plotting problems
x, y, z = zip(*sorted(zip(x, y, z)))

x = np.array(x)
y = np.array(y)
z = np.array(z)

import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
# spline=RBFInterpolator(data[:,:2], z, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
# spline = sp.interpolate.LSQBivariateSpline(x,y,z)
x_grid = np.linspace(min(x), max(x), len(x) + 200)
y_grid = np.linspace(min(y), max(y), len(y) + 200)
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
#

Z = spline(B1, B2)
fig = plt.figure(figsize=(15, 6))
ax = Axes3D(fig)
ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z, alpha=0.5)
ax.scatter3D(x, y, z, c='r')
plt.show()
# #
# #
# #
# #
# #
# #
# # # Make a PyVista/VTK mesh
surface = pv.StructuredGrid(B1, B2, Z)
#
# Plot it!
toot_t = pv.read("tooth_3.stl").subdivide(2, "loop")
toot_c = toot_t.clip_surface(surface)
clipped = surface.clip_surface(toot_t)
all_tooth = clipped.merge(toot_c)
all_tooth = all_tooth.triangulate()
all_tooth.save("ssss.vtk")
# all_tooth.plot()
pcd = pv.PolyData(np.asarray(surface.points))
p = pv.Plotter()
p.add_mesh(toot_c, color="red")
p.add_mesh(clipped)
p.show()
#
mesh = pv.PolyData(np.asarray(all_tooth.points), np.asarray(all_tooth.cells))
mesh.save("o.stl")
#


