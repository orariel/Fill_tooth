from stl import mesh
import pymeshfix as mf
from pymeshfix._meshfix import PyTMesh
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from KNN_sort import get_sorted_arr
from points_addition import add_lines_in_the_boundary
# data=np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/gvul.txt")

# mesh=o3d.io.read_triangle_mesh("ma.stl")
# pcd = mesh.sample_points_poisson_disk(300, init_factor=2, pcl=None)
# data=np.asarray(pcd.points)
toot_t=pv.read("tooth_6jo.stl").subdivide(1,"loop")

toot_t.plot()
data_b=np.asarray(toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).points)#

circle= pv.Sphere(radius=1.5,theta_resolution=15, phi_resolution=15).translate(toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).center)
circle =circle.project_points_to_plane()
circle=circle.translate((0,0,-4.8))


circle_p=np.asarray(circle.points)
data=np.concatenate((circle_p,data_b),axis=0)
# <----------------------------------fix waves ----------------------------------------->
circle_2= pv.Circle(radius=3.0).translate(toot_t.center)
circle_2=circle_2.translate((0,0,-4))
circle_p_2=np.asarray(circle_2.points)
data=np.concatenate((circle_p_2,data),axis=0)
#<-------------------------------------------------------------------------------->
# data=get_sorted_arr(data,0)
# data=add_lines_in_the_boundary(data,1)
x = data[:,0]
y = data[:,1]
z = data[:,2]

# sort data to avoid plotting problems
x, y, z = zip(*sorted(zip(x, y, z)))

x = np.array(x)
y = np.array(y)
z = np.array(z)

import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

spline = sp.interpolate.Rbf(x,y,z,function='multiquadric', smooth=1, episilon=0)
# spline=RBFInterpolator(data[:,:2], z, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
# spline = sp.interpolate.LSQBivariateSpline(x,y,z)
x_grid = np.linspace(min(x),max(x), len(x))
y_grid = np.linspace(min(y),max(y), len(y))
B1, B2= np.meshgrid(x_grid, y_grid, indexing='xy')
#

Z = spline(B1,B2)
fig = plt.figure(figsize=(15,6))
ax = Axes3D(fig)
ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z,alpha=0.5)
ax.scatter3D(x,y,z, c='r')
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

toot_c = toot_t.clip_surface(surface)
clipped = surface.clip_surface(toot_t)
all_tooth=clipped.merge(toot_t)
all_tooth=all_tooth.triangulate()
all_tooth.save("ssss.vtk")
# all_tooth.plot()
pcd=pv.PolyData(np.asarray(surface.points))
p=pv.Plotter()
p.add_mesh(toot_t,color="red")
p.add_mesh(clipped)
p.show()
#
mesh =pv.PolyData(np.asarray(all_tooth.points),np.asarray(all_tooth.cells))
mesh.save("o.stl")
#

# tooth_path = pv.read("o.stl")
# tooth = mf.MeshFix(tooth_path)
#
#
# tooth_path = pv.read("o.stl")
#
# # tooth_path.plot()
# # tooth_path=tooth_path.smooth(1000, feature_smoothing=True,boundary_smoothing=True,convergence=0.5)
# tooth = mf.MeshFix(tooth_path)
# tooth.repair(verbose=False, joincomp=True, remove_smallest_components=False)
# tooth.plot()
#
# tooth.save("o_2.stl")