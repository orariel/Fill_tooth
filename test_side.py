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
from pymeshfix._meshfix import PyTMesh
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from sklearn.neighbors import NearestNeighbors
p=pv.Plotter()

tooth=pv.read("tooth_f_6.stl")

all_data=np.loadtxt("sides_6_nonside.txt")

# all_data=np.concatenate((arr_1,arr_2))

pcd=pv.PolyData(all_data)
p.add_mesh(tooth)
p.add_mesh(pcd,color="red")
p.show()
#
# # #
#<-----------------------------------------------Create lower surface---------------------------------->
circle_l= pv.Sphere(radius=5*0.2,theta_resolution=20, phi_resolution=15).translate((pcd.center),inplace=False)
circle_l =circle_l.project_points_to_plane()
circle_l=circle_l.translate((0,0,-6*0.8),inplace=False)
circle_p=np.asarray(circle_l.points)
all_data=np.concatenate((all_data,circle_p))
data=np.concatenate((circle_p,all_data),axis=0)
pv.PolyData(data).plot(eye_dome_lighting=True)
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])
#
import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
x_grid = np.linspace(min(x), max(x), len(x))
y_grid = np.linspace(min(y), max(y), len(y))
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
Z = spline(B1, B2)
surface = pv.StructuredGrid(B1, B2, Z)
toot_c = tooth.clip_surface(surface)
clipped = surface.clip_surface(toot_c)

total_tooth_witout_sides=toot_c.merge(clipped)
p__=pv.Plotter()
p__.add_mesh(clipped,color="red")
p__.add_mesh(toot_c)
p__.show()
mesh = pv.PolyData(np.asarray(clipped.points), np.asarray(clipped.cells)).triangulate().decimate(0.955)
mesh.save("surface_6.stl")


#<-----------------------------------------------fill sides way 1 ---------------------------------->
meshes = o3d.io.read_triangle_mesh("surface_6.stl")
meshes.compute_triangle_normals()
number = 300 # TO MODIFY IF NECESSARY
source = meshes.sample_points_uniformly(number_of_points=number)
o3d.visualization.draw_geometries([source])
source_arr = np.array(source.points)
data=np.loadtxt("data_b.txt")
data=get_sorted_arr(data,0)
data=add_points_between_the_boundary(data,1)
data=np.concatenate((source_arr,data))

pv.PolyData(data).plot()
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])
#
import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D

spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
x_grid = np.linspace(min(x), max(x), len(x))
y_grid = np.linspace(min(y), max(y), len(y))
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
Z = spline(B1, B2)
surface = pv.StructuredGrid(B1, B2, Z)
clip=surface.clip_surface(tooth)
tooth_c=tooth.clip_surface(surface)
__=pv.Plotter()
__.add_mesh(clip,color="red")
__.add_mesh(tooth_c)
__.show()



#<-----------------------------------------------fill sides way 2 ---------------------------------->
arr_s1=np.loadtxt("side1_6.txt")
arr_s2=np.loadtxt("side2_6.txt")
arr_s1=get_sorted_arr(arr_s1,0)

arr_s1=add_points_between_the_boundary(arr_s1,2)
 #first side
pcd_1=pv.PolyData(arr_s1)
sur_1=pcd_1.delaunay_2d().triangulate()
sur_1.plot()
sur_1.save("side_1_sur.stl")
meshes = o3d.io.read_triangle_mesh("side_1_sur.stl")
meshes.compute_triangle_normals()
number = 1000 # TO MODIFY IF NECESSARY
source_1 = meshes.sample_points_uniformly(number_of_points=number)
# o3d.visualization.draw_geometries([source_1])

# secound side
pcd_2=pv.PolyData(arr_s2)
sur_2=pcd_2.delaunay_2d().triangulate()
sur_2.plot()
sur_2.save("side_2_sur.stl")
meshes = o3d.io.read_triangle_mesh("side_2_sur.stl")
meshes.compute_triangle_normals()
number = 1000 # TO MODIFY IF NECESSARY
source_2 = meshes.sample_points_uniformly(number_of_points=number)
o3d.visualization.draw_geometries([source_2,source_1])

arr_side_1_fill=np.asarray(source_1.points)
arr_side_2_fill=np.asarray(source_2.points)

arr_tot=np.concatenate((arr_side_2_fill,arr_side_1_fill),axis=0)
pv.PolyData(arr_tot).plot()

#add lower part
meshes = o3d.io.read_triangle_mesh("surface_6.stl")
meshes.compute_triangle_normals()
number = 8000 # TO MODIFY IF NECESSARY
source = meshes.sample_points_uniformly(number_of_points=number)
# o3d.visualization.draw_geometries([source])
source_arr = np.array(source.points)
data=np.loadtxt("data_b.txt")
data=get_sorted_arr(data,0)

data=np.concatenate((source_arr,data))
arr_tot=np.concatenate((data,arr_tot))
pv.PolyData(arr_tot).plot(eye_dome_lighting=True)

meshes = o3d.io.read_triangle_mesh("tooth_f_6.stl")
meshes.compute_triangle_normals()
number = 15000 # TO MODIFY IF NECESSARY
source_1 = meshes.sample_points_uniformly(number_of_points=number)
o3d.visualization.draw_geometries([source_1])
arr_tooth=np.asarray(source_1.points)
arr_tot=np.concatenate((arr_tooth,arr_tot))
#<------------------------------------Poisson------------------------------------------------------->
tengent_plane = 100
depth_mesh = 9
density =0.0000001  # TO MODIFY IF NECESSARY
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(arr_tot)
source.estimate_normals()

source.orient_normals_consistent_tangent_plane(tengent_plane)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth_mesh)
vertices_to_remove = densities < np.quantile(densities, density)  # 0.059
mesh.remove_vertices_by_mask(vertices_to_remove)
mesh.compute_triangle_normals()
mesh.compute_triangle_normals()

o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("tooth_f_" + str(density) + ".stl", mesh)
#
size=density
mfix = PyTMesh(False)  # False removes extra verbose output
mfix.load_file("tooth_f_" + str(size) + ".stl")
mfix.fill_small_boundaries(nbe=150, refine=True)
vert, faces = mfix.return_arrays()
triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
triangles[:, -3:] = faces
triangles[:, 0] = 3
mfix.save_file("tooth_f_" + str(size) + ".stl")

mesh = pv.PolyData("tooth_f_" + str(size) + ".stl")
mesh = mesh.decimate(0.8)
mesh.save("d_tooth_f_" + str(size) + ".stl")
# mesh=mesh.smooth(200)
mesh.plot()


