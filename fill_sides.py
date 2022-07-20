# <-----------------------------imports---------------------------->
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from points_addition import add_lines_in_the_boundary
from points_addition import add_points_between_the_boundary
from interpolation import get_curve_interpolation
from filling_holes_final import fill_holes_poisson
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from filling_holes_final import fill_holes_poisson_pcd
import scipy as sp
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D
from pymeshfix._meshfix import PyTMesh


# <-----------------------------load tooth and cut tooth gum---------------------------->

tooth_num=8
gum=pv.read("tooth_f_14.stl")
surface=gum
tooth=pv.read("d_tooth_f_"+str(tooth_num)+".stl")
# clipped = surface.clip_surface(tooth)
# surface=clipped
# <----------------------------Curve Interpolation---------------------------->
p=pv.Plotter()

data = get_curve_interpolation("d_tooth_f_"+str(tooth_num))
pv.PolyData(data).plot()
np.savetxt("data_b"+str(tooth_num)+".txt",data)
data = get_sorted_arr(data,0)
p.enable_eye_dome_lighting()
# p.add_mesh(gum)
# p.add_mesh(tooth)
tooth_b=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

tooth_b=tooth_b.connectivity(largest=True)

p.add_mesh(tooth_b,color="red")
p.show()
gum_arr=np.asarray(gum.points)
arr=[]
#
# # <----------------------------Locating the overlapping sides with another tooth---------------------------->
sphere =pv.PolyData(data)
plane = gum
_ = sphere.compute_implicit_distance(plane, inplace=True)
dist = sphere['implicit_distance']

pl = pv.Plotter()
_ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
_ = pl.add_mesh(plane, color='w', style='wireframe')
pl.show()
points_poly =gum

dist = sphere['implicit_distance']
index=[]
for i in range(data.shape[0]):
    if(abs(dist[i])<0.4):
        index.append(data[i,:])

non_sides=np.asarray(index)
np.savetxt(str(tooth_num)+"_nonside.txt",non_sides)

index=[]
for i in range(data.shape[0]):
    if(abs(dist[i])>0.4):
        index.append(data[i,:])
sides=np.asarray(index)
pl=pv.Plotter()
pl.add_mesh(sides,color="red")
pl.add_mesh(gum)# pv.PolyData(corrected_b).plot(
pl.show()
# <----------------------------clustering and separate---------------------------->
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(sides)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.8, min_points=3, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])
indexes_1=[]
indexes_2=[]
sides=np.asarray(pcd.points)
if (max_label<1):
   # np.savetxt("only_1_side"+str(tooth_num)+".txt", sides)
 ##<-----------------------------side 1---------------------------->
    line_1=pv.Line(pointa=sides[0],pointb=sides[-1],resolution=100)
    line_1_arr=np.asarray((line_1).points)
    side_1=np.concatenate((sides,line_1_arr),axis=0)
    pcd=pv.PolyData(side_1)
    sur = pcd.delaunay_2d().triangulate()
    sur.plot()
    sur.save("only_1_side_mesh" + str(tooth_num) + ".stl")
    all_data = np.loadtxt(str(tooth_num) + "_nonside.txt")
    p = pv.Plotter()
    pcd = pv.PolyData(all_data)
    p.add_mesh(tooth)
    p.add_mesh(pcd, color="red")
    p.show()
    circle_l = pv.Sphere(radius=5 * 0.2, theta_resolution=20, phi_resolution=15).translate((pcd.center), inplace=False)
    circle_l = circle_l.project_points_to_plane()
    circle_l = circle_l.translate((0, 0, -6 * 0.8), inplace=False)
    circle_p = np.asarray(circle_l.points)
    all_data = np.concatenate((all_data, circle_p))
    data = np.concatenate((circle_p, all_data), axis=0)
   # <-----------------------------------------------RBF interpolation---------------------------------->
    pv.PolyData(data).plot(eye_dome_lighting=True)
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    z = np.array(data[:, 2])
    spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
    x_grid = np.linspace(min(x), max(x), len(x))
    y_grid = np.linspace(min(y), max(y), len(y))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = spline(B1, B2)
    surface = pv.StructuredGrid(B1, B2, Z)
    toot_c = tooth.clip_surface(surface)
    clipped = surface.clip_surface(toot_c)
    total_tooth_witout_sides = toot_c.merge(clipped)
    p__ = pv.Plotter()
    p__.add_mesh(clipped, color="red")
    p__.add_mesh(toot_c)
    p__.show()
    mesh = pv.PolyData(np.asarray(clipped.points), np.asarray(clipped.cells)).triangulate().decimate(0.955)
    mesh.save("surface_" + str(tooth_num) + ".stl")
   #  <-----------------------------------------------Add lower part---------------------------------->
    meshes = o3d.io.read_triangle_mesh("surface_" + str(tooth_num) + ".stl")
    meshes.compute_triangle_normals()
    number = 8000  # TO MODIFY IF NECESSARY
    source = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source])
    source_arr = np.array(source.points)
    data = np.loadtxt("data_b" + str(tooth_num) + ".txt")
    data = get_sorted_arr(data, 0)
    data = np.concatenate((source_arr, data))

    arr_tot = np.concatenate((data, sides))
    pv.PolyData(arr_tot).plot(eye_dome_lighting=True)

    meshes = o3d.io.read_triangle_mesh("tooth_f_" + str(tooth_num) + ".stl")
    meshes.compute_triangle_normals()
    number = 15000  # TO MODIFY IF NECESSARY
    source_1 = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source_1])
    arr_tooth = np.asarray(source_1.points)
    arr_side_1_fill=pv.read("only_1_side_mesh" + str(tooth_num) + ".stl").points

    arr_tot = np.concatenate((arr_tooth, data))
    pv.PolyData(arr_tot).plot(color="blue")


   #  <-----------------------------------------------remove points from lower surface---------------------------------->
    lower_part = source_arr
    s1 = np.asarray(sur.points)

    p = pv.Plotter()
    p.enable_eye_dome_lighting()
    p.add_mesh(pv.PolyData(s1))

    p.add_mesh(pv.PolyData(lower_part))
    p.show()
    temp = s1[s1[:, 2].argsort()]
    z_min = temp[0, 2]
    points_to_remove = []
    for j in range(lower_part.shape[0]):
        if (lower_part[j, 2] > z_min):
            points_to_remove.append(j)

    source_arr = np.delete(lower_part, points_to_remove, axis=0)
    p = pv.Plotter()
    p.enable_eye_dome_lighting()
    p.add_mesh(pv.PolyData(s1))
    p.add_mesh(pv.PolyData(source_arr), color="red")
    p.show()
    pv.PolyData(arr_tot).plot(eye_dome_lighting=True)
    arr_tot = np.concatenate((source_arr, arr_tooth))
    arr_tot = np.concatenate((s1, arr_tot))
    pv.PolyData(arr_tot).plot(eye_dome_lighting=True,color="green")
    # <------------------------------------Poisson------------------------------------------------------->
    tengent_plane = 100
    depth_mesh = 9
    density = 0.0000001  # TO MODIFY IF NECESSARY
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(arr_tot)
    source.estimate_normals()

    source.orient_normals_consistent_tangent_plane(tengent_plane)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth_mesh)
    vertices_to_remove = densities < np.quantile(densities, density)  # 0.059
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_triangle_normals()
    mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("tooth_f_" + str(density) + ".stl", mesh)
    # <-----------------------------------fill small holes------------------------------------------------------->
    size = density
    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_file("tooth_f_" + str(size) + ".stl")
    mfix.fill_small_boundaries(nbe=150, refine=True)
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    mfix.save_file("tooth_f_" + str(size) + ".stl")
    mesh = pv.PolyData("tooth_f_" + str(size) + ".stl")
    mesh = mesh.decimate(0.9)
    mesh.save("full_tooth_" + str(tooth_num) + ".stl")
    mesh.plot()


else:
    for i in range(sides.shape[0]):
        if(labels[i]==1):
           indexes_1.append(i)

        else:
           indexes_2.append(i)
    indexes_1=np.asarray(indexes_1)
    side_1 = np.delete(sides, indexes_1, axis=0)
    side_2 = np.delete(sides, indexes_2, axis=0)
    np.savetxt("side1_"+str(tooth_num)+".txt",side_1)
    np.savetxt("side2_"+str(tooth_num)+".txt",side_2)
    # <-----------------------------side 2---------------------------->
    line_2 = pv.Line(pointa=side_2[0], pointb=side_2[-1], resolution=100)
    line_2_arr = np.asarray((line_2).points)
    side_2 = np.concatenate((side_2, line_2_arr), axis=0)
    np.savetxt("side2_"+str(tooth_num)+".txt", side_2)
    # pv.PolyData(side_2).plot()
    ##<-----------------------------side 1---------------------------->
    line_1=pv.Line(pointa=side_1[0],pointb=side_1[-1],resolution=100)
    line_1_arr=np.asarray((line_1).points)
    side_1=np.concatenate((side_1,line_1_arr),axis=0)
    np.savetxt("side1_"+str(tooth_num)+".txt",side_1)
    # pv.PolyData(side_1).plot()

  ##<-----------------------------create lower surface---------------------------->
    all_data = np.loadtxt(str(tooth_num)+"_nonside.txt")
    p=pv.Plotter()
    pcd = pv.PolyData(all_data)
    p.add_mesh(tooth)
    p.add_mesh(pcd, color="red")
    p.show()
    circle_l = pv.Sphere(radius=5 * 0.2, theta_resolution=20, phi_resolution=15).translate((pcd.center), inplace=False)
    circle_l = circle_l.project_points_to_plane()
    circle_l = circle_l.translate((0, 0, -6 * 0.8), inplace=False)
    circle_p = np.asarray(circle_l.points)
    all_data = np.concatenate((all_data, circle_p))
    data = np.concatenate((circle_p, all_data), axis=0)

# <-----------------------------------------------RBF interpolation---------------------------------->
    pv.PolyData(data).plot(eye_dome_lighting=True)
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    z = np.array(data[:, 2])
    spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
    x_grid = np.linspace(min(x), max(x), len(x))
    y_grid = np.linspace(min(y), max(y), len(y))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = spline(B1, B2)
    surface = pv.StructuredGrid(B1, B2, Z)
    toot_c = tooth.clip_surface(surface)
    clipped = surface.clip_surface(toot_c)
    total_tooth_witout_sides = toot_c.merge(clipped)
    p__ = pv.Plotter()
    p__.add_mesh(clipped, color="red")
    p__.add_mesh(toot_c)
    # p__.show()
    mesh = pv.PolyData(np.asarray(clipped.points), np.asarray(clipped.cells)).triangulate().decimate(0.955)
    mesh.save("surface_"+str(tooth_num)+".stl")
# <-----------------------------------------------fill sides---------------------------------->
    meshes = o3d.io.read_triangle_mesh("surface_"+str(tooth_num)+".stl")
    meshes.compute_triangle_normals()
    number = 300 # TO MODIFY IF NECESSARY
    source = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source])
    source_arr = np.array(source.points)
    data=np.loadtxt("data_b"+str(tooth_num)+".txt")
    data=get_sorted_arr(data,0)
    data=add_points_between_the_boundary(data,1)
    data=np.concatenate((source_arr,data))
    pv.PolyData(data).plot()
    arr_s1 = np.loadtxt("side1_"+str(tooth_num)+".txt")
    arr_s2 = np.loadtxt("side2_"+str(tooth_num)+".txt")
    arr_s1 = add_points_between_the_boundary(arr_s1, 2)
    # <-----------------------------------------------alpha shape side 1---------------------------------->
    pcd_1 = pv.PolyData(arr_s1)
    pcd_1.plot()
    sur_1 = pcd_1.delaunay_2d().triangulate()
    sur_1.plot()
    sur_1.save("side_1_mesh"+str(tooth_num)+".stl")
    meshes = o3d.io.read_triangle_mesh("side_1_mesh"+str(tooth_num)+".stl")
    meshes.compute_triangle_normals()
    number = 1000
    source_1 = meshes.sample_points_uniformly(number_of_points=number)
    # <-----------------------------------------------alpha shape side 2---------------------------------->
    pcd_2 = pv.PolyData(arr_s2)
    sur_2 = pcd_2.delaunay_2d().triangulate()
    sur_2.plot()
    sur_2.save("side_2_mesh"+str(tooth_num)+".stl")
    meshes = o3d.io.read_triangle_mesh("side_2_mesh"+str(tooth_num)+".stl")
    meshes.compute_triangle_normals()
    number = 1000  # TO MODIFY IF NECESSARY
    source_2 = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source_2, source_1])
    arr_side_1_fill = np.asarray(source_1.points)
    arr_side_2_fill = np.asarray(source_2.points)

    #  <-----------------------------------------------Add lower part---------------------------------->
    arr_tot = np.concatenate((arr_side_2_fill, arr_side_1_fill), axis=0)
    pv.PolyData(arr_tot).plot()
  #  <-----------------------------------------------Add lower part---------------------------------->
    meshes = o3d.io.read_triangle_mesh("surface_"+str(tooth_num)+".stl")
    meshes.compute_triangle_normals()
    number = 10000  # TO MODIFY IF NECESSARY
    source = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source])
    source_arr = np.array(source.points)
    #  <-----------------------------------------------remove points from lower surface---------------------------------->
    lower_part = source_arr
    s1 = np.loadtxt("side1_"+str(tooth_num)+".txt")
    s2 = np.loadtxt("side2_"+str(tooth_num)+".txt")
    p = pv.Plotter()
    p.enable_eye_dome_lighting()
    p.add_mesh(pv.PolyData(s1))
    p.add_mesh(pv.PolyData(s2))
    p.add_mesh(pv.PolyData(lower_part))
    p.show()

    temp = s1[s1[:, 2].argsort()]
    temp_2 = s2[s2[:, 2].argsort()]

    z_min_1 = temp[0, 2]
    z_min_2 = temp_2[0, 2]
    z_min = z_min_1
    if z_min_2 < z_min_1:
        z_min = z_min_2
    points_to_remove = []
    for j in range(lower_part.shape[0]):
        if (lower_part[j, 2] > z_min):
            points_to_remove.append(j)
    # points_to_remove=np.asarray(points_to_remove)
    source_arr = np.delete(lower_part, points_to_remove, axis=0)

    p = pv.Plotter()
    p.enable_eye_dome_lighting()
    p.add_mesh(pv.PolyData(s1))
    p.add_mesh(pv.PolyData(s2))
    p.add_mesh(pv.PolyData(source_arr), color="red")
    p.show()
    #  <-----------------------------------------------sample points uniformly---------------------------------->
    data = np.loadtxt("data_b"+str(tooth_num)+".txt")
    data = get_sorted_arr(data, 0)
    data = np.concatenate((source_arr, data))
    arr_tot = np.concatenate((data, arr_tot))
    pv.PolyData(arr_tot).plot(eye_dome_lighting=True)

    meshes = o3d.io.read_triangle_mesh("tooth_f_" + str(tooth_num) + ".stl")
    meshes.compute_triangle_normals()
    number = 15000  # TO MODIFY IF NECESSARY
    source_1 = meshes.sample_points_uniformly(number_of_points=number)
    # o3d.visualization.draw_geometries([source_1])
    arr_tooth = np.asarray(source_1.points)
    arr_tot = np.concatenate((arr_tooth, arr_tot))
    # <------------------------------------Poisson------------------------------------------------------->
    tengent_plane = 100
    depth_mesh = 9
    density = 0.0000001  # TO MODIFY IF NECESSARY
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(arr_tot)
    source.estimate_normals()

    source.orient_normals_consistent_tangent_plane(tengent_plane)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth_mesh)
    vertices_to_remove = densities < np.quantile(densities, density)  # 0.059
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_triangle_normals()
    mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("tooth_f_" + str(density) + ".stl", mesh)
    # <-----------------------------------fill small holes------------------------------------------------------->
    size = density
    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_file("tooth_f_" + str(size) + ".stl")
    mfix.fill_small_boundaries(nbe=150, refine=True)
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    mfix.save_file("tooth_f_" + str(size) + ".stl")
    mesh = pv.PolyData("tooth_f_" + str(size) + ".stl")
    mesh = mesh.decimate(0.9)
    mesh.save("full_tooth_" + str(tooth_num) + ".stl")
    mesh.plot()