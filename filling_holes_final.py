import open3d as o3d
import numpy as np
from pymeshfix._meshfix import PyTMesh
import pyvista as pv




def fill_holes_poisson(filename,tooth_num):
    tengent_plane = 100
    depth_mesh = 9
    density = 0.00008#TO MODIFY IF NECESSARY
    meshes = o3d.io.read_triangle_mesh(filename)
    meshes.compute_triangle_normals()
    # o3d.visualization.draw_geometries([meshes])
    number = 300000#TO MODIFY IF NECESSARY
    source = meshes.sample_points_uniformly(number_of_points=number)
    source_arr = np.array(source.points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_arr)
    source.estimate_normals()

    source.orient_normals_consistent_tangent_plane(tengent_plane)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth_mesh)
    vertices_to_remove = densities < np.quantile(densities, density) # 0.059
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_triangle_normals()
    mesh.compute_triangle_normals()

    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("test.stl", mesh)

    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_file("test.stl")
    mfix.fill_small_boundaries(nbe=100, refine=True)
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    mesh = pv.PolyData(vert, triangles)
    mesh = mesh.subdivide(1,"loop")
    # mesh=mesh.decimate(0.955)
    # mesh=mesh.smooth(200)
    # mesh.plot()
    mesh.save("complete_tooth"+str(tooth_num)+".stl")

def fill_holes_poisson_pcd(arr_tooth,size):
    tengent_plane = 100
    depth_mesh = 9
    density = 0.0999#TO MODIFY IF NECESSARY
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(arr_tooth)
    source.estimate_normals()

    source.orient_normals_consistent_tangent_plane(tengent_plane)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth_mesh)
    vertices_to_remove = densities < np.quantile(densities, density) # 0.059
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_triangle_normals()
    mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("tooth_f_"+str(size)+".stl", mesh)

    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_file("tooth_f_"+str(size)+".stl")
    mfix.fill_small_boundaries(nbe=150, refine=True)
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    mfix.save_file("tooth_f_"+str(size)+".stl")

    mesh=pv.PolyData("tooth_f_"+str(size)+".stl")
    mesh=mesh.decimate(0.955)
    mesh.save("d_tooth_f_"+str(size)+".stl")
    # mesh=mesh.smooth(200)
    # mesh.plot()


