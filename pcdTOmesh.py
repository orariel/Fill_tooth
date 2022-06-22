import pyvista as pv
import numpy as np
from points_addition import add_lines_in_the_boundary,add_points_between_the_boundary
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

arr_tooth=np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/tooth5.txt")
pcd_tooth=pv.PolyData(arr_tooth)
arr_sorted=np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/gvul.txt")
arr_b=add_points_between_the_boundary(arr_sorted,20)
arr=add_lines_in_the_boundary(arr_b,40)
# pcd=o3d.geometry.PointCloud()
# pcd.points=o3d.utility.Vector3dVector(arr)
# pcd.estimate_normals()
a=arr_tooth
b=arr_sorted
u = np.asarray([a[i] for i in range(a.shape[0]) if a[i] not in b])
p__=pv.Plotter()
pcd=pv.PolyData(u)
pcd.plot()


# neighbors = []
# neigh = NearestNeighbors(n_neighbors=10)
# neigh.fit(arr_tooth)
# for i in arr_b:
#     neighbors.append(neigh.kneighbors([i],return_distance=False))
# n_array = np.concatenate(neighbors)
# n_list = [n_array[0][0] for n_array in neighbors]
#
# neighbors_vrai = arr_tooth[n_list]
# #<----------------------------------------------------------------------------------------------------->
# o3d.visualization.draw_geometries([pcd])


# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
#
#
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,0.8, tetra_mesh, pt_map)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# o3d.io.write_triangle_mesh("ma.stl",mesh)





points = pv.wrap(arr)
pcd_to_mesh_1 = pv.PolyData(arr)
surf_1 = points.delaunay_2d()
surf_1.plot()
p=pv.Plotter()
p.add_mesh(surf_1)
p.add_mesh(pcd_tooth)
p.enable_eye_dome_lighting()
p.show()

sphere =pv.PolyData(arr_tooth)


plane =surf_1

_ = sphere.compute_implicit_distance(plane, inplace=True)
dist = sphere['implicit_distance']

pl = pv.Plotter()
_ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
_ = pl.add_mesh(plane, color='w')
_=pl.enable_eye_dome_lighting()
pl.show()

points_to_remove=[]
for i in range(dist.size):
        if (dist[i]<0 or dist[i]<.12):
            points_to_remove.append([i])



arr_clean = np.delete(arr_tooth, points_to_remove, axis=0)
#
reduced_sphere=pv.PolyData(arr_clean)
pl = pv.Plotter()
_ = pl.add_mesh(surf_1)
_ = pl.add_mesh(reduced_sphere, color='red')
_=pl.enable_eye_dome_lighting()
pl.show()
#
#
#
