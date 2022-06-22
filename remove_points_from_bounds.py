import pyvista as pv
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from points_addition import add_points_between_the_boundary

#<------------------------------------------Load files------------------------------------------------------->
arr_b_sorted = np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/gvul.txt")
mesh=o3d.io.read_triangle_mesh("tooth_5jo.stl")
tooth=pv.read("tooth_5jo.stl")
tooth=tooth.subdivide(2, 'loop')
p=pv.Plotter()
spline = pv.Spline(arr_b_sorted, 5000)
spline.plot(render_lines_as_tubes=True, line_width=10, show_scalar_bar=False)
p.add_mesh(tooth)
p.add_mesh(spline,color="red")
p.show()

pcd = mesh.sample_points_poisson_disk(30050)
arr_tooth=np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])
total_arr=np.concatenate((arr_tooth,arr_b_sorted),axis=0)
a=arr_tooth
b=arr_b_sorted
u =np.asarray([x for x in a if x not in b])
p__=pv.Plotter()
pcd=pv.PolyData(u)
pcd_b=pv.PolyData(arr_b_sorted)
p__.add_mesh(pcd)
p__.add_mesh(pcd_b,color="red")

#<------------------------------------------KNN-OF-SKLEAN------------------------------------------------------->

#<-----------------------------------------Iteration___1___------------------------------------------------------->

all_data_set=total_arr
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(all_data_set)
k_nn_indexes=neigh.kneighbors(arr_b_sorted,return_distance=False)
k_nn_indexes=k_nn_indexes.flatten()
points_to_remove=[]
j=0
mesh_com=np.asarray(pcd.center)
for i in range(k_nn_indexes.size):
    for j in range(5):
       z_val=total_arr[k_nn_indexes[i],2]
       z_val_refrance=arr_b_sorted[j,2]
       # x_val = total_arr[k_nn_indexes[i], 1]
       # x_val_refrance = arr_b_sorted[j, 1]
       # if(z_val_refrance>=z_val or np.abs(x_val)<np.abs(x_val_refrance)):
       #   points_to_remove.append(k_nn_indexes[i])
       a_tooth=total_arr[k_nn_indexes[i],:]
       lina_a=mesh_com-a_tooth
       lina_a=np.sqrt(lina_a[0]**2+lina_a[1]**2+lina_a[2]**2)
       a_ref=arr_b_sorted[j,:]
       lina_b = mesh_com - a_ref
       lina_b= np.sqrt(lina_b[0] ** 2 + lina_b[1] ** 2 + lina_b[2] ** 2)
       if(np.abs(z_val_refrance)<np.abs(z_val) or lina_b<lina_a):
         points_to_remove.append(k_nn_indexes[i])
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
index_to_remove = np.asarray(points_to_remove)
voxel_down_pcd=o3d.geometry.PointCloud()
arr_clean = np.delete(total_arr, index_to_remove, axis=0)

voxel_down_pcd.points=o3d.utility.Vector3dVector(arr_clean)


cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=10, radius=0.4)
display_inlier_outlier(voxel_down_pcd, ind)
pcd_clean=pv.PolyData(arr_clean)
cyl = pv.Cylinder(direction=(0.0, 0.0, 1.0),center=mesh_com)
p_=pv.Plotter()
p_.add_axes()
grid = pcd_clean.delaunay_3d(alpha=0.5)
p_.enable_eye_dome_lighting()
p_.add_mesh(pcd_clean)
p_.add_mesh(pcd_b,color="red")
p_.show()
# before_seg_r,rx=combine_mesh.remove_points(k_nn_indexes)
# before_seg_r2,rx=combine_mesh.remove_points(rx)
# #
# p_=pv.Plotter()
# p_.add_mesh(before_seg)
# p_.add_mesh(before_seg_r2,color="red")
# p_.export_gltf('KNN_15_improve.gltf',)
# p_.show()
#<-----------------------------------------Iteration___2___------------------------------------------------------->
total_arr=np.concatenate((arr_clean,arr_b_sorted),axis=0)
all_data_set=total_arr
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(all_data_set)
k_nn_indexes=neigh.kneighbors(arr_b_sorted,return_distance=False)
k_nn_indexes=k_nn_indexes.flatten()
points_to_remove=[]
j=0
mesh_com=np.asarray(pcd.center)
for i in range(k_nn_indexes.size):
    for j in range(10):
       z_val=total_arr[k_nn_indexes[i],2]
       z_val_refrance=arr_b_sorted[j,2]
       # x_val = total_arr[k_nn_indexes[i], 1]
       # x_val_refrance = arr_b_sorted[j, 1]
       # if(z_val_refrance>=z_val or np.abs(x_val)<np.abs(x_val_refrance)):
       #   points_to_remove.append(k_nn_indexes[i])
       a_tooth=total_arr[k_nn_indexes[i],:]
       lina_a=mesh_com-a_tooth
       lina_a=np.sqrt(lina_a[0]**2+lina_a[1]**2+lina_a[2]**2)
       a_ref=arr_b_sorted[j,:]
       lina_b = mesh_com - a_ref
       lina_b= np.sqrt(lina_b[0] ** 2 + lina_b[1] ** 2 + lina_b[2] ** 2)
       if(np.abs(z_val_refrance)<np.abs(z_val) or lina_b<lina_a):
         points_to_remove.append(k_nn_indexes[i])
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
index_to_remove = np.asarray(points_to_remove)
voxel_down_pcd=o3d.geometry.PointCloud()
arr_clean = np.delete(total_arr, index_to_remove, axis=0)

voxel_down_pcd.points=o3d.utility.Vector3dVector(arr_clean)


cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=10, radius=0.4)
display_inlier_outlier(voxel_down_pcd, ind)
pcd_clean=pv.PolyData(arr_clean)
cyl = pv.Cylinder(direction=(0.0, 0.0, 1.0),center=mesh_com)
p_=pv.Plotter()
p_.add_axes()
grid = pcd_clean.delaunay_3d(alpha=0.5)
p_.enable_eye_dome_lighting()
p_.add_mesh(pcd_clean)
p_.add_mesh(pcd_b,color="red")
p_.show()
