from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np


data = arr_list4.transpose()# list/arr points boundary

#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data,s=10,per=True) # plus on augmente le s, plus sa fit largement
    #here we generate the new interpolated dataset,
    #increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,500), tck, der=0)

    #now lets plot it!
fig = plt.figure()
ax = Axes3D(fig)
# ax.plot(data[0], data[1], data[2], label='originalpoints', lw =2, c='Dodgerblue')
ax.plot(new[0], new[1], new[2], label='fit', lw =2, c='red')
ax.legend()
plt.savefig('junk.png')
plt.show()

fit_cr = np.concatenate((new[0].reshape(len(new[0]),1),new[1].reshape(len(new[0]),1),new[2].reshape(len(new[0]),1)),axis=1)
np.savetxt("matana2.txt",fit_cr)



fit_cr2 = np.loadtxt("matana2.txt")
# bottom = np.loadtxt("test_5.txt")

gum_draw_pcd_f1 = o3d.geometry.PointCloud()
gum_draw_pcd_f1.points = o3d.utility.Vector3dVector(fit_cr2)
gum_draw_pcd_f1.estimate_normals()
gum_draw_pcd_f1.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([gum_draw_pcd_f1])


up = source_full_order[int(212339):int(253144),:-1]
gum_draw_pcd_f = o3d.geometry.PointCloud()
gum_draw_pcd_f.points = o3d.utility.Vector3dVector(up)
gum_draw_pcd_f.estimate_normals()
gum_draw_pcd_f.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([gum_draw_pcd_f,gum_draw_pcd_f1])


# full = np.concatenate((up,bottom),axis=0)
# source = o3d.geometry.PointCloud()
# source.points = o3d.utility.Vector3dVector(full)
# source.estimate_normals()
# pcd_down = source.voxel_down_sample(0.01)
# o3d.visualization.draw_geometries([pcd_down])



from sklearn.neighbors import NearestNeighbors

neighbors = []
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(arr_tooth)
for i in arr_b:
    neighbors.append(neigh.kneighbors([i],return_distance=False))
n_array = np.concatenate(neighbors)
n_list = [n_array[0][0] for n_array in neighbors]

neighbors_vrai = arr_tooth[n_list]


neighbors_good22 = o3d.geometry.PointCloud()
neighbors_good22.points = o3d.utility.Vector3dVector(neighbors_vrai)
neighbors_good22.paint_uniform_color([1,0,0])
neighbors_good22.estimate_normals()
o3d.visualization.draw_geometries([neighbors_good22])


corresponding_arch_point_pcd22 = o3d.geometry.PointCloud()
corresponding_arch_point_pcd22.points = o3d.utility.Vector3dVector(up)
corresponding_arch_point_pcd22.paint_uniform_color([0,1,0])
corresponding_arch_point_pcd22.estimate_normals()

o3d.visualization.draw_geometries([neighbors_good22,corresponding_arch_point_pcd22])