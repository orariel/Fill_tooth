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

# toothlist=[]
# p=pv.Plotter()
# toothlist=[]
# arr_lable=np.loadtxt("1-8-1 bas.txt")
# lable=arr_lable[0,3]
# start=0
# p=pv.Plotter()
# for i in range (arr_lable.shape[0]-1):
#     if(arr_lable[i+1,3]!=lable):
#        toothlist.append(arr_lable[start:i,0:3])
#        start=i+1
#        lable=arr_lable[i+1,3]
#
# toothlist.append(arr_lable[start:i,0:3])
# for i in range (15):
#     fill_holes_poisson_pcd(toothlist[i],i)

gum=pv.read("tooth_f_14.stl")
surface=gum
tooth=pv.read("tooth_f_"+str(6)+".stl")
for i in range (6,7):
  tooth=pv.read("tooth_f_"+str(i)+".stl")
  clipped = surface.clip_surface(tooth)
  surface=clipped
  # toot_c = tooth.clip_surface(surface)
  # p.add_mesh(tooth,color="red")
  # p.add_mesh(collision,scalars=scalars,show_scalar_bar=False,cmap='bwr')
p=pv.Plotter()
data = get_curve_interpolation("d_tooth_f_"+str(6))
np.savetxt("data_b.txt",data)
data = get_sorted_arr(data,0)
p.enable_eye_dome_lighting()
# p.add_mesh(pv.PolyData(data),color="green")
p.add_mesh(gum)
tooth_b=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
p.add_mesh(tooth_b,color="red")
p.show()
gum_arr=np.asarray(gum.points)
arr=[]

sphere =pv.PolyData(data)
plane = gum
_ = sphere.compute_implicit_distance(plane, inplace=True)
dist = sphere['implicit_distance']

pl = pv.Plotter()
_ = pl.add_mesh(sphere, scalars='implicit_distance', cmap='bwr')
_ = pl.add_mesh(plane, color='w', style='wireframe')
pl.show()
points_poly =gum

# data=np.asarray(tooth_b.points)
dist = sphere['implicit_distance']
index=[]
for i in range(data.shape[0]):
    if(abs(dist[i])>0.4):
        index.append(data[i,:])

sides=np.asarray(index)
# np.savetxt("sides_6_nonside.txt",sides)
pl=pv.Plotter()
pl.add_mesh(sides,color="red")
pl.add_mesh(gum)# pv.PolyData(corrected_b).plot(
pl.show()



# tooth_b_c=tooth_b.clip(normal='z',value=-1)
# sides=np.asarray(tooth_b_c.points)
#
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
for i in range(sides.shape[0]):
    if(labels[i]==1):
       indexes_1.append(i)
    else:
       indexes_2.append(i)
indexes_1=np.asarray(indexes_1)
side_1 = np.delete(sides, indexes_1, axis=0)
side_2 = np.delete(sides, indexes_2, axis=0)
np.savetxt("side1_6.txt",side_1)
np.savetxt("side2_6.txt",side_2)

#
#<-----------------------------side 2---------------------------->

# line_2=np.asarray(pv.Line(pointa=(side_1[0:0],side_1[0:1],side_1[0:2]),pointb=side_1[-1],resolution=60).points)
line_2=pv.Line(pointa=side_2[0],pointb=side_2[-1],resolution=100)
line_2_arr=np.asarray((line_2).points)

#
# line_2_=line_2_arr+np.asarray([0,0,-1.5])
# line_3_=line_2_arr+np.asarray([0,0,-3.5])
# line_4_=line_2_arr+np.asarray([0,0,-3.5])



side_2=np.concatenate((side_2,line_2_arr),axis=0)

np.savetxt("side_2.txt",side_2)

pv.PolyData(side_2).plot()
##<-----------------------------side 1---------------------------->


# line_2=np.asarray(pv.Line(pointa=(side_1[0:0],side_1[0:1],side_1[0:2]),pointb=side_1[-1],resolution=60).points)
line_1=pv.Line(pointa=side_1[0],pointb=side_1[-1],resolution=100)
line_1_arr=np.asarray((line_1).points)
side_1=np.concatenate((side_2,line_2_arr),axis=0)

np.savetxt("side_1.txt",side_1)



pv.PolyData(side_1).plot()

# #
# x =side_2_tot[:, 0]
# y =side_2_tot[:, 1]
# z =side_2_tot[:, 2]
# #
# # # sort data to avoid plotting problems
# # x, y, z = zip(*sorted(zip(x, y, z)))
# #
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
# spline = sp.interpolate.Rbf(x, y, z, function='multiquadric', smooth=1, episilon=0)
#
# x_grid = np.linspace(min(x), max(x), len(x))
# y_grid = np.linspace(min(y), max(y), len(y))
# B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
# #
#
# Z = spline(B1, B2)
# fig = plt.figure(figsize=(15, 6))
# ax = Axes3D(fig)
# ax.plot_wireframe(B1, B2, Z)
# ax.plot_surface(B1, B2, Z, alpha=0.5)
# ax.scatter3D(x, y, z, c='r')
# plt.show()
# #
# p__=pv.Plotter()
# surface = pv.StructuredGrid(B1, B2, Z)
# surface=surface.triangulate()
# mesh = pv.PolyData(np.asarray(surface.points), np.asarray(surface.cells))
# mesh = mesh.decimate(0.955)
# mesh.save("sur_2.stl")
# clipped=tooth.clip_surface(surface)
# clipped=surface.clip_surface(surface)
# p__.add_mesh(tooth)
# p__.add_mesh(surface)
# p__.add_mesh(gum)
# p__.show()