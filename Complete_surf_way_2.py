import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from fill_in_between import fill_between_3d
import pyvista as pv
from KNN_sort import get_sorted_arr
arr_b_sorted = np.loadtxt("C:/Users/orari/PycharmProjects/Fill_Tooth/txt_files/gvul.txt")

line_1=arr_b_sorted[0:250,:]
line_2=arr_b_sorted[250:500,:]
x_1 =line_1[:,0]
y_1 =line_1[:,1]
z_1 =line_1[:,2]
x_2 =line_2[:,0]
y_2 =line_2[:,1]
z_2 =line_2[:,2]
# y = np.linspace(-1,1,100)
list=[]

set1 = [x_1, y_1, z_1]
set2 = [x_2, y_2, z_2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*set1, lw=4)
ax.plot(*set2, lw=4)
A=fill_between_3d(ax, *set1, *set2, mode = 2)


# a=np.concatenate(A)
# print(np.array_equal(a,arr_b_sorted))
# p=pv.PolyData(a)


plt.show()