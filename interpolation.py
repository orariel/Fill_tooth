from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from KNN_sort import get_sorted_arr
from sklearn.neighbors import NearestNeighbors


def get_curve_interpolation(tooth_str):

    toot_t=pv.read(tooth_str+".stl").subdivide(2,"loop")
    data_b=toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    arr_b=np.asarray(toot_t.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).points)
    arr_b=get_sorted_arr(arr_b,0)
    arr_b[:,2] +=0.18
    data = arr_b.transpose()# np.arr points boundary after sorting

    #now we get all the knots and info about the interpolated spline
    tck, u= interpolate.splprep(data,s=5,per=True) # s- lower it fit better
    new = interpolate.splev(np.linspace(0,1,500), tck, der=0)
    fit_cr2 = np.concatenate((new[0].reshape(len(new[0]),1),new[1].reshape(len(new[0]),1),new[2].reshape(len(new[0]),1)),axis=1)
    pcd_inter=pv.PolyData(fit_cr2)
    p=pv.Plotter()


    arr_tooth=np.asarray(toot_t.points)
    all_data_set=arr_tooth
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(all_data_set)
    k_nn_indexes=neigh.kneighbors(fit_cr2,return_distance=False)
    k_nn_indexes=k_nn_indexes.flatten()
    arr_clean = np.delete(all_data_set, k_nn_indexes, axis=0)
    corrected_b = np.asarray([x for x in arr_tooth if x not in arr_clean])
    # p.add_mesh(toot_t,color="red")
    # p.add_mesh(pv.PolyData(fit_cr2))
    # p.add_mesh(pv.PolyData(corrected_b),color="blue")
    # p.show()
    np.savetxt(tooth_str+"_new_b.txt",corrected_b)
    return corrected_b