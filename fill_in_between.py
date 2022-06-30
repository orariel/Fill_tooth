# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:04:23 2019
@author: artmenlope
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def fill_between_3d(ax, x1, y1, z1, x2, y2, z2, mode=1, c='steelblue', alpha=1.0):
    """

    Function similar to the matplotlib.pyplot.fill_between function but
    for 3D plots.

    input:

        ax -> The axis where the function will plot.

        x1 -> 1D array. x coordinates of the first line.
        y1 -> 1D array. y coordinates of the first line.
        z1 -> 1D array. z coordinates of the first line.

        x2 -> 1D array. x coordinates of the second line.
        y2 -> 1D array. y coordinates of the second line.
        z2 -> 1D array. z coordinates of the second line.

    modes:
        mode = 1 -> Fill between the lines using the shortest distance between
                    both. Makes a lot of single trapezoids in the diagonals
                    between lines and then adds them into a single collection.

        mode = 2 -> Uses the lines as the edges of one only 3d polygon.

    Other parameters (for matplotlib):

        c -> the color of the polygon collection.
        alpha -> transparency of the polygon collection.

    """

    if mode == 1:
        print("lala")

    if mode == 2:
        a=[]
        verts = [(x1[i], y1[i], z1[i]) for i in range(len(x1))] +[(x2[i], y2[i], z2[i]) for i in range(len(x2))]
        a.append(verts)
        print(verts)
        mesh = Poly3DCollection([verts])
        ax.add_collection3d(mesh)




    return mesh


