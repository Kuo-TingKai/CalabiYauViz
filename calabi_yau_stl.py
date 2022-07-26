import sympy as sp

import numpy as np
import math
from sympy.utilities.lambdify import lambdify

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import Delaunay
import surf2stl as s2s

# function to bind sympy plotting to backend figure and ax
# https://stackoverflow.com/questions/60325325/putting-together-plots-of-matplotlib-and-sympy
def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    # Fix for > sympy v1.5
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)

    
N = 9
A = 0.4
writeSTL = False

# setup plotting boxes
fig = plt.figure(figsize=(16,8))
axes = []
for g in range(7):
    axes.append(fig.add_subplot(2,4,g+1, projection='3d'))
    axes[g].view_init(elev=30, azim=45)
    axes[g].set_title("n=%d" % (g+2))
    axes[g].set_xlabel('X')
    axes[g].set_ylabel('Y')


a = sp.symbols('a')
for index in range(2, N):
    
    n = sp.symbols('n_%d'%index)
    p_array = []
    count = 0
    for i in range(index):
        k1 = sp.symbols('k1_%d_%d'%(index, i))
        for j in range(index):
            k2 = sp.symbols('k2_%d_%d'%(index, j))

            # declare symbols
            z1 = sp.symbols('z1_%d_%d'%(index, count))
            z2 = sp.symbols('z2_%d_%d'%(index, count))
            x = sp.symbols('x_%d_%d'%(index, count))
            y = sp.symbols('y_%d_%d'%(index, count))
            X = sp.symbols('X_%d_%d'%(index, count))
            Y = sp.symbols('Y_%d_%d'%(index, count))
            Z = sp.symbols('Z_%d_%d'%(index, count))

            # prepare z1, z2 equations
            z1 = sp.exp(sp.I*((2*sp.pi*k1)/n)) * (sp.cos(x+y*sp.I))**(2/n)
            z2 = sp.exp(sp.I*((2*sp.pi*k2)/n)) * (sp.sin(x+y*sp.I))**(2/n)
            z1 = z1.subs([(n, index), (k1, i)])
            z2 = z2.subs([(n, index), (k2, j)])

            # set equations to symbols
            X = sp.re(z1)
            Y = sp.re(z2)
            Z = sp.im(z1)*sp.cos(a) + sp.im(z2)*sp.sin(a)
            # s2s.write('calabi_yau.stl', X, Y, Z)
            # import sys
            # sys.exit()
            Z = Z.subs(a, A)


            # draw each parts
            p = sp.plotting.plot3d_parametric_surface(X, Y, Z, (x, 0, sp.pi/2), (y, -sp.pi/2, sp.pi/2), show=False)            
            p_array.append(p)

            # write a part into a STL file
            if writeSTL:
                x0 = np.linspace(0, math.pi/2, 30)
                y0 = np.linspace(-math.pi/2, math.pi/2, 30)
                x_, y_ = np.meshgrid(x0, y0)
                x_, y_ = x_.flatten(), y_.flatten()

                X_ = lambdify((x, y), X, "numpy")
                Y_ = lambdify((x, y), Y, "numpy")
                Z_ = lambdify((x, y), Z, "numpy")

                X_npf = np.frompyfunc(X_, 2, 1)
                Y_npf = np.frompyfunc(Y_, 2, 1)
                Z_npf = np.frompyfunc(Z_, 2, 1)

                X_np = X_npf(x_, y_).astype(float)
                Y_np = Y_npf(x_, y_).astype(float)
                Z_np = Z_npf(x_, y_).astype(float)

                delaunay_tri = Delaunay(np.array([x_, y_]).T)
                s2s.tri_write('output_calabi-yau_n%d_%d.stl'%(index, count), X_np, Y_np, Z_np, delaunay_tri)

            # inclement symbol number
            count += 1

    # show all parts
    for k in range(1, index*index):
        p_array[0].append(p_array[k][0])
    #np.savetxt("p_array.txt", p_array)
    print(p_array)
    # import sys
    # sys.exit()

    move_sympyplot_to_axes(p_array[0], axes[index-2])
    # print(p_array[0])
    # p_array[0].show()
    
plt.show()