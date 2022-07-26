import numpy as np
import surf2stl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
plt.show()

surf2stl.write('3d-sinusoidal.stl', X, Y, Z)