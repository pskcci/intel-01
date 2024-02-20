import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-2, 2, 11)
f = lambda x : x ** 2
fx = f(x)
print(x)
print(fx)

plt.plot(x, fx, '-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('This is an example for ld graph')
plt.show()

x = np.linspace(-2, 2, 11)
y = np.linspace(-2, 2, 11)

print(x)
print(y)

x, y = np.meshgrid(x, y)
print(x)
print(y)

f = lambda x, y : (x-1)**2 + (y-1)**2
z = f(x, y)
print(z)

from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection='3d', elev=50, azim=-50)
ax.plot_surface(x, y, z, cmap=plt.cm.jet)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

plt.show()
