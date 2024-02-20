from numpy import linalg as LA
import numpy as np
import matplotlib.pylab as plt

A = np.array([[2, -1], [-1, 2]])
eigenvalues, eigenvectors = LA.eig(A)
x = eigenvectors
lamda = eigenvalues
y1 = A@x[:, 0]
y2 = A@x[:, 1]

print("A : \n", A, "\neigen value0 : ", lamda[0],
      "\neigen vector0 : \n", x[:, 0], "\nAx : \n", y1,
      "\neigen value1 : \n", lamda[1],
      "\neigen vector1 : \n", x[:, 1], "\nAx : \n", y2)
print("\n=========random array==========\n")

x_test = np.random.rand(2, 1)
y_test = A@x_test
print("test vector : \n", x_test, "\nAx_test : \n", y_test)

plt.subplot(3, 3, 1)
plt.plot([0, x[0, 0]], [0, x[1, 0]], 'go-', label='eigenVector0', linewidth=2)
plt.plot([0, y1[0]], [0, y1[1]], 'rs-', label='eigenVector0 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[0]}')

plt.subplot(3, 3, 2)
plt.plot([0, x[0, 1]], [0, x[1, 1]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y2[0]], [0, y2[1]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[1]}')

plt.subplot(3, 3, 3)
plt.plot([0, x_test[0, 0]], [0, x_test[1, 0]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y_test[0, 0]], [0, y_test[1, 0]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('random')

plt.show()

A = np.array([[2, 0], [0, 2]])
eigenvalues, eigenvectors = LA.eig(A)
x = eigenvectors
lamda = eigenvalues
y1 = A@x[:, 0]
y2 = A@x[:, 1]

print("A : \n", A, "\neigen value0 : ", lamda[0],
      "\neigen vector0 : \n", x[:, 0], "\nAx : \n", y1,
      "\neigen value1 : \n", lamda[1],
      "\neigen vector1 : \n", x[:, 1], "\nAx : \n", y2)
print("\n=========random array==========\n")

x_test = np.random.rand(2, 1)
y_test = A@x_test
print("test vector : \n", x_test, "\nAx_test : \n", y_test)

plt.subplot(3, 3, 1)
plt.plot([0, x[0, 0]], [0, x[1, 0]], 'go-', label='eigenVector0', linewidth=2)
plt.plot([0, y1[0]], [0, y1[1]], 'rs-', label='eigenVector0 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[0]}')

plt.subplot(3, 3, 2)
plt.plot([0, x[0, 1]], [0, x[1, 1]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y2[0]], [0, y2[1]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[1]}')

plt.subplot(3, 3, 3)
plt.plot([0, x_test[0, 0]], [0, x_test[1, 0]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y_test[0, 0]], [0, y_test[1, 0]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('random')

plt.show()

A = np.array([[0, -2], [2, 0]])
eigenvalues, eigenvectors = LA.eig(A)
x = eigenvectors
lamda = eigenvalues
y1 = A@x[:, 0]
y2 = A@x[:, 1]

print("A : \n", A, "\neigen value0 : ", lamda[0],
      "\neigen vector0 : \n", x[:, 0], "\nAx : \n", y1,
      "\neigen value1 : \n", lamda[1],
      "\neigen vector1 : \n", x[:, 1], "\nAx : \n", y2)
print("\n=========random array==========\n")

x_test = np.random.rand(2, 1)
y_test = A@x_test
print("test vector : \n", x_test, "\nAx_test : \n", y_test)

plt.subplot(3, 3, 1)
plt.plot([0, x[0, 0]], [0, x[1, 0]], 'go-', label='eigenVector0', linewidth=2)
plt.plot([0, y1[0]], [0, y1[1]], 'rs-', label='eigenVector0 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[0]}')

plt.subplot(3, 3, 2)
plt.plot([0, x[0, 1]], [0, x[1, 1]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y2[0]], [0, y2[1]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'eigen : {lamda[1]}')

plt.subplot(3, 3, 3)
plt.plot([0, x_test[0, 0]], [0, x_test[1, 0]], 'go-', label='eigenVector1', linewidth=2)
plt.plot([0, y_test[0, 0]], [0, y_test[1, 0]], 'rs-', label='eigenVector1 result')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('random')

plt.show()
