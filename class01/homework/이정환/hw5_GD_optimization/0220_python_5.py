import numpy as np

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4

for i in range(0, n):
    val = 0.0
    for j in range(0, n):
        val += A[i, j] * x[j]
    b[i] = val

print("calculater = ", b)

b = np.dot(A, x)
print("dot = ", b)

b = np.matmul(A, x)
print("matmul = ", b)

b = A@x
print("A@x = ", b)

b = A*x
print("A*x = ", b)

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
b = np.array([15, 19, 26, 44])

print("det = ", np.linalg.det(A))

x = np.linalg.solve(A, b)
print("solver = ", x)

x = np.dot(np.linalg.inv(A), b)
print("invers1 = ", x)

# A의 역행렬이 존재하지 않을 때, 해를 구하는 방법
tmp_b = np.dot(A.T, b)
tmp_T = np.dot(A.T, A)
tmp_inv = np.linalg.inv(np.dot(A.T, A))
x = np.dot(tmp_inv, tmp_b)
print("inverse2 = ", x)
