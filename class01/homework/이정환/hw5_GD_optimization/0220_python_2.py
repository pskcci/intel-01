#numpy 배열
import numpy as np
a = np.array([1, 2, 3, 4])
print(a)
print(a + a)

#일반 배열
b = [1, 2, 3, 4]
print(b + b)

#이중 배열(행렬)
a = np.array([[1, 2], [3, 4]])
print(a)

#삼중 배열(행렬)
a = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
print(a)

#배열의 모양(Shape)
a = np.array([1, 2, 3, 4])
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

#Norm
from numpy import linalg as LA
c = np.array([[1, 2, 3], [-1, 1, 4]])

print(LA.norm(c, axis=0))
print(LA.norm(c, axis=1))
print(LA.norm(c, ord=1, axis=1))
print(LA.norm(c, ord=2, axis=1))
