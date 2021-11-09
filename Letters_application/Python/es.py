import numpy as np

a = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5] ])
b = np.array([1,2,3,4,5]).reshape((5,1))

print(a.shape)
print(b.shape)

res = np.dot(a,b)
print(res)
print(res.shape)

