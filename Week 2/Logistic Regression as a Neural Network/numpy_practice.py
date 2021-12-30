import numpy as np

arr = np.zeros([3,64,64],int)

x = np.array([[[1,2,3],[3,4,5],[6,7,8],[5,5,5]],[[5,6,7],[7,8,9],[1,2,3],[6,6,6]]],f[[[1,1,1],[2,2,2],[4,5,6],[7,7,7]]])
print(x.shape)
print("sum around 0")
y=np.sum(x,axis=0)
print(y)
print(y.shape)
print("sum around 1")
y=np.sum(x,axis=1)
print(y)
print(y.shape)
print("sum around 2")
y=np.sum(x,axis=2)
print(y)
print(y.shape)