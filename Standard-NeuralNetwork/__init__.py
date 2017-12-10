import numpy as np
import pandas as pd #导入数据分析库Pandas
import Initial_NeuralNetwork as INN
import NeuralNetwork_Tools as NNT
import Neural_Network as NN
import matplotlib.pyplot as plt 

inputfile = 'C:/Users/yzc/Desktop/1.xlsx' #销量数据路径
x = np.array([[1, 2, 3], [2, 3, 4],[4, 5, 6]])
y = np.array([[1], [2], [3]])
l = np.array([3, 3, 2, 1])
L = len(l)
print(L)
alpha = 0.2
set_size = x.shape[0]
m = set_size
nn = NN.NeuralNetwork(x = x, y = y, l = l, alpha = alpha)
#print(nn.Z)

x1 = []
y1 = []
for i in range(1, 5):
    x1.append(i)
    cost = nn.train_once()
    y1.append(cost)
    print(cost)

plt.plot(x1,y1,label='Frist line',linewidth=3,color='r',marker='o', 
         markerfacecolor='blue',markersize=12)
plt.show() 

'''
l = np.append(np.array([x.shape[0]]), l)

w = []
dw = []
w.clear()
dw.clear()
w.append(np.identity(l[0]))
dw.append(np.identity(l[0]))
for i in range(1, len(l)):
    w.append(np.random.randn(l[i], l[i - 1]))
    dw.append(np.random.randn(l[i], l[i - 1]))


w[1] = np.array([[4, 3, 1], [1, 3, 1], [2, 3, 4]])
w[2] = np.array([[1, 3, 3], [1, 4, 3]])
w[3] = np.array([[3, 1]])

b = []
db = []
b.clear()
db.clear()
for i in range(0, len(l)):
    b.append(np.zeros((l[i], 1)))
    db.append(np.zeros((l[i], 1)))


x = x.T
y = y.T

Z = []
dZ = []
Z.clear()
dZ.clear()
Z.append(x)
dZ.append(x)
for i in range(1, len(l)):
    Z.append(np.zeros((l[i], set_size)))
    dZ.append(np.zeros((l[i], set_size)))

A = []
dA = []
A.clear()
dA.clear()
A.append(x)
dA.append(x)
for i in range(1, len(l)):
    A.append(np.zeros((l[i], set_size)))
    dA.append(np.zeros((l[i], set_size)))
    




for i in range(1, L):
    Z[i] = np.dot(w[i], A[i - 1]) + b[i]
    A[i] = NNT.activate_Relu(Z[i])
# Z[1] = np.dot(w[1], A[0]) + b[1]
# A[1]  = NNT.activate_Relu(Z[1])
# Z[2] = np.dot(w[2], A[1]) + b[2]
# A[2]  = NNT.activate_Relu(Z[2])
Z[L] = np.dot(w[L], A[L - 1]) + b[L]
A[L] = NNT.activate_Sigma(Z[L])
dA[L] = -y / A[L] + (1 - y) / (1 - A[L])
dZ[L] = dA[L]* NNT.derivative_Sigma(Z[L])

cost = NNT.Cost(A[L], y)
print(w)
print(Z)

for i in range(L, 0, -1):
    if(i < L):
        dZ[i] = dA[i]* NNT.derivative_Relu(Z[i])
    dw[i] = np.dot(dZ[i], A[i - 1].T) / m
    db[i] = np.sum(dZ[i], axis = 1, keepdims = True) / m
    dA[i - 1] = np.dot(w[i].T, dZ[i])
    w[i] = w[i] - alpha * dw[i]
    b[i] = b[i] - alpha * b[i]

#print(w)
#cost = np.sum(cost) / cost.shape[1]
'''