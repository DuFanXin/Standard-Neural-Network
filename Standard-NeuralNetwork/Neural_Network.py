# coding=utf-8
'''
@创建于: 2017年11月26日 上午9:31:35

@作者: yzc

@说明: 此处编写一个标准的神经网络
'''
import Initial_NeuralNetwork as INN
import NeuralNetwork_Tools as NNT
import  numpy as np

x = np.array([[1, 2, 3], [2, 3, 4],[4, 5, 6]])
y = np.array([[1], [2], [3]]).T
l = np.array([3 ,3, 2, 1])
alpha = 0.5

w, dw, b, db, Z, dZ, A, dA = [], [], [], [], [], [], [], []
m, L = INN.initial_NeuralNetwork(eigenvector = x, unitsNum_inLayer = l, labels = y, output_w = w, output_dw = dw, output_b = b, output_db = db, output_Z = Z, output_dZ = dZ, output_A = A, output_dA = dA)
#print(x.shape)
'''
print(dw)
print(b)
print(db)
print(Z)
print(dZ)
print(A)
print(dA)
'''
for i in range(1, L + 1):
    Z[i] = np.dot(w[i], A[i - 1]) + b[i]
    A[i] = NNT.activate_Sigma(Z[i])
    
#dZ[L] = A[L] - y
#dw[L] = dZ[L] / m * A[L - 1].T
#db[L] = np.sum(dZ[L], axis = 1, keepdims = True) / m

dA[L] = -y / A[L] + (1 - y) / (1 - A[L]) 


for i in range(L, 0, -1):
    dZ[i] = dA[i]* NNT.derivative_Sigma(Z[i])
    dw[i] = np.dot(dZ[i], A[i - 1].T) / m
    db[i] = np.sum(dZ[i], axis = 1, keepdims = True) / m
    dA[i - 1] = np.dot(w[i].T, dZ[i])
    w[i] = w[i] - alpha * dw[i]
    b[i] = b[i] - alpha * b[i]
    
print(w)