# coding=utf-8
'''
@创建于: 2017年11月26日 上午9:31:35

@作者: yzc

@说明: 此处编写一个标准的神经网络
'''
import Initial_NeuralNetwork as INN
import NeuralNetwork_Tools as NNT
import  numpy as np

#L = 3
#m = 1
x = np.array([1, 2, 3])
x = x.reshape((3, 1))

y = np.array([1])
y = y.reshape((1, 1)) 

l = np.array([x.shape[0], 3, 2, 1])

L = l.shape[0] - 1
m = x.shape[1]

w = INN.initial_W(unitsNum_inLayer = l)
dw = INN.initial_W(unitsNum_inLayer = l)

b = INN.initial_b(unitsNum_inLayer = l)
db = INN.initial_b(unitsNum_inLayer = l)

Z = INN.initial_Z(unitsNum_inLayer = l, set_size = m, input = x)
dZ = INN.initial_Z(unitsNum_inLayer = l, set_size = m, input = x)

A = INN.initial_A(unitsNum_inLayer = l, set_size = m, input = x)
dA = INN.initial_A(unitsNum_inLayer = l, set_size = m, input = x)

#print(Z)

for i in range(1, L + 1):
    Z[i] = np.dot(w[i], A[i - 1]) + b[i]
    A[i] = NNT.activate_Sigma(Z[i])
    
#dZ[L] = A[L] - y
#dw[L] = dZ[L] / m * A[L - 1].T
#db[L] = np.sum(dZ[L], axis = 1, keepdims = True) / m
dA[L] = -y / A[L] + (1 - y) / (1 - A[L]) 

for i in range(L, 0):
    dZ[i] = dA[i]* derivative_Sigma(Z[i])
    dW[i] = np.dot(dZ[i], A[i - 1].T) / m
    db[i] = np.sum(dZ[i], axis = 1, keepdims = True) / m
    dA[i - 1] = np.dot(w[i].T, dZ[i])
    
print(Z)