# coding=utf-8
'''
@创建于: 2017年11月26日 上午9:31:35

@作者: yzc

@说明: 此处编写一个标准的神经网络
'''
import Initial_NeuralNetwork as INN
import NeuralNetwork_Tools as NNT
import  numpy as np


class NeuralNetwork:
    
    alpha = 0
    m, L = 0, 0
    l, x, y = [], [], []
    w, dw, b, db, Z, dZ, A, dA = [], [], [], [], [], [], [], []
    
    def __init__(self, x, y, l, alpha):
        
        assert x.size > 0, 'the x is empty'
        assert l.size > 0, 'the l is empty'
        assert y.size > 0, 'the y is empty'
        assert alpha > 0, 'alpha is 0'
        
        self.l = l#np.append(np.array([x.shape[0]]), l)
        self.x = x.T
        self.y = y.T
        self.alpha = alpha
        self.m, self.L = INN.initial_NeuralNetwork(eigenvector = self.x, unitsNum_inLayer = self.l, labels = self.y, 
                                         output_w = self.w, output_dw = self.dw, 
                                         output_b = self.b, output_db = self.db, 
                                         output_Z = self.Z, output_dZ = self.dZ, 
                                         output_A = self.A, output_dA = self.dA)
        
#         self.w[1] = np.array([[4, 3, 1], [1, 3, 1], [2, 3, 4]])
#         self.w[2] = np.array([[1, 3, 3], [1, 4, 3]])
#         self.w[3] = np.array([[3, 1]])
        
    def train_once(self):
        
        #从第一层到L-1层， L层用别的激活函数
        for i in range(1, self.L):
            self.Z[i] = np.dot(self.w[i], self.A[i - 1]) + self.b[i]
            self.A[i] = NNT.activate_Relu(self.Z[i])
         
        self.Z[self.L] = np.dot(self.w[self.L], self.A[self.L - 1]) + self.b[self.L]
        self.A[self.L] = NNT.activate_Sigma(self.Z[self.L])
        self.dA[self.L] = - self.y / self.A[self.L] + (1 - self.y) / (1 - self.A[self.L])
        self.dZ[self.L] = self.dA[self.L]* NNT.derivative_Sigma(self.Z[self.L])
        
        #print(self.Z)
        cost = NNT.Cost(self.A[self.L], self.y)
        
        for i in range(self.L, 0, -1):
            if(i < self.L):
                self.dZ[i] = self.dA[i]* NNT.derivative_Relu(self.Z[i])
            self.dw[i] = np.dot(self.dZ[i], self.A[i - 1].T) / self.m
            self.db[i] = np.sum(self.dZ[i], axis = 1, keepdims = True) / self.m
            self.dA[i - 1] = np.dot(self.w[i].T, self.dZ[i])
            self.w[i] = self.w[i] - self.alpha * self.dw[i]
            self.b[i] = self.b[i] - self.alpha * self.b[i]
        
        return cost

'''
x = np.array([[1, 2, 3], [2, 3, 4],[4, 5, 6]])
y = np.array([[1], [2], [3]]).T
l = np.array([3 ,3, 2, 1])
alpha = 0.5

w, dw, b, db, Z, dZ, A, dA = [], [], [], [], [], [], [], []
m, L = INN.initial_NeuralNetwork(eigenvector = x, unitsNum_inLayer = l, labels = y, output_w = w, output_dw = dw, output_b = b, 
                                 output_db = db, output_Z = Z, output_dZ = dZ, output_A = A, output_dA = dA)
#print(x.shape)

print(dw)
print(b)
print(db)
print(Z)
print(dZ)
print(A)
print(dA)

for i in range(1, L):
    Z[i] = np.dot(w[i], A[i - 1]) + b[i]
    A[i] = NNT.activate_Relu(Z[i])
    
#dZ[L] = A[L] - y
#dw[L] = dZ[L] / m * A[L - 1].T
#db[L] = np.sum(dZ[L], axis = 1, keepdims = True) / m

Z[L] = np.dot(w[L], A[L - 1]) + b[L]
A[L] = NNT.activate_Sigma(Z[L])
NNT.Cost(A, y)
dA[L] = -y / A[L] + (1 - y) / (1 - A[L])
dZ[L] = dA[L]* NNT.derivative_Sigma(Z[L])

for i in range(L, 0, -1):
    if(i < L):
        dZ[i] = dA[i]* NNT.derivative_Relu(Z[i])
    dw[i] = np.dot(dZ[i], A[i - 1].T) / m
    db[i] = np.sum(dZ[i], axis = 1, keepdims = True) / m
    dA[i - 1] = np.dot(w[i].T, dZ[i])
    w[i] = w[i] - alpha * dw[i]
    b[i] = b[i] - alpha * b[i]
    
print(A)
'''