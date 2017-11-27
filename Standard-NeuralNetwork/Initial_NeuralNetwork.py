# coding=utf-8
'''
@创建于: 2017年11月26日 上午9:36:36

@作者: yzc

@说明: 初始化神经网络
'''
import numpy as np

'''
function: 
Initial the parameter W for Whole Network

input: 
unitsNum_inLayer    Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])

output:
W    The parameter
'''
def initial_W(unitsNum_inLayer = []):
    # 按照每一层的节点数，创建的随机数矩阵参数W[i] (unitsNum_inLayer[i - 1] × unitsNum_inLayer[i])，作为每一层的参数
    # 将每一层的矩阵参数W[i]加和，得到整个神经网络的矩阵参数W
    # W = np.random.randn(raw, column)
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    W = [np.identity(unitsNum_inLayer[0])]
    for i in range(1, len(unitsNum_inLayer)):
        W.append(np.random.randn(unitsNum_inLayer[i], unitsNum_inLayer[i - 1]))
    return W


'''
function: 
Initial the parameter b(B) for whole Network

input:    
unitsNum_inLayer    Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])

output:    
b    The parameter
'''
def initial_b(unitsNum_inLayer = []):
    # 按照每一层的节点数，创建的零矩阵b[i] (unitsNum_inLayer[i] × 1)，作为每一层的参数
    # 将每一层的矩阵参数b[i]加和，得到整个神经网络的矩阵参数b
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    b = [np.zeros((unitsNum_inLayer[0], 1))]
    for i in range(1, len(unitsNum_inLayer)):
        b.append(np.zeros((unitsNum_inLayer[i], 1)))
    return b


'''
function:    initial the middle result parameter Z for whole Network

input:	
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size						Size of the set
input							The data of input layer

output: 
Z    The middle result parameter
'''
def initial_Z(unitsNum_inLayer = [], set_size = 1, input = []):
    # 按照每一层的节点数，创建的零矩阵Z[i] (unitsNum_inLayer[i] × 1)，作为每一层的中间结果
    # 将每一层的中间结果Z[i]加和，得到整个神经网络的中间结果矩阵参数Z
    # 第0层为输入层，Z[0]为输入数据
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(input) > 0, 'input is  NONE'
    Z = [input]
    for i in range(1, len(unitsNum_inLayer)):
        Z.append(np.zeros((unitsNum_inLayer[i], set_size)))
    return Z


'''
function:    initial the final result parameter A for whole Network

input:    
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size                        Size of the set
input							The data of input layer

output: 
A    The final result parameter
'''
def initial_A(unitsNum_inLayer = [], set_size = 1, input = []):
    # 按照每一层的节点数，创建的零矩阵Z[i] (unitsNum_inLayer[i] × 1)，作为每一层的最终结果
    # 将每一层的最终结果矩阵A[i]加和，得到整个神经网络的矩阵参数A
    # 第0层为输入层，A[0]为输入数据
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(input) > 0, 'input is  NONE'
    A = [input]
    for i in range(1, len(unitsNum_inLayer)):
        A.append(np.zeros((unitsNum_inLayer[i], set_size)))
    return A


'''
function:    initial the final result parameter A for whole Network

input:    
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size                        Size of the set
input                            The data of input layer

output: 
A    The final result parameter
'''
def initial_A(unitsNum_inLayer = [], set_size = 1, input = []):
    # 按照每一层的节点数，创建的零矩阵Z[i] (unitsNum_inLayer[i] × 1)，作为每一层的最终结果
    # 将每一层的最终结果矩阵A[i]加和，得到整个神经网络的矩阵参数A
    # 第0层为输入层，A[0]为输入数据
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(input) > 0, 'input is  NONE'
    A = [input]
    for i in range(1, len(unitsNum_inLayer)):
        A.append(np.zeros((unitsNum_inLayer[i], set_size)))
    return A