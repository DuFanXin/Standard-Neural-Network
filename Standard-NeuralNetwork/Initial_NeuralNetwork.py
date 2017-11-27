# coding=utf-8
'''
@创建于: 2017年11月26日 上午9:36:36

@作者: yzc

@说明: 初始化神经网络
'''
import numpy as np

'''
function: 
Initial the matrix parameter w for Whole Network

input: 
unitsNum_inLayer    Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])

output:
output_w    The parameter
'''
def initial_W(unitsNum_inLayer = [], output_w = []):
    # 按照每一层的节点数，创建的随机数矩阵参数W[i] (unitsNum_inLayer[i - 1] × unitsNum_inLayer[i])，作为每一层的参数
    # 将每一层的矩阵参数W[i]加和，得到整个神经网络的矩阵参数W
    # output_w = np.random.randn(raw, column)
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    output_w.clear()
    output_w.append(np.identity(unitsNum_inLayer[0]))
    #output_w = [np.identity(unitsNum_inLayer[0])]
    for i in range(1, len(unitsNum_inLayer)):
        output_w.append(np.random.randn(unitsNum_inLayer[i], unitsNum_inLayer[i - 1]))
    #return output_w


'''
function: 
Initial the parameter b(B) for whole Network

input:    
unitsNum_inLayer    Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])

output:    
output_b    The parameter
'''
def initial_b(unitsNum_inLayer = [], output_b = []):
    # 按照每一层的节点数，创建的零矩阵b[i] (unitsNum_inLayer[i] × 1)，作为每一层的参数
    # 将每一层的矩阵参数b[i]加和，得到整个神经网络的矩阵参数b
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    output_b.clear()
    for i in range(0, len(unitsNum_inLayer)):
        output_b.append(np.zeros((unitsNum_inLayer[i], 1)))
    #return output_b


'''
function:    
initial the middle result parameter output_Z for whole Network

input:	
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size						Size of the set
eigenvector						The data of eigenvector layer

output: 
output_Z    The middle result parameter
'''
def initial_Z(unitsNum_inLayer = [], set_size = 1, eigenvector = [], output_Z = []):
    # 按照每一层的节点数，创建的零矩阵Z[i] (unitsNum_inLayer[i] × 1)，作为每一层的中间结果
    # 将每一层的中间结果Z[i]加和，得到整个神经网络的中间结果矩阵参数Z
    # 第0层为输入层，Z[0]为输入数据
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(eigenvector) > 0, 'eigenvector is  NONE'
    output_Z.clear()
    output_Z.append(eigenvector)
    for i in range(1, len(unitsNum_inLayer)):
        output_Z.append(np.zeros((unitsNum_inLayer[i], set_size)))
    #return output_Z


'''
function:    
initial the final result parameter A for whole Network

input:    
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size                        Size of the set
eigenvector						The data of eigenvector layer

output: 
output_A    The final result parameter
'''
def initial_A(unitsNum_inLayer = [], set_size = 1, eigenvector = [], output_A = []):
    # 按照每一层的节点数，创建的零矩阵Z[i] (unitsNum_inLayer[i] × 1)，作为每一层的最终结果
    # 将每一层的最终结果矩阵A[i]加和，得到整个神经网络的矩阵参数A
    # 第0层为输入层，A[0]为输入数据
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(eigenvector) > 0, 'eigenvector is  NONE'
    output_A.clear()
    output_A.append(eigenvector)
    for i in range(1, len(unitsNum_inLayer)):
        output_A.append(np.zeros((unitsNum_inLayer[i], set_size)))
    #print(output_A)
    #return output_A


'''
function:    
initial the whole Network with each initial function above

input:    
unitsNum_inLayer                Array of number of units in each layer, like:l = np.array([3, 3, 2, 1])
set_size                        Size of the set
eigenvector                     The data of eigenvector layer
labels							The label of set(also called Y)

output: 
output_w						The matrix parameter w
output_dw						The derivative matrix parameter dw
output_b						The parameter b
output_db						The derivative parameter db
output_Z						The middle result Z
output_dZ						The derivative middle result dZ
output_A						The final result A
output_dA						The derivative final result dA
m								Size of set
L								num of the layer of Network
'''
def initial_NeuralNetwork(eigenvector = [], unitsNum_inLayer = [], labels = [], output_w = [], output_dw = [], output_b = [], output_db = [], output_Z = [], output_dZ = [], output_A = [], output_dA = []):
	
	#判断数据是否有错误
    assert len(unitsNum_inLayer) > 0, 'unitsNum_inLayer is  less than or equal to 0'
    assert len(eigenvector) > 0, 'eigenvector is  NONE'
    assert len(labels) > 0, 'labels is  NONE'
    
	#将特征向量转置，使x得每一列为一组特征向量
    x = eigenvector.T
    x = x.reshape((x.shape[0],x.shape[1]))
    
	#确定标签矩阵行数和列数
    labels = labels.reshape((labels.shape[0],labels.shape[1]))
	
	#m为训练集的大小，L为神经网络的层数
    m = x.shape[1]
    L = len(unitsNum_inLayer)
	
    #将第0层（即：x）的单元数也加入到神经网络的每层单元数中，为了方便编程和计算
    l = np.append(np.array([x.shape[0]]),unitsNum_inLayer)
    
	#初始化神经网络参数w和dw
    initial_W(unitsNum_inLayer = l, output_w = output_w)
    initial_W(unitsNum_inLayer = l, output_w = output_dw)
	
	#初始化神经网络参数b和db
    initial_b(unitsNum_inLayer = l, output_b = output_b)
    initial_b(unitsNum_inLayer = l, output_b = output_db)
	
	#初始化神经网络参数Z和dZ
    initial_Z(unitsNum_inLayer = l, set_size = m, eigenvector = x, output_Z = output_Z)
    initial_Z(unitsNum_inLayer = l, set_size = m, eigenvector = x, output_Z = output_dZ)

    #初始化神经网络参数A和dA
    initial_A(unitsNum_inLayer = l, set_size = m, eigenvector = x, output_A = output_A)
    initial_A(unitsNum_inLayer = l, set_size = m, eigenvector = x, output_A = output_dA)
    
	#由于Python的不可变变量不能进行传址的参数传递，因此显式地将两个参数传回
    return m, L