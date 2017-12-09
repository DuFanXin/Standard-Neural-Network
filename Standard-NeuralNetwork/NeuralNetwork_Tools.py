# coding=utf-8
'''
@创建于: 2017年11月26日 下午8:44:08

@作者: yzc

@说明: 神经网络的相关函数：激活函数，导数
'''
import math
import numpy as np


'''
function:
activate the middle result Z  to final result A  of a layer in sigma way

input:
Z    Middle result of a layer

output:
A    Final result of a layer
'''
def activate_Sigma(Z = []):
    assert len(Z) > 0, 'sigma 激活函数的输入为空'
    A = 1.0 / (1.0 + np.exp(-Z))
    return A


'''
function:
activate the middle result Z  to final result A  of a layer in tanh way

input:
Z    Middle result of a layer

output:
A    Final result of a layer
'''
def activate_tanh(Z = []):
    assert len(Z) > 0, 'tanh 激活函数的输入为空'
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return A


'''
function:
activate the middle result Z  to final result A  of a layer in Relu way

input:
Z    Middle result of a layer

output:
A    Final result of a layer
'''
def activate_Relu(Z = []):
    assert len(Z) > 0, 'Relu 激活函数的输入为空'
    A = np.maximum(0, Z)
    return A


'''
function:
activate the middle result Z  to final result A  of a layer in leakyRelu way

input:
Z    Middle result of a layer

output:
A    Final result of a layer
'''
def activate_leakyRelu(Z = []):
    assert len(Z) > 0, 'leakyRelu 激活函数的输入为空'
    A = np.maximum(0.01*Z, Z)
    return A


'''
function:
get the result of derivative function sigma with input x

input:
x   The input

output:
dSigma    result of derivative function sigma with input x
'''
def derivative_Sigma(x = []):
    assert len(x) > 0, 'Sigma 导函数的输入为空'
    dSigma = activate_Sigma(x) * (1 - activate_Sigma(x))
    return dSigma


'''
function:
get the result of derivative function tanh with input x

input:
x   The input

output:
dSigma    result of derivative function tanh with input x
'''
def derivative_tanh(x = []):
    assert len(x) > 0, 'tanh 导函数的输入为空'
    dtanh = 1 - np.power(activate_tanh(x), 2)
    return dtanh


'''
function:
get the result of derivative function Relu with input x

input:
x   The input

output:
dSigma    result of derivative function Relu with input x
'''
def derivative_Relu(x = []):
    assert len(x) > 0, 'Relu 导函数的输入为空'
    dRelu = x
    dRelu[dRelu > 0] = 1
    dRelu[dRelu < 0] = 0
    dRelu[dRelu == 0] = 0
    return dRelu


'''
function:
get the result of derivative function leakyRelu with input x

input:
x   The input

output:
dSigma    result of derivative function leakyRelu with input x
'''
def derivative_leakyRelu(x = []):
    assert len(x) > 0, 'leakyRelu 导函数的输入为空'
    dleakyRelu = x
    dleakyRelu[dleakyRelu > 0] = 1
    dleakyRelu[dleakyRelu < 0] = 0.01
    dleakyRelu[dleakyRelu == 0] = 0.01
    return dleakyRelu

'''
function:
calculate the loss(difference) of predict value and real value

input:
a    The predict value
y    The real value

output:
loss    loss(difference) of predict value and real value
'''
def Loss(a = 0 , y = 0):
    loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))
    return loss

'''
function:
calculate the loss(difference) of predict value and real value

input:
a    The predict value
y    The real value

output:
loss    loss(difference) of predict value and real value
'''
def Cost(A = [] , Y = []):
    assert len(A) > 0, '预测值A 的输入为空'
    assert len(Y) > 0, '实际值Y 的输入为空'
    cost = -(Y * np.log10(A) + (1 - Y) * np.log10(1 - A))
    cost = np.sum(cost) / cost.shape[1]
    return cost