#03_03_MultiDim.py

import numpy as np
import matplotlib.pyplot as plt


"""
'1차원 배열'
A=np.array([1,2,3,4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])
"""

"""
'2차원 배열'
B=np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
"""

"""
'행렬의 내적 구하기'
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])

print(A.shape)
print(B.shape)

c=np.dot(A,B)
print(c)
"""

"""
'신경망의 내적'
X=np.array([1,2])
W=np.array([[1,3,5],[2,4,6]])
print(X.shape)
print(W.shape)

Y=np.dot(X,W)
print(Y)
"""

'3층 신경망 - 행렬의 내적을 이용하여 3층 순방향 신경망 소스'

"""
'시크모이드 메소드'
def sigmoid(x):
    return 1/(1+np.exp(-x))

'항등 함수'
def identity_funtion(x):
    return x;

'입력층'
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])

print("0층 입력층")
print(W1.shape)
print(X.shape)
print(B1.shape)

A1=np.dot(X,W1)+B1

print("1층 은닉층")
print(A1)

Z1=sigmoid(A1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([[0.1,0.2]])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

print("2층 은닉층")
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

print(Z2.shape)
print(W2.shape)
print(B2.shape)

A3=np.dot(Z2,W3)+B3
Y=identity_funtion(A3)

print("3출력층")
print(A3)
print(Y)
"""


'3층 신경망 정리 '

'시크모이드 메소드'
def sigmoid(x):
    return 1/(1+np.exp(-x))

'항등 함수'
def identity_funtion(x):
    return x;

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y


def init_network():
    network={}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=identity_funtion(a3)

    return y


network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)
