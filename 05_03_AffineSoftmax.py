#05_06_AffineSoftmax.py

import numpy as np
import numpy as np
from common.functions import *
from common.util import im2col, col2im


#Affine Layer
"""
#Affine계층 순전파 역전파 간단 예
#어파인에서 역전파는 행렬의 형상에 대해 주의 해야한다.
x_dot_W=np.array([[0,0,0],[10,10,10]])
B=np.array([1,2,3])

print(x_dot_W)

print(x_dot_W+B)

dY=np.array([[1,2,3], [4,5,6]])
print(dY)

dB=np.sum(dY,axis=0)
print(dB)
"""

"""
#Affine계층 구현하기
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
    
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        
        return out
    
    def backward(self,dout):
        dx=np.dot(dout, self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
"""


#softmax with loss계층 구현
class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx

    