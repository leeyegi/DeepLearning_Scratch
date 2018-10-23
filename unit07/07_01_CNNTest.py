import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
"""
#4차원 배열 test
x=np.random.rand(10,1,28,28)
print(x.shape)
print(x[0].shape)
print(x[1].shape)

print(x[0][0])
"""

#im2col의 함수 형태
# im2col(imput data, filter_h, filter_w, stride=1, pad=0)
"""
x1=np.random.rand(1,3,7,7)
col1=im2col(x1,5,5,stride=1,pad=0)
print(x1)
print(col1)
print(col1.shape)

x2=np.random.rand(10,3,7,7)
col2=im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)
"""

#합성곱 계층 구현
class Convolutional:
    #필터, 편향, 스트라이드, 패딩을 인수로 받아 초기화
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        FN,C,FH,FW=self.W.shape     #필터개수, 채널, 필터높이, 필터너비
        N,C,H,W=x.shape             #입력데이터의 개수, 채널, 높이 , 너비

        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col=im2col(x,FH,FW,self.stride, self.pad)   #4차원을 2차원으로 변환
        col_W=self.W.reshape(FN,-1).T               #reshape를 -1로 지정하면 다차원배열의 원소수가 변환후에도 똑같이 유지 됨
        out=np.dot(col,col_W)+self.b                #신경망 계산

        #출력데이터를 적절한 형상으로 바꿔준다.
        out=out.reshape(N,out_h,out_w, -1).transpose(0,3,1,2)

        return out


#풀링계층 구현하기
class Pooling:
    def __init__(self,pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        N,C,H,W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)

        out=np.max(col,axis=1)

        out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out

