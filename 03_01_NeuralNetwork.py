#03_02_NeuralNatwork.py
import numpy as np
import matplotlib.pylab as plt

"""
'계단 함수 구현하기'
'임계값을 경계로 출력이 바뀌는 활성화 함수 '
def step_funton(x):
    y=x>0
    return np.array(x>0,dtype=np.int)
x=np.arange(-5,5,0.1)
y=step_funton(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
"""

'시그모이드 함수 구현하기'

"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5,5,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
"""

"""
'렐루함수(ReLU - Rectified Linear Unit) - 렐루는 입력이 0을 넘으면 그대로 출력하고 0이하이면 0을 출력하는 함수 '

def relu(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.1)
y=relu(x)

plt.plot(x,y)
plt.show()
"""

