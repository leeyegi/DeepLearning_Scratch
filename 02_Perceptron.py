#02_Perceptron.py
import numpy as np
'퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력'
'인공신경망 모형의 하나로 입력신호가 보내질때 각각 고유한 가중치를 곱한다.'
'이 신호의 총합니 입계값을 넘으면 1을 출력 아니면 0또는 -1을 출력'

"""
'퍼셉트론 구현 - AND'
def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
"""

'가중치와 편향도입'
'앞에서 구한 AND게이트는 직관적이고 알기 쉽지만 앞으로를 생각해 다른 방식으로 수정'
'입력값에 가중치를 곱한것에 편향을 더해 0초과이면 1을 출력 0이하이면 -1또는 0을 출력'
"""
x=np.array([0,1])
w=np.array([0.5,0.5])
b=-0.7
print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b)
"""

"""
'편향을 도입한 AND게이트의 페셉트론'
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7

    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0

    elif tmp > 0:
        return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
"""

"""
'편향을 도입한 NAND게이트의 페셉트론'
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
"""

"""
'편향을 도입한 OR게이트의 페셉트론'
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
"""

'다층 퍼셉티콘을 이용해서 XOR게이트 만들기'

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))

