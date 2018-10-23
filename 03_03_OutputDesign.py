#03_05_OutputDesign.py

import numpy as np

"""
'소프트 맥스 함수 기본'
def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

a=np.array([0.3,2.9,4.0])
y=softmax(a)

print(y)
"""

'소프트 맥스 함수 구현시 주의점 - 지수의 제곱은 수가 빨리 커지므로 오버플로우가 발생할 수 있음'
"""
def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

a=np.array([1010,1000,999])     #오버플로우가 발생됨
y=softmax(a)

print(y)
"""

"""
'오버플로우를 방지하기 위해 수식변경 후 적용'
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

a=np.array([1010,1000,999])     #오버플로우가 발생되지 않음
y=softmax(a)
print(y)
print(np.sum(y))
"""

