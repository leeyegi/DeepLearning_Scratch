#04_LossFunction.py

import numpy as np

#평균제곱오차 - 오차의 제곱에 대해 평균을 취함
#작을 수록 오차가 작은것으로 추측한 값의 정확성이 높은 것이다
"""
def mean_squared_error(y,t):
    print((y-t)**2)
    return 0.5*np.sum((y-t)**2)

t=[0,0,1,0,0,0,0,0,0,0]

y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

result =mean_squared_error(np.array(y), np.array(t))
print(result)

y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
result2= mean_squared_error(np.array(y), np.array(t))
print(result2)
"""


#교차 엔트로피 오차
def cross_entropy_error(y,t):
    #delta=1e-7
    delta=0
    result=-np.sum(t*np.log(y+delta))
    print(-(t*np.log(y+delta)))
    return result

t=[0,0,1,0,0,0,0,0,0,0]

#y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y=[0.1,0.1,  1.0    ,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
result =cross_entropy_error(np.array(y), np.array(t))
print(result)

y=[0.001,0.001,  0.99    ,0.001,0.001,0.001,0.001,0.001,0.001,0.002]
result2= cross_entropy_error(np.array(y), np.array(t))
print(result2)

"""
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
result2= cross_entropy_error(np.array(y), np.array(t))
print(result2)
"""

#미니배치 학습 - 한번에 하나만 계산하는데 아니라 일부를 조금씩 가져와서 전체의 근사치로 이용하여 일부분만 계속사용하여 학습
#평군 손실함수를 구함
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test,t_test)=load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size, batch_size)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]

print(np.random.choice(60000,10))
"""

#배치용 교차 엔트로피 오차 구현
"""
#타겟이 원핫인코딩인경우
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.shape(1,t.size)
        y = y.shape(1, y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y))/batch_size


#타겟이 단순 레이블인경우
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.shape(1,t.size)
        y = y.shape(1, y.size)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arrange(batch_size),t]))/batch_size


"""