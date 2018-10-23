#04_04_Gradient.py

import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append(os.pardir)

#경사법 - 기울기를 이용해 함수의 최솟값이 어디에 있는지 찾는것
"""
#편미분 함수
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)   #x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val

    return grad

#경사하강법 함수
def gradient_decent(f, init_x, lr=0.01, step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad

    return x

#함수 f(x)= x0**2 + x1**2
def function_2(x):
    return x[0]**2+x[1]**2


init_x=np.array([-3.0,4.0])
result = gradient_decent(function_2,init_x, lr=0.1, step_num=100)
print(result)
"""

#경사하강법 그래프그리기
#편미분 함수
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)   #x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad

#경사하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )
        grad = numerical_gradient(f, x)     #편미분 함수로 기울기 구함
        x -= lr * grad                      #경사하강법 수식
    print(x_history)
    return x, np.array(x_history)

#함수 f(x0,x1)= x0**2 + x1**2
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()



"""
#학습률에 따라 결과가 다른 경사 하강법
#편미분 함수
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)   #x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    print(x_history)
    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

#학습률이 너무 큰 예
init_x = np.array([-3.0, 4.0])
lr = 10.0
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

#학습률이 너무 작은 예
init_x = np.array([-3.0, 4.0])
lr = 1e-10
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

#학습률이 적절하게 설정된 예
init_x = np.array([-3.0, 4.0])
lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
"""

#신경망에서의 기울기 - 가중치 매개변수에 관한 손실함수의 기울ㄱㅣ
"""
#편미분 구하는 함수
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad

#소프트 맥스 함수
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#교차엔트로피오차 편역법 함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#간단한 신경망 소스
class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss

#객체 생성
net =simpleNet()
print(net.W)        #가중치 값출력(랜덤)

x=np.array([0.6,0.9])   #1행 2열 과 가중치 의 2행 3열을 내적하여 1행 3열 출력
p=net.predict(x)
print(p)

print(np.argmax(p))     #최댓값인덱스

t=np.array([0,0,1])     #정답레이블
print(net.loss(x,t))

"""
#아래의 람다식과 같은 함수
def f(W):
    return net.loss(x,t)
"""
f=lambda w:net.loss(x,t)

dW=numerical_gradient(f,net.W)
print(dW)

"""