#05_07_BackPropagation.py

import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist

#2층 신경망 구현
class TwoLayerNet:

    def __init__(self,input_size, hidden_size,output_size,weight_init_std=0.01):

        #딕셔너리 변수로 신경망의 매개변수를 보관(가중치, 편향)
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층생성 - 순서가 있는 딕셔너리 변수로 신경망의 계층을 보관
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])

        #신경망의 마지막 계층
        self.lastLayer=SoftmaxWithLoss()


    #예측추론을 수행한다
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x

    #손실함수 구함
    def loss(self,x,t):
        y=self.predict(x)
        return self.lastLayer.forward(y,t)

    #정확도 구함
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim!=1:t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    #경사하강법 - 가중치 매개변수의 기울기를 수치미분방식으로 구한다
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)

        #결과저장
        grads={}
        grads['W1']=numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    #오차 역전파 - 가중치 매개변수의 기울기를 오차편역법으로 구한다.
    def gradient(self,x,t):
        #순전파
        self.loss(x,t)

        #역전파
        dout=1
        dout=self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout=layer.backward(dout)

        #결과저장
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

"""
(x_train,t_train), (x_test, t_test)=load_mnist(normalize=True)

network=TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch=x_train[:3]
t_batch=t_train[:3]

grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradient(x_batch,t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+" : "+str(diff))
"""

# 데이터 읽기 - 손글씨 인식 - 28*28의 이미지가 60,000장
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#2층신경망 객체 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#하이퍼파라미터 - 튜닝옵션
iters_num = 10000                   #총 반복횟수
train_size = x_train.shape[0]       #훈련레이블 사이즈
batch_size = 100                    #미니배치 크기
learning_rate = 0.1                 #학습률

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)        #에폭 구함 60,000/100=600개

#10,000번 반복
for i in range(iters_num):
    #60,000개의 데이터중 100개를 무작위로 뽑음
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 매개변수값 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #손실함수 값 구함
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #에폭이 끝날때 마다 정확도 출력
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

