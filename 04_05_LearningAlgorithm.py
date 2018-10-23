#04_05_LearningAlgorithm.py
import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

#######################################

#2층 신경망 구현 클래스
class TwoLayerNet:
    def __init__(self,input_size, hidden_size,output_size,weight_init_std=0.01):

        #가중치 초기화
        #신경망 매개변수를 보관하는 딕셔너리 변수
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        self.n=1

    #예측 추론을 수행한다.
    def predict(self,x):
        W1,W2=self.params['W1'], self.params['W2']
        b1,b2=self.params['b1'], self.params['b2']
        print(self.n)
        self.n+=1

        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)
        return y

    #손실함수의 값을 구한다.
    def loss(self,x,t):
        y=self.predict(x)
        return cross_entropy_error(y,t)

    #정확도를 구한다.
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    #경사 하강법
    #가중치 매개변수의 기울기를 구한다.
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)

        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])
        return grads

    #오차역전파법
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

"""
net=TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

#x=np.random.rand(100,784)
#y=net.predict(x)

x=np.random.rand(100,784)
t=np.random.rand(100,10)

#grads=net.numerical_gradient(x,t)   #경사하강법
grads=net.gradient(x,t)              #오차역전파법

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
"""
###############################################################



#미니배치 학습구현하기
"""
'''
미니배치의 크기 :100
매번 60,000개의 훈련데이터에서 임의로 100개의 데이터를 추려내고, 그 100개의 미니배치를
대상으로 확률적 경사하강법을 수행해 매개변수를 갱신
경사법에 의한 갱신횟수를 10,000번으로 정하고, 갱신할 때마자 훈련데이터에 대한 손실함수를 계산하고 그값을 배열에 추가
'''
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, one_hot_label=True)

train_loss_list=[]

#하이퍼파라미터
iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

network=TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #미니배치 획득
    batch_mask=np.random.choice(train_size, batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #기울기 계산
    #grad=network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    #매개변수 갱신
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]

    #학습경과 기록
    loss=network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

print(train_loss_list)
"""
#############################################################


#시험 데이터로 평가하기
# 데이터 읽기(Mnist)
#normalize - 입력이미지의 픽셀값을 0~1사이의 값으로 정규화할지 정한다.
#flatten - 입력이미지를 1차원 배열로 만들지 정한다.
#one_hot_label - 원핫코딩 형식으로 저장할지 정한다.
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#2층신경망 구현 클래스에서 객체 할당
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터 - 튜닝 옵션
iters_num = 10000                       # 반복 횟수 10,000번
train_size = x_train.shape[0]           # (60,000 , 784)
print(train_size)
batch_size = 100                        #미니 배치 크기 100개
learning_rate = 0.1                     #학습률 0.1

train_loss_list = []                    #훈련데이터 손실도
train_acc_list = []                     #훈련데이터 정확도
test_acc_list = []                      #시험데이터 정확도

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)    #60,000 / 100
print(iter_per_epoch)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)    #경사하강법(속도 느림)
    grad = network.gradient(x_batch, t_batch)               #오차역전파법(속도 빨라짐)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록(손실함수 계산)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()