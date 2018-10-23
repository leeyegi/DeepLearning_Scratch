#03_06_HandwritingCogition.py
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle
import numpy as np
from PIL import Image

"""
#(훈련이미지, 훈련레이블), (시험이미지, 시험레이블)
#인수 normalize - 입력이미지의 픽셀값을 0~1사이의 값으로 정규화할지 정한다.
#    flatten - 입력이미지를 1차원 배열로 만들지 정한다.
#    one_hot_label - 원핫코딩 형식으로 저장할지 정한다.
(x_train ,t_train), (x_test,t_test)=load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(x_test.shape)
"""

"""
#첫번째로 훈련된 이미지를 보여주는 코드
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train ,t_train), (x_test,t_test)=load_mnist(flatten=True, normalize=False)

img=x_train[0]
label=t_train[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)
"""

#MNIST데이터셋을 가지고 추론을 수행하는 신경망구현

'시크모이드 함수'
def sigmoid(x):
    return 1/(1+np.exp(-x))

'소프트 맥스 함수'
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

#MNIST데이터 셋을 내려받아 이미지를 넘파이 배열로 변환해주는 스크립트
def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#pickle파일인 simple_weight.pkl에 저장된 학습된 가중치 매개변수를 읽는다.
#이파일에는 가중치와 편향 매개변수가 딕셧너리 변수로 저장되어있다.
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network=pickle.load(f)
    return network

#입력신호를 출력으로 변환하는 처리가정을 구현
def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=softmax(a3)

    return y

"""
x,t=get_data()              # 데이터셋을 얻는다.
network=init_network()      # 네트워크를 형성한다. (학습된 가중치 매개변수를 읽는다)

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])     # 네트워크를 형성한다. (학습된 가중치 매개변수를 읽는다)
    p=np.argmax(y)              # 가장 확률이 높은 인덱스를 얻는다. (softmax 로 리턴하기 때문에 확률 파악 가능)
    if p==t[i]:                  # 예측 성공
        accuracy_cnt+=1


print("Accuracy: "+str(float(accuracy_cnt)/len(x)))
"""

#Batch처리가 구현된 처리
x,t=get_data()
network=init_network()
batch_size=100              # 배치 크기
accracy_cnt=0

for i in range(0,len(x), batch_size):       # 0에서 len(x)까지 batch_size 의 인덱스 배열 반환
    x_batch=x[i:i+batch_size]               # 반환받은 곳부터 batch_size 만큼 자름
    y_batch =predict(network, x_batch)      # 예측한다.
    p=np.argmax(y_batch, axis=1)            # 각각 배열에서 최대값의 인덱스 리턴
    accracy_cnt+=np.sum(p==t[i:i+batch_size])   # boolean 배열 생성하여 True 만 카운트. t는 get_data()에서 가져온 정확한 데이터 수치

print("Accuracy"+str(float(accracy_cnt)/len(x)))
