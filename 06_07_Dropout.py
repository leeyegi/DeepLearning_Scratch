#드롭아웃 - 가중치 감소만으로 대응하기 어려울떄 쓰는 기법
#뉴런을 임의로 삭제하여 학습하는 방법
#단 시험때는 각 뉴런에 훈련데이터때 삭제한 비율을 곱해서 출력
"""
import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None

    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask
        else:return x*(1.0-self.dropout_ratio)

    def backward(self, dout):
        return dout*self.mask
"""

#하이퍼 파라미터 최적화
#검증데이터 - 하이퍼파라미터 성능 평가


