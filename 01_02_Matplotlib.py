#01_06_Matplotlib.py
'matplotlib - 그래프 그리기와 데이터 시각화에 중요한 모듈'

import numpy as np
import matplotlib.pyplot as plt

"""
'단순한 그래프 그리기'
'data setting'
'arange(start, end, step)'
x=np.arange(0,6,0.1)
y=np.sin(x)

'그래프 그리기'
plt.plot(x,y)
plt.show()
"""

"""
'pyplot를 이용해 그래프를 그릴때는 제목, 각축의 이름, 레이블등을 설정할 수 있다.'
x=np.arange(0,6,0.1)
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="sin")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin&cos")
plt.legend()
plt.show()
"""

'이미지 표시하기'
from matplotlib.image import imread
img=imread("lena.png")

plt.imshow(img)
plt.show()