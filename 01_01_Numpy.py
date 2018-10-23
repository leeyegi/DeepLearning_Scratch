#01_05_Numpy.py
import numpy as np
import tensorflow
"""
'넘파이 - 배열을 꼐산하고 처리하는데 효율적인 기능 제공, 속도 빠름'
'산술연산시 원소수 같게 해야함'

x=np.array([1.0,2.0,3.0])
y=np.array([2.0,3.0,4.0])
print(x)
print(y)
print(type(x))
print(type(y))

print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x%y)
"""
"""
'넘파이는 다차원 배열도 처리할 수 있다.'
a=np.array([[1,2],[3,4]])
b=np.array([[10,10],[20,20]])
print(a)
print(b)
print(a.shape)
print(a.dtype)

print(b.shape)
print(b.dtype)
"""

'broadcast - 넘파이는 형상이 다른 배열끼리도 계산할 수 있다.'
'm*n - m*1'
'm*n - 1*n'
'm*1 - 1*n'
'위의 경우에만 브로드 캐스트 계산가능'
'(2,2)와 (1,2)'
a=np.array([[1,2],[3,4]])
b=np.array([[10,20]])
print(a*b)

print(a[0])
print(a[0][1])

'다차원 배열을 1차원 배열로 변환(평탄화)'
a=a.flatten()
print(a)

print(a>1)
print(a[a>1])



