#04_03_Diffrerntial.py
import numpy as np
import matplotlib.pylab as plt


#미분함수 구현의 나쁜예
#전방차분 - x+h와 x의 차분
#문제 1 - 반올림 오차를 일으킴 - 작은값이 생략되어 최종 계산결과에 오차생김
#문제 2 - 함수 f의 차분 - x의 기울기를 수해야하지만 x+h와 x사이의 기울기를 구하고 있음
"""
def numerical_diff(f,x):
    h=10e-50
    return (f(x+h)-f(x))/f

print(np.float32(1e-50))
"""

#중심차분 - x+h와 x-h의 차분
#전방차분의 문제점을 해경해줌
"""
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

print(np.float32(1e-4))
"""


#수치미분의 예
"""
#중심 차분 함수
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

print(np.float32(1e-4))

#2차 함수
def function_1(x):
    return 0.01*x**2+0.1*x

x=np.arange(0.0,20.0,0.1)
y=function_1(x)
#plt.xlabel("x")
#plt.ylabel("f(x)")
#plt.plot(x,y)
#plt.show()

result = numerical_diff(function_1,5)
result2 = numerical_diff(function_1,10)

print(result)
print(result2)
"""


#편미분 - 번수가 여럿인 함수의 대한 미분
"""
def function_2(x):
    return x[0]**2+x[1]**2
    #또는 np.sum(x**2)
"""

#편미분 구하는 문제 연습
#x0=4, x1=3일때 x0에 대한 편미분을 구하여라
"""
def function_tmp1(x0):
    return x0*x0+4.0**2.0

#중심 차분 함수
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

result = numerical_diff(function_tmp1, 3.0)
print(result)
"""

#편미분 구하는 문제 연습2
#x0=4, x1=3일때 x1에 대한 편미분을 구하여라
"""
def function_tmp2(x1):
    return 3.0**2.0+x1*x1

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

result = numerical_diff(function_tmp2, 4.0)
print(result)
"""


#편미분의 기울기 구하기
#x0,x1의 편미분을 묶어서 계산
"""
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

def function_2(x):
    return x[0]**2+x[1]**2
    #또는 np.sum(x**2)

print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))
"""

