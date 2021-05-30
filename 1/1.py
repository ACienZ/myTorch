#numpy实现两层神经网络
import random
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.utils import deprecate


#生成数据集
#y=x**2
x=np.reshape(np.array([i for i in range(0,100)]),(-1,1))
y=np.reshape(np.array([2*t+5 for t in x]),(-1,1))
# print(x)
# print(y)


#定义相关联参数
#x数量
num_data=np.shape(x)[0]
# print(num_data)
#单个x形状
shape_x=np.shape(x)[1]
#第一层神经元个数
num_1=3
#第二层神经元个数
num_2=np.shape(y)[1]

batch_size=1
le_rate=1e-6


#初始化网络参数
W1=np.random.randn(shape_x,num_1)
# print(W1)
W2=np.random.randn(num_1,num_2)
# print(W2)

for i in range(1000):
    #forward
    randomList=list(range(num_data))
    
    random.shuffle(randomList)
    select_list=randomList[0:batch_size]
    #batch_x
    batch_x=np.array([x[t].tolist() for t in select_list])
    #batch_y
    batch_y=np.array([y[t].tolist() for t in select_list])
    # print(batch_x)
    # print(batch_y)
    h=np.dot(batch_x,W1)
    #print(h)
    #ReLU
    a=np.maximum(0,h)
    #print(a)
    y_hat=np.dot(a,W2)
    # print(y_hat)

    #calc loss
    delta_y=y_hat-batch_y
    # print(delta_y)
    delta_y_powered=np.power(delta_y,2)
    # print(np.shape(delta_y_powered))
    E_t=np.sum(delta_y_powered,axis=1)
    E=np.sum(E_t,axis=0)/batch_size
    print(E)

    #back propagation
    #delta_W2
    delta_W2=-le_rate*(1/batch_size)*2*np.dot(a.T,delta_y)
    # print(delta_W2)
    #delta_W1
    K=np.dot(W2,delta_y.T)
    # print(np.shape(K))
    h_prime=np.where(h<=0,0,1)
    # print(np.shape(h_prime))
    K_h=np.multiply(K.T,h_prime)
    delta_W1=-le_rate*(1/batch_size)*2*np.dot(batch_x.T,K_h)
    # print(delta_W1)

    #update
    W1+=delta_W1
    W2+=delta_W2


h=np.dot(x,W1)
#print(h)
#ReLU
a=np.maximum(0,h)
#print(a)
y_hat=np.dot(a,W2)
# print(y_hat)

from matplotlib import pyplot as plt

plt.plot(x,y_hat)
plt.scatter(x, y_hat, marker='o')
plt.show()