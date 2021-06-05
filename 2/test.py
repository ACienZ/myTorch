# C=3
# idx=500
# # print(list(range(0,-1)))
# L=[]
# if idx==0:
#     L=[1,2,3]
# elif idx==1:
#     L=[0,2,3,4]
# elif idx ==2:
#     L=[0,1,3,4,5]
# else:
#     L=list(range(idx-C,idx))+list(range(idx+1,idx+1+C))
# print(L)


# #Tensor,可以接收一组数组作为序号,返回序号对应的元素
# import torch

# a=torch.Tensor([1,2,3,4,5,6])
# print(a[[1,2,3]])

import torch
a=torch.Tensor([1,2,3])
print(a.shape[0])