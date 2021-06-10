
#%%
C=3
idx=500
# print(list(range(0,-1)))
L=[]
if idx==0:
    L=[1,2,3]
elif idx==1:
    L=[0,2,3,4]
elif idx ==2:
    L=[0,1,3,4,5]
else:
    L=list(range(idx-C,idx))+list(range(idx+1,idx+1+C))
print(L)

#%%
#Tensor,可以接收一组数组作为序号,返回序号对应的元素
import torch

a=torch.Tensor([1,2,3,4,5,6])
print(a[[1,2,3]])

#%%
import torch
a=torch.Tensor([1,2,3])
print(a.shape[0])

#%%
#create data
x=list(range(0,11))
y=[i**2 for i in x]

print(x, y)

import torch
import torch.nn as nn
import torch.utils.data as tud

class testDataset(tud.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

test=testDataset(x,y)
print(test[3])

loader=tud.DataLoader(test, batch_size=2, shuffle=True, num_workers=0)
print(next(iter(loader)))
print(next(iter(loader)))
print(next(iter(loader)))

#%%
import torch
x = torch.tensor([1, 2, 3, 4])
print(x,x.shape,x[0])
b=torch.unsqueeze(x, 0)
print(b, b.shape,b[0])
c=torch.unsqueeze(b,2)
print(c,c.shape)

# %%
import torch
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print(x, x.shape, x[0])
b = torch.unsqueeze(x, 0)
print(b, b.shape, b[0],b[0].shape)
c = torch.unsqueeze(b, 2)
print(c, c.shape, c[0])


#%%
# https://pytorch.org/docs/stable/generated/torch.squeeze.html
import torch
x = torch.tensor([[[1, 2, 3, 4]],
                  [[5, 6, 7, 8]],
                  [[9, 10, 11, 12]]])
print(x, x.shape, x[0])
b=torch.squeeze(x,0) #leave no change
print(b,b.shape)
c=torch.squeeze(x,1)
print(c,c.shape)
d=torch.squeeze(x)
print(d,d.shape)

