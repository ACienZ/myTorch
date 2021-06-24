#%%
import os
import sys

sys_type=sys.platform

file_path = os.path.abspath(__file__)
file_path = os.path.dirname(file_path)
if sys_type=='win32':
    path=os.path.join(file_path,r'test\test.txt') #for windows
elif sys_type=='linux':
    path=os.path.join(file_path,'/test/test.txt') #for linux
with open(path,'r') as fi:
    text=fi.read()
print(text)

#%%
import numpy as np
print(*[1,2,3])
print(*(1,2,3))
a=np.multiply(*(1,2))
print(a)
b=np.multiply(*(1,2,3)) #错误
# %%
import torch
loss_fn=torch.nn.CrossEntropyLoss()
loss=loss_fn(torch.Tensor([[0.5]]),torch.LongTensor([0.5]))
print(loss.item())
# %%
torch.randint(50002, (1, 1), dtype=torch.long)
# %%
