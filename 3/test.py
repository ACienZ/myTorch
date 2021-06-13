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