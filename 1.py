import numpy as np
import torch
def pT(t):
     print(t,t.shape,t.size(),t.dim())
T1=torch.tensor([3.4])
pT(T1)
T2=torch.tensor([3.4,9.6])
pT(T2)
