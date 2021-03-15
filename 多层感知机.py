import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from itertools import cycle
import numpy as np
import sys
import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
dr=0.000001
x=np.array(np.arange(20)*0.5)
def cross_entropy(p_real,p):
    s=sum(p_real*(-np.log(p+dr*(p==0))))
    return s
mu0=6.0
p_real=np.exp(-(x-mu0)**2/2.0**2)
p_real=p_real/sum(p_real)
mu0=5.0
p=np.exp(-(x**1.1-mu0)**2/2.2**2)
p=p/sum(p)
s=str(cross_entropy(p_real,p))
ax=plt.gca()
plt.figure(1)
plt.subplot(211)
plt.plot(x,p_real,'ko-',label='Real')
plt.plot(x,p,'gx-',label='Prediction')
plt.legend(loc='best')
plt.xlim(0,10)
plt.ylim(0,0.3)
plt.ylabel('p')
ax.text(1,0.2,'Cross Entropy between The Two Distributions:'+s,fontsize=15)
plt.savefig('pwb')