import scipy as sp
import numpy as np
import cv2
from scipy.ndimage import convolve

sz=(400,400)
A=np.zeros(sz)+1
B=np.zeros(sz)

idx=sz[0]//2
length=10
print(idx)
B[idx:idx+length,idx:idx+length]=1
#B[100:105,100:105]=1
k=np.linspace(.045,.07,sz[0])
f=np.linspace(.01,.1,sz[0])
f,k=np.meshgrid(f,k)
for i in range(100000):
    dT=1.0
    Da=1.0
    Db=0.5
    f=0.0545
    k=0.062
    kernel=np.array(
            [[.05,.2,.05], 
            [.2,-1,.2], 
            [.05,.2,.05]])
    Ca=convolve(A, kernel, mode='reflect')
    Cb=convolve(B, kernel, mode='reflect')
    mat=A*B*B
    Ap=A+(Da*Ca-mat+f*(1-A))*dT
    Bp=B+(Db*Cb+mat-(k+f)*B)*dT
    A=Ap
    B=Bp
    ratio= A/(A+B)
    ratio-=np.min(ratio)
    ratio/=np.max(ratio)
    if i%10==0:
        cv2.imshow("",ratio)
        cv2.waitKey(1)
    print(i,np.max(B),np.max(A))