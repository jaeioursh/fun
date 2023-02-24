import numpy as np
import cv2 
from math import sqrt
import matplotlib.pyplot as plt
from random import sample
import numba
from time import time
'''




N=500
b=np.random.random((N,N))-0.5
z=np.linspace( 1,np.sqrt(N),N)
z1,z2=np.meshgrid(z,z)
z=z1+z2
b/=z

clip=10
if 1:
    b[clip:,:]=0
    b[:,clip:]=0
else:
    b[:clip,:]=0
    b[:,:clip]=0


c=cv2.idct(b)
c-=np.mean(c)

#c=cv2.imread("pics/mountain.jpg",0)

#c[c>0]=1
#c[c<=0]=-1
#R=np.linspace(-1,1,N)
#R1,R2=np.meshgrid(R,R)
#c=np.sqrt(R1*R1+R2*R2)
grad=np.gradient(c,edge_order=2)
D2,D1=grad

r=np.sqrt(D1*D1+D2*D2)
if 0:
    D1/=r
    D2/=r

d1=np.cumsum(D1,1)
d2=np.cumsum(D2,0)

Zhat = np.zeros(d1.shape)
for i in range(Zhat.shape[0]):
    for j in range(Zhat.shape[1]):
        Zhat[i,j] += np.sum([d2[i,0], -d2[0,0], d1[i,j], -d1[i,0]])

d=Zhat

print(np.min(c),np.max(c))
print(np.min(d),np.max(d))

plt.subplot(2,3,1)
plt.imshow(b)
plt.subplot(2,3,2)
plt.imshow(c)
plt.subplot(2,3,3)
plt.imshow(d)
plt.subplot(2,3,4)
qq=np.arctan2(D1,D2)
qq[qq<0]*=-1
plt.imshow(qq)
plt.subplot(2,3,5)
plt.imshow(D1)
plt.subplot(2,3,6)
plt.imshow(D2)
plt.show()

'''
T=time()
N=1080//2
b=np.random.random((N+2,N+2))-0.5
z=np.linspace( 1,np.sqrt(N),N+2)
z1,z2=np.meshgrid(z,z)
z=z1+z2
b/=z

clip=10
if 1:
    b[clip:,:]=0
    b[:,clip:]=0
c=cv2.idct(b)
c-=np.min(c)
c/=np.max(c)
c*=60
img=c
img+=np.random.random((N+2,N+2))
noise=img.copy()

@numba.jit(nopython=True,nogil=True)
def test(img,N):
    x,y=0,0
    X,Y=0,0
    idx=0
    IDXS=np.array([[-1,0],[1,0],[0,-1],[0,1]])
    #probs=np.array([1,1,1,1])
    probs=np.array([0,1,2,3])
    q=0
    for i in range(6000000):
        x=np.random.randint(0,100)-50+N//2
        y=np.random.randint(0,100)-50+N//2

        while 1:
            v=img[x,y]
            if x==0 or y==0 or x==N+1 or y==N+1:
                X,Y=x,y 
                break 
            q=0
            for i in range(4):
                dx=IDXS[i,0]
                dy=IDXS[i,1]
                X,Y=x+dx,y+dy 
                v0=img[X,Y]
                if v0<v:
                    probs[q]=i
                    q+=1
            
            if q==0:
                X,Y=x,y 
                break
            idx=np.random.randint(0,q)
            idx=probs[idx]
            x+=IDXS[idx,0]
            y+=IDXS[idx,1]
            
        img[x,y]+=1
    return img
img=test(img,N)
print(time()-T)
plt.subplot(1,2,1)
map=img-noise
#map=np.sin(map/2)
map=map.astype(np.uint8)
plt.imshow(map//6)
plt.subplot(1,2,2)
plt.imshow(noise)
plt.show()
cv2.imwrite("pics/surface.png",map)
