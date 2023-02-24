import numpy as np
from scipy.signal import convolve2d as conv
import cv2
from math import sin,cos,pi,atan2,sqrt
from random import randint,random
import quadprog 

SIZE=1080

def split(img,size):
    lst=[]
    R,C=img.shape
    for r in range(0,R,size):
        for c in range(0,C,size):
            lst.append(img[r:r+size,c:c+size].flatten())
    return lst

def build(img,size,lst):
    R,C=img.shape
    i=0
    for r in range(0,R,size):
        for c in range(0,C,size):
            L=lst[i]
            L=np.reshape(L,(size,size))
            img[r:r+size,c:c+size]=L
            i+=1
    return img

def solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)+np.eye(P.shape[0])*1e-3   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def SOLVE(img):
    M = 10
    N=150
    lst=np.array(split(img,M)).astype(float) #big x m*m*3
    ims=[]
    A=(np.random.randint(0,2,(M*M,N))*250).astype(float)
    #A=np.hstack((A,A,A))
    #A=np.reshape(A,(M*M*3,N))

    P=A.T@A
    G=-1*np.eye(N)
    h=np.zeros(N)
    A_=np.ones((N,1)).T
    b=np.array([1]).reshape(1,)
    for i in range(len(lst)):

        Y=np.array([lst[i,:]]).T
        
        
        
        q=-A.T@Y
        q=q.reshape(N,)
        #for i in [P,q,G,h,A_,b]:
        #    print(i.shape)
        x=solve_qp(P,q,G,h,A_,b)
        
        im=A@x 
        print(i,np.max(x))
        ims.append(im)
      
    return ims


    

fname="pics/split.png"
img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)

'''
r,c,_=img.shape
r=int((r-SIZE)/2)
c=int((c-SIZE)/2)
print(r,c)
img=img[r:r+SIZE,c:c+SIZE,:]
'''
ims=SOLVE( img)

with open('sav.npy', 'wb') as f:
    np.save(f, np.array(ims))
with open('sav.npy', 'rb') as f:
    ims = np.load(f,allow_pickle=True)

build(img,10,ims)
#img0=cv2.resize(img, (0,0), fx = 2.0, fy = 2.0)
cv2.imwrite('pics/random_rose.png', img)
cv2.imshow("blank",img)
cv2.waitKey(0)

