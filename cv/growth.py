import numpy as np
from scipy.signal import convolve2d as conv
import cv2
from math import sin,cos,pi,atan2,sqrt
from random import randint,random

SIZE=1080//2



def draw(img,mask0,t):
    mask0[img>0]=0
    img2=img.copy()
    filter=np.asarray([[0,1,0],
                    [1,0,1],
                    [0,1,0]],dtype=float)/9.0
    
    c=conv(img[:,:,0],filter,"same")
    c=np.atleast_3d(c)
    if t%9==0:
        img2[img2>0]-=1
    r=np.random.random(size=(SIZE,SIZE))
    r=np.atleast_3d(r)
    mask=np.logical_and(c>0,img2==0)
    mask=np.logical_and(r>0.9,mask)
    mask=np.logical_and(mask0,mask)
    img2[mask]=255
    return img2,mask0


        
    


height=SIZE
width =SIZE
size = (width,height)
#out = cv2.VideoWriter('pics/growth2.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
img = np.zeros((height,width,3), np.uint8)
mask= np.ones((height,width,3), dtype=bool)
img[SIZE//2,SIZE//2]=[250]*3

for t in range(2200):
    print(t)
    img,mask=draw(img,mask,t)
    
    if t%4==0:
        img0=cv2.resize(img, (0,0), fx = 2.0, fy = 2.0)
        cv2.imshow("blank",img0)
        cv2.waitKey(30)
        #out.write(img0)
cv2.imwrite('pics/growth2.png', img0)
#out.release()