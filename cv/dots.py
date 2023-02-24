from math import sqrt
from random import random
import numpy as np
import cv2

def make_circ(n):
    SCALE=0.25
    x,y,r=random()-0.5,random()-0.5,(random()+1)*SCALE
    circ=[[x,y,r]]
    for i in range(n):
        while 1:
            x,y=random()*2-1,random()*2-1
            rs=[]
            for circle in circ:
                X,Y,R=circle
                r=sqrt((X-x)**2.0+(Y-y)**2.0)
                rs.append(r-R)
            r=min(rs)
            if r>0:
                r=[r,abs(x-1),abs(y-1),abs(x+1),abs(y+1)]
                r=min(r)
                r=min(r,(random()+1)*SCALE)
                circ.append([x,y,r])
                break
    return circ 

def fill(circ,n):
    for i in range(n):
        while 1:
            x,y=random()*2-1,random()*2-1
            rs=[]
            for circle in circ:
                X,Y,R=circle
                r=sqrt((X-x)**2.0+(Y-y)**2.0)
                rs.append(r-R)
            r=min(rs)
            if r>0:
                r=[r,abs(x-1),abs(y-1),abs(x+1),abs(y+1)]
                r=min(r)
                circ.append([x,y,r])
                break
    return circ 


def fill_obs(circ,n):
    for i in range(n):
        while 1:
            x,y=random()*2-1,random()*2-1
            rs=[]
            for circle in circ:
                X,Y,R=circle
                r=sqrt((X-x)**2.0+(Y-y)**2.0)
                rs.append(r-R)
            r=min(rs)
            if r>0:
                r=[r,abs(x-1),abs(y-1),abs(x+1),abs(y+1)]
                r=min(r)
                circ.append([x,y,r])
                break
    return circ 


circ=make_circ(1)
circ=fill(circ,2000)
w=int(4000)               
            
blank=np.zeros((w,w),dtype=np.uint8)

w=float(w)

for c in circ:
    x,y,r=c
    x=int(w*(x+1)/2)
    y=int(w*(y+1)/2)
    r=int(w*(r)/2)
    print(r,x,y)
    color=(255)
    blank=cv2.circle(blank,(x,y),r,(200,200,200),-1)

blank=cv2.resize(blank,(1000,1000),interpolation=cv2.INTER_AREA)
cv2.imshow("blank",blank)
cv2.waitKey(0) 