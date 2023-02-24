import numpy as np
import cv2
from math import sin,cos,pi,atan2,sqrt
from random import randint,random
from numba import jit

SIZE=1080
RAN=20
SCALE=4

def rot(x,y,t):
    return x*cos(t)-y*sin(t),x*sin(t)+y*cos(t)

def sample():
    theta=random()*2*pi
    dx,dy=sin(theta),cos(theta)
    x,y=SIZE//2+dy*100,SIZE//2-dx*100
    c=randint(0,255)
    return [x,y,dx,dy,c]

def gen():
    Y=[]
    grid=[[[]]*(SIZE//RAN)]*(SIZE//RAN)
    for i in range(1300):
        
        
        Y.append(sample())
    return np.array(Y)


def step(xy):

    for i in range(len(xy)):
        
        x,y,dx,dy,c=xy[i]
        if 0<=x<SIZE and 0<=y<SIZE:
            dx,dy=rot(dx,dy,0.0015)
            xy[i]=[x+dx,y+dy,dx,dy,c]
        else:
            xy[i]=sample()


    return xy




def draw(img,Y):
    #img[img!=0]-=1
    for point in Y:
        x,y,_,_,c=point
        x,y=int(x*SCALE),int(y*SCALE)

        img=cv2.rectangle(img,(x,y),(x+SCALE*2,y+SCALE*2),(0,c,255),-1)
    return img,cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_AREA)


height=SIZE
width =SIZE
size = (width,height)
out = cv2.VideoWriter('pics/follow2.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
img = np.zeros((height*SCALE,width*SCALE,3), np.uint8)


y=gen()

for i in range(3000):
    y=step(y)
    img,show=draw(img,y)
    if i%100==0:
        print(i)
lst=[]
for t in range(3000):
    

    y=step(y)
    print(t)
    img,show=draw(img,y)
  

    
    
    if t%5==0:
        cv2.imshow("blank",show)
        cv2.waitKey(30)
        lst.append(show)
        #out.write(show)
for l in reversed(lst):
    out.write(l)
out.release()