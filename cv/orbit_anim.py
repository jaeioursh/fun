import cv2
import numpy as np
from matplotlib import pyplot as plt

import math
from math import sin,cos,pi,sqrt
from scipy.integrate import odeint



#(r,g,b)
height=1080
width =1080
dist=20    


def grav(y0,t):
    dY=[]
    g=200
    for i in range(0,len(y0),4):
        x,y,dx,dy=y0[i:i+4]
        ddx,ddy=0.0,0.0
        for j in range(0,len(y0),4):
            if j!=i:   
                X,Y,_,_=y0[j:j+4]
                rx=x-X
                ry=y-Y
                r=sqrt(rx**2.0+ry**2.0)+0.01
                ry/=r**3.0
                rx/=r**3.0
                ddx-=rx
                ddy-=ry
        dY+=[dx,dy,ddx*g,ddy*g]
    return dY

def rotate(vec,angle):
    theta=float(angle)/180.0*pi
    x,y=vec
    X=x*cos(theta)-y*sin(theta)
    Y=x*sin(theta)+y*cos(theta)
    return [X,Y]

def initi(n):
    y=[]
    for i in range(n):
        angle=i*360/n
        y+=rotate([0,1],angle)
        y+=rotate([1,0],angle)
    return y

def draw(img,y,idx):
    for i in range(0,len(y[0]),4):
        x0,y0,x1,y1=y[idx-1,i],y[idx-1,i+1],y[idx,i],y[idx,i+1]
        x0,y0,x1,y1=[int(i*float(height/4)+float(height/2)) for i in [x0,y0,x1,y1]]
        A,B=[x0,y0],[x1,y1]
        img= cv2.line(img,tuple(A),tuple(B),(0,0,250),2)
    return img

size = (width,height)
out = cv2.VideoWriter('pics/orbit.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
t=np.linspace(0,1.0,200)
#y0=initi(5)
y0=[-2,1,0.5,0,-2,-1,0.5,0]
print(y0)
y = odeint(grav, y0, t)
img = np.zeros((height,width,3), np.uint8)
for i in range(1,len(t)):
    img=draw(img.copy(),y,i)
    out.write(img)
out.release()
