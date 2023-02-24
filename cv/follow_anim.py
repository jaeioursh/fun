import numpy as np
import cv2
from math import sin,cos,pi,atan2,sqrt
from random import randint,random
from numba import jit

SIZE=1080
RAN=120
SCALE=4

def rot(x,y,t):
    return x*cos(t)-y*sin(t),x*sin(t)+y*cos(t)

def gen():
    Y=[]
    grid=[[[]]*(SIZE//RAN)]*(SIZE//RAN)
    for i in range(500):
        x,y=[SIZE*(0.1+random()*0.8)for j in range(2)]
        theta=random()*2*pi
        dx,dy=sin(theta),cos(theta)
        Y.append([x,y,dx,dy])
        ix,iy=[int(xy//RAN) for xy in [x,y]]
        for xx in [ix-1,ix,ix+1]:
            for yy in [iy-1,iy,iy+1]:
                if 0<=xx<SIZE//RAN and 0<=yy<SIZE//RAN:
                    grid[yy][xx].append(i)
    return np.array(Y),grid

@jit(nopython=True)
def nearest(x,y,dx,dy,i,ran,Y,grid,sz):
    near=-1
    dist=1.0e9
    ax=0
    ay=0
    for j in range(sz):
        idx=grid[j]
        if idx==i:
            continue
        ox=Y[idx,0]
        oy=Y[idx,1]
        r=sqrt((ox-x)**2.0+(oy-y)**2.0)
        if (ox-x)*dx+(oy-y)*dy>r*0.6: #check if in front
            d=((ox-x)**2.0+(oy-y)**2.0)**0.5
            if 0<d<dist and d<ran:
                dist=d
                near=idx
                ax,ay=ox-x,oy-y
    if near < 0:
        return 0
    else:
        if ax*dy-ay*dx>0:
            return -1 
        else:
            return 1


def step(xy,grid):
    ynew=[]
    gridnew=[[[]]*(SIZE//RAN)]*(SIZE//RAN)
    ii=SIZE//RAN
    for i in range(len(xy)):
        x,y,dx,dy=xy[i]
        X,Y=x+dx,y+dy 
        iX,iY=[int(xy//RAN) for xy in [X,Y]]
        for xx in [iX-1,iX,iX+1]:
            for yy in [iY-1,iY,iY+1]:
                if 0<=xx<SIZE//RAN and 0<=yy<SIZE//RAN:
                    gridnew[yy][xx].append(i)

        ix,iy=[int(xy//RAN) for xy in [x,y]]
        if 0<ix<(SIZE//RAN)-1 and 0<iy<(SIZE//RAN)-1:
            
            near=nearest(x,y,dx,dy,i,RAN,xy,np.array(grid[iy][ix],dtype=int),len(grid[iy][ix]))         
                   
                


        else:
            if (SIZE/2-x)*dx+(SIZE/2-y)*dy<0:
                near=1
            else:
                near=0
        A=0.04    
        if near<0:
            dx,dy=rot(dx,dy,-A)

        if near>0:
            dx,dy=rot(dx,dy,A)
        
        
        ynew.append([X,Y,dx,dy])


    return np.array(ynew),gridnew


def step2(xy,grid):

    for i in range(len(xy)):
        x,y,dx,dy=xy[i-1]
        X,Y,dX,dY=xy[i]
        
        if (X-x)*dy-(Y-y)*dx<0:
            near=1
        else:
            near=-1
        A=0.015   
        
        if near<0:
            dx,dy=rot(dx,dy,-A)

        if near>0:
            dx,dy=rot(dx,dy,A)
        
        
        xy[i-1]=[x+dx,y+dy,dx,dy]


    return xy,grid




def draw(img,Y):
    img[img!=0]-=8
    for point in Y:
        x,y,_,_=point
        x,y=int(x*SCALE),int(y*SCALE)

        img=cv2.rectangle(img,(x,y),(x+SCALE*2,y+SCALE*2),(240,240,240),-1)
    return img,cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_AREA)


height=SIZE
width =SIZE
size = (width,height)
out = cv2.VideoWriter('pics/follow.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
img = np.zeros((height*SCALE,width*SCALE,3), np.uint8)


y,grid=gen()

for i in range(5000):
    y,grid=step2(y,grid)

for t in range(1200):
    

    y,grid=step2(y,grid)
    print(t)
    img,show=draw(img,y)
    print(t)



    
    
    if t%2==0:
        cv2.imshow("blank",show)
        cv2.waitKey(30)
        out.write(show)
out.release()