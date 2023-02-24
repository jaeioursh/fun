import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import sin,cos,pi,sqrt

from random import sample,shuffle,random,seed 

if 0:
    seed(0)
    np.random.seed(0)

#(r,g,b)
height=1080
width =1080
dist=20

x_max=1.0
y_max=height/width

image=np.zeros((height,width,3),np.uint8)
coords=[]

def closest(group1,group2,coords):
    mindist=1e9
    connect=None
    for e1 in group1:
        for e2 in group2:
            x1,y1=coords[e1]
            x2,y2=coords[e2]
            if ( abs(x1-x2)+abs(y1-y2) )==dist:
                
                return(e1,e2)

    return -1

def rad(p1):
    w,h=p1
    w-=width/2
    h-=height/2
    w*=2.0/float(height)
    h*=2.0/float(height)
    return (w**2.0+h**2.0)**0.5

def depth(graph,start,prev):

    branches=graph[start]

    #print(branches,start,prev)
    branches=[b for b in branches if b!=prev]
    if len(branches)==0 and prev!=-1:
        return 1
    else:
        return max([depth(graph,b,start) for b in branches])+1

def depths(graph,start):
    dep=[]
    
    que=graph[start]
    que=[[q,start,0] for q in que]

    while 1:
        que2=[]
        dep+=que
        for q,prev,idx in que:
            branches=[[b,q,idx+1] for b in graph[q] if b!=prev]
            que2+=branches
        que=que2
        
        if len(que)==0:
            break
    return dep



def draw(dep,image,tt,TSCALE=10,COL=0):
    t=float(tt//TSCALE)
    t_=float(tt%TSCALE)/float(TSCALE)
    image=image.copy()
    N=10
    T=[t-i for i in range(N) if (t-i)>=0]
    V={}
    for i in range(len(T)):
        l=len(V)
        V[T[i]]=1.0-float(l)/float(N)-t_/float(N)
    for d in dep:
        start,prev,idx=d
        if idx in V:
            v=V[idx]
            v*=255
            color=[int(0),int(0),int(0)]
            color[COL]=int(v)
            color=tuple(color)
            #print(start,b,color)
            c1=tuple(coords[start])
            c2=tuple(coords[prev])
            image=cv2.line(image,c1,c2,color,3)

    return image






        
    return image




for h in range(dist,height,dist):
    for w in range(dist,width,dist):
        coords.append([w,h])





groups=[ [i] for i in range(len(coords)) ]
graph=[ [] for i in range(len(coords)) ]
conns=[]

print("creating lines")
while len(groups)>1:
    idx1,idx2=sample(range(len(groups)),2)
    g1=groups[idx1]
    g2=groups[idx2]
    conn=closest(g1,g2,coords)
    
    if conn != -1:
        groups[idx1]=g1+g2
        groups.pop(idx2)
        conns.append(conn)





for c in conns:
    p1,p2=c
    graph[p1].append(p2)
    graph[p2].append(p1)

        
    c1=tuple(coords[p1])
    c2=tuple(coords[p2])
    
    if rad(c1)<0.5 and rad(c2)<0.5:
        color=(255,255,255)
    else:
        color=(255,255,255)
    #image=cv2.line(image,c1,c2,color,3)
    




middle=int(len(graph)/2)
size = (width,height)
out = cv2.VideoWriter('pics/rain.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
deps=[]
D=[]
for mid in [0,middle,len(graph)-1]:
    #print("finding depth")
    d=depth(graph,mid,-1)
    D.append(d)
    print("depth: "+str(d))
    #print("coloring")
    dep=depths(graph,mid)
    deps.append(dep)
    print(max(dep))
d=max(D)
SCALE=4
for t in range(d*SCALE):
    img=image.copy()
    for i in range(3):  
        dep=deps[i]
        im_out=draw(dep,image,t,SCALE,COL=i)
        img=img+im_out
    #cv2.imshow("blank",im_out)
    #cv2.waitKey(0)
    out.write(img)
out.release()
