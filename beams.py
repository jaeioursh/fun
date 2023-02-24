import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos,sin,pi,atan2

from random import sample,shuffle

#(r,g,b)
SIZE=1080
height=SIZE
width =SIZE
R=20

h,w=height,width
sx,sy=width/2,height/2
scale=1.1
r,theta=0.0,3.14/1
x,y,thet=[-R*0.3+w/2],[-R*0.05+h/2],[0]
for i in range(400):
    
    
    
    
    r=R/3.1415*theta/2.0
    theta+=R/r
    dx,dy=r*math.cos(theta),r*math.sin(theta)
    if -scale*w/2<dx<scale*w/2 and -scale*h/2<dy<scale*h/2:
        x.append(dx*1.05+w/2)
        y.append(dy*1.05+h/2)
        thet.append(theta)
    
vx=[0.0]*len(x)
vy=[0.0]*len(y)
x,y,vx,vy=[np.array(i) for i in [x,y,vx,vy]]

edge=[[]for i in range(len(x))]

for i in range(len(x)):
    for j in range(len(x)):
        if i!=j:
            
            dx,dy=x[i]-x[j],y[i]-y[j]
            r=(dx**2.0+dy**2.0)**0.5
            if r<R*1.6:
                if not i in edge[j] and not j in edge[i]:
                    edge[i].append(j)
                    edge[j].append(i)
      




def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])        

def intersect(A,B,C,D): # AB and CD
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#print( intersect([1,0],[0,1],[0,0],[1,1]) )
image=np.zeros((height,width,3),np.uint8)



for i in range(len(edge)):
    for ed in edge[i]:
        A,B=[x[i],y[i]] , [x[ed],y[ed]]
        for src in edge[ed]:
            flag=0
            for dst in edge[src]:
                C,D=[x[src],y[src]] , [x[dst],y[dst]]
                if src == i or src == ed or dst == i or src ==dst:
                    continue
                
                if i==ed or src == dst:
                    print("blah")

                if intersect(A,B,C,D):
                    #print("bad")
                    print(src,dst,i,ed)
                    edge[src].remove(dst)
                    edge[dst].remove(src)
                    #flag=1
                    #break
            #if flag:
                #break

dx,dy=x[-1]-SIZE/2,y[-1]-SIZE/2
r=(dx**2+dy**2)**0.5
r-=R/3
st=min(edge[-1])
for i in range (st,len(edge)):
	dx,dy=x[i]-SIZE/2,y[i]-SIZE/2
	t=atan2(dy,dx)
	x[i]=cos(t)*r+SIZE/2
	y[i]=sin(t)*r+SIZE/2
l=[]
for i in range(len(edge)):
	l.append([])
	for j in edge[i]:
		dx,dy=x[i]-x[j],y[i]-y[j]
		r=(dx**2.0+dy**2.0)**0.5
		l[i].append(r)

def step(x,y,vx,vy,l,scale):
	scale*=0.1

	scale2=10.0
	ax=np.zeros(len(x))
	ay=np.zeros(len(y))+1.0
	for i in range(len(edge)):
		
		for j,length in zip(edge[i],l[i]):
			dx,dy=x[i]-x[j],y[i]-y[j]
			r=(dx**2.0+dy**2.0)**0.5
			d=r-length
			ax[i]-=d*dx/r*scale2
			ay[i]-=d*dy/r*scale2
	x+=vx*scale
	y+=vy*scale
	vx*=0.99
	vy*=0.99
	vx+=ax*scale
	vy+=ay*scale
	y[y>780]=780
	return x,y,vx,vy

def draw(x,y):
	img = np.zeros((height*2,width*2,3), np.uint8)
	X=np.array(x*2,dtype=int)
	Y=np.array(y*2,dtype=int)
	for i in range(len(edge)):
		for j in edge[i]:
			img=cv2.line(img,(X[i],Y[i]),(X[j],Y[j]),(200,200,200),2)
	return cv2.resize(img,(width,width),interpolation=cv2.INTER_AREA)

size = (SIZE,SIZE)	

out = cv2.VideoWriter('cv/pics/beams.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(6000):
	
	if i%10==0:
		print(i)
		img=draw(x,y)
		out.write(img)
		cv2.imshow("blank",img)
		cv2.waitKey(30)
	steps=10
	for i in range(steps):
		x,y,vx,vy=step(x,y,vx,vy,l,1.0/float(steps))	
	
	
	

out.release()
