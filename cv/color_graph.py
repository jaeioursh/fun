import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos,pi

from random import sample,shuffle

#(r,g,b)
height=1080
width =1920
R=50

h,w=height,width
sx,sy=width/2,height/2
	
r,theta=0.0,3.14/1
x,y=[-R*0.3+w/2],[-R*0.05+h/2]
for i in range(2000):
    
    
    
    
    r=R/3.1415*theta/2.0
    theta+=R/r
    dx,dy=r*math.cos(theta),r*math.sin(theta)
    if -w/2<dx<w/2 and -h/2<dy<h/2:
        x.append(dx*1.05+w/2)
        y.append(dy*1.05+h/2)
    

print(len(x))
print(max(x),min(x))
print(max(y),min(y))

edge=[[]for i in range(len(x))]

'''
for i in range(len(x)-1):
    dx,dy=x[i]-x[i+1],y[i]-y[i+1]
    r=(dx**2+dy**2)**0.5
    if r<R*1.5:
        edge[i].append(i+1)
        edge[i+1].append(i)
'''

for i in range(len(x)):
    for j in range(len(x)):
        if i!=j:
            
            dx,dy=x[i]-x[j],y[i]-y[j]
            r=(dx**2.0+dy**2.0)**0.5
            if r<R*1.6:
                if not i in edge[j] and not j in edge[i]:
                    edge[i].append(j)
                    edge[j].append(i)
      


def drawline(i,j,r,g,b,image):
    x1,y1=x[i],y[i]
    x2,y2=x[j],y[j]
    theta=math.atan2(y2-y1,x2-x1)

    x1-=math.cos(theta)
    y1-=math.sin(theta)

    while abs(x1-x2) > 1 or abs(y1-y2) > 1:

        x1+=math.cos(theta)
        y1+=math.sin(theta)
        
        X,Y=int(x1),int(y1)
        for dx in [X-2,X-1,X]:#,X+1,X+2]:
            for dy in [Y-2,Y-1,Y]:#,Y+1,Y+2]:
                if 0<=dx<w and 0<=dy<h:
                    image[dy][dx][0]=max(b,image[dy][dx][0])
                    image[dy][dx][1]=max(g,image[dy][dx][1])
                    image[dy][dx][2]=max(r,image[dy][dx][2])

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


if 0:
    for i in range(len(edge)):
        edges=edge[i]
        for e in edges:
        #    edge[e].remove(i)
        #    edge[i].remove(e)
            dx=(abs(x[i]+x[e])/2-w/2 ) / (h/2)
            dy=(abs(y[i]+y[e])/2-h/2 ) / (h/2)
            r=.707*(dx**2+dy**2)**0.5 
            #r=max(abs(dx),abs(dy))
            q=r*pi*2
            r=205*min(max(cos(q+0*pi/3)+.5,0),1)#r
            g=205*min(max(cos(q+2*pi/3)+.5,0),1)#g
            b=205*min(max(cos(q+4*pi/3)+.5,0),1)#b
            #print(r,g,b)
            drawline(i,e,r,g,b,image)
else:
    stak=[[0,0.0]]

    while len(stak)>0:
        src,q=stak.pop()
        if q>1:
            q=0.0
        if len(edge[src])>0:
            stak.append([src,q])
            shuffle(edge[src])
            dst=edge[src].pop()

            edge[dst].remove(src)
            stak.append([dst,q+1.0/100.0])

    
            r=255*min(max(cos(q+0*pi/3)+.5,0),1)#r
            g=255*min(max(cos(q+2*pi/3)+.5,0),1)#g
            b=255*min(max(cos(q+4*pi/3)+.5,0),1)#b
            
            drawline(src,dst,q*250,0,0,image)
    
image=cv2.GaussianBlur(image,(3,3),0)

plt.imshow(image)
cv2.imwrite("pics/test1.png",image)
#plt.scatter(x,y)
plt.show()





