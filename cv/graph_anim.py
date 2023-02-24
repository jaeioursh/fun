import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos,sin,pi,atan2

from random import sample,shuffle

#(r,g,b)
height=1080
width =1080
R=50

h,w=height,width
sx,sy=width/2,height/2
scale=1.1
r,theta=0.0,3.14/1
x,y,thet=[-R*0.3+w/2],[-R*0.05+h/2],[0]
for i in range(2000):
    
    
    
    
    r=R/3.1415*theta/2.0
    theta+=R/r
    dx,dy=r*math.cos(theta),r*math.sin(theta)
    if -scale*w/2<dx<scale*w/2 and -scale*h/2<dy<scale*h/2:
        x.append(dx*1.05+w/2)
        y.append(dy*1.05+h/2)
        thet.append(theta)
    


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

tris=[[0,1,7]]
print(edge[0:5])
for i in range(1,len(x)-2):
    for j in edge[i+1]:
        if j in edge[i]:
            tris.append([i,i+1,j])

vals=[]        
for tri in tris:
    a,b,c=tri
    v=[thet[j] for j in [a,b,c]]
    v=sorted(v)
    v=v[-1]-0.001*v[-2]
        
    vals.append(v)
#np.save("data",np.array(vals))
tris = [tri for _,tri in sorted(zip(vals,tris))]

size = (width,height)
out = cv2.VideoWriter('pics/spiral.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
for t in np.arange(360,0,-0.5):
    img = np.zeros((height*3,width*3,3), np.uint8)
    i=0
    for tri in tris:
        i+=1
        a,b,c=tri
        tr=[[x[j]*3,y[j]*3] for j in [a,b,c]]
        tr=np.array(tr,dtype=np.int32)

        q=(t+i*2.0)/360.0
      
        q-=float(int(q))
        #print(q)
        q*=2*pi
        
        r,g,b=0,0,0
        if 0<q<=2*pi/3:
            r=255*(-cos(q*3)*0.5+0.5)#r
        elif 2*pi/3<q<=4*pi/3: 
            g=255*(-cos(q*3)*0.5+0.5)#g
        else:
            b=255*(-cos(q*3)*0.5+0.5)#b
        
        
        
        color=[int(b),int(g),int(r)]
        
        cv2.fillPoly(img,[tr],tuple(color))
    img=cv2.resize(img,(width,width),interpolation=cv2.INTER_AREA)

    #cv2.imshow("blank",img)
    #cv2.waitKey(0)
    out.write(img)
out.release()

