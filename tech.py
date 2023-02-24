import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos,pi,sqrt

from random import sample,shuffle,random

#(r,g,b)
height=1080//5
width =1920//5

x_max=1.0
y_max=height/width

image=np.zeros((height,width,3),np.uint8)

d_lim=(0.05)**2.0
pts=[[random()*x_max,random()*y_max]]

fails=0

while fails<1000:
    bad=False
    pt=[random()*x_max,random()*y_max]
    for p in pts:
        d=(p[0]-pt[0])**2.0 + (p[1]-pt[1])**2.0
        if d<d_lim:
            fails+=1
            bad=True
            break
    if not bad:
        fails=0
        pts.append(pt)

pts=[[p[0]*width,p[1]*width] for p in pts]

for x in range(width):
    for y in range(height):
        d=[ ((p[0]-x)**2.0+(p[1]-y)**2.0)**2.0/width for p in pts]
        d=sorted(d)
        ratio=d[0]/(d[1])
        ratio*=(d[0]/(sqrt(d_lim)*width)) ** 0.5
       
        cutoff=0.3
        if ratio<cutoff:
            val=ratio/cutoff
            val=min(val,1.0)
            image[y][x][0]=int(250)
            image[y][x][1]=int(((1.0-val)**2.0)*250)
            image[y][x][2]=int(((1.0-val)**2.0)*250)


cv2.imwrite("pics/carbon.png",image)            

plt.imshow(image)
plt.show()


#plt.scatter(x,y)
#plt.show()

    