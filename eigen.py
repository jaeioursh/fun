import numpy as np
import cv2 
import matplotlib.pyplot as plt 
from scipy.spatial import KDTree as Tree

theta=np.linspace(0,np.pi*2.0,1000)
x=np.cos(theta)
y=np.sin(theta)
x+=(np.random.random(x.shape)-0.5)/100.0
y+=(np.random.random(y.shape)-0.5)/100.0

for i in range(10):
    
    data=np.array([x,y]).T 
    tree=Tree(data)
    for j in range(len(x)):
        X=x[j]
        Y=y[j]
        pts=tree.query([X,Y],k=3)
        
        dx=0.0
        dy=0.0
        for idx in pts[1][1:]:
            xx,yy=data[idx]
            dx+=X-xx
            dy+=Y-yy
        r=(dx**2.0+dy**2.0)**0.5
        
        x[j]+=dx/r/100.0
        y[j]+=dy/r/100.0


plt.plot(x,y)
plt.show()
    