import cv2
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,1,1080//4)
y=np.linspace(0,1,1080//4)
X,Y=np.meshgrid(x,y)

pts=[[0.5,0.5,1,0.5],[0.6,0.6,0.55,1]]
pts=np.random.random((12,4))
for i in range(10000):
    summ=np.zeros_like(X)
    for i in range(len(pts)):
        x,y,dx,dy=pts[i]
        r=(X-x)**2.0+(Y-y)**2.0
        
        summ+=1/(r+.001)
        x,y=x+0.01*dx,y+0.01*dy
        brdr=0.00
        if x<brdr:
            dx=abs(dx)
        if x>1-brdr:
            dx=-abs(dx)
        if y<brdr:
            dy=abs(dy)
        if y>1-brdr:
            dy=-abs(dy)
        pts[i]=x,y,dx,dy
    img=np.zeros_like(X)
    print(np.min(summ),np.max(summ))
    cutoff=100
    band=50
    img[np.logical_and(summ>(cutoff-band),summ<cutoff+band)]=1
    cv2.imshow("blank",img)
    cv2.waitKey(33)
