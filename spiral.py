import numpy as np
import cv2 
import numpy as np
from math import asin,sqrt,sin,cos
import matplotlib.pyplot as plt

def spiral2(start,n,spc):
    pts=[[1,start]]#[r,theta]
    for i in range(0,n):
        r=sqrt(i)+spc
        theta=asin(spc/r)
        start+=theta
        r=theta*spc/3.14
        pts.append([r,start])
    
    return pts 

def spiral(start,n,spc):
    pts=[[1,start]]#[r,theta]
    dtheta=0.01
    r,theta=pts[-1]
    for i in range(0,n):
        x,y=r*cos(theta),r*sin(theta)
        old_theta=theta
        while(True):
            theta+=dtheta
            r=(theta-start)/(np.pi*2)
            X,Y=r*cos(theta),r*sin(theta)
            if sqrt((x-X)**2.0+(y-Y)**2.0)>spc:
                pts.append([r,theta])
                break
        dtheta=(theta-old_theta)/10.0
        
    return pts

        

def toxy(pts):
    r,theta=np.array(pts).T
    r=r/np.max(r)
    x=r*np.cos(theta)+1
    y=r*np.sin(theta)+1
    return x/2,y/2
N=4.0
X=[]
Y=[]
for i in range(int(N)):
    x,y=toxy(spiral(i/N*2*np.pi,3000,1/N))
    X.append(x)
    Y.append(y)
    plt.plot(x,y,'o',markersize=1)
x=np.concatenate(X)
y=np.concatenate(Y)


fname="cv/pics/rose.jpg"
img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
h,w=img.shape
w=int((w-h)/2)
img=img[:,w:w+h]

n = 2
img=img.repeat(n,axis=0).repeat(n,axis=1)
h,w=img.shape

N=4
blank=np.zeros((h*N,w*N),dtype=np.uint8)

H=float(h)
for i in range(len(x)):
    X=int(x[i]*H)
    Y=int(y[i]*H)
    color=img[Y,X]
    blank=cv2.circle(blank,(Y*N,X*N),int(color/10.0),(200,200,200),-1)
print(img.shape)

blank=cv2.resize(blank,(h,w),interpolation=cv2.INTER_AREA)

#cv2.imshow("blank",blank)
#cv2.waitKey(0) 
#plt.show()
fname="cv/pics/rose2.jpg"
cv2.imwrite(fname,blank)
