import cv2
import numpy as np
from math import sin, cos
FINAL=1
if FINAL:
    r=[1030]
else:
    r=range(1000,10000)
for i in r:
    
    if not FINAL:
        seed=np.random.randint(0,10000)
        seed=i
    else:
        seed=i
    print("Seed: ",seed)
    np.random.seed(seed)


    def rot(theta):
        theta*=np.pi/180
        r=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
        return r 

    def generate(n,rots):
        
        curr=np.array([[1,0]]).T
        pts=[curr.T]
        end=rot(360/rots)
        
        mats=[]
        for i in range(n):
            m=np.random.random((2,2))*2-1 
            m/=np.max(np.abs(m))
            #m/=(np.abs(m))**0.5
            m[np.abs(m)<0.3]=0.3*np.sign(m[np.abs(m)<0.3])
            mats.append(m)
        M=mats[0]
        for m in mats[1:]:
            M=m@M 
        mats.append(end@np.linalg.inv(M))

        for i in range(rots):
            for m in mats:
                curr=m@curr
                pts.append(curr.T)
        pts=np.vstack(pts)
        pts/=np.max(np.abs(pts))
        return pts
    if FINAL:
        SIZE=1080*16
        rots=360*7
    else:
        SIZE=1080
        rots=360
    pts=generate(5,rots)
    pts=pts*(SIZE//2)+SIZE//2
    prev=pts[0]
    img=np.zeros((SIZE,SIZE,3),dtype=np.uint8)
    for pt in pts[1:]:
        x,y=int(prev[0]), int(prev[1])
        X,Y=int(pt[0]), int(pt[1])
        prev=pt
        img=cv2.line(img,(x,y),(X,Y),(255,255,255),1)


    img=cv2.resize(img,(1080,1080),interpolation=cv2.INTER_AREA)
    if FINAL:
        cv2.imwrite("pics/gen/gen"+str(seed)+".png",img)
    cv2.imshow(str(seed),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)