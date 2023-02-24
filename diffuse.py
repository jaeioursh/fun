import cv2
import numpy as np
import matplotlib.pyplot as plt

height=40
width =40

def diffuse(pts,r):
    for h in range(height):
        for w in range(width):
            sh=0.0
            sw=0.0
            val=pts[h,w]
            for dh in range(-r+h,h+r+1):
                for dw in range(-r+w,w+r+1):
                    if not (dh==h and dw ==w) and (0<=dh<height and 0<=dw<width):
                        if pts[dh,dw] == val:
                            sh+=(dh-h)/(abs(dh-h)+abs(dw-w))
                            sw+=(dw-w)/(abs(dh-h)+abs(dw-w))

            rad=(sh**2.0+sw**2.0)**0.5+.01
            
            sh/=rad
            sw/=rad
            
            sh=round(sh*0.7)
            sw=round(sw*0.7)
            
            #print(sh,sw)

            pts[h,w]=pts[sh,sw]
            pts[sh,sw]=val



image=np.zeros((height,width,3),np.uint8)

pts=np.random.randint(0,2,(height,width))

for i in range(500):
    #print(i)
    diffuse(pts,1)
    if i%1==0:
        plt.clf()
        plt.imshow(pts)
        plt.pause(0.1)
        print(i,np.sum(pts))
#for h in range(height):
#    for w in range(width):
        
