import numpy as np
import cv2 


m=cv2.imread("pics/surface.png",1)
m=m[1:541,1:541,:]
m[:,:,0]=0
m[:,:,1]=0
#m[:,:,2]=m[:,:,2]/6

print(np.max(m),np.min(m))
m*=3#*6
m=m.astype(float)
x,y=np.gradient(m[:,:,2])
m=m*.66
m[m>0]+=80
if 0:
    r=np.arctan2(y,x)+np.pi
    r/=2*np.pi
else:
    r=np.abs(np.arctan2(x,y))+0.5
    r/=np.pi+0.5

m[:,:,2]*=r 
m=m.astype(np.uint8)
#m[m>220]-=2*(m[m>220]-220)

m=m//16
m=m*16
m=cv2.resize(m, (1080,1080), interpolation = cv2.INTER_NEAREST)
print(np.max(m),np.min(m))

cv2.imwrite("pics/surface2.png",m)
print(m.shape)
cv2.imshow("",m)
cv2.waitKey(0)