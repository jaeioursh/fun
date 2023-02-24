import numpy as np
from scipy.signal import convolve2d as conv
import cv2
from math import sin,cos,pi,atan2,sqrt
from random import randint,random





img=cv2.imread('pics/growth2.png',cv2.IMREAD_COLOR)
img[:,:,2]=0
img[:,:,1]=255-img[:,:,1]
cv2.imshow("blank",img)
cv2.waitKey(0)
cv2.imwrite('pics/growth.png', img)
#out.release()