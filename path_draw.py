import pickle
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def DRAW(seed):
    SCALE=2
    color=(0,255,0)
    image=np.zeros((1080*SCALE,1080*SCALE,3),np.uint8)
    with open("paths/paths"+str(seed)+".pkl", 'rb') as handle:
        paths=pickle.load( handle)#, protocol=pickle.HIGHEST_PROTOCOL)
            

    for path in paths:
        for i in range(len(path)-1):
            c1=((path[i][0]*18+9)*SCALE,(path[i][1]*18+9)*SCALE)
            c2=((path[i+1][0]*18+9)*SCALE,(path[i+1][1]*18+9)*SCALE)
            
            image=cv2.line(image,c1,c2,color,3*SCALE)
        for pt in [path[0],path[-1]]:
            c1=((pt[0]*18+9)*SCALE,(pt[1]*18+9)*SCALE)
            cv2.circle(image, c1, 6*SCALE, color, -1)
            cv2.circle(image, c1, 3*SCALE, (0,0,0), -1)

    image=cv2.resize(image,(1080,1080),interpolation=cv2.INTER_AREA)
    cv2.imwrite("pics/paths.png",image)   
    plt.imshow(image)  
for i in range(4):
    
    plt.subplot(2,2,i+1)
    plt.title(str(i))
    DRAW(i)

plt.show()