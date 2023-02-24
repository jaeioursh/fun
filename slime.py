import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2

np.random.seed(0)
kernel=np.random.randint(0,2,(5,5))
k2=np.ones((3,3))
plt.imshow(kernel)
plt.show()

Map=np.zeros((1000,1000))
step=0.001
summ=step
Map[500:502,500:502]=1
for i in range(50):
    print(i)
    summ+=step
    C=convolve(Map, kernel, mode='reflect')
    C2=convolve(Map, k2, mode='reflect')
    rng=np.random.random(Map.shape)
    Map[np.logical_and(np.logical_and(np.logical_and(5>C, C>1),Map==0),C2>1)]=1

plt.imshow(Map)
plt.show()

