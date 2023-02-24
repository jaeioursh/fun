import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos,pi
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from random import sample,shuffle

image=img = cv2.imread('pics/star.jpg',1)
origin=image.copy()
y,x,d=image.shape 
img=img.copy()
img=img.reshape((x*y,3))
image=image.reshape((x*y,3))
canvas=np.zeros((x*y,3),dtype=np.uint8)
np.random.shuffle(img)
img=img[:100000]
print(img.shape)
print(image.shape)
if 0:
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(img)
    print(pca.get_covariance())
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    print(pca.mean_)
    image=pca.transform(image)
    maxi,mini=np.amax(image,axis=0),np.amin(image,axis=0)
    image=(image-mini)/(maxi-mini)
    image=image.reshape((y,x,3))

    print(maxi,mini)
    image=image[:,:,0]
    print(np.max(image),np.min(image))
    
else:
    kmeans = KMeans(n_clusters=500, random_state=0).fit(img)
    idxs=kmeans.predict(image)
    if 1:
        for i in range(len(kmeans.cluster_centers_)):
            center=kmeans.cluster_centers_[i]
            canvas[idxs==i]=center
    else:
        print(idxs.shape)
        print(canvas[idxs==1].shape)
        canvas[idxs==9]=[0,0,0]
        canvas[idxs==1]=[255,0,0]
        canvas[idxs==2]=[0,255,0]
        canvas[idxs==0]=[0,0,255]
        canvas[idxs==4]=[255,255,0]
        canvas[idxs==5]=[255,0,255]
        canvas[idxs==6]=[0,255,255]
        canvas[idxs==7]=[255,255,255]
    image=canvas.reshape((y,x,3))
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image2)
cv2.imwrite("pics/out2.png",image)
#cv2.imwrite("pics/out2.png",np.mean(np.abs(image-origin),axis=2))
plt.show()





