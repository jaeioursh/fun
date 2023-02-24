from copyreg import pickle
from random import shuffle
from typing_extensions import Protocol
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pickle

from python_tsp.heuristics import solve_tsp_simulated_annealing,solve_tsp_local_search
from sklearn.cluster import KMeans


img = cv2.imread('pics/IMG_2159.jpg', cv2.IMREAD_COLOR)

img = cv2.copyMakeBorder(img, 0, 0, 504, 504, cv2.BORDER_CONSTANT)
print(img.shape)

dim = (1080, 1080)
img_whole=img
img=cv2.resize(img, dim,interpolation=cv2.INTER_AREA)



print(img.shape)

#img=cv2.resize(img, (800,800),interpolation=cv2.INTER_AREA)

def show(img,flag=0):
    print(np.max(img),np.min(img))
    if flag:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('image',img)
    cv2.waitKey(0)

#show(img)

def analyze(img):
    img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2=img2.astype(float)
    img2=np.reshape(img2,(-1,3))/255
    pixel_colors = img2.tolist()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img=np.reshape(img,(-1,3))
    
    print(pixel_colors[0])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(img[:,0],img[:,1],img[:,2],c=pixel_colors)
    plt.show()

def segment(img,low,high):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(img, low, high)
    img=cv2.bitwise_and(img,img,mask)
    img=cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img,mask

def smooth(mask):
    kernel=np.ones((9,9))
    kernel/=np.sum(kernel)

    mask=cv2.filter2D(mask,-1,kernel)

    cutoff=100
    mask[mask<=cutoff]=0
    mask[mask>cutoff]=255
    
    return mask

def brdr(mask):
    r=4
    kernel=np.zeros((r*2+1,r*2+1))
    kernel=cv2.circle(kernel,(r,r),r,(255))
    kernel/=np.sum(kernel)

    mask=cv2.filter2D(mask,-1,kernel)

    
    return mask

def draw(mask,cutoff,groups,COL,ENDS=None):
    SCALE=4
    img=np.zeros((mask.shape[0]*SCALE,mask.shape[1]*SCALE,3),dtype=np.uint8)

    if ENDS is None:
        pts=np.where(np.logical_and(mask,np.random.random(mask.shape)>cutoff))
        pts=np.vstack(pts).T
        print(pts.shape)
        ends=tsp(pts,groups)
    else:
        ends=ENDS

    for end in ends:
        R,L=end
        #for i in range(len(L)):
        #    img=cv2.line(img,tuple(R[i]*SCALE),tuple(L[i]*SCALE),COL,SCALE)
        R=np.array([R])*SCALE
        img=cv2.fillPoly(img,R,COL)
    img=cv2.resize(img,(1080,1080),interpolation=cv2.INTER_AREA)
    return img,ends
    

def tsp(pts,groups):
    
    kmeans=KMeans(groups,tol=1,max_iter=200)
    kmeans.fit(pts)
    lbls=kmeans.labels_
    data=[]
    for i in range(groups):
        
        idx=(i==lbls)
        subset=pts[idx,:]
        dists=distance_matrix(subset,subset)
        #print("dist mat finished")
        rows=np.arange((len(dists)))
       

        permutation, distance = solve_tsp_local_search(dists)
        print(i,len(permutation),distance)
        data.append([subset[permutation],subset[np.roll(permutation,1)]])

    return data
    
  
        


if 0:
    imm=img[238:260,156:214,:]
    analyze(imm)



for i in range(1):
    #show(img,1)
    
    #analyze(img)

    #nails 
    img2,mask1=segment(img,(0,10,130),(5,255,255))
    img2,mask2=segment(img,(160,10,130),(180,255,255))
    mask = np.bitwise_or(mask1,mask2)
    mask=smooth(mask)
    #show(mask)
    mask_n=mask
    #ring

    #hand 
    img2,mask=segment(img,(6,30,100),(22,255,255))
    mask=smooth(mask)
    mask_h=mask

    b=brdr(mask_n)
    mask_h[b>0]=0

    #show(mask)
    
    #ring
    kernel=np.ones((3,3))/8
    kernel[1,1]=-1
    mask = cv2.filter2D(img_whole[:,:,0], -1, kernel)*10
    mask=cv2.resize(mask, mask_h.shape,interpolation=cv2.INTER_AREA)

    kernel=np.ones((5,5))
    kernel/=np.sum(kernel)

    mask=cv2.filter2D(mask,-1,kernel)
    cutoff=25
    mask[mask<=cutoff]=0
    mask[mask>cutoff]=255
    mask[:500,:]=0
    mask_r=mask
    

    b=brdr(mask_r)

    mask_h[b>0]=0

    #background:
    hand=mask_n+mask_h+mask_r

    b=brdr(hand)
    hand[:]=0
    
    hand[b==0]=255
    mask_b=hand


    #show(mask_b+mask_n+mask_h+mask_r)
def proc1(idx):
    np.random.seed(idx)
    if 0:
        img,data=draw(mask_r,0.9,3,(255,255,255))
        with open("sav/r"+str(idx)+".pkl","wb") as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("sav/r"+str(idx)+".pkl","rb") as f:
            data=pickle.load(f)
        img,data=draw(mask_h,0.9,3,(255,255,255),data)
    print(idx)
    show(img)

#h-3
#r-9


with open("sav/h3.pkl","rb") as f:
    data_h=pickle.load(f)

with open("sav/n3.pkl","rb") as f:
    data_n=pickle.load(f)

with open("sav/r9.pkl","rb") as f:
    data_r=pickle.load(f)

img1,_=draw(mask_h,0,0,(221,218,250),data_n)
img2,_=draw(mask_h,0,0,(187,207,243),data_h)
img3,_=draw(mask_h,0,0,(255,255,255),data_r)

img=img1+img2+img3
img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
img=cv2.flip(img,1)
cv2.imwrite("ring.png",img)
show(img)
'''
import multiprocessing as mp
procs=[]
for i in [8]:
    p=mp.Process(target=proc1,args=(i,))
    p.start()
    
    procs.append(p)
    #p.join()
for p in procs:
    p.join()
'''