import numpy as np
from scipy.signal import convolve2d
import cv2 


def grow(c,i):
    k=np.ones((3,3))
    k[0,0]=0
    k[0,2]=0
    k[2,0]=0
    k[2,2]=0
    q=np.zeros_like(c)
    q[c>0]=250
    con=convolve2d(q,k,"same")
    vals=np.unique(con)
    vals=vals[vals!=0]
    vals=vals[vals<(250*np.sum(k))]

    counts=[]
    for v in vals:
        counts.append(np.sum(vals[v==vals]))
    counts=1/(np.array(counts))
    counts/=np.sum(counts)
    x=np.random.choice(vals,p=counts)
    #print(x)
    #x=vals[0]
    
    c[con==x]=250-i//3
    return c


c=np.zeros((1080//2,1080//2))
c[270:272,270:272]=250
for i in range(600):
    c=grow(c,i)
    print(i)
c=c.astype(np.uint8)
c = cv2.resize(c, dsize=(1080, 1080), interpolation=cv2.INTER_NEAREST)
cv2.imshow("blank",c)
cv2.waitKey(0)


