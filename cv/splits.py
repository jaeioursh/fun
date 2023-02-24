import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate(n,iters=300):
    x=np.random.random((1,n))
    y=np.random.random((1,n))
    for i in range(iters):
        X=(x-x.T)
        Y=(y-y.T)
        R=X*X+Y*Y+np.eye(n)*100
        R=np.sqrt(R)
        X/=R*n
        Y/=R*n
        idx=np.arange(n)
        m=np.argmin(R,0)
        x-=X[idx,m]
        y-=Y[idx,m]
        x[x<0]=0
        y[y<0]=0
        x[x>1]=1
        y[y>1]=1
    return x,y


def draw(x,y):
    res=1080*2
    ax=np.linspace(0,1,res)
    XX,YY=np.meshgrid(ax,ax)
    im=[]
    for X,Y in zip(XX,YY):
        line=np.zeros(res).astype(np.uint8)

        X=np.array([X]).T-x
        Y=np.array([Y]).T-y
       
        R=np.sqrt(X*X+Y*Y)
        R=np.sort(R,1)
        
        line[(R[:,1]-R[:,0])<0.0025]=250
        im.append(line)        
    return np.array(im)
        

x,y = generate(1000,500)
print("drawing")
im=draw(x,y)
im=cv2.resize(im, (1080,1080), interpolation = cv2.INTER_AREA)
cv2.imwrite("pics/split.png",im)
cv2.imshow("blank",im)
cv2.waitKey(5000)

plt.scatter(x,y)
plt.show()

