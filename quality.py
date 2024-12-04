import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm, ticker
import sys
def matrix(P,Q,q):
    mat=np.zeros((10,10))
    for i in range(5):
        for j in range(5-i):
            if j==0:
                val=1.0+P
                rem=val*Q
                val-=rem
            else:
                val=rem
                rem=val*0.1
                val-=rem                
            j+=i+5
            if j==9:
                val+=rem
            mat[i,j]=val
    for i in range(4):
        for j in range(5-i):
            if j==0:
                val=0.25
                rem=val*q
                val-=rem
            else:
                val=rem
                rem=val*0.1
                val-=rem                
            j+=i
            if j==4:
                val+=rem
            mat[i+5,j]=val
    mat[9,9]=1.0
    return mat

def ratio(mat):
    vec=np.zeros(10)
    vec[5]=1.0
    for i in range(10):
        mat=mat@mat
    vec=mat.T@vec
    leg=vec[-1]
    return leg

def plot(recycleQ=0.01):
    P=np.linspace(0,3.,100)
    Q=np.linspace(0,.5,50)
    mat=[]
    for p in P:
        vec=[]
        for q in Q:
            m=matrix(p,q,recycleQ)
            vec.append(1.0/ratio(m))
            print(vec[-1],p,q)
        mat.append(vec)
    mat=np.array(mat)
    #plt.imshow(mat,extent=[min(P),max(P),min(Q),max(Q)],norm=LogNorm(1,10000), aspect=1)
    plt.xlabel("Productivity")
    plt.ylabel("Quality")
    plt.title("Recycled With Quality: "+str(recycleQ))
    print(np.sum(mat,axis=1))
    PP,QQ = np.meshgrid(P,Q,indexing="ij")
    tix1=ticker.LogLocator(base=2,numticks=100)
    tix2=ticker.LogLocator(base=np.sqrt(2),numticks=100)
    fmt = lambda x: '{:.0f}'.format(x)
    plt.contourf(PP,QQ,mat, locator=tix2)
    CS=plt.contour(PP,QQ,mat, locator=tix1,colors=["black"])
    plt.gca().clabel(CS, CS.levels, fmt=fmt, fontsize=10)
    x4=np.arange(5)
    x5=np.arange(6)
    x8=np.arange(9)
    plt.plot(x4*0.1,x4[::-1]*.025 ,marker="v",label="Assemblers")
    plt.plot(x4*0.25,x4[::-1]*.062,marker="v",label="Assemblers++")
    plt.plot(x5*0.1+0.5,x5[::-1]*.025 ,marker="P",label="Foundry")
    plt.plot(x5*0.25+0.5,x5[::-1]*.062,marker="P",label="Foundry++")
    plt.plot(x8*0.1,x8[::-1]*.025,marker="s",label="Cryo")
    plt.plot(x8*0.25,x8[::-1]*.062,marker="s",label="Cryo++")
    plt.legend()
    plt.show()
    
                
def plot_mat(m):      
    plt.imshow(m)
    plt.show()
    print(1.0/ratio(m))

plot(float(sys.argv[1]))

