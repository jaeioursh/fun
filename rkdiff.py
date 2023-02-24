import numpy as np

def f(t,y):
    return np.array([y[1],-y[0]])

def rk(dt,t0,y0,f):
    y0=np.array(y0,dtype=float)
    T=[t0[0]]
    Y=[y0]
    y=y0
    t=t0[0]
    h=dt
    while t<t0[1]:
        k1=f(t,y)
        k2=f(t+h/2,y+h*(k1)/2)
        k3=f(t+h/2,y+h*(k2)/2)
        k4=f(t+h,y+h*(k3))
        y=y+h*(k1+2*k2+2*k3+k4)/6
        t+=h
        Y.append(y)
        T.append(t)
    return T,np.array(Y) 

t,y=rk(.01,[0,1000],[1,0],f)
print(len(t))
import matplotlib.pyplot as plt 
plt.plot(t,y[:,0])
plt.show()