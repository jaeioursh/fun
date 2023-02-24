from scipy.integrate import ode
import numpy as np
from math import exp,sin
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def f(t, y):
    return [-10*y[0]+exp(t)*y[1],-2*y[1]]


def f(t, y):
    return [-2*abs(y)*y+y-10]

def f(t, y):
    p=28
    o=10
    B=8./3.
    return [o*(y[1]-y[0]),y[0]*(p-y[2])-y[1],y[0]*y[1]-B*y[2]]


r = ode(f).set_integrator('dopri5')

y0, t0 = [10.0,10.0,0], 0
t1=100
dt=.01

r.set_initial_value(y0, t0)

Y=[]

while r.successful() and r.t < t1:
    val=r.integrate(r.t+dt)
    Y.append(val)

print(Y)
Y=np.array(Y)
x,y,z=Y.T
plt.plot(x, y, z)
#plt.plot(x,y)

#plt.plot(Y.T[0])
#x=np.linspace(-10,10,1000)
#plt.plot(x,[f(x_,x_) for x_ in x])

plt.show()