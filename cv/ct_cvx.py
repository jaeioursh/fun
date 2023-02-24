import numpy as np
import matplotlib.pyplot as plt
import quadprog 
from math import sin,cos,pi
import cvxpy as cp
#min .5 xPx + qx
#subject to Gx <= h
#           Ax  =  b

def solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)+np.eye(P.shape[0])*1e-3   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

n=10

a=np.zeros((n,n))


for i in range(360):
    x,y=[int( n/2+0.39*n*fn(i*pi/180) ) for fn in [cos,sin]]
    a[x,y]=1

im1=np.sum(a,axis=1)
im0=np.sum(a,axis=0)
N=n*n

b=np.arange(N)
b=b.reshape(a.shape)
b=np.vstack([b,b.T])


A=np.zeros((n*2,N))
for i in range(n*2):
    for idx in b[i]:
        A[i,idx]=1.0
B=np.hstack([im0,im1])
B=np.array([B]).T

print(A.shape,B.shape,np.linalg.matrix_rank(A))

#least squares 0.5 (Ax-B)^2
P=A.T@A
q=-A.T@B

G=-1*np.eye(N)
h=np.zeros(N)

if 0:
    A_=None#np.ones((N,1)).T
    b=None#np.array([1]).reshape(1,)

    x=solve_qp(P,q,G,h,A_,b)

else:
    vals=np.array([[0,1.0]]).T
    x = cp.Variable((N,1),boolean=True)#,len(vals)))
    print(B.shape)
    L=np.ones((len(vals),1))
    R=np.ones((N,1))
    objective = cp.Minimize(cp.norm( A @ x  - B ,1))
    #objective = cp.Minimize(cp.norm(x,1))
    constraints = [0.0 <= x]#, x@L == R]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    print(x.value)
    print(np.round(x.value,2))
    print(objective.value)
    
    x=x.value

x=x.reshape((n,n)).T

plt.subplot(2,2,1)
plt.imshow(a)
plt.subplot(2,2,2)
plt.imshow(x)
plt.subplot(2,2,3)
plt.imshow(np.vstack([im1,im0]))
plt.subplot(2,2,4)
im1=np.sum(x,axis=1)
im0=np.sum(x,axis=0)
plt.imshow(np.vstack([im1,im0]))
plt.show()