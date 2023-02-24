import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from math import cos,atan 
from sklearn.cluster import KMeans

def pareto_optimal(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


class ea:
    def __init__(self,s,n):
        self.pop=np.random.normal(size=(n,s+1))/100+np.array([[.1,-.2,0]]*(n))
        self.n=n
        self.s=s
    def pts(self):
        return self.pop[:,:-1]

    def evolve(self,fn):
        for i in range(self.n):
            vals=self.pop[i,:-1]
            self.pop[i,-1]=fn(vals)
        groups=4
        self.n_best=groups
        pops=[[] for i in range(groups)]
        kmeans = KMeans(n_clusters=groups, random_state=0).fit(self.pts())
        n=kmeans.labels_
        for i in range(self.n):
            pops[n[i]].append(self.pop[i])
        trash=[]
        pop=[]
        for i in range(groups):
            p=sorted(pops[i],key=lambda x:-x[-1])
            pop.append(p[0])
            if len(p)>1:
                trash+=p[1:]

        for i in range(len(trash)):
            best=pop[i%groups]
            worst=best.copy()
            worst+=np.random.normal(size=(self.s+1))/100
            pop.append(worst)
        pop=np.array(pop)
        self.pop=pop
        self.best=sorted(self.pop[:groups],key=lambda x: x[-1],reverse=True)[0]
        return self.best
        '''
        dists=distance_matrix(self.pop[:,:-1],self.pop[:,:-1])
        #dists[dists==0.0]=np.inf
        dst=np.max(dists,axis=0)
        fit=self.pop[:,-1]
        costs=np.vstack([dst,fit]).T
        dom=pareto_optimal(costs)
        #print(dom,costs)
        #print(costs.shape,dom)
        best=list(self.pop[dom])
        n_best=len(best)
        self.n_best=n_best
        for i in range(self.n-n_best):
            best.append(best[i%n_best].copy())
            best[-1]+=np.random.normal(size=(self.s+1))/100
        self.pop=np.array(best)
        #print(best)
        '''

    def best(self):
        top=max(self.pop,key=lambda x:x[-1])
        return top[:-1],top[-1]


def simion(vals):
    #print(vals)
    x,y=vals
    if x**2.0+y**2.0 <= (1+0.2*cos(8*atan(x/y)))**2.0:
        return 0.1*x*y
    else:
        return -10

def fn2img(fn,rng):
    x=np.linspace(-rng,rng,200)
    y=x.copy()
    xx,yy=np.meshgrid(x,y)
    zz=np.array([list(map(fn,np.vstack(([X],[Y])).T)) for X,Y in zip(xx,yy)])
    zz[zz==-10]=np.nan
    return xx,yy,zz

def plot_fn(img,rng,pts):
    xx,yy,zz=img
    plt.ion()
    #print(zz)
    plt.contourf(xx,yy,zz,20)
    plt.scatter(pts[:,0],pts[:,1],color="k",s=4)
    plt.pause(0.2)

img=fn2img(simion,1.5)
learner=ea(2,100)
for i in range(1000):

    learner.evolve(simion)
    if i%10==0:
        plot_fn(img,1.5,learner.pts())
        print(learner.n_best,len(learner.pop))