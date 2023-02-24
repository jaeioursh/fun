import pickle
import matplotlib.pyplot as plt
import numpy as np
from neuro_evo import Sim
with open('data/MAP.pkl', 'rb') as inp:
    cret,learner = pickle.load(inp)

blue_print,body_idx,limits,powers,name,itr=cret

reward=0  #0:speed , 1:height
time=12
sim=Sim(blue_print,body_idx,limits,powers,reward,time)

n=learner.n
x=learner.X

ii,iii=0,0
for i in range(n):
    for j in range(n):
        if not learner.X[i][j] is None:
            ii+=1
        if not np.isneginf(learner.P[i,j]):
            iii+=1
print(ii,iii,n,n*n)
fig1=plt.figure(1)
v=learner.P.copy()
#v[np.isneginf(v)]=np.nan
plt.imshow(v)

def onclick(event):
        
    encs=[int(event.xdata+0.5),int(event.ydata+0.5)]
    j,i=encs
    wb=learner.X[i][j]
    if wb is None:
        print(":(",i,j)
    else:
        print(learner.P[i,j])
        learner.ai.set(wb)
        sim.env(learner.ai.feed,True,"i "+str(i)+",  j "+str(j))
        fig1.canvas.draw()
        
        
fig1.canvas.mpl_connect('button_press_event', onclick)
plt.show()