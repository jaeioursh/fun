
from collections import deque
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from genagent import agent,net
from time import sleep

class Game:

    def __init__(self):
        self.speed=0.05
        self.hof=deque(maxlen=10)
        self.b_delay=10
        self.reset()
        self.player_r=0.005
        self.b_speed=0.10
        self.p_speed=0.05
        self.draw_flag=0
        
        if self.draw_flag:
            plt.ion()

    def reset(self):
        self.pos1=np.array([-1.0,0.0])
        self.pos2=np.array([1.0,0.0])
        self.delay1=self.b_delay
        self.delay2=self.b_delay
        self.bullets1=[]
        self.bullets2=[]
        self.minr=1e9

    def draw(self,i,r):

        plt.clf()
        plt.title(str(i)+"   "+str(r))
        plt.scatter(self.pos1[0],self.pos1[1],c="k")
        plt.scatter(self.pos2[0],self.pos2[1],c="k")
        for bull,C in zip([self.bullets1,self.bullets2],["r","b"]):
            blt=np.array(bull)
            #print(blt)
            if len(blt)>0:
                plt.scatter(blt.T[0,:],blt.T[1,:],c=C)
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.pause(0.05)

    def collide(self):

        if (self.pos1[0]**2+self.pos1[1]**2)>4:
            return -1
        
        if (self.pos2[0]**2+self.pos2[1]**2)>4:
            return 1
        
        if len(self.bullets2)>0:
            diff=np.array(self.bullets2)[:,:2]-self.pos1
            r=diff[:,0]**2+diff[:,1]**2
            #print(r)
            if np.any(r<self.player_r):
                return -1
        if len(self.bullets1)>0:
            
            diff=np.array(self.bullets1)[:,:2]-self.pos2
            r=diff[:,0]**2+diff[:,1]**2
            #print(self.bullets1,self.pos2)
            if np.any(r<self.player_r):
                return 1
            
            if np.min(r)<self.minr:
                self.minr=np.min(r)
                #print(self.minr)
        
        return min(1,1/(self.minr*1000))
    
    def step(self,player,a):
        a=(a+1)/2
        ang,vel,fire=a[0]
        ang*=pi*2
        if player==1:
            self.bullets1=[b for b in self.bullets1 if -2<b[0]<2 and -2<b[1]<2]
            plyr=self.pos1
            blt=self.bullets1
            enemy=self.pos2
            if self.delay1>0:
                self.delay1-=1
            if fire>0.5 and self.delay1==0:
                self.delay1=self.b_delay

            dly=self.delay1
        else:
            self.bullets2=[b for b in self.bullets2 if -2<b[0]<2 and -2<b[1]<2]
            plyr=self.pos2
            blt=self.bullets2
            enemy=self.pos1
            if self.delay2>0:
                self.delay2-=1
            
            if fire>0.5 and self.delay2==0:
                self.delay2=self.b_delay

            dly=self.delay2
            ang+=pi
            if ang>2*pi:
                ang-=2*pi 
        
        if dly==self.b_delay:
            blt.append([plyr[0],plyr[1],ang])

        for i in range(len(blt)):
            blt[i][0]+=self.b_speed*np.cos(blt[i][2])
            blt[i][1]+=self.b_speed*np.sin(blt[i][2])
        
        
        
        plyr[0]=plyr[0]+vel*self.p_speed*np.cos(ang)
        plyr[1]=plyr[1]+vel*self.p_speed*np.sin(ang)
        
        


    def state(self,player):
        if player==1:
            plyr=self.pos1
            blt=self.bullets2
            enemy=self.pos2
            dly=self.delay1
        else:
            plyr=self.pos2
            blt=self.bullets1
            enemy=self.pos1
            dly=self.delay2 

        diff=plyr-enemy
        d1=np.sum((diff)**2.0)/8
        t1=np.arctan2(diff[1],diff[0])+pi
        d2=np.sum(plyr**2.0)/4
        t2=np.arctan2(plyr[1],plyr[0])+pi
        if player==1:
            t1+=pi
            if t1>2*pi:
                t1-=2*pi
            t2+=pi
            if t2>2*pi:
                t2-=2*pi

        if len(blt)>0:
            diff=plyr-np.array(blt)[:,:2]
            if player==1:
                diff=-diff

            dist=(diff[:,0]**2.0+diff[:,1]**2.0)*10

            ang=np.arctan2(diff[:,1],diff[:,0])+pi
            temp=ang.copy()
            ang*=4/(2*pi)
            ang=ang.astype(int)
            #print(ang,temp,i,dist)
            vals=[1/(1+np.sum(np.maximum(np.zeros_like(dist[ang==i]),dist[ang==i]))) for i in range(4)]
        else:
            vals=[1 for i in range(4)]
            
        return np.hstack((d1,d2,t1/7,t2/7,vals,not dly))

    def play(self,pol1,pol2,draw=0):
        self.reset()
        for i in range(100):
            s1=self.state(1)
            s2=self.state(2)
            #print("space")
            #print(s1,s2)
            a1=pol1(s1)
            a2=pol2.feed(s2)
            #print(s1,s2,a1,a2)
            #a1=np.array([[np.random.random()*2-1,0,1]])
        
            #a1=np.array([[1,.15,.75]])
            #a2=a1.copy()
            self.step(1,a1)
            self.step(2,a2)
            r=self.collide()
            if draw:
                self.draw(i,r)
                #sleep(1)
            if abs(r)==1:
                break
        
        #print(r)
        return r

    def add_plyr(self,plyr):
        cp=net(plyr.shape)
        cp.copy(plyr)
        self.hof.append(cp)

    def eval(self,plyr,draw=0):
        f=0
        for p in self.hof:
            f+=self.play(plyr,p,draw)
        return f 

shape=[9,10,3]
env=Game()
ai=agent(shape,20,env.eval)  


env.add_plyr(ai.best)
#env.draw_flag=1
for i in range(1000):
    best=ai.train()
    print(best.e,i,len(env.hof))
        
    if best.e==len(env.hof):
        env.add_plyr(best)
    if i%50==0 and i > 5:
        env.eval(best.feed,1)





