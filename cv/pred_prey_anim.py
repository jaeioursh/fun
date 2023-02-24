
import numpy as np
import cv2 
from random import randint,random,seed
from math import atan2,pi
#state, close, far, none
class learner():
    def __init__(self):
        self.epsilon=0.1
        self.lr=0.01
        self.gamma=0.99
        self.Q=np.zeros(((3**8),5))
        self.vec=np.power(np.ones(8)*3,np.arange(8))
        #print(self.vec)

    def s2i(self,s):
        s=np.array(s)
        i=np.dot(s,self.vec)
        return int(i)
    def act_greedy(self,s):
        i=self.s2i(s)
        return np.argmax(self.Q[i])

    def act_explore(self,s):
        if random()<self.epsilon:
            return np.random.randint(0,5)
        else:
            return self.act_greedy(s)

    def learn(self,s,a,r,sp):
        i=self.s2i(s)
        if sp == None:
            self.Q[i,a]=self.Q[i,a]*(1-self.lr)+(self.lr)*r
        else:
            ip=self.s2i(sp)
            self.Q[i,a]=self.Q[i,a]*(1-self.lr)+(self.lr)*(r+self.gamma*np.max(self.Q[ip]))
    def save(self,fname):
        with open(fname,"wb") as f:
            np.save(f,self.Q)

    def load(self,fname):
        with open(fname,"rb") as f:
            self.Q=np.load(f)

class Sim():
    def __init__(self,side=0,Z=5000,H=0,B=100):
        self.side=side
        self.H=1080//1
        self.W=1080//1
        self.STEPS=10000

        self.base=np.zeros((self.H,self.W,3),dtype=np.uint8)
        self.buildings(B)
        
        self.cur=self.base.copy()

        self.Nz=Z
        self.Nh=H

        self.z=[]
        self.h=[]
        self.d=[]
        self.zlearn=learner()
        self.hlearn=learner()

    def draw(self):
        self.cur=self.base.copy()
        for z in self.z:
            x,y=z
            self.cur = cv2.rectangle(self.cur, (x,y),(x+1,y+1), (0,0,255), -1)
        for h in self.h:
            x,y=h
            self.cur = cv2.rectangle(self.cur, (x,y),(x+1,y+1), (255,0,0), -1)
        for d in self.d:
            x,y=d
            self.cur = cv2.rectangle(self.cur, (x,y),(x+1,y+1), (100,100,100), -1)
        

    def reset(self):
        self.z=[]
        self.d=[]
        self.h=[]

        for j in range(self.Nz):
            while 1:
                x=randint(0,self.W-1)
                y=randint(0,self.H-1)
                #x,y=5,5
                if self.base[y,x,0]==0:
                    self.z.append([x,y])
                    break

        for j in range(self.Nh):
            while 1:
                x=randint(0,self.W-1)
                y=randint(0,self.H-1)
                #x,y=25,25
                if self.base[y,x,0]==0:
                    self.h.append([x,y])
                    break

    def sense_wall(self,x,y):
        wall=[]
        for dx,dy in [[0,1],[1,0],[0,-1],[-1,0]]: 
            for i in range(0,20,9):
                Y=y+i*dy
                X=x+i*dx
                if X<0 or X>=self.W or Y<0 or Y>=self.H or self.base[Y,X,0] == 255:
                    break
            if i<10:
                wall.append(0)
            #elif i<35:
            #    wall.append(1)
            else:
                wall.append(2)
        return wall

    def sense_other(self,x,y,others):
        close=[1e9 for i in range (4)]
        person=[0 for i in range (4)]
        for i in range(len(others)):
            X,Y=others[i]
            dx=X-x
            dy=Y-y
            r=(dx**2.0+dy**2.0)**0.5
            theta=(atan2(dy,dx)+pi)/(2*pi)*0.99999
            #print(theta)
            theta+=0.125
            if theta>=1.0:
                theta-=1.0
            idx=int(theta*4)
            #print(idx,theta)
            if close[idx]>r:
                close[idx]=r
                person[idx]=i
        for i in range(4):
            if close[i]<10:
                close[i]=0
            elif close[i]<100:
                close[i]=1
                person[i]=-1
            else:
                close[i]=2
                person[i]=-1
        return close,person

    def state(self):
        sh=[]
        sz=[]
        ph=[]
        pz=[]
        for h in self.h:
            x,y=h
            wall=self.sense_wall(x,y)
            #close,person=self.sense_other(x,y,self.z)
            close,person=[0,0,0,0],[0,0,0,0]
            sh.append(close+wall)
            ph.append(person)
        for z in self.z:
            x,y=z
            wall=self.sense_wall(x,y)
            close,person=self.sense_other(x,y,self.h)
            #print(close+wall,person)
            sz.append(close+wall)
            pz.append(person)
        return sh,sz,ph,pz


    def action(self,az,ah,pz,ph):
        turned=[-1]*len(self.z)
        killed=[-1]*len(self.h)
        zspeed=4
        hspeed=3
        delta=[[0,1],[1,0],[0,-1],[-1,0]]
        newz=[]
        newh=[]
        for i in range(len(self.h)):
            x,y=self.h[i]
            a=ah[i]
            p=ph[i]
            if a==0:
                X,Y=x,y
                '''
                for person in p:
                    if person>=0:
                        killed[i]=p[a] 
                        break    
                '''       
            else:
                a-=1
                dx,dy=delta[a]
                X,Y=x+dx*hspeed,y+dy*hspeed
                if X<0 or X>=self.W or Y<0 or Y>=self.H or self.base[Y,X,0] == 255:
                    X,Y=x,y
            newh.append([X,Y])

        for i in range(len(self.z)):
            x,y=self.z[i]
            a=az[i]
            p=pz[i]
            if a==0:
                X,Y=x,y
                for person in p:
                    if person>=0:
                        turned[i]=p[a]     
                        break       
            else:
                a-=1
                dx,dy=delta[a]
                X,Y=x+dx*zspeed,y+dy*zspeed
                if X<0 or X>=self.W or Y<0 or Y>=self.H or self.base[Y,X,0] == 255:
                    X,Y=x,y
            newz.append([X,Y])
        self.h=[]
        self.z=[]
        for i in range(len(newz)):
            if i in killed:
                self.d.append(newz[i])
            else:
                self.z.append(newz[i])

        for i in range(len(newh)):
            if i in turned:
                self.z.append(newh[i])
            else:
                self.h.append(newh[i])

        return killed,turned
             

    def buildings(self,num): 
        for i in range(num):
            x=randint(0,self.W-1)
            y=randint(0,self.H-1)
            dx=randint(10,50)
            dy=randint(10,50)
            self.base = cv2.rectangle(self.base, (x,y),(x+dx,y+dy), (255,255,255), -1)

    def show(self):
        self.draw()
        cv2.imshow("blank",self.cur)
        cv2.waitKey(100)

    def actions(self,sh,sz):
        ah=[]
        az=[]
        for s in sh:
            ah.append(self.hlearn.act_explore(s))
        for s in sz:
            az.append(self.zlearn.act_explore(s))
        return ah,az
    def update(self,oldh,oldz,ah,az,sh,sz,turned,killed):
        offset=0
        for i in range(len(oldh)):
            s,a=oldh[i],ah[i]
            r=0.01
            if killed[i]>=0:
                r+=0.25
            if i in turned:
                r-=1
                sp=None
                offset-=1
            else:
                sp=sh[i+offset]
            if self.side>=0:
                self.hlearn.learn(s,a,r,sp)
        offset=0
        
        for i in range(len(oldz)):
            s,a=oldz[i],az[i]
            
            r=-0.00
            if turned[i]>=0:
                r+=1.0
            if i in killed:
                r-=1
                sp=None
                offset-=1
            else:
                sp=sz[i+offset]
            if self.side<=0:
                self.zlearn.learn(s,a,r,sp)


    def play(self,show=False):
        self.reset()
        sh,sz,ph,pz=self.state()
        for i in range(self.STEPS):
        
            #ah,az=np.random.randint(0,8,size=len(self.h)),np.random.randint(0,8,size=len(self.z))
            ah,az=self.actions(sh,sz)

            killed,turned=self.action(az,ah,pz,ph)

            oldh,oldz=sh,sz
            sh,sz,ph,pz=self.state()
            self.update(oldh,oldz,ah,az,sh,sz,turned,killed)
            if show:
                self.show()

if __name__=="__main__":
    sim=Sim(-1)
    fname="sav/sav0"
    data=[]
    sim.zlearn.load(fname+"z.np")
    sim.hlearn.load(fname+"h.np") 
    N=50000  
    for i in range(N):
        
        sim.play(1)
        sim.hlearn.epsilon=0.5-0.5*float(i)/float(N)
        sim.zlearn.epsilon=0.5-0.5*float(i)/float(N)
        data.append(len(sim.z))
        if i%10==0:
            print(np.max(sim.zlearn.Q))
            print(np.min(sim.zlearn.Q))
            print(np.mean(sim.zlearn.Q)) 
            print(i,len(sim.z),len(sim.h),len(sim.d))
        if i%1000==0:
            sim.zlearn.save(fname+"z.np")
            sim.hlearn.save(fname+"h.np")
            np.save(fname+"d",np.array(data)) 
    #sim.show()
