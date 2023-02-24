
from math import sqrt
import random
from copy import deepcopy as copy
from numba import jit
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import colorsys

class bot:

    def __init__(self,actions,states,r_max,hunger_max):
        self.x=0
        self.y=0
        self.life=0
        self.hunger=0
        self.size=0
        self.policy=[]
        self.r_delay=0
        self.id=0
        self.artificial=0

        self.actions=actions
        self.states=states
        self.r_max=r_max
        self.hunger_max=hunger_max
        self.reset()

    def generate(self,depth):
        
        if random.random()<0.4 or depth==0:
            idx=random.randint(0,self.actions-1)
            return [idx]
        else:
            idx=random.randint(0,self.states-1)
            val=random.random()
            return [idx,val,self.generate(depth-1),self.generate(depth-1)]        

    def crossover(self,a,b):
        pass 
    
    def mutate(self):
        q=[self.policy]
        while len(q)>0:
            piece=q.pop()
            if len(piece)==1:
                if random.random()<0.1:
                    piece[0]=random.randint(0,self.actions-1)
            if len(piece)==4:
                idx,val,L,R=piece
                if random.random()<0.1:
                    piece[0]=random.randint(0,self.states-1)
                if random.random()<0.1:
                    piece[1]=random.random()
                q.append(L)
                q.append(R)
        

    def mate(self):
        self.r_delay=self.r_max
        a=bot(self.actions,self.states,self.r_max,self.hunger_max)
        a.x=self.x+10*random.random()-5
        a.y=self.y+10*random.random()-5
        a.id=self.id
        a.size=self.size
        a.color=self.color.copy()
        a.policy=copy(self.policy)
        a.mutate()
        return a

    def act(self,s):
        pol=self.policy
        #print(pol)
        while len(pol)>1:
            idx,val,L,R=pol
            if s[idx]<val:
                pol=L 
            else:
                pol=R
        return pol[0]
        

    def reset(self):
        x=random.random()*2-1
        y=random.random()*2-1
        while x*x+y*y > 1:
            x=random.random()*2-1
            y=random.random()*2-1
        self.id=random.random()
        self.color=np.array([i*255 for i in colorsys.hsv_to_rgb(self.id,1.0,1.0)],dtype=np.uint8)
        self.x=x*100
        self.y=y*100
        self.life=0
        self.hunger=self.hunger_max
        self.size=random.random()
        self.r_delay=self.r_max
        self.policy=self.generate(4)

    def step(self):
        self.life+=1
        self.hunger-=1
        if self.hunger==0:
            self.life=0
        if self.r_delay>0:
            self.r_delay-=1
        

    def eat(self):
        self.hunger=self.hunger_max
        #self.size+=0.02

def tree2str(tree):
    A=["MoveToSmall","MoveAwaySmall","MoveToLarge","MoveAwayLarge","FindFood","Reproduce"]
    S=["DistToSmall","DistToLarge","DistToFood","Hunger","RepDelay"]
    lines=[]
    q=[[tree,0]]
    while len(q)>0:
        piece,depth=q.pop()
        space="|---"*depth
        if len(piece)==1:
            idx=piece[0]
            lines.append(space+"Action: "+A[idx])
        if len(piece)==4:
            
            idx,val,L,R=piece
            lines.append(space+"If "+S[idx]+" < "+str(round(val,2)))
            q.append([R,depth+1])
            q.append([L,depth+1])

    return "\n".join(lines)

@jit(nopython=True)
def dists(n_pop,n_food,posr,posf):
    vals=np.zeros((n_pop,12))-1
    for i in range(n_pop):
        large=1e9
        small=1e9
        X,Y,S,ID=posr[i]
        for j in range(n_pop):
            if i==j:
                continue
            x,y,s,id=posr[j]
            if ID==id:
                continue
            dx=x-X
            dy=y-Y
            d=sqrt(dx*dx+dy*dy)+0.0001
            if s<S and d<small:
                small=d
                vals[i,0]=d
                vals[i,1]=dx/d
                vals[i,2]=dy/d
                vals[i,3]=j
            if S<s and d<large:
                large=d
                vals[i,4]=d
                vals[i,5]=dx/d
                vals[i,6]=dy/d
                vals[i,7]=j
        close=1e9
        for j in range(n_food):
            x,y=posf[j]
            dx=x-X
            dy=y-Y
            d=sqrt(dx*dx+dy*dy)+0.0001
            if d<close:
                close=d
                vals[i,8]=d
                vals[i,9]=dx/d
                vals[i,10]=dy/d
                vals[i,11]=j
    return vals


class sim:
    def __init__(self):
        self.img=np.zeros((804,804,3),dtype=np.uint8)
        self.min_pop=50
        self.max_pop=200
        self.n_food=10

        self.a=6
        self.s=5
        self.hunger_max=100
        self.r_max=120
        self.best=0
        self.best_policy=[]
        self.pop=[bot(self.a,self.s,self.r_max,self.hunger_max) for i in range(self.min_pop)]
        self.food=np.array([self.food_spawn() for i in range(self.n_food)])

    def states(self):

        posr=np.array([[p.x,p.y,p.size,p.id] for p in self.pop])
        return dists(len(self.pop),self.n_food,posr,self.food)
    #         0        1      2     3       4      
    #state smaller, larger, food, hunger, r_delay
    #            0        1         2         3          4        5
    #action mov2sm, mov away sm, mov to lg, move away, food, reproduce
    def actions(self,S):
        dead=[]

        for i in range(len(self.pop)):
            
            s,rob=S[i],self.pop[i]

            rob.step()
            if rob.life<=0:
                dead.append(i)    
                continue

            state=[s[0]/100,s[4]/100,s[8]/100,rob.hunger/rob.hunger_max,rob.r_delay/rob.r_max]
            a=rob.act(state)
            if a==0: #move to smallest
                dx,dy,idx=s[1],s[2],int(s[3])
                distance=s[0]
                if idx>=0:
                    rob.x+=dx
                    rob.y+=dy
                if distance<1.5:
                    rob.eat()
                    self.pop[idx].life=-1
                    dead.append(idx)
            if a==1: # move away from smallest
                dx,dy,idx=s[1],s[2],s[3]
                if idx>=0:
                    rob.x+=dx
                    rob.y+=dy
                
            if a==2: #move to largest
                dx,dy,idx=s[5],s[6],s[7]
                if idx>=0:
                    rob.x+=dx
                    rob.y+=dy
            if a==3: #move away
                dx,dy,idx=s[5],s[6],s[7]
                if idx>=0:
                    rob.x+=dx
                    rob.y+=dy
            if a==4: #eat food
                dx,dy,idx=s[9],s[10],int(s[11])
                distance=s[8]
                if idx>=0:
                    rob.x+=dx
                    rob.y+=dy
                if distance<1.5:
                    rob.eat()
                    self.food[idx,:]=self.food_spawn()
            if a==5: #reproduce
                if len(self.pop)<self.max_pop and rob.r_delay==0:
                    self.pop.append(rob.mate())
        self.pop=[self.pop[i] for i in range(len(self.pop)) if i not in dead]
        too_low=self.min_pop-len(self.pop)
        if too_low>0:
            for i in range(too_low):
                self.pop.append(bot(self.a,self.s,self.r_max,self.hunger_max))
    def step(self):
        S=self.states()
        self.actions(S)
        best=self.get_best()
        if best.life>self.best:
            self.best=best.life
            self.best_policy=best.policy

    def food_spawn(self):  
        x=random.random()*2-1
        y=random.random()*2-1
        while x*x+y*y > 1:
            x=random.random()*2-1
            y=random.random()*2-1
        return [x*100,y*100]          
    def get_best(self):
        return max(self.pop,key=lambda x:x.life)

    def draw(self):
        img=self.img
        img[img>25]-=25
        img[img<=25]=0
        red=[250,0,0]
        blue=[0,0,250]
        yellow=[250,250,0]
        white=[250,250,250]
        black=[0,0,0]
        for p in self.pop:
            x=int(p.x*3+400)
            y=int(p.y*3+400)
            if 0<=x<800 and 0<=y<800:
                sz=int(p.size*5+1)
                if p.artificial:
                    sz+=3
                    img[y:y+sz,x:x+sz]=white
                    img[y+1:y+sz-1,x+1:x+sz-1]=black
                else:
                    img[y:y+sz,x:x+sz]=p.color

        for p in self.food:
            x,y=p
            
            x=int(x*3+400)
            y=int(y*3+400)
            if 0<=x<800 and 0<=y<800:
                img[y:y+4,x:x+4]=blue
        p=self.get_best()
        x=int(p.x*3+400)
        y=int(p.y*3+400)
        sz=int(p.size*5+1)+3
        if 0<=x<800 and 0<=y<800:
            img[y:y+sz,x:x+sz]=yellow
            img[y+1:y+sz-1,x+1:x+sz-1]=black

        return img

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.time=0
        self.pol=[]
        self.best=0
        # configure the root window     \
        self.title('Life')
        self.geometry('1100x800')
        Lframe=tk.Frame(self)
        Lframe.pack(side=tk.LEFT)
        Rframe=tk.Frame(self)
        Rframe.pack(side=tk.RIGHT)
        # label
        self.label = ttk.Label(Rframe, text='Hello, Tkinter!')
        self.label.pack(side=tk.TOP)
        
        # button
        self.button1 = ttk.Button(Lframe, text='1,000')
        self.button1['command'] = self.onek
        self.button1.pack(side=tk.TOP)

        # button
        self.button2 = ttk.Button(Lframe, text='10,000')
        self.button2['command'] = self.tenk
        self.button2.pack(side=tk.TOP)

        # button
        self.button3 = ttk.Button(Lframe, text='Reset')
        self.button3['command'] = self.reset
        self.button3.pack(side=tk.TOP)

        spacer1=tk.Label(Lframe,text="\n\n")
        spacer1.pack(side=tk.TOP)

        self.text=tk.Text(Lframe, height=1,width=10)
        self.text.pack(side=tk.TOP)

        self.label3=tk.Label(Rframe, anchor="w", justify=tk.LEFT)
        self.label3.pack(side=tk.TOP)

        self.label2=tk.Label(Rframe, anchor="w", justify=tk.LEFT)
        self.label2.pack(side=tk.TOP)

        

        self.button4 = ttk.Button(Lframe, text='Insert')
        self.button4['command'] = self.insert
        self.button4.pack(side=tk.TOP)

        self.button5 = ttk.Button(Lframe, text='Insert Best')
        self.button5['command'] = self.insert_best
        self.button5.pack(side=tk.TOP)

        #canvas
        self.canvas = tk.Canvas(self,width=804,height=804)
        self.canvas.pack(side=tk.BOTTOM)
        self.game=sim()

    def update_frame(self):
        self.game.step()
        #print(self.game.get_best().life)
        best=self.game.get_best()
        pop=self.game.pop

        self.label["text"]="Oldest: "+str(self.game.best)+"\nTime: "+str(self.time)+"\nPop Size: " +str(len(self.game.pop))+"\n\n"
        
        txt="Ranking\n"
        for i in range(min(len(pop),5)):
            txt+=str(i+1)+") "+str(pop[i].life)+"\n"
        
        self.label3["text"]=txt

        self.time+=1
        img=self.game.draw()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(10,10,anchor="nw",image=self.img)
        self.after(1000//30,self.update_frame)

        self.label2["text"]="Current Best:\n"+tree2str(best.policy)+"\n\nRecord Holder:\n\n"+tree2str(self.game.best_policy)




    def onek(self):
        for i in range(1000):
            self.time+=1
            self.game.step()
    def tenk(self):
        for i in range(10000):
            self.time+=1
            self.game.step()

    def insert(self):
        txt = self.text.get("1.0",'end-1c')
        y=json.loads(txt)
        self.game.pop.append(bot(self.game.a,self.game.s,self.game.r_max,self.game.hunger_max))
        self.game.pop[-1].policy=y
        self.game.pop[-1].artificial=1
        self.game.pop[-1].color=np.array([255,255,255],dtype=np.uint8)
        
    def insert_best(self):
        self.game.pop.append(bot(self.game.a,self.game.s,self.game.r_max,self.game.hunger_max))
        self.game.pop[-1].policy=self.game.best_policy
        self.game.pop[-1].artificial=1
        self.game.pop[-1].color=np.array([255,255,255],dtype=np.uint8)

    def reset(self):
        self.time=0
        self.best=0
        self.pol=[]
        self.game=sim()
if __name__ == "__main__":
    
    app = App()
    app.update_frame()
    app.mainloop()
    '''
    a=[4,0.5,[1],[3,0.4,[1],[2]]]
    print(a)
    tree2str(a)
    '''
    