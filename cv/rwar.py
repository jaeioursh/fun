from imp import reload
from math import atan2,log
from pickle import REDUCE
from pickletools import read_unicodestringnl
from sys import setprofile
import cv2
import numpy as np
from random import random,choice

RED=(255,100,100)
BLUE=(100,100,255)
LRED=(255,140,140)
LBLUE=(140,140,255)
BLACK=(0,0,0)
RGREY=(150,0,0)
BGREY=(0,0,150)
YELLOW=(150,255,255)
class bullet:
    def __init__(self,pos,dmg,team,size,speed):
        x,y,t=pos
        self.x=x
        self.y=y
        self.x_=x
        self.y_=y
        self.t=t+(random()-0.5)/20
        self.dmg=dmg
        self.team=team
        self.size=size
        self.speed=speed
        self.hit=0
    
    def dist(self,X,Y):
        x3,y3=X,Y
        x1,y1=self.x,self.y
        x2,y2=self.x_,self.y_
        px = x2-x1
        py = y2-y1

        norm = px*px + py*py
        u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = (dx*dx + dy*dy)**.5

        return dist


    def step(self):
        self.x_=self.x
        self.y_=self.y
        self.x=self.x_+np.cos(self.t)*self.speed
        self.y=self.y_+np.sin(self.t)*self.speed

    def draw(self,img):
        start=(int(self.x),int(self.y))
        end=(int(self.x_),int(self.y_))
        if self.team==0:
            color=LRED
        else:
            color=LBLUE
        #color=YELLOW
        img=cv2.line(img,start,end,color,self.size)
        return img


class robot:
    def __init__(self,team,health,model,dmg,pos,size,speed,clip,rps,lim,rload):
        self.team=team
        self.health=health
        self.hp_max=health
        self.speed=speed
        self.model=model
        self.dmg=dmg
        self.rps=rps
        self.pos=pos
        self.size=size
        self.clip=clip
        self.mag=clip
        self.reload_delay=rload
        self.dead=0
        self.lim=lim
        self.lock_on=None
        self.offset=np.random.randint(0,60)

    def rot_trans(self):
        x,y,t=self.pos
        t-=np.pi/2
        rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        mdl=rot@self.model.T
        mdl=mdl.T
        mdl[:,0]+=x
        mdl[:,1]+=y
        return mdl

    def shoot(self):
                                            #size   speed
        return bullet(self.pos,self.dmg,self.team,2,4)

    def draw(self,img):
        mdl=self.rot_trans()
        num=mdl.shape[0]//4
        boxes=np.split(mdl,num,axis=0)
        color2=(255,255,255)
        if self.team==0:
            color=RED
        else:
            color=BLUE
        if self.dead:
            color=BLACK
            if self.team==0:
                color2=RGREY
            else:
                color2=BGREY
        for box in boxes:
            vertices=np.array(box,dtype=np.int32)
            img=cv2.fillPoly(img, [vertices],color)

            img=cv2.polylines(img, [vertices],1,color2,2)
        return img

    def draw_hp(self,img):
        x,y,t=self.pos
        sz=20
        box1=np.array([[-sz,0],[-sz,6],[sz,6],[sz,0]],dtype=int)
        
        hp=self.health/self.hp_max*2*sz
        box2=np.array([[-sz,0],[-sz,6],[-sz+hp,6],[-sz+hp,0]],dtype=int)

        box1[:,1]+=int(y+self.size*0.7)
        box1[:,0]+=int(x)
        
        box2[:,1]+=int(y+self.size*0.7)
        box2[:,0]+=int(x)

        img=cv2.polylines(img, [box1],1,(255,255,0),2)
        img=cv2.fillPoly(img, [box2],(255,255,255))

        return img


    def move(self,ang,vel):
        x,y,t=self.pos
        t+=ang
        x+=vel*np.cos(t)*self.speed
        y+=vel*np.sin(t)*self.speed
        if t>np.pi:
            t-=np.pi*2
        if t<-np.pi:
            t+=np.pi*2
        if y>self.lim-1:
            y=self.lim-1
        if y<0:
            y=0
        if x>self.lim-1:
            x=self.lim-1
        if x<0:
            x=0
        self.pos=(x,y,t)
    
    def target(self,x,y,enemies):
        best=None
        highest=1e9
        for r in enemies:
            if not r.dead:
                X,Y,T=r.pos
                R=(x-X)**2.0+(y-Y)**2.0
                if R<highest:
                    highest=R
                    best=r
        return best

    def weighted(self,x,y,enemies):
        P=[]
        R=[]
        for r in enemies:
            if r.dead==0:
                R.append(r)
                X,Y,_=r.pos
                H=r.hp_max
                dist=(x-X)**2.0+(y-Y)**2.0
                P.append(log(H)/dist)
        if len(R)==0:
            return None
        P=np.array(P)/sum(P)
        idx=np.arange(0,len(R))
        idx=np.random.choice(idx,p=P)
        return R[idx]

    def ai(self,enemies):
        x,y,t=self.pos
        if self.lock_on==None or self.lock_on.dead==1:
            #self.lock_on=self.target(x,y,enemies)
            self.lock_on=self.weighted(x,y,enemies)
        best=self.lock_on
        if best is None:
            return (0.0,0.0)
        X,Y,T=self.lock_on.pos
        highest=(x-X)**2.0+(y-Y)**2.0
        vel=1.0
        if highest**0.5<200:
            vel=0.0

        ang2=atan2(Y-y,X-x)
        diff=-(t-ang2)
        
        if diff>np.pi:
            diff-=np.pi*2
        if diff<-np.pi:
            diff+=np.pi*2

        if diff>0.1:
            diff=0.1
        if diff<-0.1:
            diff=-0.1
        
        return diff,vel
    def alt(self):
        return None

class large(robot):
    def __init__(self,pos,team,lim):
        size=8*4
        model=np.array([[0,0],[0,8],[2,8],[2,0],  [2,1],[6,1],[6,3],[2,3],  [6,0],[6,8],[8,8],[8,0]])-np.array([[4,4]]*12)
        model*=4
        clip_size=4
        rload=4
        super().__init__(team,3000,model,100,pos,size,2,clip_size,6,lim,rload)

    def shoot(self):
        self.clip-=1
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None

        x,y,t=self.pos
        
        if self.clip%2==0:
            x+=np.cos(np.pi/4+t)*self.size*0.6
            y+=np.sin(np.pi/4+t)*self.size*0.6
            pos=x,y,t
        else:
            x+=np.cos(-np.pi/4+t)*self.size*0.6
            y+=np.sin(-np.pi/4+t)*self.size*0.6
        pos=(x,y,t)
                                            #size   speed
        return bullet(pos,self.dmg,self.team,8,8)


class medium(robot):
    def __init__(self,pos,team,lim):
        size=6*4
        model=np.array([[0,1],[0,4],[2,4],[2,1],  [2,0],[2,6],[4,6],[4,0],  [4,1],[4,4],[6,4],[6,1]])-np.array([[3,3]]*12)
        model*=4
        clip_size=6
        rload=4
        super().__init__(team,1000,model,80,pos,size,3,clip_size,4,lim,rload)

    def shoot(self):
        self.clip-=1
        x,y,t=self.pos
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None

        x+=np.cos(t)*self.size*0.6
        y+=np.sin(t)*self.size*0.6
        pos=x,y,t
        
                                            #size   speed
        return bullet(pos,self.dmg,self.team,3,20)

class turret(robot):
    def __init__(self,pos,team,lim):
        size=4*4
        model=np.array([[0,0],[4,0],[4,1],[0,1],  [1,1],[3,1],[3,4],[1,4]])-np.array([[2,2]]*8)
        model*=4
        clip_size=1
        rload=1
        super().__init__(team,1000,model,500,pos,size,0.0,clip_size,15,lim,rload)

    def shoot(self):
        self.clip-=1
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None
        return bullet(self.pos,self.dmg,self.team,2,50)

class uber(robot):
    def __init__(self,pos,team,lim):
        size=10*6
        model=np.array([[0,4],[1,4],[1,8],[0,8],
                        [1,6],[2,6],[2,7],[1,7],
                        [2,5],[3,5],[3,9],[2,9],
                        [3,4],[7,4],[7,7],[3,7],
                        [7,5],[8,5],[8,9],[7,9],
                        [8,6],[9,6],[9,7],[8,7],
                        [9,4],[10,4],[10,8],[9,8],
                        [2,-1],[8,-1],[8,4],[2,4]])-np.array([[5,4]]*32)
        model*=6
        clip_size=8
        rload=8
        super().__init__(team,10000,model,300,pos,size,1,clip_size,6,lim,rload)

    def shoot(self):
        
        self.clip-=1
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None

        x,y,t=self.pos
        
        idx=self.clip%4
        if idx>1:
            idx+=1
        if idx==3:
            idx=1
        elif idx==1:
            idx=3
        ang=-np.pi/4+idx*np.pi/8+t
        x+=np.cos(ang)*self.size*0.6
        y+=np.sin(ang)*self.size*0.6
        pos=(x,y,t)
                                            #size   speed
        return bullet(pos,self.dmg,self.team,12,8)

class organic(robot):
    def __init__(self,pos,team,lim):
        size=np.random.randint(4,10)
        model=rng_mdl(size)
        size*=8
        model*=8
        clip_size=np.random.randint(2,10)
        rload=np.random.randint(2,10)
        speed=(np.random.random()*3+1)
        health=10**speed
        dmg=100*(np.random.random()*2+1)
        fire_delay=np.random.randint(3,10)
        super().__init__(team,health,model,dmg,pos,size,5-speed,clip_size,fire_delay,lim,rload)
        
    def shoot(self):
        
        self.clip-=1
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None

        x,y,t=self.pos
        
        idx=np.random.random()
        ang=-np.pi/2+idx*np.pi/2+t
        x+=np.cos(ang)*self.size*0.6
        y+=np.sin(ang)*self.size*0.6
        pos=(x,y,t)
                                            #size   speed
        #print(self.speed)
        return bullet(pos,self.dmg,self.team,int(self.speed*2),int((5-self.speed)*3))

class replicator(robot):
    def __init__(self,pos,team,lim,delay=0):
        size=6*4
        model=np.array([[0,0],[6,0],[6,2],[0,2],  [2,2],[2,4],[4,4],[4,2], [0,4],[2,4],[2,6],[0,6], [4,4],[6,4],[6,6],[4,6]])-np.array([[3,3]]*16)
        model*=4
        clip_size=6
        rload=6     
        self.replicate=45
        self.delay=delay
        self.counter=np.random.randint(0,self.replicate//2)
        super().__init__(team,150,model,70,pos,size,2,clip_size,5,lim,rload)

    
    def ai(self,enemies):
        self.delay-=1
        if self.delay>0:
            return (0.0,6.0)
        else:
            return robot.ai(self,enemies)

    def shoot(self):
        if self.delay>0:
            return None
        self.clip-=1
        x,y,t=self.pos
        if self.clip<0:
            if -self.clip==self.reload_delay:
                self.clip=self.mag
            return None

        x+=np.cos(t)*self.size*0.6
        y+=np.sin(t)*self.size*0.6
        pos=x,y,t
        
                                            #size   speed
        return bullet(pos,self.dmg,self.team,3,12)

    def alt(self):
        if self.delay>0:
            return None
        self.counter+=1
        if self.counter%self.replicate==0:
            theta=(np.random.random()-0.5)*np.pi*2
            x,y,t=self.pos
            #x+=np.cos(theta)*self.size*2.5
            #y+=np.sin(theta)*self.size*2.5
            return replicator((x,y,theta),self.team,self.lim,6)
        else:
            return None


    
    
def rng_mdl(size):
    N=(size*size)//4
    design=np.zeros((size+2,size+2),dtype=bool)
    center=design[1:-1,1:-1]
    L=design[:-2,1:-1]
    R=design[2:,1:-1]
    U=design[1:-1,:-2]
    D=design[1:-1,2:]
    center[size//2,size//2]=1
    for i in range(N):
        x,y=np.random.randint(0,size,2,dtype=int)
        while  center[x,y] or not (U[x,y] or D[x,y] or L[x,y] or R[x,y]):
            x,y=np.random.randint(0,size,2,dtype=int)
        center[x,y]=1
        
    boxes=[]
    for I in range(size):
        for J in range(size):
            if center[I,J]:
                i=I-size//2
                j=J-size//2
                boxes+=[[i,j],[i+1,j],[i+1,j+1],[i,j+1]]
    return np.array(boxes)

class game:
    def __init__(self,specs1,specs2):
        self.map_size=2160
        self.res=216//2
        self.team1=self.gen_teams(specs1,0)
        self.team2=self.gen_teams(specs2,1)
        self.idx=0
        self.bullets=[]
        self.fps=30
        self.rob_limit=800

    def gen_teams2(self,specs,team):
        robs=[]
        for spec in specs:
            x=random()*self.map_size
            y=(random()+1)*self.map_size*0.1
            t=np.pi/2
            if team==1:
                y+=0.7*self.map_size
                t*=-1
            pos=(x,y,t)
            robs.append(spec(pos,team,self.map_size))
        return robs

    def gen_teams(self,specs,team):
        robs=[]
        slots=np.zeros((5,25),dtype=bool)

        for spec in specs:
            r,c=choice(np.argwhere(~slots))
            slots[r,c]=1
            x=(c+0.5)*(self.map_size/25)
            y=(r+0.5)*(self.map_size/25)
            t=np.pi/2
            if team==1:
                y+=self.map_size*20/25
                t*=-1
            pos=(x,y,t)
            robs.append(spec(pos,team,self.map_size))
        return robs

    def collisions(self):
        N=self.map_size//self.res+6
        for team,idx in [[self.team1,1],[self.team2,0]]:
            bmap=[[[] for i in range(N)] for j in range(N)]
            for b in self.bullets:
                if b.team==idx:
                    X=int(b.x//self.res)+3
                    Y=int(b.y//self.res)+3
                    bmap[X][Y].append(b)
            for r in team:
                if not r.dead:
                    x,y,t=r.pos
                    X=int(x//self.res)+3
                    Y=int(y//self.res)+3
                    bullets=bmap[X][Y]
                    for b in bullets:
                        if b.dist(x,y)<r.size:
                            b.hit=1
                            r.health-=b.dmg
                        if r.health<0:
                            r.dead=1
                            continue
    def winner(self):
        for team in [self.team1,self.team2]:
            alive=0
            for r in team:
                if not r.dead:
                    alive+=1
            if alive==0:
                return 1
        return 0
    def bullet_step(self):
        for b in self.bullets:
            b.step()
            if not ( 0<b.x<self.map_size-1 and 0<b.y<self.map_size-1 ):
                b.hit=1
        
        self.bullets=[b for b in self.bullets if not b.hit]

    def step(self):
        self.idx+=1
        self.collisions()
        if self.winner():
            if len(self.bullets)>0:
                self.bullet_step()
                return self.draw()
            return None
        for team1,team2 in [[self.team1,self.team2],[self.team2,self.team1]]:
            for r in team1:
                if not r.dead:
                    if len(team1)<self.rob_limit:
                        new_r=r.alt()
                        if not new_r is None:
                            team1.append(new_r)
                    ang,vel=r.ai(team2)
                    r.move(ang,vel)
                    if (self.idx+r.offset) % r.rps==0:
                        b=r.shoot()
                        if not b is None:
                            self.bullets.append(b)
        
        self.bullet_step()
        return self.draw()

    def draw(self,hp_bar=1):
        img=np.zeros((self.map_size,self.map_size,3),dtype=np.uint8)
        robs=self.team1+self.team2

        for r in robs:
            if r.dead:
                img=r.draw(img)

        for r in robs:
            if not r.dead:
                img=r.draw(img)
                if hp_bar:
                    img=r.draw_hp(img)

        for b in self.bullets:
            img=b.draw(img)
        img = cv2.resize(img, (self.map_size//4, self.map_size//4),interpolation=cv2.INTER_AREA)
        cv2.imshow('sim', img)
        cv2.waitKey(1)#int(1000/self.fps))
        return img

specs1=[large]*22+[medium]*90+[uber]*3+[turret]*5+[organic]*5+[replicator]*0
specs2=[replicator]*75+[medium]*48+[uber]*2
#specs=[organic]*30
sim=game(specs1,specs2)
frames=[]
for i in range(1000):
    f=sim.step()
    frames.append(f)

if 1:
    itr=1
    writer = cv2.VideoWriter("vids/rwar"+str(itr)+".mp4", cv2.VideoWriter_fourcc(*'MP4V'),sim.fps,(sim.map_size//4,sim.map_size//4))
    for f in frames:
        writer.write(f)
    writer.release()
    #ffmpeg -i vids/rwar1.mp4 -vcodec libx264 -crf 24 vids/rwar2.mp4 
    #use 24-30 // lower= higher quality, lower compression

    
        