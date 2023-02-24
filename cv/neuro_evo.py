
from http.client import NOT_ACCEPTABLE
from re import S
import numpy as np
import cv2
from math import sin,cos,pi,atan2,sqrt
import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody,vec2,filter)
from genagent import agent,mapelite
import matplotlib.pyplot as plt
# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
class Sim:
    def __init__(self,blue_print,body_idx,limits,powers,reward,time):
        self.PPM = 30.0  # pixels per meter
        self.TARGET_FPS = 30
        self.TIME_STEP = 1000 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1080, 1080
        self.viewing=0
        self.body_idx=body_idx
        self.powers=powers
        self.reward=reward
        self.time=time
        # --- pybox2d world setup ---
        # Create the world
        self.world = world(gravity=(0, -10), doSleep=False)

        # And a static body to hold the ground shape
        
        for i in range(20):
            ground_body = self.world.CreateStaticBody(
                position=(i*20, 0),
                angle=0,
                shapes=polygonShape(box=(10, 1)),
            )
            #print(ground_body)
        # Create a couple dynamic bodies
        bodies=[]
        conditions=[]
        blue_print,conns=blue_print
        for i in range(len(blue_print)):
            x,y,X,Y=blue_print[i]
            
            dx,dy=X-x,Y-y
            mx,my=(x+X)/2,(y+Y)/2
            r=sqrt(dx*dx+dy*dy)
            theta=atan2(dy,dx)
            conditions.append([mx,my,theta])
            body = self.world.CreateDynamicBody(position=(mx, my), angle=theta)
            if i==body_idx:
                thickness=0.5
            else:
                thickness=0.25
            box = body.CreatePolygonFixture(box=(r/2, thickness), density=1, friction=0.8,groupIndex=-1)
            #print(box)
            bodies.append(body)

        
        #bodyA = self.world.CreateDynamicBody(position=(20, 15))
        #circle = bodyA.CreateCircleFixture(radius=0.5, density=1, friction=0.93)
        
        
        #bodyB = self.world.CreateDynamicBody(position=(30, 15), angle=0)
        #box = bodyB.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)
        motors=[]
        for i in range(len(conns)):
            if conns[i]<0:
                continue
            A=bodies[i]
            B=bodies[conns[i]]
            anch=vec2(blue_print[i][2:])

            motor = self.world.CreateRevoluteJoint(
                bodyA=A,
                bodyB=B,
                anchor=anch,
                lowerAngle=-limits[i] * pi / 180.0,
                upperAngle=limits[i] * pi / 180.0,
                enableLimit=True,
                enableMotor=True,
                maxMotorTorque=int(powers[i]*10),
            )
            motors.append(motor)
            
    
        #print(body)

        self.bodies=bodies
        self.conditions=conditions
        self.motors=motors
        self.n_state=2+len(motors)
        self.n_action=len(motors)

        # Let's play with extending the shape classes to draw for us.


        def my_draw_polygon(polygon, body, img,offset,color=[255,255,255],dx=0,dy=0):

            vertices = [(body.transform * v- offset) * self.PPM  for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            vertices=np.array(vertices,dtype=np.int32)
            return cv2.polylines(img, [vertices],1,color,2)
        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, img,offset,color=(255,255,255),dx=0,dy=0):
            position = (body.transform * circle.pos-offset) * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            ang=body.transform.angle
            r=circle.radius
            pos=circle.pos
            center=tuple([int(x) for x in position])
            end=(int(center[0]+cos(-ang)*r* self.PPM),int(center[1]+sin(-ang)*r* self.PPM))
            
            img=cv2.circle(img,center, int(r * self.PPM),color,-1)
            img=cv2.line(img,center,end,(0,0,0),2)
            return img
            # Note: Python 3.x will enforce that pygame get the integers it requests,
            #       and it will not convert from float.
        circleShape.draw = my_draw_circle
        

        # --- main game loop ---
    def vis(self,string=""):
        if self.viewing==0:
            cv2.namedWindow('sim')
            self.viewing==1
        img=np.zeros((self.SCREEN_HEIGHT,self.SCREEN_WIDTH,3),dtype=np.uint8)
        torso=self.bodies[self.body_idx].position-vec2((self.SCREEN_WIDTH/self.PPM/2,self.SCREEN_HEIGHT/self.PPM/5))
        torso[1]-=self.bodies[self.body_idx].position[1]
        for body in self.world.bodies:
            for fixture in body.fixtures:
                img=fixture.shape.draw(body, img,torso)
        # font
        font = cv2.FONT_HERSHEY_COMPLEX
        # org
        org = (10, 60)
        # fontScale
        fontScale = 2
        # Blue color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        image = cv2.putText(img, string, org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow('sim', img)
        cv2.waitKey(int(self.TIME_STEP))
        return img

    def step(self):
            #                   time        vel pos
        self.world.Step(self.TIME_STEP/1000, 50, 50)
    
    def reset(self):
        for i in range(len(self.bodies)):
            body=self.bodies[i]
            x,y,t=self.conditions[i]

            body.angularVelocity=0.0
            body.linearVelocity=vec2((0.0,0.0))
            body.position=vec2((x,y))
            body.angle=t

        for m in self.motors:
            m.motorSpeed=0
    
    def state(self):
        s=[]
        for m in self.motors:
            s.append(m.angle)
        body=self.bodies[self.body_idx]
        s.append(body.angle)
        s.append(body.position[1])
        return s 
    
    def action(self,act):
        for a,m in zip(act,self.motors):
            m.motorSpeed=int(a*10)

    def env(self,pol,view=False,string=""):
        self.reset()
        body=self.bodies[self.body_idx]
        highest=-1e9
        frames=[]
        for i in range(self.TARGET_FPS*self.time):
            s=self.state()
            s=np.array([s])
            act=pol(s)[0]
            self.action(act)
            
            if body.position[1]>highest:
                highest=body.position[1]
            self.step()
            if view:
                frames.append(self.vis(string))
        if view:
            return frames
        if self.reward==0:
            return body.position[0]
        if self.reward==1:
            return highest
        
def creature(idx):
    if idx==0:
        blue_print=[[0,0],[1,1],[0,3],[5,3],[4,1],[5,0]]
        blue_print=[[bp[0]+1,bp[1]+1] for bp in blue_print]
        blue_print=[blue_print[i]+blue_print[i+1] for i in range(len(blue_print)-1) ]
        conns=[i+1 for i in range(len(blue_print)-1)]
        limits=[40]*(len(blue_print)-1)
        powers=[35]*(len(blue_print)-1)
        blue_print=[blue_print,conns]
        body_idx=2
        name="dog"
        itr=1
    if idx==1:
        blue_print=[[i*2,0] for i in range(10)]
        blue_print=[[bp[0]+1,bp[1]+1] for bp in blue_print]
        blue_print=[blue_print[i]+blue_print[i+1] for i in range(len(blue_print)-1) ]
        conns=[i+1 for i in range(len(blue_print)-1)]
        limits=[40]*(len(blue_print)-1)
        powers=[25]*(len(blue_print)-1)
        body_idx=len(blue_print)-1
        blue_print=[blue_print,conns]
        
        name="snake"
        itr=0
    if idx==2:
        blue_print=[[1,0],[0,1],[1,3],[0,5],[5,5],[6,3],[5,1],[6,0]]
        blue_print=[[bp[0]+1,bp[1]+1] for bp in blue_print]
        blue_print=[blue_print[i]+blue_print[i+1] for i in range(len(blue_print)-1) ]
        conns=[i+1 for i in range(len(blue_print)-1)]
        limits=[40]*(len(blue_print)-1)
        powers=[25]*(len(blue_print)-1)
        blue_print=[blue_print,conns]
        body_idx=3
        name="cheetah"
        itr=0
    if idx==3:    #    0          1          2         3        4         5          6         7        8            
        blue_print=[[1,1,2,2],[2,2,1,3],[1,3,5,3],[1,1,2,2],[2,2,1,3],[5,1,6,2],[6,2,5,3],[5,1,6,2],[6,2,5,3]]
        conns=[         1,        2,        -1,        4,        2,        6,        2,        8,       2     ]
        
        body_idx=2
        limits=[40]*len(conns)
        powers=[25]*len(conns)
        blue_print=[blue_print,conns]
        name="crab"
        itr=2
    if idx==4:    #        0          1          2         3        4         5          6         7          8             9         10       11
        blue_print=[[1.5,1.5,1,2],[1,2,2,3],[2,3,1,4],[1.5,1.5,1,2],[1,2,2,3],[2,3,1,4], [1,4,1,7],[3,7,2,6],[2,6,1,7],[3,7,2,6],[2,6,1,7],[1,8,1,7]]
        conns=[         1,           2,         6,         4,         5,         6,        -1,        8,         6,          10,      6,         6    ]
        powers=[       35,          35,         35,       35,          35,       35,       -1,       10,        10,         10,      10,       10    ]
        body_idx=6
        limits=[45]*len(conns)
        #powers=[25]*len(conns)
        blue_print=[blue_print,conns]
        name="person"
        itr=0
    return blue_print,body_idx,limits,powers,name,itr

if __name__=="__main__":
    cret=creature(idx=4)
    blue_print,body_idx,limits,powers,name,itr=cret

    reward=0  #0:speed , 1:height
    time=12
    sim=Sim(blue_print,body_idx,limits,powers,reward,time)


    s=[sim.n_state,12,sim.n_action]
    n=32
    #learner=agent(s,n,sim.env)
    learner=mapelite(s,env=sim.env)
    frequency=100
    frames=[]
    tests=[0,50,100,500,1000,5000,10000]
    for i in range(20001):
        learner.train()
        #learner.dtrain(groups=4)
        if i%100==0:
            print(i,learner.policy().e)

    if 1:
        import pickle
        with open('data/MAP2.pkl', 'wb') as outp:
            learner.environ=None
            pickle.dump([cret,learner], outp, pickle.HIGHEST_PROTOCOL)

    v=learner.P.copy()
    v[np.isneginf(v)]=0
    plt.imshow(v)
    plt.show()
        #if i in tests:
        #    frames+=sim.env(learner.policy().feed,True,"Generation: "+str(i))


            

    if len(frames)>0:
        writer = cv2.VideoWriter("vids/"+name+str(itr)+".mp4", cv2.VideoWriter_fourcc(*'MP4V'),sim.TARGET_FPS,(sim.SCREEN_WIDTH,sim.SCREEN_HEIGHT))
        for f in frames:
            writer.write(f)
        writer.release()



    '''
    for i in range(2):
        sim.reset()
        for i in range(200):
            if i%2==0:
                sim.action([1,1,1,1])
            else:
                sim.action([-1,-1,-1,-1])
            sim.step()
        print(sim.bodies[sim.body_idx])
    '''
