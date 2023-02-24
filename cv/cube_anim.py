
import cv2 
import numpy as np
from math import sin,cos,pi

def gen_cube(height):
    pts=[]
    for i in range(6):
        ang=pi*(i/3+1/6)
        pts.append([cos(ang),sin(ang)+1])
    pts.append([0,1])
    for i in [0,1,2,6]:
        pts[i][1]+=height-1 

    return np.array(pts)

def draw_cube(img,cube,color):
    vs,center=cube[:-1],cube[-1]
    
    pts = np.array(vs, np.int32)
    center=np.array(center,np.int32)
    #print(pts)
    pts = pts.reshape((-1,1,2))
    img = cv2.fillPoly(img,[pts],color)
    
    
    for i in [0,2,4]:
        pt=pts[i][0]
        img= cv2.line(img,tuple(center),tuple(pt),(100,100,100),1)
    

    img = cv2.polylines(img,[pts],True,(0,0,0),2)

    return img

def rotate(cube,theta):
    theta*=pi/180
    R=[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]
    R=np.array(R)
    #print(cube.T,R)
    pts=np.matmul(cube,R)
    return pts
def Value(x,y,t):
    r=(x**2.0+y**2.0)**0.5
    v1=[x/r,y/r]
    v2=[cos(t*pi/180),sin(t*pi/180)]
    V=np.dot(v1,v2)
    v=max(V,0.0)
    return v**2.0

def all_cubes(t):
    t=t+90
    cubes=[]
    for q in range(3):
        for i in range(4,-1,-1):
            for j in range(4,-1,-1):
                theta=t+q*120
                dy=sin(pi/6)
                dx=cos(pi/6)
                y=(i+j)*dy+0.01
                x=(i-j)*dx
                r=(x**2.0+y**2.0)**0.5
                V=Value(x,y,theta)
                cube=gen_cube(V*(0.8*r+.2))
                cube[:,0]+=x
                cube[:,1]+=y
                color=[255,255,255]
                color[q-1]=int(255-V*255)
                color[q-2]=int(255-V*255)
                cube=rotate(cube,q*120)
                cubes.append([cube,tuple(color)])

    #cub=rotate(cub,90)
    return cubes

height=1000
width =1000
size = (width,height)
out = cv2.VideoWriter('pics/project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
 

for t in range(0,360,2):
    

    img = np.zeros((height,width,3), np.uint8)



    cubes=all_cubes(t)

    for cube in cubes:
        cub,color=cube

        cub*=50
        cub[:,0]+=width/2
        cub[:,1]+=height/2



        img=draw_cube(img,cub,color)

    img=np.flipud(img)
    #cv2.imshow("blank",img)
    #cv2.waitKey(0) 
    out.write(img)
out.release()

