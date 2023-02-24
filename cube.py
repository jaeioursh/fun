import cv2
import numpy as np 
from random import random

'''


#img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

#img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.fillPoly(img,[pts],(0,255,255))
'''

def rand_color():
    a=[int(random()**(0.5)*255) for i in range(3)]
    print(tuple(a))
    a=cv2.cvtColor(np.array([[[int(random()*255),int(random()*150+100.0),255]]],np.uint8),cv2.COLOR_HSV2BGR)[0][0]
    a=tuple([int(c) for c in a])
    print(a)
    return a

def gen_boxs():
    low=.20
    high=.25
    boxes=[[[0.0,0.0],[1.0,1.0]]]
    done=[]
    while len(boxes)>0:
        box=boxes.pop()
        p1,p2=box
        x1,y1=p1
        x2,y2=p2

        dx,dy=x2-x1,y2-y1
        if dx<high and dy<high:
            done.append(box)
        else:
            div=random()*0.5+0.25
            if random()<0.5:
                if dx>low:
                    x3=x1+div*dx
                    boxes.append([p1,[x3,y2]])
                    boxes.append([[x3,y1],p2])
                else:
                    boxes.append(box)
            else:
                if dy>low:
                    y3=y1+div*dy
                    boxes.append([p1,[x2,y3]])
                    boxes.append([[x1,y3],p2])
                else:
                    boxes.append(box)
    return done

def cubify(box):
    vs=[]

    p1,p2=np.array(box)
    dx,dy=p2[0]-p1[0],p2[1]-p1[1]

    height=(min(dx,dy)) * .5
    
    if 1:
        over=max(p2+height)
        if over>1.0:
            height-=over-1.0

    p3,p4=p1+height,p2+height

    


    vs.append(p1)
    vs.append([p1[0],p2[1]])
    vs.append([p3[0],p4[1]])
    vs.append(p4)
    vs.append([p4[0],p3[1]])
    vs.append([p2[0],p1[1]])

    vs.append(p3)

    return np.array(vs)

def draw_cube(img,cube):
    vs,center=cube[:-1],cube[-1]
    
    pts = np.array(vs, np.int32)
    center=np.array(center,np.int32)
    #print(pts)
    pts = pts.reshape((-1,1,2))
    img = cv2.fillPoly(img,[pts],rand_color())
    
    
    for i in [0,2,4]:
        pt=pts[i][0]
        img= cv2.line(img,tuple(center),tuple(pt),(100,100,100),1)
    

    img = cv2.polylines(img,[pts],True,(0,0,0),2)

    return img


def add_plane(img,scale,mat,dx,dy):
    boxes=gen_boxs()
    boxes=sorted(boxes,key=lambda x:(x[1][0]+x[0][0])/2+(x[0][1]+x[1][1])/2,reverse=True)
    for box in boxes:
        

        cube=cubify(box)*scale
        
        cube=np.matmul(cube,mat)
        
        cube[:,0]+=dx
        cube[:,1]+=dy

        img=draw_cube(img,cube)

    return img

def add_cube(img,scale,dx,dy):
    #right
    mat=np.array([[1,-.5],[0,1]])
    img=add_plane(img,scale,mat,dx+10,dy)

    #left
    mat=np.array([[-1,-.5],[0,1]])
    img=add_plane(img,scale,mat,dx-10,dy)

    #top
    mat=np.array([[1,-.5],[-1,-.5]])
    img=add_plane(img,scale,mat,dx,dy-20)

    return img


height=1337
width =750

img = np.zeros((height,width,3), np.uint8)
dx=width//4
dy=height//8
scale=min(width,height)//2

scale/=2.25
SHIFT=1.72

img=add_cube(img,scale,dx*0,dy*(4-2*SHIFT))
img=add_cube(img,scale,dx*2,dy*(4-2*SHIFT))
img=add_cube(img,scale,dx*4,dy*(4-2*SHIFT))

img=add_cube(img,scale,dx*0,dy*4)
img=add_cube(img,scale,dx*2,dy*4)
img=add_cube(img,scale,dx*4,dy*4)

img=add_cube(img,scale,dx*1,dy*(4-1*SHIFT))
img=add_cube(img,scale,dx*3,dy*(4-1*SHIFT))

img=add_cube(img,scale,dx*0,dy*(4+2*SHIFT))
img=add_cube(img,scale,dx*2,dy*(4+2*SHIFT))
img=add_cube(img,scale,dx*4,dy*(4+2*SHIFT))

img=add_cube(img,scale,dx*1,dy*(4+1*SHIFT))
img=add_cube(img,scale,dx*3,dy*(4+1*SHIFT))



'''


for box in boxs:
    p1,p2=box
    color=rand_color()
    print(p1,p2,color)
    img=cv2.rectangle(img,tuple(p1),tuple(p2),tuple(color),-1)
    img=cv2.rectangle(img,tuple(p1),tuple(p2),(0,0,0),2)
print(boxs)
print(pts)
'''
import matplotlib.pyplot as plt
# load image using cv2....and do processing.
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()

cv2.imwrite("pics/cube.png",img)

