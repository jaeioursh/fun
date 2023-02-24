import numpy as np
import cv2 
RES=6
SQRS=20
PIX=1080

ref=np.full([SQRS, SQRS, 3], 0 , dtype=np.uint8)
#Note I fill with 0 instead of 255
ref[0][SQRS-1] = (255,1,1)
ref[0][0] = (255,255,1)
ref[SQRS-1][SQRS-1] = (1,1,255)
ref[SQRS-1][0] = (1,255,255)


from scipy.interpolate import griddata

#we format the data to feed in griddata
points=np.where(ref != 0)
values=ref[points]
grid_x,grid_y,grid_z=np.mgrid[0:SQRS,0:SQRS,0:3]

#we compute the inperpolation
filled_grid=griddata(points, values, (grid_x, grid_y, grid_z), method='linear')
filled_grid=np.array(filled_grid,dtype=np.uint8)



points=np.where(filled_grid != 0)
values=filled_grid[points]
near_grid=griddata(points, values, (grid_x, grid_y, grid_z), method='nearest')
completed_grid=(near_grid*(filled_grid == 0))+filled_grid

colors=np.reshape(completed_grid,(-1,3))

def transform1(sqrs,t):
    sqrs=sqrs.copy()
    for sqr in sqrs:
        for i in range(len(sqr[0])):
            X,Y=sqr[0][i],sqr[1][i]
            x=X*np.sqrt(1-Y*Y/2)*0.8*t+(1-t)*X
            y=Y*np.sqrt(1-X*X/2)*0.8*t+(1-t)*Y
            sqr[0][i],sqr[1][i]=x,y
    return sqrs

def transform2(sqrs,t):
    sqrs=sqrs.copy()
    for sqr in sqrs:
        for i in range(len(sqr[0])):
            X,Y=sqr[0][i],sqr[1][i]
            r=X*X+Y*Y
            x=(X*np.sin(r)-Y*np.cos(r))*0.8*t+(1-t)*X
            y=(X*np.cos(r)+Y*np.sin(r))*0.8*t+(1-t)*Y
            sqr[0][i],sqr[1][i]=x,y
    return sqrs

def transform3(sqrs,t):
    sqrs=sqrs.copy()
    for sqr in sqrs:
        for i in range(len(sqr[0])):
            X,Y=sqr[0][i],sqr[1][i]
            r=np.sqrt(X*X+Y*Y)+0.1
            x=(X-Y)*(X+Y)/r*0.8*t+(1-t)*X
            y=X*Y/r*0.8*t+(1-t)*Y
            sqr[0][i],sqr[1][i]=x,y
    return sqrs

def transform4(sqrs,t):
    sqrs=sqrs.copy()
    for sqr in sqrs:
        for i in range(len(sqr[0])):
            X,Y=sqr[0][i],sqr[1][i]
            r=X*X+Y*Y
            x=(X*np.sin(r)+Y*np.cos(r))*0.8*t+(1-t)*X
            y=(X*np.cos(r)+Y*np.sin(r))*0.8*t+(1-t)*Y
            sqr[0][i],sqr[1][i]=x,y
    return sqrs

def transform5(sqrs,t):
    sqrs=sqrs.copy()
    for sqr in sqrs:
        for i in range(len(sqr[0])):
            X,Y=sqr[0][i],sqr[1][i]
            thet=np.arctan2(Y,X)
            r=np.sqrt(X*X+Y*Y)+0.1
            x=(np.sin(thet)/r)*0.8*t+(1-t)*X
            y=(r*np.cos(thet))*0.8*t+(1-t)*Y
            sqr[0][i],sqr[1][i]=x,y
    return sqrs

a=np.linspace(0,1,RES)
b=np.ones(RES-1)
s=np.vstack
f=np.flip

sqrs=[]
for i in range(SQRS):
    for j in range(SQRS):
        sqr=np.hstack((s((a[:-1],b*0)),s((b,a[:-1])),s((f(a)[:-1],b)),s((b*0,f(a)[:-1]))))

        sqr[0]+=i
        sqr[1]+=j
        sqrs.append(sqr)

sqrs=np.array(sqrs)
sqrs/=SQRS/2
sqrs-=1.0
SQRS=sqrs
size=(1080,1080)
out = cv2.VideoWriter('pics/transform.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)           

for tf in [transform1,transform2,transform3,transform4,transform5]:
    for q in range(120):
        thet=np.pi*2/120*q
        t=1-(np.cos(thet)+1)/2
        sqrs=tf(SQRS.copy(),t)
        img=np.zeros((PIX*2,PIX*2,3),np.uint8)
        for k in range(2):
            for sqr,clr in zip(sqrs.copy(),colors):

                sqr=sqr.T

                sqr=sqr.reshape((-1,1,2))

                sqr+=1.0
                sqr*=PIX
                sqr=sqr.astype(np.int32)
                if k:
                    cv2.polylines(img,[sqr],True,(0,0,0),4)
                else:

                    cv2.fillPoly(img,[sqr],tuple(clr.astype(int).tolist()))

        img=cv2.resize(img,(PIX,PIX),interpolation=cv2.INTER_AREA)
        out.write(img)

        #cv2.imshow("txt",img)
        #cv2.waitKey(0)
        print(q,thet,t)
out.release()
