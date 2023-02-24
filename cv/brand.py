import cv2
import numpy as np



def brand(img,sz,lr=1,ud=1):
    r,c=img.shape[0]*2,img.shape[1]*2

    if len(img.shape)<3:
        white = 1
        black = 0
    else:
        white = (1,1,1)
        black = (0,0,0) 

    wid=min(r,c)
    mask=np.zeros_like(img,dtype=np.float)
    mask=cv2.resize(mask, (0,0), fx = 2, fy = 2)

    

    offset=wid*sz*0.25
    scale=wid*sz
    
    mask=cv2.circle(mask, (int(scale/2+offset) , int(scale/2+offset)), int(scale/2), white, -1)

    box=np.array([[0.3,0],[0.7,0],[0.3,1],[0.7,1]])*scale+offset
    box=box.astype(int)
    cv2.fillPoly(mask, pts = [box], color =black)


    box=np.array([[.1,0],[1,0.9],[.9,1],[0,.1]])*scale+offset
    box=box.astype(int)
    cv2.fillPoly(mask, pts = [box], color = white)

    box=np.array([[-0.15,0.43],[-0.15,0.57],[1.15,.57],[1.15,.43]])*scale+offset
    box=box.astype(int)
    #cv2.fillPoly(mask, pts = [box], color = white)

    mask=cv2.circle(mask, (int(scale/2+offset) , int(scale/2+offset)), int(scale/2*0.75), black, -1)

    

    box=np.array([[.1,1],[0,0.9],[.9,0],[1,.1]])*scale+offset
    box=box.astype(int)

    cv2.fillPoly(mask, pts = [box], color = white)

    mask=cv2.resize(mask, (c//2,r//2), interpolation=cv2.INTER_AREA)
    if lr:
        mask=np.fliplr(mask)
    if ud:
        mask=np.flipud(mask)
    
    img2=img.copy()
    img2[mask>0]=255-img[mask>0]*mask[mask>0]
    return img2

def brand_vid(fname):
    vidcap = cv2.VideoCapture(fname)
    success,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fname2=fname.split(".")
    fname2=fname2[0]+"brand."+fname2[1]
    writer = cv2.VideoWriter(fname2, cv2.VideoWriter_fourcc(*'MP4V'),fps,(image.shape[0],image.shape[1]))
    while success:
        image=brand(image,0.05)
        writer.write(image)
        success,image = vidcap.read()
    writer.release()

    
if __name__=="__main__":
    #brand_vid("vids/save.mp4")
    pic=np.zeros((600,800,3),dtype=np.uint8)
    pic=brand(pic,0.05*3)
    cv2.imshow("test",pic)
    cv2.waitKey(0)