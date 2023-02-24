from conv import Autoencoder,Autoencoder2
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch 
import cv2
import matplotlib.pyplot as plt

model = Autoencoder2()
PATH="here3.sav"

def ART():
    
    model.load_state_dict(torch.load(PATH))

    def image(x1,y1):
        encs=[x1,y1]
        
        x=np.random.random((1,2,1,1))
        x[0,0,0,0]=encs[0]
        x[0,1,0,0]=encs[1]

        x=torch.tensor(x,dtype=torch.float)
        return model.decode(x).detach().numpy()[0,0]

    X=np.linspace(-10,20,50)
    Y=np.linspace(-10,20,len(X))

    IMS=[]
    for x2 in X:
        ims=[]
        for y2 in Y:
            ims.append(image(x2,y2))
        IMS.append(np.hstack(ims))
    ims=np.vstack(IMS)
    #ims=torch.tensor(ims)
    #ims=torch.reshape(ims,(len(X)*len(Y),1,28,28))
    #ims=torchvision.utils.make_grid(ims,len(X),padding=0)
    #ims=ims.numpy()
    #ims=ims.reshape((28*len(X),28*len(X),3))
    #print(np.max(ims[:,:,:]))
    print(ims.shape)
    cv2.imwrite("pics/mnist.png",ims*255)

def VIS():
    mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    mnist_data = list(mnist_data)[:4096]
    
    model.load_state_dict(torch.load(PATH))
        
    

    encs=[10.0,-10.0]
        
    fig1=plt.figure(1)
    
    x=np.random.random((1,2,1,1))
    x[0,0,0,0]=encs[0]
    x[0,1,0,0]=encs[1]
    
    x=torch.tensor(x,dtype=torch.float)
    img2=model.decode(x).detach().numpy()[0,0]
    
    IMAGE=plt.imshow(img2)
    #plt.show()
    imgs=[img for img,_ in mnist_data]
    lbls=[lbl for _,lbl in mnist_data]
    lbls=np.array(lbls)
    encs=[]
    fig2=plt.figure(2)
    for i in imgs:
        img=torch.reshape(i,(1,1,28,28))
        enc=model.encode(img)
        enc=enc.detach().numpy()
        S=enc.shape[1]
        encs.append(enc.reshape((S)))
    encs=np.array(encs)
    if len(encs[0])==3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    markers=["o","v","1","s","*","|","x","_","d","^"]
    for i in range(10):
        enc=encs[lbls==i]
        enc=enc.T
        if len(enc)==3:
            
            
            ax.scatter(enc[0],enc[1],enc[2])
        else:
            plt.scatter(enc[0],enc[1],marker=markers[i]) 
    plt.legend([str(i) for i in range(10)])
    
    def onclick(event):
        
        encs=[event.xdata,event.ydata]
    
    
        
        x=np.random.random((1,2,1,1))
        x[0,0,0,0]=encs[0]
        x[0,1,0,0]=encs[1]

        x=torch.tensor(x,dtype=torch.float)
        img2=model.decode(x).detach().numpy()[0,0]
        
        IMAGE.set_data(img2)        
        fig1.canvas.draw()
        
        
    #fig2.canvas.mpl_connect('button_press_event', onclick)
    fig2.canvas.mpl_connect('motion_notify_event', onclick)
    plt.show()

if __name__ =="__main__":
    #ART()
    VIS()