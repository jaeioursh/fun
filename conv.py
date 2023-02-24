import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

mnist_data = list(mnist_data)[:4096]
#https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
TRAIN=1
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.S1=100
        self.S2=100
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, self.S1, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Conv2d(self.S1, self.S2, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Conv2d(self.S2, 2, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, self.S2, 7),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.ConvTranspose2d(self.S2, self.S1, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.ConvTranspose2d(self.S1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def decode(self,x):
        x = self.decoder(x)
        return x

    def encode(self,x):
        x = self.encoder(x)
        return x


class Autoencoder2(Autoencoder):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.S1=128
        self.S2=128
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, self.S1, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Conv2d(self.S1, self.S2, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Conv2d(self.S2, 2, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, self.S2, 7),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.ConvTranspose2d(self.S2, self.S1, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.ConvTranspose2d(self.S1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )



def train(model,fname, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)#, 
                                 #weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        

        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch%(num_epochs/10)==0:
            print("saving")
            torch.save(model.state_dict(), fname)

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    torch.save(model.state_dict(), fname)
    return outputs
if __name__ =="__main__":
    model = Autoencoder2()
    print(model.encoder)
    max_epochs = 500
    PATH="here4.sav"
    if TRAIN:
        outputs = train(model,PATH, num_epochs=max_epochs)
        

        for k in range(0, max_epochs, max_epochs//4):
            plt.figure(figsize=(9, 2))
            imgs = outputs[k][1].detach().numpy()
            recon = outputs[k][2].detach().numpy()
            for i, item in enumerate(imgs):
                if i >= 9: break
                plt.subplot(2, 9, i+1)
                plt.imshow(item[0])
                
            for i, item in enumerate(recon):
                if i >= 9: break
                plt.subplot(2, 9, 9+i+1)
                plt.imshow(item[0])
        plt.show()
    