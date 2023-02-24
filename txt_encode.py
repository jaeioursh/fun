import numpy as np
from nltk.corpus import words
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.S1=100
        self.S2=10
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Linear(26*6, self.S1),
            nn.Sigmoid(),
            nn.Linear(self.S1, self.S2),
            nn.Sigmoid(),
            nn.Linear(self.S2, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, self.S2),
            nn.Sigmoid(),
            nn.Linear(self.S2, self.S1),
            nn.Sigmoid(),
            nn.Linear(self.S1, 26*6),
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

def train(model,fname, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)#, 
                                 #weight_decay=1e-5) # <--

    outputs = []
    for epoch in range(num_epochs):
        for set in torch.split(data[:17000],1000):
            recon = model(set)
            loss = criterion(recon, set)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch%(num_epochs/10)==0:
            print("saving")
            torch.save(model.state_dict(), fname)

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        
    torch.save(model.state_dict(), fname)
    return outputs


data=[]
for w in words.words():
    if len(w)==6:
        word=w.lower()
        vars=[]
        
        for c in word:
            vec=np.zeros(26)
            c=ord(c)-ord("a")
            vec[c]=1.0
            vars.append(vec)

        vars=np.hstack(vars)
        data.append(vars)

def onclick(event):
        
        encs=[event.xdata,event.ydata]
        x=np.random.random((1,2))
        x[0,0]=encs[0]
        x[0,1]=encs[1]

        x=torch.tensor(x,dtype=torch.float)
        word=model.decode(x).detach().numpy()[0]
        word=np.reshape(word,(6,26))
        word=np.argmax(word,axis=1)
        word=[chr(w+ord("a")) for w in word]
        word="".join(word)
        print(word)


def onclick2(event):
        
        encs=[event.xdata,event.ydata]
        x=np.random.random(2)
        x[0]=encs[0]
        x[1]=encs[1]

        x=torch.tensor(x,dtype=torch.float)
        word=pca.inverse_transform(x)
        word=np.reshape(word,(6,26))
        word=np.argmax(word,axis=1)
        word=[chr(w+ord("a")) for w in word]
        word="".join(word)
        print(word)



data=np.array(data)
pca=PCA(n_components=2)
latent=pca.fit_transform(data)
print(latent.shape)
fig=plt.figure()
plt.scatter(latent[:,0],latent[:,1],s=4)
fig.canvas.mpl_connect('motion_notify_event', onclick2)
plt.show()
'''
print(data.shape)
data=torch.Tensor(data)
model = Autoencoder()
print(model.encoder)
max_epochs = 5000
PATH="data/txt2.sav"
if 0:
    train(model,PATH, num_epochs=max_epochs)
else:
    model.load_state_dict(torch.load(PATH))

fig=plt.figure(1)
latent=model.encode(data).detach().numpy()
plt.scatter(latent[:,0],latent[:,1],s=4)
fig.canvas.mpl_connect('motion_notify_event', onclick)
plt.show()
'''