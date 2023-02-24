#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from random import randint,gauss,shuffle,random,seed
from sklearn.cluster import KMeans

 
 
#np.random.seed(123)
#seed(123)
 
class net:
	def __init__(self,s):
		self.shape=s
		self.depth=len(s)-1

	   
		self.shuffle()
		self.e=0.0
		
	   
		
	def shuffle(self):
		s=self.shape
		self.w=[np.random.normal(size=[s[i],s[i+1]])/10 for i in range(len(s)-1)]
		self.b=[np.random.normal(size=[1,s[i+1]])/10 for i in range(len(s)-1)]
 
	def get(self):
		return [[w.copy() for w in self.w],[b.copy() for b in self.b]]
	
	def set(self,wb):
		self.w=[w.copy() for w in wb[0]] 
		self.b=[b.copy() for b in wb[1]]

   
	def cross(self,p,p1,p2):
		for i in range(len(p)):
			P=np.random.random(p[i].shape)<.5
			nP=np.logical_not(P)
			p[i][P]=p1[i][P]
			p[i][nP]=p2[i][nP]
	def copy(self,p):
		for i in range(len(self.w)):
			self.w[i]=p.w[i].copy()
			self.b[i]=p.b[i].copy()
			 
	   
	def crossover(self,p1,p2):
		self.cross(self.w,p1.w,p2.w)
		self.cross(self.b,p1.b,p2.b)

 
	def mut(self,p,m,rad):
		for i in range(len(p)):
			P=np.random.random(p[i].shape) > m
			if(self.bloom<0.95):
				d=np.random.normal(0,rad,p[i].shape)
			else:
				d=np.random.normal(0,1.0/rad,p[i].shape)
			d[P]=0
			p[i]+=d
				
	def mutate(self,mut,rad):
		self.bloom=random()
		self.mut(self.w,mut,rad)
		self.mut(self.b,mut,rad)

		   
	def s(self,x):
		return 1.0/(1.0+np.exp(-x))
   
	def h(self,x):
		return np.tanh(x)
 
	def l(self,x):
		return x
   
	def flatten(self):
		return np.hstack([w.flatten() for w in self.w]+[b.flatten() for b in self.b])
	   
	
	def feed(self,x):
		
		for w,b in zip(self.w,self.b):
			x=self.h(np.matmul(x,w)+b)
		return x
 
	def error(self,x,y):
		Y=self.feed(x)
		self.e=np.sum((Y-y)**2)
		return self.e
	



class agent:
	def __init__(self,s,n,env=None,N=.1,L=.1,MUT=.1,RAD=.2):


		
		self.POP=n
		self.NEXT=int(N*n) # elite
		self.LUCK=int(L*n)
		self.CHILDREN=self.POP-self.NEXT-self.LUCK
		self.MUT=MUT
		self.RAD=RAD
		self.environ=env
		
		

		self.pop=[net(s) for i in range(self.POP)]
		
		self.best=self.pop[0]

	def policies(self):
		return [p.feed for p in self.pop]

	def train(self,rewards=None,tournament=True):
	
	
		if self.environ != None:
			for p in self.pop:
				p.e=self.environ(p.feed)	
		else:
			for p,r in zip(self.pop,rewards):
				p.e=r
	   
		self.pop=sorted(self.pop,key=lambda x: x.e,reverse=True)
	    
		self.best=self.pop[0]
	    
		if tournament:
			mid=self.POP//2
			shuffle(self.pop)
			new=[]

			old1=self.pop[:mid]
			old2=self.pop[mid:]

			for p1,p2 in zip(old1,old2):
				if p1.e > p2.e: 
					best = p1
					worst= p2
				else: 
					best = p2
					worst= p1
				worst.copy(best)
				worst.mutate(self.MUT,self.RAD)

				new.append(best)
				new.append(worst)
				

		else:
			new=self.pop[:self.NEXT]
			old=self.pop[self.NEXT:]
		
			shuffle(old)
			new=new+old[:self.LUCK]
			old=old[self.LUCK:]
			
		
		
			for i in range(len(old)):
				p1,p2=randint(0,self.NEXT+self.LUCK-1),randint(0,self.NEXT+self.LUCK-1)
				Net=old.pop()
				Net.copy(new[p1])
				#Net.crossover(new[p1],new[p1])
				Net.mutate(self.MUT,self.RAD)
				new.append(Net)
				
		self.pop=new
		
		return self.best
	
	def dtrain(self,rewards=None,groups=4):
		if self.environ != None:
			for p in self.pop:
				p.e=self.environ(p.feed)	
		else:
			for p,r in zip(self.pop,rewards):
				p.e=r
		pops=[[] for i in range(groups)]
		X=[p.flatten() for p in self.pop]
		kmeans = KMeans(n_clusters=groups, random_state=0).fit(X)
		n=kmeans.labels_
		for i in range(self.POP):
			pops[n[i]].append(self.pop[i])
		trash=[]
		pop=[]
		for i in range(groups):
			p=sorted(pops[i],key=lambda x:-x.e)
			pop.append(p[0])
			if len(p)>1:
				trash+=p[1:]

		for i in range(len(trash)):
			best=pop[i%groups]
			worst=trash[i]
			worst.copy(best)
			worst.mutate(self.MUT,self.RAD)
			pop.append(worst)
		self.pop=pop
		self.best=sorted(self.pop,key=lambda x: x.e,reverse=True)[0]
		return self.best



	def policy(self,index=-1):
		if index<0:
			return self.best
		else:
			return self.pop[index]

class mapelite:
	def __init__(self,s,n=None,env=None,N=.1,L=.1,MUT=.1,RAD=.2):
		
		self.ai=net(s)

		self.res=4
		if n is None:
			n=(len(self.ai.flatten())+1)//self.res

		self.n=n
		self.X=[[None for i in range(n)] for j in range(n)]
		self.P=np.zeros((n,n))
		self.P[:]=np.NINF

		self.MUT=MUT
		
		self.RAD=RAD
		self.environ=env

		self.start=32
		self.idx=0

	def sample_idx(self):
		idxs=np.argwhere(~np.isneginf(self.P))
		vals=self.P[~np.isneginf(self.P)]
		vals-=np.min(vals)
		vals/=np.sum(vals)
		i = np.random.randint(0,len(idxs))
		ran=np.arange(0,len(idxs))
		i=np.random.choice(ran,p=vals)
		return idxs[i]

	def vec2idx(self,vec):
		mag=np.sum(vec*vec)
		mag=np.sqrt(mag)
		#print(mag,np.max(np.abs(vec)))

		mag/=2
		mag=np.min([mag,self.n-1])
		
		mag=int(mag)

		num=np.sum(np.abs(vec)>0.2)/self.res
		num=int(num)
		return mag,num
		
	def policy(self):
		i,j=np.unravel_index(self.P.argmax(),self.P.shape)
		wb=self.X[i][j]
		self.ai.set(wb)
		self.ai.e=self.P[i,j]
		return self.ai

	def train(self,rewards=None):
		self.idx+=1
		if self.idx<self.start:
			self.ai.shuffle()
		else:
			i,j=self.sample_idx()
			wb=self.X[i][j]
			self.ai.set(wb)
			self.ai.mutate(self.MUT,self.RAD)

		if self.environ != None:
			f=self.environ(self.ai.feed)	
		else:
			f=rewards
	   
		i,j=self.vec2idx(self.ai.flatten())


		if self.P[i,j]<f:
			#print(f,i,j)
			self.P[i,j]=f
			self.X[i][j]=self.ai.get()
			
		return np.max(self.P)

if __name__=="__main__":
	a=net([5,3,1])
	print(a.flatten())

	def env(feed):
		X=np.array([[1,0],[0,1],[1,1],[0,0]])
		Y=np.array([[0],[0],[1],[1]])
		y=feed(X)
		return -np.sum((y-Y)**2.0)
	
	b=agent([2,2,1],32,env=env)
	for i in range(1000):
		a=b.dtrain()
		print(a.e)