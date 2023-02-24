
"""
colors for pic
blue  - boundary
green - F vert (0 - down, 127 - none, 255 up) 
red   - F horz (0 - left, 127 - none, 255 right)
white - metal
black - space
@author: josh
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import collections  as mc
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from time import sleep

def mplot(tri,pts,T):
	#print(tri,pts,T)
	lines=[]
	for t,v in zip(tri,T):
		x,y,z = t
		if v==0:
			continue
		for i,j in [[x,y],[y,z],[z,x]]:
			lines.append([(pts[i][0],pts[i][1]),(pts[j][0],pts[j][1])])
	plt.subplot(2,2,2)
	lc = mc.LineCollection(lines)	
	ax = plt.gca()
	ax.add_collection(lc)
	ax.autoscale()


									
def ispart(num):
	#return np.sum(num)!=0
	return ((num[0]==num[1] and num[1]==num[2]) and np.sum(num)!=0)
										
def principle(s):
	mat=np.matrix([[s[0,0],s[2,0]],[s[2,0],s[1,0]]])
	
	eig=np.linalg.eigvals(mat)
	 
	return (((eig[0]-eig[1])**2.0 + eig[1]**2.0 + eig[0]**2.0)/2.0)**.5
	
def principle2(s):
	mat=np.matrix([[s[0,0],s[2,0]],[s[2,0],s[1,0]]])
	eig=np.linalg.eigvals(mat)
	 
	return (abs(eig[1]-eig[0]))/2.0
	
		
def maxx(s):
	mat=np.matrix([[s[0,0],s[2,0]],[s[2,0],s[1,0]]])
	
	eig=np.linalg.eigvals(mat)
	
	if abs(eig[1])>abs(eig[0]):
		return eig[1]
	else:
		return eig[0]		
		


'''
data
0 = clamped
1 = forced
2 = f 1
3 = f 2
'''
def info(part,i,j):

	Z=[[1,0],[0,1],[-1,0],[0,-1]]
	data=[False,False,0.0,0.0]
	for z in Z: 
		n=np.sum(part[i+z[0]][j+z[1]])
		if ispart(part[i+z[0]][j+z[1]]) or n==0:
			pass
		elif part[i+z[0]][j+z[1]][0]>250:
			data[0]=True
			break
		elif part[i+z[0]][j+z[1]][0]==0:
			data[1]=True
			data[2]=-1.0*float(part[i+z[0]][j+z[1]][2]-127)
			data[3]=1.0*float(part[i+z[0]][j+z[1]][1]-127)
			break
	return data

class evofea:
	def __init__(self):
		####settings#####
		sz=20
		File='part/ppbox.png'
		self.thickness=.2
		self.M=.0015
		
		#material properties
		self.E=210e9
		self.v=.3
		
		#population params
		self.POP=20
		self.ELITE=3
		self.LUCKY=2
		self.CHILDREN=self.POP-self.ELITE-self.LUCKY
		
		####start######
		part = cv2.imread(File,1)
		#part[np.sum(part,axis=2)/3.0==]=np.array[255,255,255]
	
		shrink=np.arange(0,len(part),sz)
		
		#part = cv2.resize(part,None,fx=1.0/sz, fy=1.0/sz)#, interpolation = cv2.INTER_AREA)
		part=part[np.ix_(shrink,shrink)]
			
			
		X=[]
		Y=[]
		#x,y coords
		for i in xrange(len(part)):
			for j in xrange(len(part[i])):
				if ispart(part[i][j]):
					Y.append(i)
					X.append(j)
		
		names=np.zeros((len(part),len(part[0])),dtype=int)-1
		nodes=[] #nodes
		pts=[]   #points x,y coords
		tri=[]   #triangles
		data=[]  #data
		T=[]

		Z=[[-1,0],[-1,-1],[0,-1],[1,-1]]
		print('making model')
		scaler=float(len(part))
		for i in xrange(len(X)):
			
			nodes.append([])
			#if edge(part,Y[i],X[i]):
			pts.append([float(X[i])/scaler,float(Y[i])/scaler])
			#else:
			#	pts.append([float(X[i])+random.uniform(-A,A),float(Y[i])+random.uniform(-A,A)])
			names[Y[i]][X[i]]=i
			data.append(info(part,Y[i],X[i]))
			for j in xrange(4):
				num = names[Y[i]+Z[j][1]][X[i]+Z[j][0]]
				if num>=0:
					if j!=1:
						nodes[i].append(num)
						nodes[num].append(i)
		
					else:
						n1=names[Y[i]-1][X[i]]
						n2=names[Y[i]][X[i]-1]
						if n1>=0 and n2>=0:
							if not(n2 in nodes[n1]):
								nodes[i].append(num)
								nodes[num].append(i)
		
						else:
							nodes[i].append(num)
							nodes[num].append(i)
			if len(nodes[i])>=2:
				for j in xrange(1,len(nodes[i])):
					if nodes[i][j] in nodes[nodes[i][j-1]]:
						tri.append([i,nodes[i][j],nodes[i][j-1]])
						T.append(1)
		self. T=np.array(T)
		self. tri=tri
		self.nodes=nodes
		self.names=names
		self.pts=np.array(pts)
		self.data=data
		
		
		self.populate()
		for i in range(100):
			print(i,self.repop())
		#self.calc(-1)
		mplot(self.tri,self.pts[0],self.T[0])
		
	def mut(self,t,p):
		for i in range(len(p)):
			if(random.random()>.1):
				continue
			x,y=p[i] #each pt
			dx,dy=x+random.gauss(0,self.M),y+random.gauss(0,self.M)
			for j in self.nodes[i]: #loop through near pts
				nx,ny=p[j]
				if dx==x:
					continue
				if dx>x and dx>nx>x: #check if in between
					dx,dy=x,y
				if dx<x and dx<nx<x:
					dx,dy=x,y
				if dy>y and dy>ny>y:
					dx,dy=x,y
				if dy<y and dy<ny<y:
					dx,dy=x,y
			p[i][0]=dx
			p[i][1]=dy
			   
		if(False):
			touching=np.zeros(len(self.nodes)) #number of tries touching each node
			for i in range(len(t)):
				for tri in self.tri[i]:
					if t[i]==1:
						touching[tri]+=1
						
			for i in range(len(t)):
				x,y,z = self.tri[i]
				dx,dy,dz=self.data[x][0]+self.data[x][1],self.data[y][0]+self.data[y][1],self.data[z][0]+self.data[z][1]
				if dx+dy+dz != 0:
					continue
				dx,dy,dz=min(1,touching[x]),min(1,touching[y]),min(1,touching[z])
				if random.random()<.01:
					if t[i]==1:
						t[i]=0
					else:
						if dx+dy+dz > 1:
							t[i]=1
			
		return t,p
		
		
	def populate(self):
		T=[]
		pts=[]
		for i in range(self.POP):
			if i == 0:
				t=self.T.copy()
				p=self.pts.copy()
			else:
				t,p=self.mut(self.T.copy(),self.pts.copy())
			T.append(t)
			pts.append(p)
		self.T=T
		self.pts=pts
		
		
	def repop(self,minimize=1):
		
		R=[self.calc(i) for i in range(self.POP)]
		
		dat=sorted(zip(R,self.T,self.pts),key=lambda x: x[0],reverse=not minimize)
		
		parents=[]
		for i in range(self.ELITE):
			parents.append([dat[i][1],dat[i][2]])
		for i in range(self.LUCKY):
			j=random.randint(0,self.POP-1)
			parents.append([dat[j][1],dat[j][2]])
		for i in range(self.CHILDREN):
			j=random.randint(0,self.ELITE+self.LUCKY-1)
			t,p=parents[j]
			t,p=self.mut(t.copy(),p.copy())
			parents.append([t,p])
		
		self.T=[P[0] for P in parents]
		self.pts=[P[1] for P in parents]	
		return dat[0][0]
		
	def calc(self,i):
		
		tri=self. tri
		nodes=self.nodes
		#names=self.names
		data=self.data
		pts=self.pts[i]
		T=self.T[i]
		
		E=self.E
		v=self.v
		
		D=E/(1.0-v**2.0)*np.matrix([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
		th=self.thickness
		K=np.zeros((2*len(nodes),2*len(nodes)),dtype=float)
		for t,v in zip(tri,T):
			if v == 0:
				continue
			
			x=[0,pts[t[0]][0],pts[t[1]][0],pts[t[2]][0]]
			y=[0,pts[t[0]][1],pts[t[1]][1],pts[t[2]][1]]
			detJ=(x[1]-x[3])*(y[2]-y[3])-(x[2]-x[3])*(y[1]-y[3])
			A=.5*abs(detJ)
			B=1.0/detJ*np.matrix([[y[2]-y[3],0,y[3]-y[1],0,y[1]-y[2],0],
						     [0,x[3]-x[2],0,x[1]-x[3],0,x[2]-x[1]],
							[x[3]-x[2],y[2]-y[3],x[1]-x[3],y[3]-y[1],x[2]-x[1],y[1]-y[2]]])
			k=th*A*B.T*D*B
			n=[t[0]*2,t[0]*2+1,t[1]*2,t[1]*2+1,t[2]*2,t[2]*2+1]
			for i in xrange(6):
				for j in xrange(6):
					K[n[i]][n[j]]+=k[i,j]
			
		F=np.zeros((2*len(nodes),1))
		static=[]
		#print('   building K')
		for i in xrange(len(nodes)):
			if data[i][0]==True:
				static.append(2*i)
				static.append(2*i+1)
				pass
			elif data[i][1]==True:
				F[i*2][0]=data[i][2]	
				F[i*2+1][0]=data[i][3]
	
		
		STAT=[i for i in xrange(len(nodes)*2) if i not in static]
		STAT=np.array(STAT)
		F=F[STAT]
		
		K=K[np.ix_(STAT,STAT)]
			
		
		#print('   solve eqns')		
		#q0=np.linalg.solve(K,F)
		
		K=csr_matrix(K)
		q=spsolve(K, F)
		q=np.array([q]).T
		
		Q=np.zeros((len(nodes)*2,1),dtype=float)
		Q[STAT]=q
		

		
		mz=[]
		#print('calculating stresses')
		for t in tri:
			x=[0,pts[t[0]][0],pts[t[1]][0],pts[t[2]][0]]
			y=[0,pts[t[0]][1],pts[t[1]][1],pts[t[2]][1]]
			detJ=(x[1]-x[3])*(y[2]-y[3])-(x[2]-x[3])*(y[1]-y[3])
		
			B=1.0/detJ*np.matrix([[y[2]-y[3],0,y[3]-y[1],0,y[1]-y[2],0],
						     [0,x[3]-x[2],0,x[1]-x[3],0,x[2]-x[1]],
							[x[3]-x[2],y[2]-y[3],x[1]-x[3],y[3]-y[1],x[2]-x[1],y[1]-y[2]]])
			n=[t[0]*2,t[0]*2+1,t[1]*2,t[1]*2+1,t[2]*2,t[2]*2+1]
			q=Q[n]
			
			STRESS=D*B*np.matrix(q)
			if np.isnan(STRESS).any():
				return(1e10)
			mz.append(principle2(STRESS))	
		
		return max(mz)





fea=evofea()