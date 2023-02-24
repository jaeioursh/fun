#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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
#             0        1     2    3    4       5
setting=['principle','max','Sx','Sy','Sxy','max_shear']
####settings#####
FULL=0
SHOW_FORCE=0
PLOT=setting[1]
sz=3
File='part/sherri3.png'
thickness=.2
varience=.15

#material properties
E=210e9
v=.3


####start######
part = cv2.imread(File,1)
#part[np.sum(part,axis=2)/3.0==]=np.array[255,255,255]
plt.clf()
res=part
shrink=np.arange(0,len(part),sz)

#part = cv2.resize(part,None,fx=1.0/sz, fy=1.0/sz)#, interpolation = cv2.INTER_AREA)
part=part[np.ix_(shrink,shrink)]

def slep(num):
	print('i is slep')
	sleep(num)										
	print('i all done')									
def ispart(num):
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
		
def edge(part,i,j):
	if not ispart(part[i-1][j]):
		return True
	if not ispart(part[i+1][j]):
		return True
	if not ispart(part[i][j-1]):
		return True
	if not ispart(part[i][j+1]):
		return True
	return False

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

	
x=[]
y=[]
xf=[]
yf=[]	
X=[]
Y=[]
for i in xrange(len(part)):
	for j in xrange(len(part[i])):
		if ispart(part[i][j]):
			Y.append(i)
			X.append(j)

names=np.zeros((len(part),len(part[0])),dtype=int)-1
nodes=[]
pts=[]
tri=[]
data=[]

lines=[]
A=varience
Z=[[-1,0],[-1,-1],[0,-1],[1,-1]]
print('making model')
for i in xrange(len(X)):
	nodes.append([])
	if edge(part,Y[i],X[i]):
		pts.append([float(X[i]),float(Y[i])])
	else:
		pts.append([float(X[i])+random.uniform(-A,A),float(Y[i])+random.uniform(-A,A)])
	pts[-1][0]/=float(len(part))
	pts[-1][1]/=float(len(part))
	names[Y[i]][X[i]]=i
	data.append(info(part,Y[i],X[i]))
	for j in xrange(4):
		num = names[Y[i]+Z[j][1]][X[i]+Z[j][0]]
		if num>=0:
			if j!=1:
				nodes[i].append(num)
				nodes[num].append(i)
				lines.append([(pts[i][0],pts[i][1]),(pts[num][0],pts[num][1])])
			else:
				n1=names[Y[i]-1][X[i]]
				n2=names[Y[i]][X[i]-1]
				if n1>=0 and n2>=0:
					if not(n2 in nodes[n1]):
						nodes[i].append(num)
						nodes[num].append(i)
						lines.append([(pts[i][0],pts[i][1]),(pts[num][0],pts[num][1])])
				else:
					nodes[i].append(num)
					nodes[num].append(i)
					lines.append([(pts[i][0],pts[i][1]),(pts[num][0],pts[num][1])])
	if len(nodes[i])>=2:
		for j in xrange(1,len(nodes[i])):
			if nodes[i][j] in nodes[nodes[i][j-1]]:
				tri.append([i,nodes[i][j],nodes[i][j-1]])

mx=[]
my=[]
for i in xrange(len(tri)):
	sumx=0.0
	sumy=0.0
	for j in tri[i]:
		sumx+=pts[j][0]
		sumy+=pts[j][1]
	mx.append(sumx/3.0)
	my.append(sumy/3.0)
				

	
plt.subplot(2,2,2)
c = np.array([(1, 0, 0, 1)])
lc = mc.LineCollection(lines)	
ax = plt.gca()
ax.add_collection(lc)
ax.autoscale()

'''
plt.subplot(2,2,3)
plt.plot(mx,my,'o')
'''

plt.subplot(2,2,1)
plt.imshow(part,interpolation="nearest")


#####math#######
print('matrix math')

D=E/(1.0-v**2.0)*np.matrix([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
th=thickness
K=np.zeros((2*len(nodes),2*len(nodes)),dtype=float)
for t in tri:
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
Fy=np.zeros(len(nodes))
Fx=np.zeros(len(nodes))
T=np.arange(0,len(nodes))
static=[]
print('   building K')
for i in xrange(len(nodes)):
	if data[i][0]==True:
		static.append(2*i)
		static.append(2*i+1)
		pass
	elif data[i][1]==True:
		F[i*2][0]=data[i][2]	
		F[i*2+1][0]=data[i][3]
		Fx[i]=data[i][2]	
		Fy[i]=data[i][3]

STAT=[i for i in xrange(len(nodes)*2) if i not in static]
STAT=np.array(STAT)
F=F[STAT]
if SHOW_FORCE:
	plt.subplot(4,2,5)
	plt.plot(T,-1.0*Fx)
	plt.subplot(4,2,7)
	plt.plot(T,-1*Fy)
K=K[np.ix_(STAT,STAT)]
	

print('   solve eqns')		
#q0=np.linalg.solve(K,F)

K=csr_matrix(K)
q=spsolve(K, F)
q=np.array([q]).T

Q=np.zeros((len(nodes)*2,1),dtype=float)
Q[STAT]=q
	
mz=[]
print('calculating stresses')
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
	if PLOT=='max':
		mz.append(maxx(STRESS))
	elif PLOT=='principle':
		mz.append(principle(STRESS))
	elif PLOT=='Sx':	
		mz.append(STRESS[0,0])
	elif PLOT=='Sy':	
		mz.append(STRESS[1,0])
	elif PLOT=='Sxy':	
		mz.append(STRESS[2,0])
	elif PLOT=='max_shear':	
		mz.append(principle2(STRESS))	
print('interpolating')

points=np.vstack((mx,my))
points=points.T
values=np.array(mz)
grid_x, grid_y = np.mgrid[0:1:400j, 0:1:400j]
grid_z2 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2=grid_z2.T
grid_z2[np.sum(res,axis=2)!=765]=np.nan
								
if not SHOW_FORCE:
	grid=grid_z2.copy()
	grid[np.abs(grid)<1.0]=1.0
	
	grid=np.sign(grid)*np.log10(np.abs(grid))
	plt.subplot(2,2,3)	
	plt.imshow(grid, extent=(0,1,0,1),interpolation=None)
	plt.colorbar()

								
plt.subplot(2,2,4)								
if not FULL: plt.clf()
plt.imshow(grid_z2, extent=(0,1,0,1),interpolation=None)
plt.colorbar()

plt.tight_layout()
plt.show()





