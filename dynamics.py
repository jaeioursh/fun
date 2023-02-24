import numpy as np
import numpy.linalg as la
def Rx(theta):
	theta*=np.pi / 180.
	return np.matrix([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def Ry(theta):
	theta=float(theta)*np.pi / 180.
	return np.matrix([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
	
def Rz(theta):
	theta*=np.pi / 180.
	return np.matrix([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

	
a=np.matrix([[1],[0],[0]])
	
print()