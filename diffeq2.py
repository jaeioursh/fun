#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:58:25 2017

@author: josh
"""


import numpy as np
import matplotlib.pyplot as plt



from scipy.integrate import odeint

def dydt(y,t):
	return [y[1],-.1*y[1]-5*y[0]]
	
def grav(y,t,g,m1,m2):
	return[y[2],
		y[3],
		-g*m2*(y[0]-y[4])/((y[1]-y[5])**2.0+(y[0]-y[4])**2.0)**1.5,
		-g*m2*(y[1]-y[5])/((y[1]-y[5])**2.0+(y[0]-y[4])**2.0)**1.5,
		y[6],
		y[7],
		g*m1*(y[0]-y[4])/((y[1]-y[5])**2.0+(y[0]-y[4])**2.0)**1.5,
		g*m1*(y[1]-y[5])/((y[1]-y[5])**2.0+(y[0]-y[4])**2.0)**1.5
		]
	
y=[0,5,0,5.4,0,10,5.4,0]
t=np.arange(0.0,10.0,.01)

sol=odeint(grav,y,t,args=(.2,1000,1000))
plt.clf()
plt.plot(sol[:,0],sol[:,1])
plt.plot(sol[:,4],sol[:,5])
plt.show()