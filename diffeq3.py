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
	return [y[1],-.1*y[1]-2000*y[0]]

y=[0,5]
t=np.arange(0.0,10.0,.01)

sol=odeint(dydt,y,t)
plt.clf()
plt.subplot(211)
plt.plot(t,sol[:,0])

f=np.fft.rfft(sol[:,0])
f=np.abs(f[:len(f)/2])
t2=t[:len(f)]
plt.subplot(212)
plt.plot(t2,f)
plt.show()