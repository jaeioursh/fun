#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
c-shader/02-fragment.py

mandelbrot fragment shader
"""

# imports ####################################################################

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GL import shaders

from math import exp, log, sqrt

from utils import *

GLUT_WHEEL_UP   = 3
GLUT_WHEEL_DOWN = 4


# shaders ####################################################################
file = open('gl.frag',mode='r')
frag_shader_source = file.read()
file.close()

# constants ##################################################################

WINDOW_SIZE = 640, 640


# display ####################################################################

center = 0, 0.
scale = 1.0

def reshape(width, height):
	"""window reshape callback."""
	glViewport(0, 0, width, height)
	
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	r = float(max(width, height))
	w, h = width/r, height/r
	glOrtho(-w, w, -h, h, -1, 1)
	
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

def display():
	"""window redisplay callback."""
	global center
	global scale
	global center_location
	global scale_location

	glUniform1f(center_location1,center[0])
	glUniform1f(center_location2,center[1])
	glUniform1f(scale_location ,scale)


	glClear(GL_COLOR_BUFFER_BIT)
	glBegin(GL_TRIANGLE_STRIP)
	cx, cy = center
	for x in [-1, 1]:
		for y in [-1, 1]:
			#glTexCoord2d(x/scale-cx, y/scale-cy)
			glVertex(x, y)

	

	
	glEnd()
	glutSwapBuffers()


# interaction ################################################################

def keyboard(c, x, y):
	"""keyboard callback."""
	if c in ["q", chr(27)]:
		sys.exit(0)
	glutPostRedisplay()


panning = False
zooming = False
x0,x1=0,0
y0,y1=0,0
def mouse(button, state, x, y):
	global x0, y0, panning
	global xz, yz, zooming
	x0, y0 = xz, yz = x, y
	
	if button == GLUT_LEFT_BUTTON:
		panning = (state == GLUT_DOWN)
	
	elif button == GLUT_RIGHT_BUTTON:
		zooming = (state == GLUT_DOWN)
	
	elif button == GLUT_WHEEL_UP:
		zoom(x, y, 1)
	elif button == GLUT_WHEEL_DOWN:
		zoom(x, y, -1)


def motion(x1, y1):
	global x0, y0, panning
	global xz, yz, zooming
	dx, dy = x1-x0, y1-y0
	x0, y0 = x1, y1
	
	if panning:
		pan(dx, dy)
	
	elif zooming:
		zoom(xz, yz, dx-dy)


def pan(dx, dy):
	global center, scale
	cx, cy = center
	r = max(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT))
	dx *=  2./r
	dy *= -2./r
	center = cx+dx*scale, cy+dy*scale
	glutPostRedisplay()

def zoom(x, y, s):
	global scale, center
	
	cx, cy = center
	width, height = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
	r = max(width, height)
	x = (2.*x-width)/r*scale
	y = -(2.*y-height)/r*scale
	
	ds = 1.01**s
	scale *= ds
	#cx += cx*(1-ds)
	#cy += cy*(1-ds)
	cx -= x*(1-ds)
	cy -= y*(1-ds)


	center = cx, cy
	
	
	#glUniform1i(max_i_location, int(4.*(log(scale)/log(2)+1))+32)
	glutPostRedisplay()


# setup ######################################################################

def init_glut(argv):
	"""glut initialization."""
	glutInit(argv)
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE)
	glutInitWindowSize(*WINDOW_SIZE)
	
	glutCreateWindow(argv[0])
	
	glutReshapeFunc(reshape)
	glutDisplayFunc(display)
	glutKeyboardFunc(keyboard)
	glutMouseFunc(mouse)
	glutMotionFunc(motion)


def init_opengl():
	# program
	frag_shader = shaders.compileShader(frag_shader_source, GL_FRAGMENT_SHADER)
	program = shaders.compileProgram(frag_shader)
	glUseProgram(program)

	global max_i_location
	max_i_location = glGetUniformLocation(program, "max_i")
	glUniform1f(max_i_location,float( WINDOW_SIZE[0]))

	global center_location1
	global center
	center_location1 = glGetUniformLocation(program, "center1")
	glUniform1f(center_location1,center[0])
	global center_location2
	
	center_location2 = glGetUniformLocation(program, "center2")
	glUniform1f(center_location1,center[1])

	global scale_location
	global scale
	scale_location = glGetUniformLocation(program, "scale")
	glUniform1f(scale_location ,scale)


	

	



# main #######################################################################

import sys

def main(argv=None):
	if argv is None:
		argv = sys.argv
	init_glut(argv)
	init_opengl()
	return glutMainLoop()	

if __name__ == "__main__":
	sys.exit(main())


