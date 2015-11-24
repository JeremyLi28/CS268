# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
from math import e
import math
from sympy import *

#graph-2-coloring
e_t = 0.1
dim = 25
ratio = 0.98
N = 1
G = np.array([[0,1,0,0,0],
			  [1,0,1,1,0],
			  [0,1,0,0,1],
			  [0,1,0,0,1],
			  [0,0,1,1,0]])
X0 = np.zeros(dim)
T = 50
s = 0.1
e_abs = 1e-4
e_f = 1e-6
e_r = np.sqrt(1e-15)

def genGraph(N):
	d = int(N*N)
	m = np.zeros((d,d))
	for i in range(d):
		for j in range(d):
			if abs(i-j) == 1:
				if j > i and j%N != 0:
					m[i][j] = 1
				elif i > j and i%N !=0:
					m[i][j] = 1
			if abs(i-j) == N:
				m[i][j] = 1
	return m


def graphColor(G,X,N):
	dim = len(X)
	if N==1:
		return (np.trace(np.dot(G,np.outer(X,X.T).T))+np.trace(np.dot(G,np.outer(1-X,(1-X).T).T)))/2;
	else:
		counter = 0
		for i in range(dim):
			for j in range(dim):
				if X[i]==X[j] and i!=j and G[i][j] !=0:
					counter += 1
		return counter*1.0/2

def Boltsman(E,G,X,T,N):
	return e**(-E(G,X,N)*1.0/T)

def flip(X,N):
	dim = len(X)
	i = np.random.randint(dim)
	if N == 1:
		X[i] = 1 - X[i]
	else:
		while True:
			tmp = np.random.randint(N+1)
			if X[i]!= tmp:
				X[i] = tmp
				break
	return X
def flipReal(X,step):
	dim = len(X)
	direction = [-1,1]
	d = direction[np.random.randint(2)]
	i =  np.random.randint(dim)
	X[i] += d*step
	# while True:
	# 	step = 0
	# 	d = direction[np.random.randint(2)]
	# 	q = 1
	# 	i =  np.random.randint(dim)
	# 	print "outer: ", i
	# 	counter = 0
	# 	while abs(accept-q) > 0.1:		 	
	# 		if counter > 10/step_size:
	# 			break
	# 		X_prime = X.copy()
	# 		X_prime[i] = X[i]+d*step
	# 		if E(X) >= E(X_prime):
	# 			y,z = X_prime,X
	# 		else:
	# 			y,z = X,X_prime
	# 		delatE = E(y) - E(z)
	# 		q = e**(delatE*1.0/T)
	# 		step += step_size
	# 		counter += 1
	# 	if abs(accept-q) <= 0.1:
	# 		break
	# print "Step:", step
	return X

def SA(E,G,X,T,N):
	while T >= e_t:
		X_prime = flip(X.copy(),N)
		if E(G,X,N) >= E(G,X_prime,N):
			y,z = X_prime,X
		else:
			y,z = X,X_prime

		delatE = E(G,y,N) - E(G,z,N)
		q = e**(delatE*1.0/T)
		accept = 1.0/(1+q)
		if np.random.uniform()<=accept:
			X = y
		else:
			X = z
		# K = np.array([[e**(-deltaE*1.0/T),e**(-deltaE*1.0/T)],[1,1]])*1.0/(1+e**(-deltaE*1.0/T)) # Heat bath
		# print X, E(G,X,N),T
		T = T*ratio
	return X,E(G,X,N)

def SAReal(E,X,T):
	step = 0.5
	r = 1.2
	while T >= e_t:
		X_prime = flipReal(X.copy(),step)
		if E(X) >= E(X_prime):
			y,z = X_prime,X
		else:
			y,z = X,X_prime
		delatE = E(y) - E(z)
		q = e**(delatE*1.0/T)
		if E(X) >= E(X_prime):
			X = X_prime
		else:
			if np.random.uniform()<=q and abs(q-0.4)<0.1:
				# print "accept"
				X = X_prime
		# print "X: ",X,'\t',"E(X): ",E(X),'\t',"q: ",q,'\t',"Step: ",step,'\t',"T: ",T
		if q>0.5:
			step *= r
		elif q<0.3:
			step /= r
		else:
			pass
		T *= ratio
	return X,E(X)

def GoldenSection(f):
	global s
	global e_abs
	global e_f
	global e_r
	stop = False
	t = (np.sqrt(5)-1)/2
	x1 = 10*np.random.randn(1)
	f1 = f(x1)
	x2 = x1+s
	f2 = f(x2)
	if f2>f1:
		x1,x2 = x2,x1
		f1,f2 = f2,f1
		s = -s
	i=0
	while(True):
		s = s/t
		x4 = x2+s
		f4 = f(x4)
		if f4>f2:
			break
		x1,x2 = x2,x4
		f1,f2 = f2,f4
		i = i + 1
	#         if i%1e3 == 0:
	#             print x1,x4
		if i>1e6:
			break
	f_old = (f1+f2+f4)/3
	i=0
	while(True):
		x3 = t*x4+(1-t)*x1
		f3 = f(x3)
		if f2<f3:
			x4,x1 = x1,x3
			f4,f1 = f1,f3
		else:
			x1,x2 = x2,x3
			f1,f2 = f2,f3
		f_new = (f1+f2+f4)/3
#         print "fc: ",abs(f_new-f_old),e_r*abs(f2)+e_abs
		if i%2 == 0 and abs(f_new-f_old) <= e_r*abs(f2)+e_abs:
			# print "G_FC"
			break
		f_old = f_new
	#     print "xc: ",abs(x1-x4),e_r*abs(x2)+e_abs
		if abs(x1-x4) <= e_r*abs(x2)+e_abs:
			# print "G_XC"
			break
		i = i+1
	#         if i%1e3 == 0:
	#             print x1,x2,x4
		if i>1e6:
			# print "G_Stop"
			stop = True
			break
	# print "x2: ",x2
	# print "f2: ",f2
	return x2[0],f2[0],i,stop

def CoorDes(f):
	global s
	stop = False
	global x
	global y
	global e_abs
	global e_f
	global e_r
	def f_x(x):
		return f(x,y)
	def f_y(y):
		return f(x,y)
	i = 0
	f_old = f(x,y)
	x_old = x
	y_old = y
	i = 0
	while True:
		x,f_new,iterNum_x,stop_x = GoldenSection(f_x)
		y,f_new,iterNum_y,stop_y = GoldenSection(f_y)
		# print x,y,f_new
		if stop_x or stop_y:
			stop = True
			break
		if abs(x-x_old) < e_abs and abs(y-y_old) < e_abs:
			# print "XC"
			break;
		if abs(f_new-f_old) < e_f+e_r*abs(f_old):
			# print "FC"
			break
		if i>1e6:
			stop = True
			break
		f_old = f_new
		x_old = x
		y_old = y
		i = i+1
	return f_new



if __name__ == "__main__":

	# Problem 1
	# 1(a) test on a tiny graph
	dim = 5
	print SA(graphColor,G,np.zeros(dim),T,1)
	# test on different size check board
	result = DataFrame(columns = ["Size","error","error_bar"])
	for i in range(2,11):
		m = genGraph(i)
		err = []
		for j in range(10):
			X, error = SA(graphColor,m,np.zeros(i**2),T,N)
			err.append(error)
		print "Size: ",i**2, "error:", np.mean(err), "error_bar: ", np.sqrt(1.0/9)*np.std(err)
		result.loc[len(result)] = [i**2,np.mean(err),np.sqrt(1.0/9)*np.std(err)]
	result.to_csv("test_results/check-board-coloring.csv")

	# 1(b) test on multicolor
	print SA(graphColor,G,np.zeros(dim),T,3)


	# Problem 2
	def f1(x):
		return (x[0]-1)**2 + (x[1]-1)**2
	def f1_2(x,y):
		return (x-1)**2 + (y-1)**2
	# def RB(x):
	# 	return pow(1-x[0],2)+100*pow(x[1]-pow(x[0],2),2)
	# sareal = DataFrame(columns = ['error','error_bar','time','time_bar'])
	# coor = DataFrame(columns = ['error','error_bar','time','time_bar'])
	se = []
	st = []
	ce = []
	ct = []

	for i in range(1):
		x = 0
		y = 0
		a = time.time()
		X,e1 = SAReal(f1,np.zeros(2),T)
		b = time.time()
		e2 = CoorDes(f1_2)
		c = time.time()
		se.append(e1)
		st.append(b-a)
		ce.append(e2)
		se.append(c-b)
	print "SAReal:", "error: ",np.mean(se), "+=", np.sqrt(1.0/9)*np.std(se),"time: ",np.mean(st), "+=", np.sqrt(1.0/9)*np.std(st)
	print "CoorDes:", "error: ",np.mean(ce), "+=", np.sqrt(1.0/9)*np.std(ce),"time: ",np.mean(ct), "+=", np.sqrt(1.0/9)*np.std(ct)



