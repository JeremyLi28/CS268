# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
from math import e
import math
from sympy import *

e_abs = 1e-4
e_f = 1e-6
e_r = np.sqrt(1e-15)

# RosenBrock Function
x_rb = [Symbol('x'+str(i)) for i in range(2)]
y_rb = (1-x_rb[0])**2 + 100*(x_rb[1]-x_rb[0]**2)**2
RosenBrock = lambdify(x_rb,y_rb,'numpy')

x10 = [Symbol('x'+str(i)) for i in range(10)]
y10 = x10[0]**2+x10[1]**2+x10[2]**2+x10[3]**2+x10[4]**2+x10[5]**2+x10[6]**2+x10[7]**2+x10[8]**2+x10[9]**2
f10 = lambdify(x10,y10,'numpy')

def drange(start, stop, step):
	t = start
	r = []
	while t <= stop:
		r.append(t)
		t += step
	return r

# Used for line search
def GoldenSection(f):
	global e_abs
	global e_f
	global e_r
	s = 1
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
	# return x2[0],f2[0],i+1,stop
	return x2[0],stop

def SteepestDescent(f,x,y,dim):
	global e_abs
	global e_f
	global e_r
	e_G = np.linalg.norm([e_abs for i in range(dim)]) # undecided
	# print e_G
	k = 0
	# np.random.seed(111)
	x_0 = np.random.randn(dim)
	fprime = [y.diff(x[i]) for i in range(dim)]
	gradient = lambdify(x[:dim],fprime,'numpy')
	x_k = x_0
	f_old = f(*x_0)
	counter = 0
	while True:
		g_k = np.array(gradient(*x_k))
		# print x_k,f_old,g_k,np.linalg.norm(g_k)
		if(np.linalg.norm(g_k) < e_G):
			# print "EG"
			break
		d_k = -g_k/np.linalg.norm(g_k)
		def f_Alpha(alpha):
			return f(*x_k+alpha*d_k)
		alpha_k, stop = GoldenSection(f_Alpha)                                             
		x_k_plus_1 = x_k + alpha_k*d_k
		f_new = f(*x_k_plus_1)
		if abs(f_new-f_old) <= e_f + e_r*abs(f_old):
			counter = counter + 1
			if counter == 2:
				# print "FG"
				break
		else:
			counter = 0
		
		k = k+1
		x_k = x_k_plus_1
		f_old = f_new

	return x_k, f_old, k


def ConjugateGradient(f,x,y,CG_iter,e_f):
	dim = len(x)
	global e_abs
	# global e_f
	global e_r
	e_G = np.linalg.norm([e_abs for i in range(dim)]) # undecided
	k = 0
	x_0 = np.random.randn(dim)
	# x_0 = [-0.65505324,-0.89369433]
	fprime = [y.diff(x[i]) for i in range(dim)]
	gradient = lambdify(x[:dim],fprime,'numpy')
	x_out = x_0
	f_old = f(*x_0)
	counter = 0
	iter_counter = 0
	while True:
		# print "====outer==="
		x_k = x_out
		g_k = np.array(gradient(*x_k))
		if(np.linalg.norm(g_k) < e_G):
			# print "e_G"
			break
		d_k = -g_k
		# print x_k,f_old,k
		for i in range(CG_iter):
			# print "====inner==="
			# alpha_k = -np.dot(d_k.T,g_k)/np.dot(np.dot(d_k.T,A),d_k)
			def f_Alpha(alpha):
				return f(*x_k+alpha*d_k)
			alpha_k,stop = GoldenSection(f_Alpha)
			# if stop:
			# 	print "Stop"
			# 	break
			x_k_plus_1 = x_k + alpha_k*d_k
			g_k_plus_1 = np.array(gradient(*x_k_plus_1))
			beta_k = np.dot(g_k_plus_1.T,g_k_plus_1)/np.dot(g_k.T,g_k)
			d_k_plus_1 = -g_k_plus_1+beta_k*d_k

			d_k = d_k_plus_1
			x_k = x_k_plus_1
			g_k = g_k_plus_1
			iter_counter = iter_counter + 1
			# print x_k,f(*x_k)


		f_new = f(*x_k)
		if f_new >= f_old:
			# print "Bad"
			# print f_new,f_old
			g_out =  np.array(gradient(*x_out))
			d_out = -g_out/np.linalg.norm(g_out)
			def f_Alpha(alpha):
				return f(*x_out+alpha*d_out)
			alpha_out,stop = GoldenSection(f_Alpha)
			# print x_out.shape, alpha_out
			if(f(*x_out + alpha_out*d_out)<f_new):
				x_out = x_out + alpha_out*d_out
				f_new = f(*x_out)
			else:
				x_out = x_k
			# print f_new
		else:
			x_out = x_k
		# x_out = x_k
		if abs(f_new-f_old) <= e_f + e_r*abs(f_old):
			counter = counter + 1
			if counter == 2:
				# print "e_f",f_new,f_old
				break
		else:
			counter = 0

		k = k+1
		f_old = f_new
	# print "===Result==="
	# print x_0
	return x_out, f_new, iter_counter



def Test(f,x,y,CG_iter_range,e_f_range,x_optimum,y_optimum,function):
	testNum = 10
	Result = DataFrame(columns=['Function','CG_iter','e_G_range','x_err','x_err_bar','y_err','y_err_bar','CG_iter_avg','CG_iter_err_bar'])
	for cg in CG_iter_range:
		for ef in e_f_range:
			# e_G = np.linalg.norm([eg for i in range(len(x))])
			x_errs = []
			y_errs = []
			cg_iters = []
			for i in range(testNum):
				# print i
				xx, yy, cg_iter = ConjugateGradient(f,x,y,cg,ef)
				x_errs.append(np.linalg.norm(xx-x_optimum))
				y_errs.append(abs(yy-y_optimum))
				cg_iters.append(cg_iter)
				# print i,xx,yy,cg_iter
			Result.loc[len(Result)] = [function,cg,ef,np.mean(x_errs),np.sqrt(1.0/(testNum-1))*np.std(x_errs) \
				,np.mean(y_errs),np.sqrt(1.0/(testNum-1))*np.std(y_errs),np.mean(cg_iters),np.sqrt(1.0/(testNum-1))*np.std(cg_iters)]
			print function,cg,ef,np.mean(x_errs),np.sqrt(1.0/(testNum-1))*np.std(x_errs) \
				,np.mean(y_errs),np.sqrt(1.0/(testNum-1))*np.std(y_errs),np.mean(cg_iters),np.sqrt(1.0/(testNum-1))*np.std(cg_iters)
	# result = open("./test_results/"+f.__name__+'_'+str(cg)+'_'+str(eg)+".txt","w")
	# result.write("CDest\nClockTime: "+str(Result.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.clockTime.std())+'\n' \
	# 	"Distance: "+str(Result.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.Distance.std())+'\n' \
	# 	"iterNum: "+str(Result.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.iterNum.std())+'\n')
	# result.close()
	# print str(Result.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.clockTime.std())
	# print str(Result.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.Distance.std())
	# print str(Result.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*Result.iterNum.std())
	Result.to_csv('./test_results/'+f.__name__+'.csv',index=False)



if __name__ == "__main__":
	

	Test(RosenBrock,x_rb,y_rb,range(len(x_rb),len(x_rb)+5),[1e-6,1e-5],np.array([1,1]),0,"RosenBrock")
	Test(f10,x10,y10,range(len(x10),len(x10)+5),[1e-6,1e-5],np.zeros(10),0,"X10")

	# steepest = DataFrame(columns=['Function','err','error_bar','time','time_error_bar'])
	e = []
	t = []
	it = []
	i = 0
	while i<10:
		a = time.time()
		x, y, k= SteepestDescent(RosenBrock,x_rb,y_rb,2)
		b = time.time()
		if abs(0-y)< 1:
			e.append(abs(0-y))
			t.append(b-a)
			it.append(k)
			print abs(0-y), b-a, k
			i = i+1
	print "steepest", np.mean(e),np.sqrt(1.0/9)*np.std(e),np.mean(t),np.sqrt(1.0/9)*np.std(t),np.mean(it),np.sqrt(1.0/9)*np.std(it)

	e = []
	t = []
	it = []
	for i in range(10):
		a = time.time()
		x, y,k = ConjugateGradient(RosenBrock,x_rb,y_rb,2,1e-6)
		b = time.time()
		e.append(abs(0-y))
		t.append(b-a)
		it.append(k)
		print abs(0-y), b-a, k
	print "conjugate", np.mean(e),np.sqrt(1.0/9)*np.std(e),np.mean(t),np.sqrt(1.0/9)*np.std(t),np.mean(it),np.sqrt(1.0/9)*np.std(it)











