# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import time
from math import e
import math

def f1(x):
    return x*x-2*x+1
def f2(x):
    return pow(e,x)-5*x
def f3(x):
    return 5+pow(x-2,6)
f1_opt = 0
f2_opt = -3.0472
f3_opt = 5

def ff1(x,y):
    return x*x+y*y
def ff2(x,y):
    return x*x+2*y*y+2*x*y
def RB(x,y):
    return pow(1-x,2)+100*pow(y-pow(x,2),2)
ff1_opt = 0
ff2_opt = 0
RB_opt = 0

x = 0
y = 0
e_abs = 1e-4
e_f = 1e-6
e_r = np.sqrt(1e-15)

gs_test = DataFrame(columns=['Function','s','time_mean','time_std','dist_mean','dist_std','iterNum_mean','iterNum_std']);

def GoldenSection(f,s):
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

def CoorDes(f,s):
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
		x,f_new,iterNum_x,stop_x = GoldenSection(f_x,s)
		y,f_new,iterNum_y,stop_y = GoldenSection(f_y,s)
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
	return x,y,f_new,i,stop

def GSTest(f,testNum,optimum,s,analysis=False):
	global gs_test;
	GST = DataFrame(columns=['Function','testNum','clockTime','x_opt','f_opt','Distance','iterNum'])
	i = 0
	while i<testNum:
		start = time.time()
		x_opt,f_opt,iterNum,stop = GoldenSection(f,s)
		end = time.time()
		elapsed = end - start
		if not stop:
			i = i+1
			dist = abs(optimum-f_opt)
			# print f.__name__,iterNum,elapsed,dist
			# print x_opt,f_opt
			GST.loc[len(GST)] = [f.__name__,testNum,elapsed,x_opt,f_opt,dist,iterNum]
	result = open("./test_results/GSTest_Result"+'_'+f.__name__+".txt","w")
	result.write("GTest\nClockTime: "+str(GST.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.clockTime.std())+'\n' \
		"Distance: "+str(GST.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.Distance.std())+'\n' \
		"iterNum: "+str(GST.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.iterNum.std())+'\n')
	result.close()
	print str(GST.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.clockTime.std())
	print str(GST.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.Distance.std())
	print str(GST.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*GST.iterNum.std())
	if analysis:
		GST.to_csv('./results_analysis/GSTest'+'_'+f.__name__+'_'+str(s)+'_'+str(e_f)+'.csv',index=False)
		gs_test.loc[len(gs_test)] = [f.__name__,s,GST.clockTime.mean(),np.sqrt(1.0/(testNum-1))*GST.clockTime.std(), \
		GST.Distance.mean(),np.sqrt(1.0/(testNum-1))*GST.Distance.std(),GST.iterNum.mean(),np.sqrt(1.0/(testNum-1))*GST.iterNum.std()]
	else:
		GST.to_csv('./test_results/GSTest'+'_'+f.__name__+'.csv',index=False)

def CDTest(f,testNum,optimum,s):
	CDT = DataFrame(columns=['Function','testNum','clockTime','x_opt','y_opt','f_opt','Distance','iterNum'])
	i = 0
	while i<testNum:
		# print i;
		start = time.time()
		x_opt,y_opt,f_opt,iterNum,stop = CoorDes(f,s)
		# print x_opt,y_opt,f_opt,stop
		end = time.time()
		elapsed = end - start
		if not stop:
			i = i+1
			dist = abs(optimum-f_opt)
			# print f.__name__,iterNum,elapsed,dist
			CDT.loc[len(CDT)] = [f.__name__,testNum,elapsed,x_opt,y_opt,f_opt,dist,iterNum]
	result = open("./test_results/CDTest_Result"+'_'+f.__name__+".txt","w")
	result.write("CDest\nClockTime: "+str(CDT.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.clockTime.std())+'\n' \
		"Distance: "+str(CDT.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.Distance.std())+'\n' \
		"iterNum: "+str(CDT.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.iterNum.std())+'\n')
	result.close()
	print str(CDT.clockTime.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.clockTime.std())
	print str(CDT.Distance.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.Distance.std())
	print str(CDT.iterNum.mean())+" +- "+str(np.sqrt(1.0/(testNum-1))*CDT.iterNum.std())
	CDT.to_csv('./test_results/CDTest'+'_'+f.__name__+'.csv',index=False)



if __name__  == "__main__":
	GSTest(f1,100,f1_opt,1)
	GSTest(f2,100,f2_opt,1)
	GSTest(f3,100,f3_opt,1)

	CDTest(ff1,100,ff1_opt,1)
	CDTest(ff2,100,ff2_opt,1)
	CDTest(RB,100,RB_opt)

	s = 0.01
	GSTest(f1,100,f1_opt,s,analysis=True)
	GSTest(f2,100,f2_opt,s,analysis=True)
	GSTest(f3,100,f3_opt,s,analysis=True)
	s = 0.1
	GSTest(f1,100,f1_opt,s,analysis=True)
	GSTest(f2,100,f2_opt,s,analysis=True)
	GSTest(f3,100,f3_opt,s,analysis=True)
	s = 1
	GSTest(f1,100,f1_opt,s,analysis=True)
	GSTest(f2,100,f2_opt,s,analysis=True)
	GSTest(f3,100,f3_opt,s,analysis=True)
	s = 10
	GSTest(f1,100,f1_opt,s,analysis=True)
	GSTest(f2,100,f2_opt,s,analysis=True)
	GSTest(f3,100,f3_opt,s,analysis=True)

	gs_test.to_csv("./results_analysis/results_analysis.csv")
	
	


	


