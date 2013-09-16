import numpy as np
from common import *


class RBM:
	def __init__(self,vlen,hlen):
		mu,sigma = 0,0.01
	#	W = np.random.normal(mu,sigma,vlen*hlen)
		W = 0.01*np.mat(np.random.randn(vlen,hlen))
		self.W = W         #nxm
		self.b = np.zeros(hlen)
		self.h = []
		self.vsize = vlen
		self.hsize = hlen
	
	def initBias(self,x):
		if x> 0 and x < 1:
			return (np.log(x)-np.log(1-x))
		return 0
	
	def calc_forward(self,vnode):
		output = []
		for j in range(self.hsize):
			sumdot = np.dot(vnode,self.W[:,j])
			hout = sigmoid(self.b[j]+sumdot)
			output.append(hout[0,0])
		return output
	
	def calc_backward(self,hnode):
		output = []
		for i in range(self.vsize):
			sumdot = np.dot(hnode,self.W[i,:].T)
			vout = sigmoid(self.a[i]+sumdot)
			output.append(vout[0,0])
		return output
	
	def sample(self,p_arr):
		ret = []
		for p in p_arr:
			value = np.random.binomial(1,p)
			ret.append(value)
		return ret
	
	def train_CD(self,v,T,e): 
		v1 = v.tolist()[0]                     #(1xn)
		self.a = map(self.initBias,v1);
		for k in range(T):
			#print "rbm train loop%d is running" %(k)
			v1 = self.sample(v1)
			p_hv_arr1 = self.calc_forward(v1)     #get P(h1j=1 | v1)   (1xm)
			h1 = self.sample(p_hv_arr1)           #binary hidden
			p_vh_arr1 = self.calc_backward(h1)
			v2 = p_vh_arr1                        #self.sample(p_vh_arr1)
			p_hv_arr2 = self.calc_forward(v2)
			h2 = p_hv_arr2
			self.updateParam(p_hv_arr1,p_hv_arr2,v1,v2,e)
			
	def updateParam(self,p_hv_arr1,p_hv_arr2,v1,v2,e):
		wd1 = np.mat(v1).T * np.mat(p_hv_arr1)        #(nxm)
		wd2 = np.mat(v2).T * np.mat(p_hv_arr2)        #(nxm)
		deltaW = e * np.subtract(wd1,wd2)              #(nxm)
		self.W += deltaW                            #update W
		deltaA = e * np.subtract(v1,v2)
		self.a += deltaA                           #update a
		deltaB = e * np.subtract(p_hv_arr1,p_hv_arr2)
		self.b += deltaB                           #update b
		
	def printRBM(self):
		print self.W
		print self.a
		print self.b
