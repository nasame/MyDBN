import numpy as np
from common import *
import sys
#import matplotlib.pyplot as plt

class SoftMax:
	def __init__(self,STOPNUM=100,step=0.01,landa=0.01):   #x,y => np.mat()
		self.STOPNUM = STOPNUM
		self.step = step
		self.landa = landa
		
	def load_theta(self,datapath):
		self.theta = cPickle.load(open(datapath,'rb'))
	
	def train(self,datapath,typenum):
		trainnum,MaxTrainNum = 0 , 2000
		train_set,valid_set,test_set = load_data(datapath)
		x,y = train_set[0],train_set[1]
		trainnum = len(x)
		if(trainnum > MaxTrainNum):
			trainnum = MaxTrainNum
		self.process_train(x[:trainnum],y[:trainnum],typenum)
	
	def process_train(self,x,y,typenum):                           # x =>(trainnum x n)     y => (trainnum x 1)
		xtypenum = np.zeros(typenum)
		costval = np.zeros(self.STOPNUM)
		for val in y:
			xtypenum[val]+=1
		print xtypenum
		trainnum = x.shape[0]
		bias = np.mat(np.ones(trainnum))
#		x = np.concatenate((bias.T,x),axis=1)                     # x => (trainnum x n)
		featurenum = x.shape[1]
		print "Trainnum = %d, featurenum = %d" %(trainnum,featurenum)    #featurenum = n+1(bias)
		self.theta = 0.001*np.mat(np.random.randn(typenum,featurenum))
#		self.theta[0] = np.ones(featurenum);
		lastcostJ = 1000
		for m in range(self.STOPNUM):
			############ Loop #############
			costs = np.zeros((typenum,trainnum))
			grads = np.zeros((typenum,featurenum))
			for j in range(typenum):
				jvalues = np.zeros((trainnum,featurenum))
				for i in range(trainnum):
					datas = np.zeros(typenum)
					hval = self.h(x[i])
					ptype = hval[0,j]
					delta = -ptype
					if j == y[i]:
						delta = 1-ptype
						costs[j,i] = np.log(ptype)
					jvalue = np.multiply(x[i],delta)   #(1xn)
					jvalues[i] = jvalue
				grads[j] = -np.mean(jvalues,axis=0)+self.landa * self.theta[j] #gradJ => (1xn)
			for k in range(typenum):
				self.theta[k] = self.theta[k] - self.step*grads[k]
			costJ = -np.sum(costs)/trainnum +(self.landa/2)*np.sum(np.square(self.theta))
			costval[m] = costJ
			if(costJ > lastcostJ):
				print "costJ is increasing !!!"
				break
			print "Loop(%d) cost = %.3f diff=%.4f" %(m,costJ,costJ-lastcostJ)
			lastcostJ = costJ
		dump_data("data/softmax.pkl",self.theta)
				
	def h(self,x):                #  x=>(1xn)
		m = np.exp(np.dot(np.mat(x),self.theta.T))   #   e(thetaT*x)      1xn * nxk  
		sump = np.sum(m)
		ret = m/sump
		return ret
		
	def predict(self,x):
		pv = self.h(np.mat(x))
		return np.argmax(pv)               # return predict type with max p(y|x)
		
	def test(self,datapath,typenum):
		train_set,valid_set,test_set = load_data(datapath)
		x,y = test_set[0],test_set[1]
		testnum = 1000
		x = x[:testnum]
		y = y[:testnum]
		#x,y = load_minst_data(datapath)
		testnum = len(x)
		bias = np.mat(np.ones(testnum))
#		x = np.concatenate((bias.T,x),axis=1)
		rightnum = 0
		corrects=np.zeros(typenum)
		print "Test sample number:%d" %(testnum)
		for i in range(testnum):
			type = softmax.predict(x[i])	
			if(y[i] == type):
				corrects[type] += 1
				rightnum += 1
		rate = float(rightnum)/testnum
		print corrects
		print "Accuracy rate = %.4f,rightnum = %d" %(rate,rightnum)


########## main #############
#if sys.argv[1] == "train":
#    TRAIN = 1
#else:
#    TRAIN = 0
#MAXT,step,landa,typenum = 100,0.1,0.01,10
#softmax = SoftMax(MAXT,step,landa)
#if TRAIN:
#	trainpath = "data/mnist.pkl.gz"
#	softmax.train(trainpath,typenum)
#else:
#	testpath = "data/mnist.pkl.gz"
#	softmax.load_theta("data/softmax.pkl")
##	print softmax.theta
#	softmax.test(testpath,typenum)
