from common import *
import matplotlib.pyplot as plt

def createMyData(datapath,ptnum):
	mean1 = [10,-10]
	mean2 = [10,10]
	mean3 = [-10,-10]
	mean4 = [-10,10]
	cov = [[5,0],[0,5]]
	x1,y1 = np.random.multivariate_normal(mean1,cov,ptnum).T
	x2,y2 = np.random.multivariate_normal(mean2,cov,ptnum).T
	x3,y3 = np.random.multivariate_normal(mean3,cov,ptnum).T
	x4,y4 = np.random.multivariate_normal(mean4,cov,ptnum).T
	x = np.mat(np.concatenate((x1,x2,x3,x4)))
	y = np.mat(np.concatenate((y1,y2,y3,y4)))
	trainset=np.concatenate((x.T,y.T),axis=1)
	labelset=np.concatenate((np.repeat(0,ptnum),np.repeat(1,ptnum),np.repeat(2,ptnum),np.repeat(3,ptnum)))
	dump_data(datapath,[trainset,labelset])
	plt.plot(x1,y1,'x'); 
	plt.plot(x2,y2,'x');  
	plt.plot(x3,y3,'x'); 
	plt.plot(x4,y4,'x'); 
	plt.axis('equal'); 
	plt.show()
#	plt.plot(trainset[:,0],trainset[:,1],'x'); 
#	plt.axis('equal'); 
#	plt.show()

######## main #########
Train = 1
trainpath = "data/mytrain.pkl"
testpath = "data/mytest.pkl"
if Train:
	createMyData(testpath,200)
else:
	set_data,set_label= load_minst_data(trainpath)
	print set_label