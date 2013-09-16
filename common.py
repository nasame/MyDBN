import numpy as np
import cPickle
import gzip,os,sys,time

def sigmoid(x):
	return 1/(1+np.exp(-x))

def load_data(datapath):
	f=gzip.open(datapath,'rb')
	train_set,valid_set,test_set = cPickle.load(f)
	f.close()
	return train_set,valid_set,test_set
	
def dump_data(datapath,data):
	f= open(datapath,'wb')
	cPickle.dump(data,f)
	f.close()

def load_minst_data(datapath):
	set_x,set_y = cPickle.load(open(datapath,'rb'))
	return np.mat(set_x),np.mat(set_y).T

def get_part_data():
	trainnum,testnum = 600,200
	train_set,valid_set,test_set = load_data("data/mnist.pkl.gz")
	dump_data("data/trainmnist.pkl",[train_set[0][:trainnum],train_set[1][:trainnum]])
	dump_data("data/testmnist.pkl",[test_set[0][:testnum],test_set[1][:testnum]])