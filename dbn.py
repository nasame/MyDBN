from rbm import RBM
from softmax import SoftMax
from common import *

class DBN:
    def __init__(self,nlayers,ntype):
        self.rbm_layers = []
        self.nlayers = nlayers
        self.softmax_layer = None
        self.trainflag = False
        self.ntype = ntype
    def calcRBMForward(self,x):
        layerid = 1
        for rbm in self.rbm_layers:
            x = rbm.calc_forward(x)
            if layerid < self.nlayers:
                x = rbm.sample(x)
            layerid += 1
        return x
    
    def load_dbn_param(self,dbnpath,softmaxpath):
        weights = cPickle.load(open(dbnpath,'rb'))
        vlen,hlen = 0,0
        self.nlayers = len(weights)
        for i in range(self.nlayers):
            weight = weights[i]
            vlen,hlen = weight.shape[0],weight.shape[1]
            rbm = RBM(vlen,hlen)
            rbm.W = weight
            self.rbm_layers.append(rbm)
            print "RBM layer%d shape:%s" %(i,str(rbm.W.shape))
        self.softmax = SoftMax()
        self.softmax.load_theta(softmaxpath)
        print "softmax parameter: "+str(self.softmax.theta.shape)
        
    def pretrainRBM(self,trainset):
        trainv = np.mat(trainset[1])   # 1xn
        vlen = trainv.shape[1]
        trainnum = len(trainset)
        hlen = 500
        weights = []
        print "vlen = %d" %(vlen)
        print "Trainnum = %d" %(trainnum)
        for i in range(self.nlayers):
            rbm = RBM(vlen,hlen)
            T,e = 3,0.05
            if i == 0:
                traindata = trainset
            else:
                traindata = outdata
            outdata = np.zeros((trainnum,hlen))
            for j in range(trainnum):
                print "layer:%d CD sample %d..." %(i,j)
                trainv = np.mat(traindata[j])
                rbm.train_CD(trainv,T,e)
                outdata[j] = np.mat(rbm.sample(rbm.calc_forward(trainv)))   # 1xhlen
            self.rbm_layers.append(rbm)
            weights.append(rbm.W)
            vlen = hlen
#            hlen -= 100
        dump_data("data/dbn.pkl",weights)
        print "========= pretrainRBM complete ==========="
    
    def fineTune(self,trainset,labelset):
        trainnum = len(trainset)
        if trainnum > 1000:
            trainnum = 1000
        print "Trainnum = %d" %(trainnum)
        rbm_output = np.zeros((trainnum,self.rbm_layers[-1].hsize))
        for i in range(trainnum):
            x = trainset[i]
            rbm_output[i] = self.calcRBMForward(x)   #rbm_output  0,1,0,1,0,0.....
        MAXT,step,landa = 800,0.02,0.01
        self.softmax = SoftMax(MAXT,step,landa)
        self.softmax.process_train(rbm_output,labelset,self.ntype)
        print "======== fineTune Complete ==========="
        
    def predict(self,x):
        rbm_output = self.calcRBMForward(x)
        ptype = self.softmax.predict(rbm_output)
        return ptype
        
    def validate(self,testset,labelset):
        rate = 0
        testnum = len(testset)
        correctnum = 0
        for i in range(testnum):
            x = testset[i]
            testtype = self.predict(x)
            orgtype = labelset[i]
            print "Testype:%d\tOrgtype:%d" %(testtype,orgtype)
            if testtype == orgtype:
                correctnum += 1
        rate = float(correctnum)/testnum
        print "correctnum = %d, sumnum = %d" %(correctnum,testnum)
        print "Accuracy:%.2f" %(rate)
        return rate
        
###### main #########

dbn = DBN(1,10)
if sys.argv[1] == "train":
    Train = 1
else:
    Train = 0

set_x,set_y = load_minst_data("data/trainmnist.pkl")
if Train:
    dbn.pretrainRBM(set_x)
    dbn.fineTune(set_x,set_y)
else:
    dbn.load_dbn_param("data/dbn.pkl","data/softmax.pkl")
    set_x,set_y = load_minst_data("data/testmnist.pkl")
    dbn.validate(set_x,set_y)