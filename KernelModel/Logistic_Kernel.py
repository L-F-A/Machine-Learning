import numpy as np
from scipy.optimize import minimize
from MachineLearning.KernelModel import KernelCalc


class KerLogisReg:
##########################################################################################################################
#                                    Kernel Logistic Regression with 'L2' regularization                                 #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#  
#   X   is a ndarray representing a matrix with Nlearn lines with all known 
#       examples and the columns
#       are the dimension of one x
#   y   is a 1d array representing a NlearnX1 vector with values 0 or 1 for each known
#       examples
#   Xt  is an array representing with the inputs to be  queried
#



        def __init__(self):
                pass
        def __logLike_logistic_L(self,w,y,lam):
		#Like Eq.4 of http://papers.nips.cc/paper/2059-kernel-logistic-regression-and-the-import-vector-machine.pdf
                WdotX = self.Ker.dot(w[:self.Nl]) + w[self.Nl]
                l1 = -np.sum( np.log( 1.+np.exp(WdotX) ) )
                l2 = (y.dot(WdotX))
                l3 = -0.5*lam*w[:self.Nl].dot(self.Ker).dot(w[:self.Nl])
                return -(l1+l2+l3)
        def __logLike_logistic_der_L2(self,w,y,lam):
        	WdotX = self.Ker.dot(w[:self.Nl]) + w[self.Nl]
                p = 1./(1.+np.exp(-WdotX))
                dyp = y-p-lam*w[:self.Nl]
		dl = np.zeros_like(w)
		dl[:self.Nl]=-self.Ker.dot(dyp)
		dl[self.Nl]=-np.sum(y-p)
		return dl
	#def __logLike_logistic_der2_L2(self,w,y,lam):
	#	WdotX = self.Ker.dot(w)
	#	Gamma = 1./( 2.*( 1+np.cosh(WdotX) ) )
	#	return self.Ker.dot( np.diag(Gamma).dot(self.Ker) ) + lam*self.Ker
        def train(self,X,y,lam,var,typeD,typeK,w_0=None):
		self.typeD = typeD
		self.typeK = typeK
		self.var = var.copy()
		self.Nl = X.shape[0]
		self.X = X.copy()
		self.Ker = KernelCalc(self.X,self.X,self.Nl,self.Nl,var=self.var,typeK=self.typeK,T=True)
                tupARG = (y,lam)
		if w_0==None:
			#The guess weight vector is set to zero
			w_0 = np.zeros((1,self.Nl+1))
               	res = minimize(self.__logLike_logistic_L,w_0,args=tupARG,method='BFGS',jac=self.__logLike_logistic_der_L2,tol=1e-8,options={'gtol': 1e-08})
                self.w = res.x
	#def train_withKer(self,y,lam,var,Ker,typeK,typeD,w_0=None):
	#	self.var=var
	#	self.typeD=typeD
	#	self.typeK=typeK
        #       self.Ker = Ker
	#	self.Nl = Ker.shape[0]
        #        tupARG = (y,lam)
        #        if w_0==None:
                        #The guess weight vector is set to zero
        #                w_0 = np.zeros((1,self.Nl))
	#	res = minimize(self.__logLike_logistic_L,w_0,args=tupARG,method='Newton-CG',jac=self.__logLike_logistic_der_L2,hess=self.__logLike_logistic_der2_L2,tol=1e-8,options={'xtol': 1e-08})
        #        self.w = res.x
        def query(self,Xt,threshold=0.5):
		KerTest = KernelCalc(self.X,Xt,self.Nl,Xt.shape[0],var=self.var,typeK=self.typeK,typeD=self.typeD,T=False) 
                WdotX = KerTest.dot(self.w[:self.Nl]) + w[self.Nl]
                p = 1./(1.+np.exp(-WdotX))
                predic = np.zeros(len(p))
                predic[p>threshold] = 1.
		#predic = np.rint(p)
                ProbPred = np.zeros((2,len(p)))
                ProbPred[0,0:len(p)] = p
                ProbPred[1,0:len(p)] = predic
                return ProbPred
        def score(self,Xt,ytrue):
                #Return the % of correct predictions
                Predic = self.query(Xt)
                ypredic = Predic[1,:]
                return 1-np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
