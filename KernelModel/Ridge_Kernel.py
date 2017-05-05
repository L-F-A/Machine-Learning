import numpy as np
from MachineLearning.KernelModel import KernelCalc

class KerRidgeReg:
##########################################################################################################################
#                                         Kernel Ridge Regression/Gaussian process                                       #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2013-2017)                     #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#  															 #
##########################################################################################################################
        def __init__(self):
                pass

	def train(self,X,y,lam,var,typeK,typeD):

		self.Nlearn = X.shape[0]
		self.Nfeat = X.shape[1]
		self.X = X.copy()
		self.lam = lam
		self.var = var.copy()
		self.typeK = typeK
		self.typeD = typeD
		self.Ker = KernelCalc(X,X,self.Nlearn,self.Nlearn,self.var,typeK=self.typeK,typeD=self.typeD,T=True)
		Klam = self.Ker + lam*np.identity(self.Nlearn)

		#Is it faster to do it at once ( np.linalg.solve(Klam,y)  )?
		try:
			self.L = np.linalg.cholesky(Klam)
			nu = np.linalg.solve(self.L,y)
			self.alpha = np.linalg.solve(self.L.transpose(),nu)
		except np.linalg.linalg.LinAlgError:
                        print 'K+lambda*I not positive definite, solving anyway, but beware!!' #postvar in query will not work, need to be corrected
                        self.alpha = np.linalg.solve(Klam,y)

	def train_withKer(self,X,y,lam,var,Ker,typeK,typeD):
	#If the Kernel matrix is already provided

                self.Nlearn = X.shape[0]
                self.Nfeat = X.shape[1]
                self.X = X.copy()
                self.lam = lam
                self.var = var.copy()
                self.typek = typeK
                self.typeD = typeD
                self.Ker = Ker
                Klam = self.Ker + lam*np.identity(self.Nlearn)

                #Is it faster to do it at once ( np.linalg.solve(Klam,y)  ) ?
		try:
                	self.L = np.linalg.cholesky(Klam)
                	nu = np.linalg.solve(self.L,y)
                	self.alpha = np.linalg.solve(self.L.transpose(),nu)
		except np.linalg.linalg.LinAlgError:
			print 'K+lambda*I not positive definite, solving anyway, but beware!!'#postvar in query will not work, need to be corrected
			self.alpha = np.linalg.solve(Klam,y)

	def query(self,Xt,postVar=False):

		KerTest = KernelCalc(self.X,Xt,self.Nlearn,Xt.shape[0],self.var,typeK=self.typeK,typeD=self.typeD,T=False)
		if postVar is False:
			return KerTest.dot(self.alpha)
		elif postVar is True: #return the Gaussian process posterior variance change k_test^T(K+lambda*I)^-1k_test
			v=np.linalg.solve(self.L,KerTest.transpose())
			print v.shape
			return KerTest.dot(self.alpha), v.transpose().dot(v)
	
	def query_withKer(self,Xt,KerTest,postVar=False):

                if postVar is False:
                        return KerTest.dot(self.alpha)
                elif postVar is True: #return the Gaussian process posterior variance change k_test^T(K+lambda*I)^-1k_test
                        v=np.linalg.solve(self.L,KerTest.transpose())
                        print v.shape
                        return KerTest.dot(self.alpha), v.transpose().dot(v)

	def score(self,ypredic,ytrue,metric='MAE'):
		#Calculate the mean absolute error MAE by default, mean square error MSE
		#ypredic = self.query(Xt)

		if metric == 'MAE':
			#need to implement for multiple outputs
			return np.mean(np.absolute(ypredic-ytrue))
		elif metric == 'MSE':
			ydiff = (ypredic-ytrue)
                        return np.mean(ydiff**2)
		elif metric == 'MDAE':
                        #need to implement for multiple outputs
                        return np.median(np.absolute(ypredic-ytrue))

	#Adding the possibility of hyperparameters determination by maximizing log-likelihood.
	def __LogLike_neg(self,mult=False,Nout=1.):
		#negative log-likelyhood
		if mult==True:
			return 0.5*self.Nlearn*np.linalg.det(self.y.transpose().dot(self.alpha)) + 2.*Nout*np.sum(np.log(np.diag(self.L)))
		else:
			return 0.5*self.y.transpose().dot(self.alpha) + np.sum(np.log(np.diag(self.L)))




