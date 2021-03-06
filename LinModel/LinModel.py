import numpy as np
import warnings
#Everything else is imported locally when needed


class LinReg:
#########################################################################################
#                                   Linear regression                                   #
#                                                                                       #
#        Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)       #
#########################################################################################
#                                                                                       #
#       INPUTS:                                                                         #
#       	X : A ndarray representing a matrix with Nlearn lines with all known	# 
#		    examples and the columns are the dimension of one x			#
#   		y : a 1d array representing a NlearnX1 vector with values for each known# 
#		    examples								#
#	OUTPUTS: 									#


#Verify multiple outputs

	def __init__(self):
		pass

	def train(self,X,y):
		#We treat the constant term as part of the coeffs. w and thus
		#need to add 1 at the end of every input
		OneVec = np.ones((X.shape[0],1))
		XX = np.concatenate((X, OneVec), axis=1)
		
		#How many intances, dimensions of the inputs and how many outputs
		SizeXX = XX.shape
		self.pdim = SizeXX[1]
		self.Nlearn = SizeXX[0]
		if y.ndim == 2:
			self.Nout = y.shape[1]
		else:
			self.Nout = 1

		if self.pdim <= self.Nlearn:
			A = np.dot(XX.transpose(),XX)
			self.w = np.linalg.solve(A,XX.transpose().dot(y))
			self.method = "pxp"
		else:
			A = np.dot(XX,XX.transpose())
			wp = np.linalg.solve(A,y)
			self.w = wp.dot(XX)
			self.method = "NxN"

	def query(self,Xt):
		OneVec = np.ones((Xt.shape[0],1))
                XX = np.concatenate((Xt, OneVec), axis=1)
                return XX.dot(self.w)

	def score(self,Xt,ytrue,metric='MAE'):
	#Calculate the mean absolute error MAE by default, mean square error MSE or
	# correlation coeff CORR
		ypredic = self.query(Xt)
		if metric == 'MAE':
			return np.sum(np.absolute(ypredic-ytrue),axis=0)/len(ytrue)
		elif metric == 'MSE':
			ydiff = (ypredic-ytrue)
                        return np.sum(ydiff**2,axis=0)/len(ytrue)

#########################################################################################

class RidgeReg:
#########################################################################################
#                                   Ridge regression	                                #
#                                                                                       #
#        Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)       #
#########################################################################################
#                                                                                       #
#       INPUTS:                                                                         #
#	   train:									#
#               X      : A ndarray representing a matrix with Nlearn lines with all	#
#			 known examples and the columns are the dimension of one x      #
#               y      : a 1d array representing a NlearnX1 vector with values for each	# 
#		         known examples                                                 #
#		lam    : Regularization parameter					#
#	   query:									#
#		Xt     : A ndarray representing a matrix with Ntest lines with all the 	#
#			 test examples and the columns are the dimension of one x	#
#	   score:									#
#		Xt     : Same as in query						#
#		ytrue  : A 1d array with the values for the test set			#
#		metric : The error metric 'MAE' (default) or 'MSE'			#
#											#
#       OUTPUTS:                                                                        #


	def __init__(self,solver='exact',eta=None,tol=None,Nbatch=None,epochMax=None):
		self.solver=solver
		if solver is 'sg':
			self.Nbatch=Nbatch
                        self.eta=eta
                        self.epochMax=epochMax
                        self.tol=tol
		elif solver is not 'exact':
			raise ValueError('solver must be exact or sg only')

	def __deriv_forSG(self,w,XX,y,lam):
		nfeat=len(XX)
		Imat = np.identity(nfeat)
                Imat[nfeat-1][nfeat-1] = 0.
		return -y*XX + (np.outer(XX,XX)+lam*Imat).dot(w)

	def __deriv(self,w,XX,y,lam):
		nfeat=XX.shape[1]
		Imat = np.identity(nfeat)
                Imat[nfeat-1][nfeat-1] = 0.
		return -np.dot(XX.T,y) + (np.dot(XX.T,XX)+lam*Imat ).dot(w)

	def train(self,X,y,lam):
                #We treat the constant term as part of the coeffs. w and thus
                #need to add 1 at the end of every input
                OneVec = np.ones((X.shape[0],1))
                XX = np.concatenate((X, OneVec), axis=1)

		#How many intances, dimensions of the inputs and how many outputs
                SizeXX = XX.shape
                self.pdim = SizeXX[1]
                self.Nlearn = SizeXX[0]
		self.lam=lam

		if self.solver is 'exact':
			if y.ndim == 2:
                        	self.Nout = y.shape[1]
                	else:
                        	self.Nout = 1

                	if self.pdim <= self.Nlearn:
                        	Imat = np.identity(self.pdim)
                        	Imat[self.pdim-1][self.pdim-1] = 0.
                        	A = np.dot(XX.transpose(),XX) + lam*Imat
                        	self.w = np.linalg.solve(A,XX.transpose().dot(y))
                        	self.method = "pxp"
                	else:
			#In the case where there are less examples N than features dimensions
			#p, switch to a NxN approach, giving the solution with the smallest l2 
			#norm among the infinite possible solutions
                        	Imat = np.identity(self.Nlearn)
                        	Imat[self.Nlearn-1][self.Nlearn-1] = 0.
                        	A = np.dot(XX,XX.transpose()) + lam*Imat
                        	wp = np.linalg.solve(A,y)
                        	self.w = wp.dot(XX)
                        	self.method = "NxN"
		else:
			#Using stochastic gradient descent found in NumMethos directory
                        from NumMethods import StochGrad
                        if self.Nbatch==1:
                                w_0,ep,ite,mess=StochGrad(self.__deriv_forSG,w_0,XX,y,self.Nlearn,(lam/self.Nlearn),lam/self.Nlearn,eta=self.eta,tol=self.tol,epochMax=self.epochMax,Nbatch=self.Nbatch)
                        else:
                                w_0,ep,ite,mess=StochGrad(self.__deriv,w_0,XX,y,self.Nlearn,(lam/self.Nlearn),lam/self.Nlearn,eta=self.eta,tol=self.tol,epochMax=self.epochMax,Nbatch=self.Nbatch)
                        self.w=w_0
                        self.epoch=ep

        def query(self,Xt):
		OneVec = np.ones((Xt.shape[0],1))
                XX = np.concatenate((Xt, OneVec), axis=1)
                return XX.dot(self.w)

	def score(self,Xt,ytrue,metric='MAE'):
        #Calculate the mean absolute error MAE by default, mean square error MSE or 
	#correlation coeff CORR
                ypredic = self.query(Xt)
                if metric == 'MAE':
                        return np.sum(np.absolute(ypredic-ytrue),axis=0)/len(ytrue)
                elif metric == 'MSE':
                        ydiff = (ypredic-ytrue)
                        return np.sum(ydiff**2,axis=0)/len(ytrue)

#########################################################################################
