import numpy as np
import warnings
#Everything else is imported locally when needed

class LogisRegRegu:
#########################################################################################################################
#                                     Logistic regression with L2 or L1 regularization                                  #
#                                                                                                                       #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2013-2017)               #
#########################################################################################################################
#                                                                                                                       #
#       INPUTS:														#
#		type     : Regularization type. The choice are 'L2' (default) and 'L1'                                  #
#		solver   : Which solver: 'sp' uses scipy minimized, NEWTON-CG for L2 and POWELL for L1			#
#				         'gd' gradient descent								#
#				         'sg' stochastic gradient descent						#
#		eta      : The value to multiply the gradient in 'gd' or 'sg'						#
#		tol and  : Tolerance and relative tolerance for w in 'gd' or 'sg'. Will convergence if any 		#
#		tol_rel    respected											#
#		Nbatch   : When 'sg', the approximate size of the mini batchs						#
#		epochMax : When 'sg', maximum number of epochs to go							#
#															#
#	OUTPUTS:													#
#		self.w 	 : The vector w. w[-1] is the bias								#
#		self.solver												#
#		self.TYPE												#
#		self.Nlearn												#
#															#
#		COMMON TO 'gd' and 'sg':										#
#		self.eta 												#
#		self.tol												#
#		self.tol_rel												#
#															#
#		UNIQUE TO 'gd'												#
#		self.ite : Number of iteration that were necessary							#
#															#
#		UNIQUE TO 'sg'												#
#		self.Nbatch 												#
#		self.epochMax												#
#		self.epoch : How many epochs were necessary								#
#########################################################################################################################

        def __init__(self,type='L2',solver='sp',eta=None,tol=None,tol_rel=None,Nbatch=None,epochMax=None):
		self.solver=solver	
		if solver is 'sp':
                	#Reguralized logistic regression. The possible types are 'L2' (default) and 'L1'
                	self.TYPE = type
		elif solver is 'gd':
			self.TYPE='L2'
			self.eta=eta
			self.tol=tol
                        self.tol_rel=tol_rel
		elif solver is 'sg':#Solving using stochastic gradient descent with mini batch of size around Nbatch (no$self.type='L2'#Only L2 for the moment for stochastic gradient
			self.TYPE='L2'
			self.Nbatch=Nbatch
                        self.eta=eta
                        self.epochMax=epochMax
                        self.tol=tol
                        self.tol_rel=tol_rel
		else:
			raise ValueError('solver must be sp, gd or sg only ')

	def __MSE(self,w,XX,y):
		WdotX = XX.dot(w)
                p = 1./(1.+np.exp(-WdotX))
                return np.mean( (y-p)**2 )

	def __logLike_logistic_L(self,w,XX,y,lam):
                WdotX = XX.dot(w)
                l1 = -np.sum( np.log( 1.+np.exp(WdotX) ) )
                l2 = (y.dot(WdotX))
                if self.TYPE == 'L2':
                        l3 = -0.5*lam*np.sum(w[0:-1]**2)
                elif self.TYPE == 'L1':
                        l3 = -lam*np.sum(np.absolute(w[0:-1]))
                return -(l1+l2+l3)

        def __logLike_logistic_der_L2(self,w,XX,y,lam):
                WdotX = XX.dot(w)
                p = 1./(1.+np.exp(-WdotX))
                dyp = y-p
		dl=-dyp.dot(XX) + lam*w
		dl[-1]=dl[-1]-lam*w[-1]
                return dl

	def __logLike_logistic_der_L2_forSG(self,w,XX,y,lam):
                WdotX = XX.dot(w)
                p = 1./(1.+np.exp(-WdotX))
                dyp = y-p
		dl=-dyp*XX + lam*w
                dl[-1]=dl[-1]-lam*w[-1]
                return dl

	def __logLike_logistic_der2_L2(self,w,XX,y,lam):
                WdotX = XX.dot(w)
                p = 1./(1.+np.exp(-WdotX))
		#Can I avoid creating the identity matrix every time? Pass as parameter?
		Imat = np.identity(XX.shape[1])
                Imat[XX.shape[1]-1][XX.shape[1]-1] = 0
		Qmat = np.diag(p*(1-p)) 
		return XX.transpose().dot( Qmat.dot(XX) )+lam*Imat

	def train(self,X,y,lam):
		
		OneVec = np.ones((X.shape[0],1))
                XX = np.concatenate((X, OneVec), axis=1)
		w_0 = np.zeros(XX.shape[1])
		self.Nlearn=XX.shape[0]
		
		if self.solver is 'sp':
		#Using Scipy minimizers
			from scipy.optimize import minimize
			tupARG = (XX,y,lam)
                	if self.TYPE == 'L2':
                        	res = minimize(self.__logLike_logistic_L,w_0,args=tupARG,method='Newton-CG',jac=self.__logLike_logistic_der_L2,tol=1e-8,options={'xtol': 1e-08})
                        	self.w = res.x
                	elif self.TYPE == 'L1':
                        	res = minimize(self.__logLike_logistic_L,w_0,args=tupARG,method='Powell',tol=1e-8)
                        	self.w = res.x
				self.w[np.abs(self.w)<=1e-10]=0.

		elif self.solver is 'gd':
		#Using homemade gradient descent found in NumMethods directory
			from NumMethods import GradDescSteep
			w_0,ite=GradDescSteep(self.__logLike_logistic_der_L2,w_0,self.eta,self.tol,self.tol_rel,10000,args=(XX,y,lam))
			self.w=w_0
			self.ite=ite

		else:
		#Using stochastic gradient descent found in NumMethos directory
			from NumMethods import StochGrad
			if self.Nbatch==1:
				w_0,ep,ite,mess=StochGrad(self.__logLike_logistic_der_L2_forSG,w_0,XX,y,self.Nlearn,(lam/self.Nlearn),lam/self.Nlearn,eta=self.eta,tol=self.tol,epochMax=self.epochMax,Nbatch=self.Nbatch)
			else:
				w_0,ep,ite,mess=StochGrad(self.__logLike_logistic_der_L2,w_0,XX,y,self.Nlearn,(lam/self.Nlearn),lam/self.Nlearn,eta=self.eta,tol=self.tol,epochMax=self.epochMax,Nbatch=self.Nbatch)
			self.w=w_0
                        self.epoch=ep


        def query(self,x,threshold=0.5):
		OneVec = np.ones((x.shape[0],1))
                XX = np.concatenate((x, OneVec), axis=1)
		WdotX = XX.dot(self.w)
                p = 1./(1.+np.exp(-WdotX))
		predic = np.zeros(len(p))
		predic[p>threshold] = 1.
		ProbPred = np.zeros((2,len(p)))
		ProbPred[0,:] = p
		ProbPred[1,:] = predic
                return ProbPred

	def score(self,X,ytrue):
        #Return the % of correct predictions
                Predic = self.query(X)
                ypredic = Predic[1,:]
                return 1-np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
##########################################################################################################################
