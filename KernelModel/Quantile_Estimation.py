import numpy as np
from MachineLearning.KernelModel import KernelCalc
from NumMethods import quadprog

class QuantEst:
##########################################################################################################################
#        				     Quantile estimation		                                         #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#															 #
#	INPUTS:														 #
#															 #
##########################################################################################################################
        def __init__(self):
                pass
        def train(self,X,y,tau,typeK='poly',var=None,typeD=None,lam=0.1,tolbias=1e-3,xinterval=None):
	#Solving the dual with quad. prog.

		#Number of instances in the training set and different parameters needed to keep the model.				
		SizeX = X.shape
                pdim = SizeX[1]
                Nlearn = SizeX[0]

		C = 1./lam/Nlearn

		#Tolerance for the bias
                self.tolbias=tolbias*C

                self.Nlearn = Nlearn
		self.typeK = typeK
		self.typeD = typeD
		self.var = var
                self.X = X.copy()
		self.lam = lam
		self.tau = tau
			
		#Building the matrices for the quadratic prog. problem
		Kmat = KernelCalc(X,X,Nlearn,Nlearn,var=self.var,typeK=self.typeK,typeD=self.typeD,T=True,xinterval=xinterval)

		#Quadratic form
		H1 = Kmat
		q1 = y.copy()

		#Equality constraint
		Aeq1=np.ones(Nlearn)
		beq1=np.array([0.])
		
		#Lower and upper bound
		lb1 = C*(tau-1)*np.ones(Nlearn)
		ub1 = C*tau*np.ones(Nlearn)

		#Solve the quad. prog.
		results=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1,returnVar='no')
		self.alpha=np.array(results['x']).reshape(Nlearn)
		self.b=np.array(results['y'])[0,0]

	def train_withKer(self,X,y,Ker,tau,typeK='poly',var=None,typeD=None,lam=0.1,tolbias=1e-3):

		#Number of instances in the training set and different parameters needed to keep the model.				
		SizeX = X.shape
                pdim = SizeX[1]
                Nlearn = SizeX[0]

		C = 1./lam/Nlearn

		#Tolerance for the bias
                self.tolbias=tolbias*C

                self.Nlearn = Nlearn
		self.typeK = typeK
		self.typeD = typeD
		self.var = var
                self.X = X.copy()
		self.lam = lam
		self.tau = tau
			

		#Quadratic form
		H1 = Ker.copy()
		q1 = y.copy()

		#Equality constraint
		Aeq1=np.ones(Nlearn)
		beq1=np.array([0.])
		
		#Lower and upper bound
		lb1 = C*(tau-1)*np.ones(Nlearn)
		ub1 = C*tau*np.ones(Nlearn)

		#Solve the quad. prog.
		results=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1,returnVar='no')
		self.alpha=np.array(results['x']).reshape(Nlearn)
		self.b=np.array(results['y'])[0,0]

	def query(self,Xt,xinterval=None):
		#Prediction for matrix Xt
		Ktest = KernelCalc(self.X,Xt,self.Nlearn,Xt.shape[0],var=self.var,typeK=self.typeK,typeD=self.typeD,T=False,xinterval=xinterval)
		return Ktest.dot(self.alpha) + self.b
	def query_withKer(self,Ktest):
                #Prediction for matrix Xt
                return Ktest.dot(self.alpha)+self.b
##########################################################################################################################
