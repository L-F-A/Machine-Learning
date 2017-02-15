import  numpy as np
from MachineLearning.KernelModel import KernelCalc
from NumMethods import quadprog

class SVMC:
##########################################################################################################################
#        				      Support vector machine with Kernel                                         #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#															 #
#	INPUTS:														 #
#															 #

        def __init__(self):
                pass
	def train(self,X,y,typeK,var,typeD=None,C=1.,tolsvm=1e-4):
				#Solving the dual with quad. prog.

				#Tolerance for the choice of support vectors
				self.tolsvm=tolsvm*C
				#Number of instances in the training set as well as the dimension of final w				
				SizeX = X.shape
                                Nlearn = SizeX[0]
                                self.Nlearn = Nlearn
				self.C = C
				self.var = var
				self.typeK=typeK
				self.typeD=typeD
				self.X=X
				self.y=y
				#Building the matrices for the quadratic prog. problem
				Kmat = KernelCalc(X,X,Nlearn,Nlearn,var=self.var,typeK=self.typeK,typeD=self.typeD,T=True)
				H1 = np.diag(y).dot( Kmat.dot(np.diag(y)) )
				q1 = -np.ones(Nlearn)
				Aeq1 = y.copy()
				beq1 = np.array([0.])
				lb1 = np.zeros(Nlearn)
				ub1 = C*np.ones(Nlearn)

				#Solve the quad. prog.
				alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)
				self.alpha = alpha.transpose()[0]

				#Find the support vectors and only use them
				#indsv = np.where((alpha > self.tolsvm) & (alpha < (C-self.tolsvm)))[0].astype(int)
				indsv = np.where(alpha > self.tolsvm)[0].astype(int)
                                self.indsv = indsv
				#Keep only the nopn zero alpha
				self.alpha = self.alpha[self.indsv]
				self.alpha[self.alpha >= 0.999*C]=C
				KmatTemp = Kmat[self.indsv,:]
                		Kmat = KmatTemp[:,self.indsv]

				ind_no_upperBound = np.where(self.alpha != C)[0].astype(int)
				yKmat = np.diag(self.y[self.indsv]).dot(Kmat)
                		self.b = -np.mean(self.alpha.dot(yKmat[:,ind_no_upperBound]))
	def train_withKer(self,X,y,Ker,typeK,var,typeD=None,C=1.,tolsvm=1e-4):
				#Solving the dual with quad. prog.

				#Tolerance for the choice of support vectors
				self.tolsvm=tolsvm*C
				#Number of instances in the training set as well as the dimension of final w				
				SizeX = X.shape
                                Nlearn = SizeX[0]
                                self.Nlearn = Nlearn
				self.C = C
				self.var = var
				self.typeK=typeK
				self.typeD=typeD
				self.X=X
				self.y=y
				#Building the matrices for the quadratic prog. problem
				Kmat = Ker.copy()
				H1 = np.diag(y).dot( Kmat.dot(np.diag(y)) )
				q1 = -np.ones(Nlearn)
				Aeq1 = y.copy()
				beq1 = np.array([0.])
				lb1 = np.zeros(Nlearn)
				ub1 = C*np.ones(Nlearn)

				#Solve the quad. prog.
				alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)
				self.alpha = alpha.transpose()[0]

				#Find the support vectors and only use them
				#indsv = np.where((alpha > self.tolsvm) & (alpha < (C-self.tolsvm)))[0].astype(int)
				indsv = np.where(alpha > self.tolsvm)[0].astype(int)
                                self.indsv = indsv
				#Keep only the nopn zero alpha
				self.alpha = self.alpha[self.indsv]
				self.alpha[self.alpha >= 0.999*C]=C
				KmatTemp = Kmat[self.indsv,:]
                		Kmat = KmatTemp[:,self.indsv]

				ind_no_upperBound = np.where(self.alpha != C)[0].astype(int)
				yKmat = np.diag(self.y[self.indsv]).dot(Kmat)
                		self.b = -np.mean(self.alpha.dot(yKmat[:,ind_no_upperBound]))

	def query(self,Xt):
		#Prediction for matrix Xt
		Ktest = KernelCalc(self.X,Xt,self.Nlearn,Xt.shape[0],var=self.var,typeK=self.typeK,typeD=self.typeD,T=False)

                return np.sign( np.dot(diag(self.y[self.indsv]).dot(Ktest),dot(self.alpha))+b)
	def score(self,X,ytrue):
		#Return the % of correct predictions
		ypredic = self.query(X)
                return 1-0.5*np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
##########################################################################################################################



class SVM_OneClass:
##########################################################################################################################
#        				     Unsupervided nu-svm for anomaly	                                         #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#															 #
#	INPUTS:														 #
#															 #

        def __init__(self):
                pass
        def train(self,X,typeK,var,typeD=None,nu=0.1,tolsvm=1e-4):
	#Solving the dual with quad. prog.
        #The constant term is included in w at the end as the last value

		#Tolerance for the choice of support vectors
		self.tolsvm=tolsvm
		#Number of instances in the training set and different parameters needed to keep the model.				
		SizeX = X.shape
                pdim = SizeX[1]
                Nlearn = SizeX[0]
                self.Nlearn = Nlearn
		self.typeK = typeK
		self.typeD = typeD
		self.var = var
                self.X = X.copy()
		self.nu = nu			
		#Building the matrices for the quadratic prog. problem
		Kmat = KernelCalc(X,X,Nlearn,Nlearn,var=self.var,typeK=self.typeK,typeD=self.typeD,T=True)
		#Quadratic form
		H1 = Kmat
		q1 = np.zeros(Nlearn)
		#Equality constraint
		Aeq1 = np.ones(Nlearn)
		beq1 = np.array([self.nu*Nlearn])
		#Lower and upper bound
		lb1 = np.zeros(Nlearn)
		ub1 = np.ones(Nlearn)

		#Solve the quad. prog.
		alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)

		#Find the support vectors and only use them
		indsv = np.where(alpha > self.tolsvm)[0].astype(int)
                self.indsv = indsv
		
		#Keep only the support vector alpha
		alphaTemp = alpha.transpose()[0]
		self.alpha = alphaTemp[self.indsv]
		#Set some alpha to 1 (those that are at least 0.999)
		self.alpha[self.alpha >= 0.999]=1.
		KmatTemp = Kmat[self.indsv,:]
		Kmat = KmatTemp[:,self.indsv]
		
		#Find the constant term in the decision function
		#Start by finding the indices for alpha_i not on the bounds
		ind_no_upperBound = np.where(self.alpha != 1.)[0].astype(int)
		self.rho = np.mean(self.alpha.dot(Kmat[:,ind_no_upperBound]))

	def train_withKer(self,X,Ker,typeK,var,nu=0.1,tolsvm=1e-4):
	#Solving the dual with quad. prog.
        #The constant term is included in w at the end as the last value

		#Tolerance for the choice of support vectors
		self.tolsvm=tolsvm
		#Number of instances in the training set and different parameters needed to keep the model.				
		SizeX = X.shape
                pdim = SizeX[1]
                Nlearn = SizeX[0]
                self.Nlearn = Nlearn
		self.typeK = typeK
		#self.typeD = typeD
		self.var = var
                self.X = X.copy()
		self.nu = nu			
		#Building the matrices for the quadratic prog. problem
	
		#Quadratic form
		Kmat=Ker.copy()
		H1 = Kmat
		q1 = np.zeros(Nlearn)
		#Equality constraint
		Aeq1 = np.ones(Nlearn)
		beq1 = np.array([self.nu*Nlearn])
		#Lower and upper bound
		lb1 = np.zeros(Nlearn)
		ub1 = np.ones(Nlearn)

		#Solve the quad. prog.
		alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)

		#Find the support vectors and only use them
		indsv = np.where(alpha > self.tolsvm)[0].astype(int)
                self.indsv = indsv
		
		#Keep only the support vector alpha
		alphaTemp = alpha.transpose()[0]
		self.alpha = alphaTemp[self.indsv]
		#Set some alpha to 1 (those that are at least 0.999)
		self.alpha[self.alpha >= 0.999]=1.
		KmatTemp = Kmat[self.indsv,:]
		Kmat = KmatTemp[:,self.indsv]
		
		#Find the constant term in the decision function
		#Start by finding the indices for alpha_i not on the bounds
		ind_no_upperBound = np.where(self.alpha != 1.)[0].astype(int)
		self.rho = np.mean(self.alpha.dot(Kmat[:,ind_no_upperBound]))


	def query(self,Xt):
		#Prediction for matrix Xt
		#Find the Kernel only for the points that are support vectors
		Ktest = KernelCalc(self.X[self.indsv],Xt,len(self.indsv),Xt.shape[0],var=self.var,typeK=self.typeK,typeD=self.typeD,T=False)

		return np.sign( Ktest.dot(self.alpha)-self.rho)
	def query_withKer(self,Xt,Ktest):
                #Prediction for matrix Xt
                #Find the Kernel only for the points that are support vectors
		Ktest1=Ktest[:,self.indsv].copy()
                return np.sign( Ktest1.dot(self.alpha)-self.rho)
	def score(self,X,ytrue):
		#Return the % of correct predictions
		ypredic = self.query(X)
                return 1-0.5*np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
##########################################################################################################################


class SVM_nu:
##########################################################################################################################
#        				     Supervided nu-svm for anomaly	                                         #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#															 #
#	INPUTS:														 #
#															 #

       	def __init__(self):
                pass
       	def train(self,X,y,typeK,var,typeD=None,nu=0.1,tolsvm=1e-4):
	#Solving the dual with quad. prog.
        #The constant term is included in w at the end as the last value
		
		#Tolerance for the choice of support vectors
		self.tolsvm=tolsvm
		#Number of instances in the training set as well as the dimension of final w				
		SizeX = X.shape
                pdim = SizeX[1]
                Nlearn = SizeX[0]
                self.Nlearn = Nlearn
		self.typeK = typeK
		self.typeD = typeD
		self.var = var
                self.X = X
		self.y = y
		self.nu = nu				
		#Building the matrices for the quadratic prog. problem
		Kmat = KernelCalc(X,X,Nlearn,Nlearn,var=self.var,typeK=self.typeK,typeD=self.typeD,T=True)
		#Quadratic form
		H1 = np.diag(y).dot( Kmat.dot( np.diag(y) ) )
		q1 = np.zeros(Nlearn)
		#Equality constraint
		Aeq1 = np.ones((2,Nlearn))
		Aeq1[0,:] = y.reshape(Nlearn,)
		beq1 = np.array([[0.],[self.nu*self.Nlearn]])
		#Lower and upper bound
		lb1 = np.zeros(Nlearn)
		ub1 = np.ones(Nlearn)

		#Solve the quad. prog.
		alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)

		#Find the support vectors and only use them
		indsv = np.where(alpha > self.tolsvm)[0].astype(int)
                self.indsv = indsv

		#Keep only the support vector alpha
		alphaTemp = alpha.transpose()[0]
                self.alpha = alphaTemp[self.indsv]
		#Set some alpha to 1 and then rescale to 0<=alpha<=1/Nlearn
                self.alpha[self.alpha > 0.999]=1.
		self.alpha = self.alpha/self.Nlearn
                KmatTemp = Kmat[self.indsv,:]
                Kmat = KmatTemp[:,self.indsv]

		#Find the constant term in the decision function
                #Start by finding the indices for alpha_i not on the bounds
		ind_no_upperBound = np.where(self.alpha != 1./self.Nlearn)[0].astype(int)
                yKmat = np.diag(self.y[self.indsv]).dot(Kmat)
                ysv = self.y[indsv]
                self.b = np.mean(ysv[ind_no_upperBound] - self.alpha.dot(yKmat[:,ind_no_upperBound]))

	def query(self,Xt):
		#Prediction for matrix Xt
		Ktest = KernelCalc(self.X,Xt,self.Nlearn,Xt.shape[0],var=self.var,typeK=self.typeK,typeD=self.typeD,T=False)

		return np.sign( np.dot(np.diag(self.y[self.indsv]).dot(Ktest),dot(self.alpha))+b)

	def score(self,X,ytrue):
		#Return the % of correct predictions
		ypredic = self.query(X)
               	return 1-0.5*np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
