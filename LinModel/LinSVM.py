#The svm class uses the package CVXOPT, Python Software for Convex Optimization, that is not part of the standard python, 
#but is included in Anaconda for example. 
import numpy as np
import warnings
#Everything else is imported locally when needed

class SVMClass:
##########################################################################################################################
#        				     Linear only support vector machine                                          #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#															 #
#	INPUTS:														 #
#															 #
#	  typeL : 'norm2' (default) is the standard svm using ||w||_2^2 while 'norm1' is a 1-norm svm using ||w||_1 	 #
#		   rather with objective function as: ttp://papers.nips.cc/paper/2450-1-norm-support-vector-machines.pdf #
#															 #
#	  X 	:  A ndarray array([x_11,x_12,...,x_1p],[x_21,x_22,...,x_2p],...,[x_Nlearn1,x_Nlearn2,...,x_Nlearnp])    #
#		   representing a matrix with Nlearn lines with all known examples of the training set and the columns 	 #
#		   are the dimensions p of one x.									 #
#															 #
#   	  y 	:  A 1d array numpy.array([y_1,y_2,...,y_Nlearn]) representing a vector of size Nlearn with values for   #
#		   each examples of the training set in the form -1 and 1.						 #
#															 #
#	  C	:  Value of the slack variable to be used. The default value is set to 0, meaning a separable problem	 #
#															 #
#	  tolsvm : In the 2-norm case with non-zero C, what tolerance to define below which we have a support vector. 	 #
#		   The final tolerance is tolsvm = tolsvm*C.								 #
#															 #
#	OUTPUTS:													 #
#															 #
#	  self.Nlearn : Number of instances in training									 #
#															 #
#	  self.pdim : How many dimensions for the linear model given by vector w; equal to p+1 (constant term included)	 #
#															 #
#	  self.w : This is the linear model. A vector with p+1 elements. A prediction is sign( w[0:-1]^T*Xt + w[-1] ).	 #
#															 #
#	  In the 2-norm case with non-zero C:										 #
#															 #
#	  self.alpha : The alpha vector for the dual model								 #
#															 #
#	  self.indsv : The indices of the support vectors 								 #
##########################################################################################################################

        def __init__(self,typeL='norm2'):
                self.typeL = typeL
        def train(self,X,y,C=0,tolsvm=1e-5):
		if self.typeL is 'norm2': #2-norm svm
		#The 2-norm svm is the standard approach where the min. approach has the ||w||_2^2 term
			from NumMethods import quadprog 
			if C==0:
				#Solving the linearly separable problem with quad. prog.
				#We treat the constant term as part of the coeffs. w and thus
                        	#need to add 1 at the end of every input
                        	OneVec = np.ones((X.shape[0],1))
                        	XX = np.concatenate((X, OneVec), axis=1)

				#Number of instances in the training set as well as the dimension of final w
                        	SizeXX = XX.shape
                        	pdim = SizeXX[1]
                        	Nlearn = SizeXX[0]
                        	self.Nlearn = Nlearn
                        	self.pdim = pdim
                        	
				#Building the matrices for the quadratic prog. problem
				H1 = np.identity(pdim)
                        	H1[pdim-1,pdim-1] = 0.
                        	bineq1 = -np.ones((Nlearn,1))
                        	Aineq1 = -np.diag(y).dot(XX)
				q1 = np.zeros((1,pdim))[0]

				#Solve the quad. prog.
				self.w = quadprog(H1,q1,Aineq=Aineq1,bineq=bineq1)
				self.w.shape = (self.pdim,)
			else:
				#Solving the dual with quad. prog.
                                #The constant term is included in w at the end as the last value

				#Tolerance for the choice of support vectors
				self.tolsvm=tolsvm*C
				#Number of instances in the training set as well as the dimension of final w				
				SizeX = X.shape
                                pdim = SizeX[1]
                                Nlearn = SizeX[0]
                                self.Nlearn = Nlearn
                                self.pdim = pdim+1
				
				#Building the matrices for the quadratic prog. problem
				Hinter = np.dot(X,X.transpose())
				H1 = np.diag(y).dot( Hinter.dot( np.diag(y) ) )
				q1 = -np.ones(Nlearn)
				Aeq1 = y.copy()
				beq1 = np.array([0.])
				lb1 = np.zeros(Nlearn)
				ub1 = C*np.ones(Nlearn)

				#Solve the quad. prog.
				alpha=quadprog(H1,q1,Aeq=Aeq1,beq=beq1,lb=lb1,ub=ub1)
				self.alpha = alpha.transpose()[0]

				#Find the support vectors and only use them
				indsv = np.where((alpha > self.tolsvm) & (alpha < (C-self.tolsvm)))[0].astype(int)
                                self.indsv = indsv
				
				#The linear model is completely specified by the vector w
				self.w = np.zeros(pdim+1)
				
				#Change this step by using np.einsum !!!!!!!!!!!!!!!!!
				self.w[0:-1] = self.alpha[indsv].dot(np.diag(y[indsv]).dot(X[indsv]))
				
				#The constant term obtained by the value for each support vector and then average them
				bias = np.mean(y[indsv]- X[indsv,:].dot(self.w[0:-1]))
				self.w[-1] = bias

		elif self.typeL is 'norm1': #1-norm svm
				#The 1-norm svm uses ||w||_1 rather than ||w||_2^2
				#For details, see http://papers.nips.cc/paper/2450-1-norm-support-vector-machines.pdf 
				from cvxopt.modeling import variable as cvxvar, op, sum as cvxsum, max as cvxmax
				from cvxopt.solvers import options as cvxopt
                                from cvxopt import matrix as cvxmat
				cvxopt['show_progress'] = False

				#Number of instances in the training set as well as the dimension of final w
				self.Nlearn = X.shape[0]
				self.pdim = X.shape[1]+1

				#Necessary matrices for the unconstrained problem
				AL1 = np.diag(y).dot(X)
				BL1 = cvxmat(C*y)
                                ALL1 = cvxmat(C*AL1)

				#The variables to be found
                                WW = cvxvar(ALL1.size[1],'WW')
				bb = cvxvar(1,'bb')

				#Calling the solver
                                op( cvxsum(abs(WW)) + cvxsum(cvxmax(0,1-(ALL1*WW + BL1*bb)))).solve()
				self.w = np.zeros(self.pdim)
				self.w[0:-1] = np.array(WW.value).reshape((self.pdim-1,))
				self.w[-1] = np.array(bb.value).reshape((1,))

	def query(self,Xt):
		#Prediction for matrix Xt
		OneVec = np.ones((Xt.shape[0],1))
                XXt = np.concatenate((Xt, OneVec), axis=1)
		return np.sign(XXt.dot(self.w))

	def score(self,X,ytrue):
	#Return the % of correct predictions
		ypredic = self.query(X)
                return 1-0.5*np.sum(np.absolute(ypredic-ytrue))/len(ytrue)
##########################################################################################################################
