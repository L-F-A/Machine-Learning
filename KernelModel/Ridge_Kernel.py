import numpy as np
import scipy as sp
from MachineLearning.KernelModel import KernelCalc,KernelCalcDer,KernelCalcDer_withD
from MachineLearning.Distances import DistancesMatrix
from scipy.optimize import minimize
import time

class KerRidgeReg:
##########################################################################################################################
#                                         Kernel Ridge Regression/Gaussian process                                       #
#                                                                                                                        #
#                      Louis-Francois Arsenault, Columbia Universisty (2013-2017), la2518@columbia.edu		 	 #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#  															 #
##########################################################################################################################

        def __init__(self):
		pass

	def train(self,X,y,lam,var,typeK,typeD,xinterval=None):

		self.Nlearn=X.shape[0]
		self.Nfeat=X.shape[1]
		self.X=X.copy()
                self.y=y.copy()
		self.lam=lam
		self.var=var.copy()
		self.typeK=typeK
		self.typeD=typeD
		self.xinterval=xinterval
		
		self.Ker = KernelCalc(X,X,self.Nlearn,self.Nlearn,self.var,typeK=self.typeK,typeD=self.typeD,T=True,xinterval=self.xinterval)
		Klam=self.Ker+lam*np.identity(self.Nlearn)

		try:
			self.L=np.linalg.cholesky(Klam)
			self.alpha=sp.linalg.cho_solve((self.L,True),y)

		except np.linalg.linalg.LinAlgError:
                      
		  	print 'K+lambda*I not positive definite, solving anyway, but beware!!'
                        self.alpha = np.linalg.solve(Klam,y)
			self.L=None

	def train_withKer(self,X,y,lam,var,Ker,typeK,typeD):
	#If the Kernel matrix is already provided

                self.Nlearn=X.shape[0]
                self.Nfeat=X.shape[1]
                self.X=X.copy()
		self.y=y.copy()
                self.lam=lam
                self.var=var.copy()
                self.typek=typeK
                self.typeD=typeD
                self.Ker=Ker
                Klam=self.Ker + lam*np.identity(self.Nlearn)

		try:
                	self.L = np.linalg.cholesky(Klam)
			self.alpha=sp.linalg.cho_solve((self.L,True),y)

		except np.linalg.linalg.LinAlgError:

			print 'K+lambda*I not positive definite, solving anyway, but beware!!'
			self.alpha = np.linalg.solve(Klam,y)
			self.L=None

	def query(self,Xt,postVar=False):

		if Xt.ndim==1:
			Xt=Xt.reshape(1,len(Xt))

		KerTest = KernelCalc(self.X,Xt,self.Nlearn,Xt.shape[0],self.var,typeK=self.typeK,typeD=self.typeD,T=False,xinterval=self.xinterval)

		if postVar is False:
			return KerTest.dot(self.alpha)

		elif postVar is True: #Return the Gaussian process posterior variance change k_test^T(K+lambda*I)^-1k_test
			
			if self.L is None:
				Klam=self.Ker+lam*np.identity(self.Nlearn)
				v=np.linalg.solve(Klam,KerTest.transpose())
				return KerTest.dot(self.alpha), np.einsum('ij,ij->j',KerTest,v)
			else:
				v=np.linalg.solve_triangular(self.L,KerTest.transpose())
				return KerTest.dot(self.alpha), np.einsum('ij,ij->j',v,v)
	
	def query_withKer(self,KerTest,postVar=False):

                if postVar is False:
                        return KerTest.dot(self.alpha)

                elif postVar is True: #return the Gaussian process posterior variance change k_test^T(K+lambda*I)^-1k_test

			if self.L is None:
				Klam=self.Ker+lam*np.identity(self.Nlearn)
                                v=np.linalg.solve(Klam,KerTest.transpose())
				return KerTest.dot(self.alpha), np.einsum('ij,ij->j',KerTest,v)
			else:
                        	v=np.linalg.solve(self.L,KerTest.transpose())
	                        return KerTest.dot(self.alpha), np.einsum('ij,ij->j',v,v)

	def score(self,ypredic,ytrue,metric='MAE'):

		#Calculate the mean absolute error MAE by default, mean square error MSE
		#ypredic = self.query(Xt)

		if metric == 'MAE':
			return np.mean(np.absolute(ypredic-ytrue),axis=0)
		elif metric == 'MSE':
                        return np.mean((ypredic-ytrue)**2,axis=0)
		elif metric == 'MDAE':
                        return np.median(np.absolute(ypredic-ytrue),axis=0)

	#log marginal likelihood.
	def LogLike_neg(self,mult=False,Nout=1):

		#negative marginal log-likelihood
		if mult==True:
			if L is None:
                                return (0.5*np.einsum("ik,ik->k",self.y,self.alpha)+0.5*np.log(np.linalg.det(self.Ker + lam*np.identity(self.Nlearn)))+0.5*self.Nlearn*np.log(2*np.pi)).sum()
                        else:
                                return (0.5*np.einsum("ik,ik->k",self.y,self.alpha)+np.sum(np.log(np.diag(self.L)))+0.5*self.Nlearn*np.log(2*np.pi)).sum()
			
		else:
			if L is None:
                                return 0.5*self.y.transpose().dot(self.alpha) + 0.5*np.log(np.linalg.det(self.Ker + lam*np.identity(self.Nlearn))) + 0.5*self.Nlearn*np.log(2*np.pi)
                        else:
                                return 0.5*self.y.transpose().dot(self.alpha) + np.sum(np.log(np.diag(self.L))) + 0.5*self.Nlearn*np.log(2*np.pi)
	

	#################################################################################################################
	#		We also add to this class the possibility of consider it as a Gaussian process			#
	#														#
	#	Series of functions to obtain values for hyperparameters by minimizing the log- marginal likelihood	#
	#################################################################################################################


	def fit_LogLike(self,X,y,vars0,typeK,typeD,tol=1e-7,solver='SLSQP',restart=None,Mess=True,bmin=1e-12,bmax=1e3,xinterval=None):

		tK=['Gau','Exp','Matern52']
		tD=['Euc','Euc_Cont','Man']
		if solver=='L-BFGS-B' and ( (typeK not in tK) or (typeD not in tD) ):

			print 'Solver with gradient not implemented for the choosen pair of Kernel and distance\n'
			print 'Will use SLSQP solver instead\n'
			solver='SLSQP'

		if (restart is not None) and ( (typeK not in tK) or (typeD not in tD) ):

			print 'For the moment, multiple restarts is not available for the choosen pair of Kernel and distance metric\n'
			print 'Doing the calculation only for vars0'
			restart=None
	
		#Is it a multi-outputs problem?
		Nlearn=X.shape[0]
		if y.ndim != 1:
			mult=True
			Nout=y.shape[1]
		else:
			mult=False
			Nout=1

		def __constr1(x,bmin):
			#Every hyperparameters is postive and at least bmin
			return x-bmin
		def __constr2(x,bmax):
			#Every hyperparameters is postive and at most bmax
			return bmax-x

		if restart is None:

			if solver=='SLSQP':
				#Variables that need to be passed to the solver
                		tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,xinterval)
				b=len(vars0)*[(bmin,bmax)]
				return minimize(self.__func_LL,vars0,args=tupARG,method=solver,bounds=b,tol=tol,options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
		
			elif solver=='COBYLA':
				#Variables that need to be passed to the solver
                		tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,xinterval)
				return minimize(self.__func_LL,vars0,args=tupARG,method='COBYLA',constraints=({'type':'ineq','fun':__constr1,'args':([bmin])},{'type':'ineq','fun':__constr2,'args':([bmax])}),tol=tol,options={'iprint': 1, 'disp': False, 'maxiter': 1000, 'catol': 1e-7, 'rhobeg': 1.})

			elif solver=='L-BFGS-B':
				if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                        		sqr=False
                        		sqEuc=True
				else:
					sqr=True
					sqEuc=False
				D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
				#Variables that need to be passed to the solver
				tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)		
				b=len(vars0)*[(bmin,bmax)]
				return minimize(self.__func_LL_dLL,vars0,args=tupARG,method='L-BFGS-B',jac=True,bounds=b,tol=tol,options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

			else:
				print 'This solver is not implemented'
		else:
			LL=1e6
			for rrestart in xrange(restart+1):
				
				if rrestart!=0:
					#We assume that the length scale sigma of the Kernel shall not be too small and that the regularization term shall not be
					#greater than 1
					vars0=np.array([10.**np.random.uniform(-2,np.log10(bmax)),10.**np.random.uniform(np.log10(bmin),0.)])
				
				if Mess==True:
                                        tstart=time.time()
					print 'vars0 = '+str(vars0)

				if solver=='SLSQP':
					#Variables that need to be passed to the solver
                                	tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,xinterval)
					b=len(vars0)*[(bmin,bmax)]
					res=minimize(self.__func_LL,vars0,args=tupARG,method=solver,bounds=b,tol=tol,options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
		
				elif solver=='COBYLA':
					#Variables that need to be passed to the solver
                                	tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,xinterval)
					res=minimize(self.__func_LL,vars0,args=tupARG,method='COBYLA',constraints=({'type':'ineq','fun':__constr1,'args':([bmin])},{'type':'ineq','fun':__constr2,'args':([bmax])}),tol=tol,options={'iprint': 1, 'disp': False, 'maxiter': 1000, 'catol': 1e-7, 'rhobeg': 1.})

				elif solver=='L-BFGS-B':
					if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                                        	sqr=False
                                        	sqEuc=True
					else:
                                        	sqr=True
                                        	sqEuc=False
                                	D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
                                	#Variables that need to be passed to the solver
                                	tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)
					b=len(vars0)*[(bmin,bmax)]
					res=minimize(self.__func_LL_dLL,vars0,args=tupARG,method='L-BFGS-B',jac=True,bounds=b,tol=tol,options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

				else:
					print 'This solver is not implemented'

				if res['fun'] < LL:
					sol=res.copy()
					LL=res['fun']

				if Mess==True:
					tstop=time.time()
					print 'Restart number '+str(rrestart+1)+' completed in '+str(tstop-tstart)+' secs'
			return sol

	def __func_LL(self,vars,X,y,Nlearn,typeK,typeD,mult,Nout,xinterval):
		
		#Althought in the minimizer we have positivity constraints for all hyperparameters, if along the way 
		#it tries a negative value, the covariance will not be positive definite. So we just avoid the negative
		#possibility by taking positive every time
		lam=np.abs(vars[-1])
		var=np.abs(vars[:-1])

		Ker=KernelCalc(X,X,Nlearn,Nlearn,var,typeK=typeK,typeD=typeD,T=True,xinterval=xinterval)
		Klam=Ker+lam*np.identity(Nlearn)

                try:
                        L=np.linalg.cholesky(Klam)
                        alpha=sp.linalg.cho_solve((L,True),y)

                except np.linalg.linalg.LinAlgError:

			print 'K+lambda*I not positive definite, solving anyway, but beware!!'
			alpha=np.linalg.solve(Klam,y)
                        L=None
			
		if mult==True:

			if L is None:
				return (0.5*np.einsum("ik,ik->k",y,alpha)+0.5*np.log(np.linalg.det(Klam))+0.5*Nlearn*np.log(2*np.pi)).sum()
			else:
				return (0.5*np.einsum("ik,ik->k",y,alpha)+np.sum(np.log(np.diag(L)))+0.5*Nlearn*np.log(2*np.pi)).sum()
                else:

			if L is None:
				return 0.5*y.transpose().dot(alpha) + 0.5*np.log(np.linalg.det(Klam)) + 0.5*Nlearn*np.log(2*np.pi)
			else:
                        	return 0.5*y.transpose().dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5*Nlearn*np.log(2*np.pi)

	def __func_LL_dLL(self,vars,D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc):#,xinterval):
	
	#Return derivatives as well! Only implemented for Kernels with unique hyperparameters sig: 'Gau','Exp','Matern52' with distances 'Euc'
	#'Euc_cont' and 'Man'
			
                #Althought in the minimizer we have positivity constraints for all hyperparameters, if along the way
                #it tries a negative value, the covariance will not be positive definite. So we just avoid the negative
                #possibility by taking positive every time
                lam=np.abs(vars[-1])
                var=np.abs(vars[:-1])

		Ker,dKer=KernelCalcDer_withD(D,var=var,typeK=typeK,sqEuc=sqEuc)
                Klam=Ker+lam*np.identity(Nlearn)

                try:
                        L=np.linalg.cholesky(Klam)
                        alpha=sp.linalg.cho_solve((L,True),y)
			Km1=sp.linalg.cho_solve((L,True),np.identity(Nlearn)) #Compute the inverse of K, hard to avoid when dealing with derivatives

                except np.linalg.linalg.LinAlgError:

                        print 'K+lambda*I not positive definite, solving anyway, but beware!!'
			alpha=np.linalg.solve(Klam,y)
                        L=None
			Km1=sp.linalg.solve(Klam,np.identity(Nlearn))
			
		dLL=np.zeros(len(vars))

                if mult==True:

			dLL[0]=-0.5*( np.einsum("ik,kj,ji",alpha,alpha.T,dKer)-Nout*np.einsum("ij,ji",Km1,dKer) )
			dLL[1]=-0.5*( np.einsum("ij,ji",alpha,alpha.T)-Nout*np.trace(Km1)  )				

                        if L is None:
                                return (0.5*np.einsum("ik,ik->k",y,alpha)+0.5*np.log(np.linalg.det(Klam))+0.5*Nlearn*np.log(2*np.pi)).sum(), dLL
                        else:
				
                                return (0.5*np.einsum("ik,ik->k",y,alpha)+np.sum(np.log(np.diag(L)))+0.5*Nlearn*np.log(2*np.pi)).sum(), dLL
                                
                else:

			dLL[0]=-0.5*( np.einsum("i,j,ji",alpha,alpha,dKer)-np.einsum("ij,ji",Km1,dKer) )
                        dLL[1]=-0.5*( np.einsum("i,i",alpha,alpha)-np.trace(Km1) )

                        if L is None:
                                return 0.5*y.transpose().dot(alpha) + 0.5*np.log(np.linalg.det(Klam)) + 0.5*Nlearn*np.log(2*np.pi), dLL
                        else:
                                return 0.5*y.transpose().dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5*Nlearn*np.log(2*np.pi), dLL



