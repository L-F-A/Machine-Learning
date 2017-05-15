import numpy as np
import scipy as sp
from MachineLearning.KernelModel import KernelCalc,KernelCalcDer,KernelCalc_withD,KernelCalcDer_withD
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


	def fit_LogLike(self,X,y,vars0,typeK='Gau',typeD='Euc',tol=1e-7,solver='SLSQP',restart=None,Mess=True,b_sig=(1e-4,1e3),b_lam=(1e-12,1.),xinterval=None):

		Nlearn,Np=X.shape

		#Is it a multi-outputs problem?
		if y.ndim != 1:
			mult=True
			Nout=y.shape[1]
		else:
			mult=False
			Nout=1

		if restart is None:

			if solver=='SLSQP':
				#Variables that need to be passed to the solver
				if typeK=='Gau_Diag':
					tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,False)
				else:
					if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                                                sqr=False
                                                sqEuc=True
                                        else:
                                                sqr=True
                                                sqEuc=False

                                        D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
                                        tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)
					
				b=len(vars0)*[tuple(np.log(b_sig))]
				b[-1]=tuple(np.log(b_lam))
				return minimize(self.__func_LL,np.log(vars0),args=tupARG,method='SLSQP',bounds=b,tol=tol,options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})

			elif solver=='L-BFGS-B':
				#Variables that need to be passed to the solver
				if typeK=='Gau_Diag':
					tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,False)
				else:
					if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                        			sqr=False
                        			sqEuc=True
					else:
						sqr=True
						sqEuc=False

					D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
					tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)
		
				b=len(vars0)*[tuple(np.log(b_sig))]
				b[-1]=tuple(np.log(b_lam))
				return minimize(self.__func_LL_dLL,np.log(vars0),args=tupARG,method='L-BFGS-B',jac=True,bounds=b,tol=tol,options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

			else:
				print 'This solver is not implemented'
		else:
			LL=1e6
			for rrestart in xrange(restart+1):
				
				if rrestart!=0:
					if typeK=='Gau_Diag':
						vars0=np.concatenate((10.**np.random.uniform(np.log10(b_sig[0]),np.log10(b_sig[1]))*np.ones(Np),[10.**np.random.uniform(np.log10(b_lam[0]),np.log10(b_lam[1]))]))
					else:
						vars0=np.array([10.**np.random.uniform(np.log10(b_sig[0]),np.log10(b_sig[1])),10.**np.random.uniform(np.log10(b_lam[0]),np.log10(b_lam[1]))])
				if Mess==True:
                                        tstart=time.time()
					print 'vars0 = '+str(vars0)

				if solver=='SLSQP':
					#Variables that need to be passed to the solver
					if typeK=='Gau_Diag':
                                        	tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,False)
                                	else:
                                        	if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                                                	sqr=False
                                                	sqEuc=True
                                        	else:
                                                	sqr=True
                                                	sqEuc=False

                                        	D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
                                        	tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)
					
					b=len(vars0)*[tuple(np.log(b_sig))]
                                        b[-1]=tuple(np.log(b_lam))
                                        res=minimize(self.__func_LL,np.log(vars0),args=tupARG,method=solver,bounds=b,tol=tol,options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})

				elif solver=='L-BFGS-B':
					#Variables that need to be passed to the solver
					if typeK=='Gau_Diag':
                                        	tupARG=(X,y,Nlearn,typeK,typeD,mult,Nout,False)
                                	else:
						if (typeD=='Euc' or typeD=='Euc_Cont') and typeK=='Gau':
                                        		sqr=False
                                        		sqEuc=True
						else:
                                        		sqr=True
                                        		sqEuc=False
                                		D=DistancesMatrix(X,X,Nlearn,Nlearn,typeD=typeD,T=True,sqr=sqr,xinterval=xinterval)
                                		tupARG=(D,y,Nlearn,typeK,typeD,mult,Nout,sqEuc)

					b=len(vars0)*[tuple(np.log(b_sig))]
                                	b[-1]=tuple(np.log(b_lam))
					res=minimize(self.__func_LL_dLL,np.log(vars0),args=tupARG,method='L-BFGS-B',jac=True,bounds=b,tol=tol,options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
				else:
					print 'This solver is not implemented'

				if res['fun'] < LL:
					sol=res.copy()
					LL=res['fun']

				if Mess==True:
					tstop=time.time()
					print 'Restart number '+str(rrestart+1)+' completed in '+str(tstop-tstart)+' secs'

			return sol

	def __func_LL(self,logVars,X,y,Nlearn,typeK,typeD,mult,Nout,sqEuc):
		
		lam=np.exp(logVars[-1])
		var=np.exp(logVars[:-1])

		
		if typeK=='Gau_Diag':
			Ker=KernelCalc(X/var,X/var,Nlearn,Nlearn,np.array([1.]),typeK='Gau',typeD='Euc',T=True)
		else:
 			Ker=KernelCalc_withD(X,var=var,typeK=typeK,sqEuc=sqEuc)
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

	def __func_LL_dLL(self,logVars,X,y,Nlearn,typeK,typeD,mult,Nout,sqEuc):
	
	#Return derivatives as well! Implemented for Kernels with unique hyperparameters sig: 'Gau','Exp','Matern52' with distances 'Euc'
	#'Euc_cont' and 'Man' as well as 'Gau' with diagonal covariance. 
			
                lam=np.exp(logVars[-1])
                var=np.exp(logVars[:-1])
		

		if typeK=='Gau_Diag':
                	Ker,dKer=KernelCalcDer(X,X,Nlearn,Nlearn,var=var,typeK=typeK,T=True,dK_dlntheta=True)
		else:
			Ker,dKer=KernelCalcDer_withD(X,var=var,typeK=typeK,sqEuc=sqEuc,dK_dlntheta=True)
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

		dLL=np.zeros(len(logVars))

                if mult==True:

			if typeK=='Gau_Diag':
				dLL[:-1]=-0.5*( np.einsum("ik,kj,pji",alpha,alpha.T,dKer.T)-Nout*np.einsum("ij,pji",Km1,dKer.T) )
			else:
				dLL[0]=-0.5*( np.einsum("ik,kj,ji",alpha,alpha.T,dKer)-Nout*np.einsum("ij,ji",Km1,dKer) )
			dLL[-1]=-0.5*( np.einsum("ij,ji",alpha,alpha.T)-Nout*np.trace(Km1)  )*lam				

                        if L is None:
                                return (0.5*np.einsum("ik,ik->k",y,alpha)+0.5*np.log(np.linalg.det(Klam))+0.5*Nlearn*np.log(2*np.pi)).sum(), dLL
                        else:
                                return (0.5*np.einsum("ik,ik->k",y,alpha)+np.sum(np.log(np.diag(L)))+0.5*Nlearn*np.log(2*np.pi)).sum(), dLL
                else:
			if typeK=='Gau_Diag':
				dLL[:-1]=-0.5*( np.einsum("i,j,pji",alpha,alpha,dKer.T)-np.einsum("ij,pji",Km1,dKer.T) )
			else:
				dLL[0]=-0.5*( np.einsum("i,j,ji",alpha,alpha,dKer)-np.einsum("ij,ji",Km1,dKer) )
                        dLL[-1]=-0.5*( np.einsum("i,i",alpha,alpha)-np.trace(Km1) )*lam

                        if L is None:
                                return 0.5*y.transpose().dot(alpha) + 0.5*np.log(np.linalg.det(Klam)) + 0.5*Nlearn*np.log(2*np.pi), dLL
                        else:
                                return 0.5*y.transpose().dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5*Nlearn*np.log(2*np.pi), dLL



