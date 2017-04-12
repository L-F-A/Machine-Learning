import numpy as np
import warnings
from scipy.sparse import lil_matrix
from scipy.sparse.linalg.dsolve import linsolve


class LSMF_Explicit:

	def __init__(self,A,k,sig2=1.,lam=1.,tol=1e-3,MaxIter=1000):

		A=A.astype(float)
		A[np.where(np.isnan(A)==True)]=0.
		A[np.where(A<0)]=0.

		self.A=lil_matrix(A)

		self.row=A.shape[0]
                self.col=A.shape[1]
		self.k=k
		self.tol=tol
		self.MaxIter=MaxIter
		self.lam=lam
		self.sig2=sig2
		self.idrc=self.A.nonzero()

	def __LeastSquare_iteration(self,X,Y):

		users,latent=X.shape

		for u in range(users):
			if users==self.A.shape[0]:
				index=self.A.getrowview(u).nonzero()[1]
				Mij=self.A[u,index]
			elif users==self.A.shape[1]:
				index=self.A.T.getrowview(u).nonzero()[1]
                                Mij=self.A.T[u,index]
			AA=np.dot(Y[index,:].T,Y[index,:])
			BB=Mij*Y[index,:]
			X[u,:]=np.linalg.solve(AA+self.lam*self.sig2*np.eye(latent),BB[0])

	def Ojective_Function(self,X,Y):

		err=np.sum((self.A[self.idrc].toarray()-np.dot(X,Y.T)[self.idrc])**2)
		L=-0.5/self.sig2*err-0.5*self.lam*(np.sum(X**2)+np.sum(Y**2))
		return L,err

	def factor(self,U=None,V=None,FixedIte=False):

		if U is None:
			U=np.sqrt(1./self.lam)*np.random.randn(self.row,self.k)
                if V is None:
			V=np.sqrt(1./self.lam)*np.random.randn(self.col,self.k)

		if FixedIte==False:

			ite=1
                	err=1.
                	while err > self.tol:
			
				self.__LeastSquare_iteration(U,V)
				self.__LeastSquare_iteration(V,U)

				L,err=self.Ojective_Function(U,V)

                        	if ite==self.MaxIter:
 					warnings.warn('The function stopped because the number of iterations exceeded the maximum')                                        
					break
 				else:
 					ite+=1
			return U,V,L
		else:
			L=np.zeros(self.MaxIter)

			for r in range(self.MaxIter):

				self.__LeastSquare_iteration(U,V)
                                self.__LeastSquare_iteration(V,U)
				
				L[r],err=self.Ojective_Function(U,V)

			return U,V,L
