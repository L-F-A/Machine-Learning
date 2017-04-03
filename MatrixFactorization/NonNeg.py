import numpy as np
import warnings

class NMF:

	def __init__(self,A,k,type='Fro',tol=1e-4,MaxIter=500,alpha=None,rho=None):
		self.A=A.copy()
		self.type=type
		self.row=A.shape[0]
		self.col=A.shape[1]
		self.k=k
		self.tol=tol
		self.MaxIter=MaxIter

		#Need to add regularization to the factorization
		self.alpha=alpha
		self.rho=rho
		if self.alpha is not None:
			self.reg=True
		else:
			self.reg=False

	def factor(self,W=None,H=None):
		
		if W is None:
			W=np.random.randn(self.row,self.k)
			W=W*np.sqrt(self.A.mean()/self.k)
			np.abs(W,W)
		if H is None:
			H=np.random.randn(self.k,self.col)
			H=H*np.sqrt(self.A.mean()/self.k)
			np.abs(H,H)
		if self.type=='Fro':
			ite=1
			err=1.
			while err > self.tol:
			
				H = H*np.dot(W.T,self.A)/np.dot(np.dot(W.T,W),H)
				W=W*np.dot(self.A,H.T)/np.dot(np.dot(W,H),H.T)

				err=np.mean((self.A-np.dot(W,H))**2)
				print err
				if ite==self.MaxIter:
					warnings.warn('The function stopped because the number of iterations exceeded the maximum number allowed.')
					break
				else:
					ite+=1
			self.W=W
			self.H=H
			self.ite=ite
			self.err=err
		elif self.type=='Div':
			ite=1
			err=1.
                        while err > self.tol:
			
				H=H*np.dot(W.T/np.sum(W.T,axis=1).reshape(self.k,1),self.A/np.dot(W,H))
				W=W*np.dot(self.A/np.dot(W,H),H.T/np.sum(H.T,axis=0))

				WH=np.dot(W,H)
				err=-np.sum(self.A*np.log(WH)-WH)
				if ite==self.MaxIter:
                                        warnings.warn('The function stopped because the number of iterations exceeded the maximum number allowed.')
                                        break
                                else:
                                        ite+=1
			self.W=W
                        self.H=H
                        self.ite=ite
			self.err=err
		else:
			warnings.warn('The algorithms are either Fro or Div.')
