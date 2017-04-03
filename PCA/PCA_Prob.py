import numpy as np
import scipy as sp
import warnings

class PPCA:

	def __init__(self,axis_features=1,centered=True,std_scaled=False,sig2=None,W=None,iteMax=100,tol=1e-4):

		self.axis_features=axis_features#Along which axis are the features (n_features,n_samples) = 0 or (n_samples,n_features) = 1
                self.centered=centered #Center the data in X?
                self.std_scaled=std_scaled #Normalize the data in X by standard deviation
                self.sig2=sig2 #The prior variance
		self.W=W
		self.iteMax=iteMax
		self.tol=tol

		#Tells what variable are contained in a class object PPCA
		self.contained=['W','mui','d_converged','axis_features','centered','std_scaled','sig2','iteMax','tol','avg','std']

	def Rotation(self,X,K):
	
		#Using EM algo
		#need to add a calculation if sig2 from data
		#And also deal with missing value which is mostly the point of prob pca

		#Work on matrix of data to have the right format
		if self.axis_features== 1:
			X=X.T

		if self.centered is False and self.std_scaled is True:
			warnings.warn('Normalizing by std without centering is a weird move, will force centering')
			self.centered=True

		if self.centered is True:
			avg=np.mean(X,axis=1)
			self.avg=avg
			if self.std_scaled is True:
				std=np.std(X,axis=1)
				self.std=std
				X=(X-avg.reshape(X.shape[0],1))/std.reshape(X.shape[0],1)
			else:
				X=X-avg.reshape(X.shape[0],1)

		#Startin matrix W
		if self.W is None:
			self.W=np.random.randn((X.shape[0],K))

		#Starting likelihood calculation
		L=np.linalg.cholesky(self.sig2*np.identity(X.shape[0])+np.dot(self.W,self.W.T))
                etai=np.linalg.solve(L,X)
		d0=-0.5*np.trace(np.dot(etai.T,etai)) - X.shape[1]*np.trace(np.log(L)) - 0.5*X.shape[1]*X.shape[0]*np.log(2*np.pi)

		ite=1
		err=1.
		
		I_K=np.identity(K)
		I_d=np.identity(X.shape[0])

		while err > tol:
			
			#E-step
			Sig=np.linalg.solve(I_K+np.dot(self.W.T,self.W)/self.sig2,I_K)
			self.mui=np.dot(Sig,np.dot(self.W.T,X)/self.sig2)

			#M-step
			B=self.sig2*I_K + np.dot(self.mui,self.mui.T) + X.shape[1]*Sig
			WT=np.solve(B.T,np.dot(X,self.mui.T).T)
			W=WT.T

			#Likelihood calculation and change with respect of previous iteration
			L = np.linalg.cholesky(self.sig2*I_d+np.dot(self.W,self.W.T))
			etai=np.linalg.solve(L,XX)
			d=-0.5*np.trace(np.dot(etai.T,etai)) - X.shape[1]*np.trace(np.log(L)) - 0.5*X.shape[1]*X.shape[0]*np.log(2*np.pi)
			err=abs(d-d0)
			d0=d

			if err > tol and ite==ite_max:
				warnings.warn('Maximum number of iterations reached and no convergenge')
				break
			ite+=1

		self.d_converged=d
