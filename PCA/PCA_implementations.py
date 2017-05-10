import numpy as np
import scipy as sp
import warnings

from MachineLearning.KernelModel import KernelCalc

class PCA:

	def __init__(self,axis_features=1,type='Standard',centered=True,std_scaled=False,type_K=None,type_D=None,var=None):

		self.axis_features=axis_features #Along which axis are the features (n_features,n_samples) = 0 or (n_samples,n_features) = 1
		self.type=type #'Standard' or 'Kernel' PCA
		self.centered=centered #Center the data in X?
		self.std_scaled=std_scaled #Standarized the data in X by standard deviation
		self.type_K=type_K#Type of Kernel if Kernel PCA
		self.type_D=type_D#Distance metric if Kernel PCA
		self.var=var #parameters of the Kernel

	def Rotation(self,X):


		if self.type=='Standard':

			if self.axis_features== 1:
                	#The matrix X has the standard form (n_samples,n_features). Therfore,
                	#for PCA, need to shape where the columns are the samples
				X=X.T

			if self.centered is False and self.std_scaled is True:
				warnings.warn('Dividing by std without centering is a weird move, will force centering')
				self.centered=True

			if self.centered is True:
				self.avg=np.mean(X,axis=1)
		
				if self.std_scaled is True:
					self.std=np.std(X,axis=1)
					X=(X-self.avg.reshape(X.shape[0],1))/self.std.reshape(X.shape[0],1)
				else:
					X=X-self.avg.reshape(X.shape[0],1)
					self.std=None
			else:
				self.avg=None

			#Each column is a PCA basis vector
			self.basis,self.sv=np.linalg.svd(X,full_matrices=True)[0:2]
			self.score=X.T.dot(self.basis)
			self.exp_var=(self.sv**2)/X.shape[1]
			self.exp_var_percent=self.exp_var/self.exp_var.sum()
			#var_score=np.var(self.score,axis=0)
			#self.exp_var=np.cumsum(var_score)
			#self.exp_var_percent=self.exp_var/np.sum(var_score)	
			

		elif self.type=='Kernel':

			if self.axis_features== 0:
				X=X.T #We need the matrix X the have the standard (n_samples,n_features) shape

			self.X=X.copy()#Will be needed to calculate the Kerel in projection
			Ker=KernelCalc(X,X,X.shape[0],X.shape[0],var=self.var,typeK=self.type_K,typeD=self.type_D,T=True)
			Ker0=Ker.copy()

			#Centered the Kernel matrix
			onen = np.ones((X.shape[0],X.shape[0]))/X.shape[0]
			Ker=Ker-onen.dot(Ker)-Ker.dot(onen)+onen.dot(Ker).dot(onen)

			self.eigVals,self.basis=np.linalg.eigh(Ker)
			self.basis=self.basis[:,self.eigVals>0]
			self.eigenVals=self.eigVals[self.eigVals>0]
			#The eigenvalues are in ascending order, we want descending
			self.basis=self.basis[:,::-1]
			self.eigVals=self.eigenVals[::-1]
			self.Xtransf=self.basis*np.sqrt(self.eigVals)

			#For projection, put in memory necessary values for centering
			self.K_avg_rows = np.sum(Ker0,axis=0)/Ker.shape[0]
        		self.K_avg_all = self.K_avg_rows.sum()/Ker.shape[0]
		else:
			warnings.warn('Must choose between Standard or Kernel PCA')
	
	def proj(self,XX):
	
		if self.type=='Standard':

			## XX should have the same form as X
			if self.axis_features== 1:
				XX=XX.T

			if self.avg is not None:
				XX=XX-self.avg.reshape(XX.shape[0],1)
				if self.std is not None:
					XX=XX/self.std.reshape(XX.shape[0],1)

			#the return projected matrix XX as the same shape as the original. Remember that self.basis 
			#contains basis vectors as each column a basis vector
			if self.axis_features==1:
				return np.dot(XX.T,self.basis)
			else:
				return np.dot(self.basis.T,XX)

		elif self.type=='Kernel':

			if self.axis_features== 0:
                                XX=XX.T

			Ntest=XX.shape[0]
			Ntrain=self.X.shape[0]
		
			Kert=KernelCalc(self.X,XX,Ntrain,Ntest,var=self.var,typeK=self.type_K,typeD=self.type_D)
			#Centering
			oneNtrainR=np.ones((Ntrain,Ntrain))/Ntrain
                        Kert=Kert-Kert.dot(oneNtrainR)-self.K_avg_rows+self.K_avg_all
			return np.dot(Kert,self.basis/np.sqrt(self.eigVals))
		else:
			warnings.warn('Must choose between Standard or Kernel PCA')
