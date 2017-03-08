import numpy as np

class BayesGau:
#################################################################################################
#		Bayes Gaussian classifier. Either Full covariance or Naive Bayes		#
#												#
#	Louis-Francois Arsenault, Columbia Universisty (2013-2017), la2518@columbia.edu		#
#################################################################################################
#												#
#	INPUTS:											#
#		X 	    : Training data features matrix					#
#		y           : Training data classes vector					#
#		Xt          : Testing data features vector or matrix				#
#		ClassOnly   : False, output the probabilities for each classes; True output only#
#			      which class is the most probable					# 
#												#
#	FUNCTIONS:										#
#		train_Full  : Train a classifier with Gaussian distributions for each class k, 	#
#			      with full covariance matrices among features Sigma_k		#
#		Query_Full  : Query the model with full covariance matrix			#
#		train_Naive : Naive Bayes assumption, each features are independent such that 	#
#			      for each class k, the covariance matrix is diagonal		#
#		Query_Naive : Query the model with independent features 			#
#												#
#	OUTPUTS:										#
#		P	    : Probabilities for different classes if ClassOnly=False		#
#		classes     : Which class is the most probable if ClassOnly=True		#
#################################################################################################

        def __init__(self,X,y):

                self.X=X.copy()
                self.y=y.copy()
		self.Nn=X.shape[0]#How many training example
		self.Np=X.shape[1]#How many dimensions (fetaures)
                self.__class_distribution()#Split training data into their classes

        def __class_distribution(self):
		#Unique classes values and how many time for each
                nClass,countClass=np.unique(self.y,return_counts=True)
		#Classes are numbered 0 to K
                K=len(nClass)-1
		#Split the indices of rows of training set among the uniques classes
                idx=np.split(np.argsort(self.y),np.cumsum(countClass))
		#Probability of each class p(k)
                pi_k=np.zeros(K+1)
		#List of np arrays containing the training data split by classes.
                XX=[]
                for r in range(K+1):
                        XX.append(self.X[idx[r],:])
                        pi_k[r] = self.X[idx[r],:].shape[0]/float(self.Nn)
                self.K=K
                self.XX=XX
                self.pi_k=pi_k
                self.idx=idx

        def train_Full(self):
                mu_k=np.zeros((self.K+1,self.Np))
                Sig_k=[]
                L_k=[]
                for r in range(self.K+1):
			#Mean of each class for the gaussian distribution
                        mu_k[r,:]=np.mean(self.XX[r],axis=0)
			#Covariance matrix for each class for the Gaussian distribution
			XMU=self.XX[r]-mu_k[r,:]
			sig_temp=np.dot(XMU.T,XMU)/float(self.XX[r].shape[0])
			Sig_k.append(sig_temp)
                        #Sig_k.append(np.cov(self.XX[r],rowvar=False,bias=True))
			#Cholesky factorization of the covariance matrix
                        L_k.append( np.linalg.cholesky(Sig_k[r])  )
                self.mu_k=mu_k
                self.Sig_k=Sig_k
                self.L_k=L_k

	def train_Naive(self):
		mu_k=np.zeros((self.K+1,self.Np))
                Sig_k=[]
		for r in range(self.K+1):
			#Mean of each class for the Gaussian distribution
			mu_k[r,:]=np.mean(self.XX[r],axis=0)
			#Variances for each feature for each class for the Gaussian distribution
			XMU=self.XX[r]-mu_k[r,:]
                        sig_temp=np.sum(XMU**2,axis=0)/float(self.XX[r].shape[0])
                        Sig_k.append(sig_temp)
			#Sig_k.append(np.var(self.XX[r],axis=0,ddof=0))
		self.mu_k=mu_k
                self.Sig_k=Sig_k

	def Query_Full(self,Xt,ClassOnly=False):
		if Xt.ndim==1:
                	logP=np.zeros(self.K+1)
                	for r in range(self.K+1):
                        	a=Xt-self.mu_k[r,:]
                        	eta=np.linalg.solve(self.L_k[r],a)
                        	d2=np.dot(eta,eta)
				#log of the unormalized probability
                        	logP[r]=np.log(self.pi_k[r]) -0.5*np.log(np.linalg.det(self.Sig_k[r])) - 0.5*d2
                	delta=np.max(logP)
                	sumP=delta+np.log(np.sum(np.exp(logP-delta)))
			#log of normalized probability
                	logP_norm=logP-sumP
		else:
			logP=np.zeros((Xt.shape[0],self.K+1))
			for r in range(self.K+1):
                                a=Xt-self.mu_k[r,:]
				eta=np.linalg.solve(self.L_k[r],a.T)
				d2=np.sum(eta**2,axis=0)
				logP[:,r]=np.log(self.pi_k[r]) -0.5*np.log(np.linalg.det(self.Sig_k[r])) - 0.5*d2
			delta=np.amax(logP,axis=1)
			sumP=delta+np.log(np.sum(np.exp(logP-delta.reshape(Xt.shape[0],1)),axis=1))
			logP_norm=logP-sumP.reshape(Xt.shape[0],1)
		if ClassOnly is False:
			return np.exp(logP_norm)#return P the probability
		else:
			return np.argmax(logP,axis=1)

	def Query_Naive(self,Xt,ClassOnly=False):
                if Xt.ndim==1:
                        logP=np.zeros(self.K+1)
                        for r in range(self.K+1):
                                a=Xt-self.mu_k[r,:]
				d2=np.sum(a**2/self.Sig_k[r])
				sig2_prod=np.prod(self.Sig_k[r])
                                logP[r]=np.log(self.pi_k[r]) -0.5*np.log(sig2_prod) - 0.5*d2
                        delta=np.max(logP)
                        sumP=delta+np.log(np.sum(np.exp(logP-delta)))
                        logP_norm=logP-sumP
                else:
                        logP=np.zeros((Xt.shape[0],self.K+1))
                        for r in range(self.K+1):
                                a=Xt-self.mu_k[r,:]
				d2=np.sum(a**2/self.Sig_k[r],axis=1)
				sig2_prod=np.prod(self.Sig_k[r])
                                logP[:,r]=np.log(self.pi_k[r]) -0.5*np.log(sig2_prod) - 0.5*d2
                        delta=np.amax(logP,axis=1)
                        sumP=delta+np.log(np.sum(np.exp(logP-delta.reshape(Xt.shape[0],1)),axis=1))
                        logP_norm=logP-sumP.reshape(Xt.shape[0],1)
		if ClassOnly is False:
			return  np.exp(logP_norm)#return P the probability
		else:
			return np.argmax(logP,axis=1)
