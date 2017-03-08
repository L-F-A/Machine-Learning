import numpy as np
from MachineLearning.Distances import DistancesMatrix
from MachineLearning.Clustering import Kmeans
import warnings

#Gaussian mixture using EM algorithm considering a unique variance per cluster k

class GMM:

        def __init__(self,K,CovType='spherical',tol=1e-7,tol_rel=5e-4,initM='kmeans',iter=100):
                self.K=K
                self.tol=tol
		self.tol_rel=tol_rel
                self.iter=iter
                self.initM=initM
		self.CovType=CovType

        def __CinitPlusPlus(self,X):
        #Using Kmeans++ starting means positions
                ind_mean_start = np.random.randint(self.Nl,size=1)
                indNew=ind_mean_start[0]
                mean_start=[]
                for r_loop in range(self.K):
                        mean_start.append( list(X[indNew,:]) )
                        Dist = DistancesMatrix(np.array(mean_start),X,1,self.Nl,'Euc',T=False,sqr=False)
                        Dist=np.min(Dist,axis=1)
                        Prob = Dist/Dist.sum()
                        cumProd = Prob.cumsum()
                        r = np.random.random()
                        indNew = np.where(cumProd >= r)[0][0]
                return np.array(mean_start)

	def __initialize(self,X,means_K=None):
		#How are the parameters initialized
                if self.initM == 'km++':
                #kmean++ starting point only
                        means_K = self.__CinitPlusPlus(X)
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        sigk2=np.zeros(self.K)
                        Thetak=np.zeros(self.K)
                        for rk in range(0,self.K):
                                ind_NinClustk = np.where(ind_cluster == rk)[0]
                                Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
				#if self.CovType=='spherical':
                                DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustk),'Euc',T=False,sqr=False)
                                sigk2[rk]=np.mean(DD)/X.shape[1]
				#elif self.CovType=='Diag':
				#	XMU=X[ind_NinClustk,:]-means_K[rk,:]
				#	sig_temp=np.sum(XMU**2,axis=0)/float(XMU.shape[0])
				#elif self.CovType=='Full':
				#	XMU=X[ind_NinClustk,:]-means_K[rk,:]
				#	sig_temp=np.dot(XMU.T,XMU)/float(XMU.shape[0])
                elif self.initM == 'Ll':
                #Lloyd's
                        possible_ind = range(0,self.Nl)
                        #I shuffle twice
                        np.random.shuffle(possible_ind)
                        np.random.shuffle(possible_ind)
                        ind_mean_start = possible_ind[0:self.K]
                        means_K = X[ind_mean_start,:]
			Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        sigk2=np.zeros(self.K)
                        Thetak=np.zeros(self.K)
                        for rk in range(0,self.K):
                                ind_NinClustk = np.where(ind_cluster == rk)[0]
                                Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustK),'Euc',T=False,sqr=False)
                                sigk2[rk]=np.mean(DD)/X.shape[1]
                elif self.initM == 'kmeans':
                #Use a full kmeans calculation to initialize the means
                        cl=Kmeans(self.K)
                        cl.train(X)
                        means_K=cl.means
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        sigk2=np.zeros(self.K)
                        Thetak=np.zeros(self.K)
                        for r_k in range(len(cl.Xc)):
                                Thetak[r_k]=cl.Xc[r_k].shape[0]/float(self.Nl)
                                DD = DistancesMatrix(means_K[r_k,:].reshape(1,X.shape[1]),cl.Xc[r_k],1,cl.Xc[r_k].shape[0],'Euc',T=False,sqr=False)
                                sigk2[r_k]=np.mean(DD)/X.shape[1]
                elif self.initM == 'init_given':
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        sigk2=np.zeros(self.K)
                        Thetak=np.zeros(self.K)
                        for r_k in range(0,self.K):
                                ind_NinClustk = np.where(ind_cluster == r_k)[0]
                                Thetak[r_k]=len(ind_NinClustk)/float(self.Nl)
                                DD = DistancesMatrix(means_K[r_k,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustK),'Euc',T=False,sqr=False)
                                sigk2[r_k]=np.mean(DD)/X.shape[1]
		else:
                        strWarn='This initialization '+self.initM+' is not implemented or does not exist. km++ will be used!'
                        warnings.warn(strWarn)
                        means_K = self.__CinitPlusPlus(X)
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        sigk2=np.zeros(self.K)
                        Thetak=np.zeros(self.K)
                        for r_k in range(0,self.K):
                                ind_NinClustk = np.where(ind_cluster == r_k)[0]
                                Thetak[r_k]=len(ind_NinClustk)/float(self.Nl)
                                DD = DistancesMatrix(means_K[r_k,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustK),'Euc',T=False,sqr=False)
                                sigk2[r_k]=np.mean(DD)/X.shape[1]
		return means_K,Thetak,sigk2,Dist

	def train(self,X,means_K=None):
                self.Nl=X.shape[0]
		#Initial values of parameters
		if self.initM=='init_given':
			means_k,Thetak,sigk2,Dist= self.__initialize(X,means_K)
		else:
			means_k,Thetak,sigk2,Dist= self.__initialize(X)
		means_k0=means_k
		Thetak0=Thetak
		sigk20=sigk2
		count = 1
		looping = 1
		#Likelihood initialized to large number
		LL0=1e16
		znk=Thetak*np.exp(-0.5*Dist/sigk2)/(np.sqrt((2.*np.pi*sigk2)**X.shape[1]))
		#logznk=np.log(Thetak) -0.5*X.shape[1]*np.log(2.*np.pi*sigk2) - -0.5*Dist/sigk2
		#################################################################################################################	
		#				Starting Expectation Maximization loops						#
		#################################################################################################################
		while looping==1:
			#normalize along k
			#delta=np.max(logznk,axis=0)
			#logznk_norm=logznk-
			znk=1./np.sum(znk,axis=1).reshape((self.Nl,1))*znk
			#sum along n
			zk=np.sum(znk,axis=0)
			#Probability of cluster k
			Thetak=1./self.Nl*zk
			#New means of clusters
			means_K=1./zk.reshape((self.K,1))*np.dot(znk.T,X)
			#New variances
			Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
			sigk2=np.sum(znk*Dist,axis=0)/zk/X.shape[1]
			znk=Thetak*np.exp(-0.5*Dist/sigk2)/(np.sqrt((2.*np.pi*sigk2)**X.shape[1]))
			#Likelihood used as cost function, convergence when stop changing
			LL=np.sum(np.log(np.sum(znk,axis=1)))/self.Nl
			err=np.abs(LL-LL0)
			errR=err/np.abs(LL0)
			if (err<=self.tol) or (errR<=self.tol_rel) or (count >= self.iter):
				if count >= self.iter:
					warnings.warn('The function stopped because the number of iterations exceeded the maximum number allowed.')
				looping = 0
			else:
				count = count+1
			LL0=LL
		#Final converged probability that example n is member of cluster k
		self.znk=1./np.sum(znk,axis=1).reshape((self.Nl,1))*znk
		#How many iterations to convergence
		self.count=count
		#Probability of cluster
		self.Thetak=Thetak
		#means
		self.means=means_K
		#Variances
		self.sigk2=sigk2
		#Final likelihood
		self.LL=LL

		#self.means0=means_k0
		#self.Thetak0=Thetak0
		#self.sigk20=sigk20
