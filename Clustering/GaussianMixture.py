import numpy as np
from MachineLearning.Distances import DistancesMatrix
from MachineLearning.Clustering import Kmeans
import warnings

#Gaussian mixture using EM algorithm

class GMM:

        def __init__(self,K,CovType='spherical',tol=1e-7,tol_rel=1e-4,initM='kmeans',iter=100):
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
                     
                        Thetak=np.zeros(self.K)
			if self.CovType=='spherical':
				sigk2=np.zeros(self.K)
				for rk in range(0,self.K):
                                	ind_NinClustk = np.where(ind_cluster == rk)[0]
                                	Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
					DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustk),'Euc',T=False,sqr=False)
					sigk2[rk]=np.mean(DD)/X.shape[1]
			elif self.CovType=='Diag':
				pass
			elif self.CovType=='Full':
				sigk2=[]
				L_k=[]
				for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
					XMU=X[ind_NinClustk,:]-means_K[rk,:]
                        		sigk2.append(np.dot(XMU.T,XMU)/float(XMU.shape[0]))
                        		L_k.append( np.linalg.cholesky(sigk2[rk]))

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
                        Thetak=np.zeros(self.K)

			if self.CovType=='spherical':
                                sigk2=np.zeros(self.K)
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustK),'Euc',T=False,sqr=False)
                                        sigk2[rk]=np.mean(DD)/X.shape[1]
                        elif self.CovType=='Diag':
                                pass
                        elif self.CovType=='Full':
                                sigk2=[]
                                L_k=[]
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        XMU=X[ind_NinClustk,:]-means_K[rk,:]
                                        sigk2.append(np.dot(XMU.T,XMU)/float(XMU.shape[0]))
                                        L_k.append( np.linalg.cholesky(sigk2[rk]))

                elif self.initM == 'kmeans':
                #Use a full kmeans calculation to initialize the means
                        cl=Kmeans(self.K)
                        cl.train(X)
                        means_K=cl.means
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
			ind_cluster = Dist.argmin(axis=1)
                        Thetak=np.zeros(self.K)
			if self.CovType=='spherical':
                                sigk2=np.zeros(self.K)
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustk),'Euc',T=False,sqr=False)
                                        sigk2[rk]=np.mean(DD)/X.shape[1]
                        elif self.CovType=='Diag':
                                pass
                        elif self.CovType=='Full':
                                sigk2=[]
                                L_k=[]
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        XMU=X[ind_NinClustk,:]-means_K[rk,:]
                                        sigk2.append(np.dot(XMU.T,XMU)/float(XMU.shape[0]))
                                        L_k.append( np.linalg.cholesky(sigk2[rk]))
                elif self.initM == 'init_given':
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        Thetak=np.zeros(self.K)
			if self.CovType=='spherical':
                                sigk2=np.zeros(self.K)
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustk),'Euc',T=False,sqr=False)
                                        sigk2[rk]=np.mean(DD)/X.shape[1]
                        elif self.CovType=='Diag':
                                pass
                        elif self.CovType=='Full':
                                sigk2=[]
                                L_k=[]
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        XMU=X[ind_NinClustk,:]-means_K[rk,:]
                                        sigk2.append(np.dot(XMU.T,XMU)/float(XMU.shape[0]))
                                        L_k.append( np.linalg.cholesky(sigk2[rk]))
		else:
                        strWarn='This initialization '+self.initM+' is not implemented or does not exist. km++ will be used!'
                        warnings.warn(strWarn)
                        means_K = self.__CinitPlusPlus(X)
                        Dist = DistancesMatrix(means_K,X,self.K,self.Nl,'Euc',T=False,sqr=False)
                        ind_cluster = Dist.argmin(axis=1)
                        Thetak=np.zeros(self.K)
			if self.CovType=='spherical':
                                sigk2=np.zeros(self.K)
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        DD = DistancesMatrix(means_K[rk,:].reshape(1,X.shape[1]),X[ind_NinClustk,:],1,len(ind_NinClustk),'Euc',T=False,sqr=False)
                                        sigk2[rk]=np.mean(DD)/X.shape[1]
                        elif self.CovType=='Diag':
                                pass
                        elif self.CovType=='Full':
                                sigk2=[]
                                L_k=[]
                                for rk in range(0,self.K):
                                        ind_NinClustk = np.where(ind_cluster == rk)[0]
                                        Thetak[rk]=len(ind_NinClustk)/float(self.Nl)
                                        XMU=X[ind_NinClustk,:]-means_k[rk,:]
                                        sigk2.append(np.dot(XMU.T,XMU)/float(XMU.shape[0]))
                                        L_k.append( np.linalg.cholesky(sigk2[rk]))

		if self.CovType=='Full':
			return means_K,Thetak,sigk2,L_k
		else:
			return means_K,Thetak,sigk2,Dist

	def train(self,X,means_K=None):

                self.Nl=X.shape[0]
		count = 1
		looping = 1
		#Likelihood initialized to large number
		LL0=1e16

		if self.CovType=='spherical':

			#Initial values of parameters
                	if self.initM=='init_given':
                        	means_K,Thetak,sigk2,Dist=self.__initialize(X,means_K)
                	else:
                        	means_K,Thetak,sigk2,Dist=self.__initialize(X)
			Thetak0=Thetak
			means_K0=means_K
			sigk20=sigk2
			znk=Thetak*np.exp(-0.5*Dist/sigk2)/(np.sqrt((2.*np.pi*sigk2)**X.shape[1]))
	                #logznk=np.log(Thetak) -0.5*X.shape[1]*np.log(2.*np.pi*sigk2) - -0.5*Dist/sigk2
                	#########################################################################################$
                	#                               Starting Expectation Maximization loops                  $
                	#########################################################################################$
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
		elif self.CovType=='Diag':
			pass
		elif self.CovType=='Full':
			
			#Initial values of parameters
                        if self.initM=='init_given':
                                means_K,Thetak,sigk2,L_k=self.__initialize(X,means_K)
                        else:
                                means_K,Thetak,sigk2,L_k=self.__initialize(X)
			Thetak0=Thetak
                        means_K0=means_K
                        sigk20=sigk2
			znk=np.zeros((self.Nl,self.K))
                	for r_k in range(self.K):
                        	XMU=X-means_K[r_k,:]
                        	ETA=np.linalg.solve(L_k[r_k],XMU.T)
                        	detSig=np.prod(np.diag(L_k[r_k])**2)
                        	znk[:,r_k]=Thetak[r_k]*np.exp(-0.5*np.diag(np.dot(ETA.T,ETA)))/(np.sqrt( (2.*np.pi)**X.shape[1]*detSig ) )
			while looping==1:
				#normalize along k
                                znk=1./np.sum(znk,axis=1).reshape((self.Nl,1))*znk
                                #sum along n
                                zk=np.sum(znk,axis=0)
                                #Probability of cluster k
                                Thetak=1./self.Nl*zk
                                #New means of clusters
                                means_K=1./zk.reshape((self.K,1))*np.dot(znk.T,X)
				sigk2=[]
		                L_k=[]
                		for r_k in range(self.K):
                        		XMU=X-means_K[r_k,:]
                        		sigk2.append( XMU.T.dot( np.diag(znk[:,r_k]).dot(XMU) )/zk[r_k] )
					L_k.append(np.linalg.cholesky(sigk2[r_k]))
					ETA=np.linalg.solve(L_k[r_k],XMU.T)
                                	detSig=np.prod(np.diag(L_k[r_k])**2)
                                	znk[:,r_k]=Thetak[r_k]*np.exp(-0.5*np.diag(np.dot(ETA.T,ETA)))/(np.sqrt( (2.*np.pi)**X.shape[1]*detSig ) )

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
