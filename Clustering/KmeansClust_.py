import numpy as np
from MachineLearning.Distances import DistancesMatrix
import warnings

class Kmeans:

	def __init__(self,k,tol=1e-6,ninit=10,initM='km++',iter=500,typeD='Euc'):
		self.k=k
		self.tol=tol
		self.iter=iter
		self.typeD=typeD
		self.ninit=ninit
		self.initM=initM
	
	def __CinitPlusPlus(self,X):
	#For Kmeans++, generate the starting means positions
		#self.Nl = X.shape[0]
                ind_mean_start = np.random.randint(self.Nl,size=1)
                indNew=ind_mean_start[0]
                mean_start=[]

                for r_loop in range(self.k):
                        mean_start.append( list(X[indNew,:]) )
                        if self.typeD == 'Euc':
                                Dist = DistancesMatrix(np.array(mean_start),X,1,self.Nl,self.typeD,T=False,sqr=False)
                        elif self.typeD == 'Man':
                                Dist = DistancesMatrix(np.array(mean_start),X,1,self.Nl,self.typeD)
                       
                        Dist=np.min(Dist,axis=1)
                        Prob = Dist/Dist.sum()
                        cumProd = Prob.cumsum()
                        r = np.random.random()
                        indNew = np.where(cumProd >= r)[0][0]
		return np.array(mean_start)

        def train(self,X):

		self.Nl=X.shape[0]
		inertia0=0.
		for rinit in range(0,self.ninit):
			
			if self.initM == 'km++':
			#kmean++
				means_K = self.__CinitPlusPlus(X)
			elif self.initM == 'Ll':
			#Lloyd's
				possible_ind = range(0,self.Nl)
                        	#I shuffle twice
                        	np.random.shuffle(possible_ind)
                        	np.random.shuffle(possible_ind)
                       	 	ind_mean_start = possible_ind[0:self.k]
                        	means_K = X[ind_mean_start,:]
			else:
				strWarn='This initialization '+self.initM+' is not implemented or does not exist. kmean++ will be used!'
				warnings.warn(strWarn)
				means_K = self.__CinitPlusPlus(X)

                	count = 1
                	looping = 1

                	while looping == 1:
				if self.typeD == 'Euc':
					Dist = DistancesMatrix(means_K,X,self.k,self.Nl,self.typeD,T=False,sqr=False)
				elif self.typeD == 'Man':
					Dist = DistancesMatrix(means_K,X,self.k,self.Nl,self.typeD)

				ind_cluster = Dist.argmin(axis=1)
				means_K0 = means_K.copy()
				Xc = [] #List that will have k different elements
				indc = []
				for rk in range(0,self.k):
					ind_NinClustk = np.where(ind_cluster == rk)[0]
					indc.append(ind_NinClustk)
					Xc.append(X[ind_NinClustk,:])
					means_K[rk] = np.mean(Xc[rk],axis=0)

				err = np.sqrt( np.sum( (means_K-means_K0)**2,axis=1)  )
				#If each centers of clusters is converged to within the tolerance,
                                #the calculation is stopped or if the number of iterations becomes
                                #too large
				if ( np.sum((err<=self.tol).astype(int)) == self.k) or (count >= self.iter):
            				if count >= self.iter:
                				warnings.warn('The function stopped because the number of iterations exceeded the maximum number allowed.')
            				looping = 0
        			else: 
            				count = count+1

			inertia=0
                        for r_inertia in range(0,self.k):
                                inertia = inertia+np.sum(DistancesMatrix(means_K[r_inertia,:].reshape(1,X.shape[1]),Xc[r_inertia],1,Xc[r_inertia].shape[0],self.typeD,T=False,sqr=False))

                        if (inertia<inertia0) or (rinit == 0):
                                #print rinit
                                self.means = means_K
                                self.Xc = Xc
				self.indc = indc
                                self.count = count
                                self.iterinit = rinit
				self.inertia=inertia
                        	inertia0 = inertia
	
	def query(self,Xt):
	#Gives the cluster's indice of each member of Xt

		if Xt.ndim == 1:
			Xt=Xt.reshape(1,len(Xt))

		if self.typeD == 'Euc':
                	DistT = DistancesMatrix(self.means,Xt,self.k,Xt.shape[0],self.typeD,T=False,sqr=False)
                elif self.typeD == 'Man':
                        DistT = DistancesMatrix(self.means,Xt,self.k,Xt.shape[0],self.typeD)

		return list(DistT.argmin(axis=1))

