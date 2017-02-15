import numpy as np
from collections import Counter
from MachineLearning.Distances import DistancesMatrix

class NearestNeighbor:

	def __init__(self,k):
		self.k=k

	def train(self,X,y):
	#X is matrix NlearnXp y is 1d-vector of length Nlearn
		self.X=X
		self.y=y
		self.Nl=X.shape[0]

	def query(self,Xt,typeD='Euc'):

		Ntest=Xt.shape[0]
		ypred = np.zeros(Ntest)
		tD=typeD
		Dist=DistancesMatrix(self.X,Xt,self.Nl,Ntest,typeD=tD,T=False)

		if self.k != 1:
			ind=np.argsort(Dist,axis=1)
			ypred = [Counter(self.y[ind[rt,0:self.k]]).most_common()[0][0] for rt in range(Ntest)]
			#for rt in range(Ntest):
			#	ypred[rt]=Counter(self.y[ind[rt,0:self.k]]).most_common()[0][0]		
		else:
			ind=np.argmin(Dist,axis=1)
			ypred=self.y[ind]

		return ypred
