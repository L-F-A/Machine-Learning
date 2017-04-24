import numpy as np
from MachineLearning.SplittingData import AdaBSample
from sklearn.tree import DecisionTreeClassifier

class AdaBoostForest:
#################################################################################################
#                               AdaBoost on decision tree stumps                                #
#                                                                                               #
#       Louis-Francois Arsenault, Columbia Universisty (2013-2017), la2518@columbia.edu         #
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               NA          : Number of AdaBoost run		                                #
#                                                                                               #
#       FUNCTION train:                                                                         #
#               X           : Matrix containing the training data. Size (n_samples,n_features)  #
#               y           : Vector containing the targets binary classes that must be -1 or 1	#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               Nn          : How many sample in training set                                   #
#               Np          : How many features in total                                        #
#               models      : A list where each entry contains the model for that iteration     #
#               alphas      : A list of weights for each iteration			        #
#                                                                                               #
#       FUNCTION query:                                                                         #
#               Xt          : Matrix with testing set. Size (n_test,n_features)                 #
#                                                                                               #
#       OUTPUT:                                                                                 #
#               pred        : Vector with the prediction of the class of each member of test set#
#                                                                                               #
#       FUNCTION score:                                                                         #
#               Xt          : Matrix with testing set. Size (n_test,n_features)                 #
#               yt          : Vector containing the targets classes for test set                #
#                                                                                               #
#       OUTPUT:                                                                                 #
#               score       : Percentage of corect prediction                                   #
#                                                                                               #
#################################################################################################

#Binary classification, the targets should be casted as -1 or 1
	def __init__(self,NA):
		self.NA=NA

        def train(self,X,y):

		self.Nn,self.Np=X.shape
		w=np.ones(self.Nn)/self.Nn

		self.models=[]
		self.alphas=[]

		for r_AdA in xrange(self.NA):

			#Decision Stump
			clf=DecisionTreeClassifier(max_depth=1)

			idx=AdaBsample(self.Nn,w)			
			Xr=X[idx,:]
			yr=y[idx]
			clf.fit(Xr,yr)

			pred=clf.predict(X)
			epsilon=(pred!=y).dot(w)
			alpha=0.5*(np.log(1.-epsilon)-np.log(epsilon))
			w=w*np.exp(-alpha*y*pred)

			w=w/np.sum(w)
			self.models.append(clf)
			self.alphas.append(alpha)

	def query(self,Xt):

		Nt=Xt.shape[0]
		F=np.zeros(Nt)
		for rA in xrange(self.NA):
			F+=self.alphas[rA]*self.models[rA].predict(Xt)
		return np.sign(F)

	def score(self,Xt,yt):

		pred=self.query(Xt)
		return np.mean(pred==yt)
