import numpy as np
from MachineLearning.SplittingData import BootSample
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class CentralParkForest:
#################################################################################################
#                       	A quasi random forest classifier 	                        #
#            Contrary to real om forest, the random features are choosen once for each tree     #
#                                                                                               #
#       Louis-Francois Arsenault, Columbia Universisty (2013-2017), la2518@columbia.edu         #
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               NB          : Number of bootstrap sample to draw                                #
#                                                                                               #
#       FUNCTION train:                                                                         #
#               X           : Matrix containing the training data. Size (n_samples,n_features)  #
#               y           : Vector containing the targets classes	                        #
#		d	    : How many features to use						#
#		P	    : Vector with probability of each sample if wanted			#
#		balanced    : If True will build P such that each class has about same prob	#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               Nn          : How many sample in training set				        #
#               Np          : How many features in total	                                #
#               Nc          : How many classes		                                        #
#               d           : How many features were used for each tree                         #
#               models      : A list where each entry contains the model for that boostrap	#
#		features    : A list where each entry contains the indices of the features used	#
#												#
#	FUNCTION query:										#
#		Xt	    : Matrix with testing set. Size (n_test,n_features)			#
#												#
#	OUTPUT:											#
#		pred	    : Vector with the prediction of the class of each member of test set#
#												#
#	FUNCTION score:										#
#		Xt	    : Matrix with testing set. Size (n_test,n_features)			#
#		yt	    : Vector containing the targets classes for test set		#
#												#
#	OUTPUT:											#
#		score	    : Percentage of corect prediction					#	
#												#
#################################################################################################

	def __init__(self,NB):
		self.NB=NB

	def train(self,X,y,d=None,P=None,balanced=False):

		self.Nn=X.shape[0]
		self.Np=X.shape[1]
		self.Nc=len(np.unique(y))

		#How many features to be used in training
		if d is None:
			self.d=int(np.floor(np.sqrt(self.Np)))
		else:
			self.d=d

		if balanced==True:
		#In this case, we will built a probability for each sample such that it will give a 
		#class with less samples a larger probability to be picked in a bootstrap sample
			PP=float(self.Nn)/(self.Nc*np.bincount(y))
			P=PP[y]
			P=P/np.sum(P)
				
		#Construct the bootstrap sample
		if P is None:
			sample=BootSample(self.NB,self.Nn)
		else:
			sample=BootSample(self.NB,self.Nn,P)
	
		self.models=[]
		self.features=[]

		for r_sample in xrange(self.NB):
			#Define a decision tree that will go full depth --> low bias, high variance
			clf=DecisionTreeClassifier()
			#The training set from the bagging
			Xr=X[sample[r_sample],:]
			yr=y[sample[r_sample]]
			#We choose a number d of features only
			id_feat=list(np.random.choice(self.Np,size=self.d,replace=False))
			Xr=Xr[:,id_feat]
			#Put the model in memory
			self.models.append(clf.fit(Xr,yr))
			self.features.append(id_feat)

	def query(self,Xt):
		
		Nt=Xt.shape[0]
		
		if Nt==1:
			pred_temp=np.zeros(self.NB)
			for rB in xrange(self.NB):
				pred_temp[rB]=self.models[rB].predict(Xt[0,self.features[rB]])
			val,count=np.unique(pred_temp,return_counts=True)
			return val[np.argmax(count)]
		else:
			pred_temp=np.zeros((Nt,self.NB))
			for rB in xrange(self.NB):
				pred_temp[:,rB]=self.models[rB].predict(Xt[:,self.features[rB]])

			return np.array([Counter(pred_temp[rt,:]).most_common()[0][0] for rt in xrange(Nt)])

	def score(self,Xt,yt):

		Pred=self.query(Xt)
		return np.mean(Pred==yt)

