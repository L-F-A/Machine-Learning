import numpy as np
######################################################################################################
#      Splitting the database for training testing purpose. Either in Nfolds or in two fold 	     #
#                       with one with nu% of the data and the other (1-nu)%  			     #
#                                                                                                    #
#                       							                     #
#                                                                                                    #
#                   Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)         #
######################################################################################################
#   INPUT:											     #
#												     #
#   Constructor:										     #
#   X  :   a numpy array containing the inputs X's, where one X, X_i is a line vector with p 	     #
#          dimensions. Must be in the form 							     #
#          numpy.array([[x11,x12,...,x1p],[x21,x22,...,x2p],...,[xn1,xn2,...,xnp]])         	     #
#          so shape is (n,p)           !!!!!!VERY IMPORTANT!!!!!!				     #
#												     #
#   Y  :   same form as X but containing the output data. Shape is (n,q) where q is the dimension of #
#          the outputs if multi-outputs							 	     #
#												     #
#   Ninstances : Number of data points in the database						     #
#												     #
#   If split in N folds:									     #
#   Nfolds :  how many folds to consider. If Ninstances/Nfolds is not an integer, will create as     #
#	      equal as possible folds								     #
#												     #
#	Ex:											     #
#	DataSp = TrainTestSplitNfolds(X,Y,Ninstances,Nfolds=10)					     #
#   If split in two:										     #
#										     		     #
#   PropTest : proportion used for the test set							     #
#	Ex:											     #
#	DataSp = TrainTestSplitPer(X,Y,Ninstances,PropTest=0.4)					     #
#												     #
#   OUTPUT:											     #
#												     #
#   DataSp.X : a list with with either 2 or Nfolds elements of type numpy array. If two, the first   #
#              is the test									     #
#   DataSp.Y : a list with with either 2 or Nfolds elements of type numpy array. If two, the first   #
#	       is the test									     #
#   DataSp.indperm : a vector of integers containing the indices of the X_i in the differents folds  #
#   Depending:											     #
#   DataSp.Nfolds : number of folds								     #
#   DataSp.Ntest : number of instances in the test set						     #
#   DataSp.Nlearn : number of instances in training set						     #
######################################################################################################

class TrainTestSplitNfolds:

	def __init__(self,X,Y,Ninstances,Nfolds=5):
		self.SplitType = 'Nfolds'
		self.Nfolds = Nfolds
		self.indPerm = np.array_split(np.random.permutation(Ninstances),Nfolds)
		XX = []
		YY = []
		for rf in range(0,Nfolds):
			XX.append(X[self.indPerm[rf],:])
			YY.append(Y[self.indPerm[rf],:])
		self.X = XX
		self.Y = YY
		

class TrainTestSplitPer:

	def __init__(self,X,Y,Ninstances,PropTest=0.2):
		self.SplitType = 'LearnTest'
		self.Ntest = int(np.ceil(PropTest*Ninstances))
		self.Nlearn = Ninstances-self.Ntest
		indperm = np.random.permutation(Ninstances)
		self.indPerm = [indperm[0:self.Ntest],indperm[self.Ntest:]]
		#self.X[0] and self.Y[0] form the test set
		self.X = [X[self.indPerm[0]],X[self.indPerm[1]]]
		self.Y = [Y[self.indPerm[0]],Y[self.indPerm[1]]]

class TrainTuneTestSplitPer:

        def __init__(self,X,Y,Ninstances,PropTest=0.2,PropTune=0.2):
                self.SplitType = 'LearnTuneTest'
                self.Ntest = int(np.ceil(PropTest*Ninstances))
		self.Ntune = int(np.ceil(PropTune*Ninstances))
                self.Nlearn = Ninstances-self.Ntest-self.Ntune
                indperm = np.random.permutation(Ninstances)
                self.indPerm = [indperm[0:self.Ntest],indperm[self.Ntest:self.Ntest+self.Ntune],indperm[self.Ntest+self.Ntune:]]
                #self.X[0] and self.Y[0] form the test set
                self.X = [ X[self.indPerm[0]] , X[self.indPerm[1]], X[self.indPerm[2]] ]
                self.Y = [Y[self.indPerm[0]],Y[self.indPerm[1]],Y[1]]
