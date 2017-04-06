import numpy as np

def BootSample(NB,Nn,P=None):
#########################################################################
#	Generates NB bootstrap samples for training set of size Nn	#
#########################################################################
#	Inputs:								#
#		NB     : Number of bootstrap samples			#
#		Nn     : How many instances in the training set		#
#		P      : If there is a probability vector		#
#									#
#	Outputs:							#
#		sample : A list containing NB lists of Nn indices	#
#									#
#########################################################################

	sample=[]
	for b in range(NB):
		sample.append(list(np.random.choice(Nn,size=Nn,p=P)))
	return sample

