import numpy as np

def BootSample(NB,Nn):
#########################################################################
#	Generates NB bootstrap samples for training set of size Nn	#
#########################################################################
#	Inputs:								#
#		NB     : Number of bootstrap samples			#
#		Nn     : How many instances in the training set		#
#									#
#	Outputs:							#
#		sample : A list containing NB lists of Nn indices	#
#									#
#########################################################################

	sample=[]
	
	for b in range(NB):
		sample.append(list(np.random.randint(0,Nn,size=Nn)))

	return sample

