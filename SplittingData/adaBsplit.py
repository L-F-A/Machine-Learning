import numpy as np

def AdaBSample(Nn,w):
#########################################################################
#       Generates a sample of size Nn for one AdaBoost iteration        #
#########################################################################
#       Inputs:                                                         #
#               Nn      : How many instances in the training set        #
#               w       : Vector of weights for the instances           #
#                                                                       #
#       Outputs:                                                        #       
#               sample  : One sample for adaBoost                       #
#                                                                       #
#########################################################################

        return np.random.choice(range(Nn),size=Nn,p=w)
