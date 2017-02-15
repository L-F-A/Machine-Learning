import numpy as np
from MachineLearning.Distances import DistancesMatrix

def KernelCalc(X,Xt,Nl,Nt,var=None,typeK='Poly',typeD=None,T=False,xinterval=None):
##########################################################################################################################
#                                                  Kernel		                                                 #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#                                                                                                                        #
##################################################################################
#    var   :  An array containing the hyperparameters of the choosen kernel, its #
#             length will depend upon the specific kernel			 #
#    typeK :  Which kernel : polynomial 'Poly', Gaussian 'Gau', Exponential 'Exp'# 
#             or Matern 5/2 'Matern52' are possible				 #
#    typeD :  Which distance metric to use for non-poly kernel: Euclidean 'Euc', #
#             Euclidean for continuous functions 'Euc_C' or Manhattan 'Man'	 #
##################################################################################
	if typeK == 'Poly':
		return ( var[1] + var[0]*Xt.dot(X.transpose()) )**var[2]
	else:
		if (typeD == 'Euc') and (typeK == 'Gau'):
			D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc',T=T,sqr=False)
			return np.exp(-0.5*D/(var[0]**2))
		elif (typeD == 'Euc_Cont') and (typeK == 'Gau'):
			D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc_Cont',T=T,sqr=False,xinterval=xinterval)
                        return np.exp(-0.5*D/(var[0]**2))
		else:
			if   typeD == 'Euc':
				D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc',T=T,sqr=True) 
			elif typeD == 'Man':
				D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Man',T=T)
			elif typeD == 'Euc_Cont':
				D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc_Cont',T=T,sqr=True,xinterval=xinterval)

			if typeK == 'Gau':
				return np.exp(-0.5*D**2/(var[0]**2))
			elif typeK == 'Exp':
				return np.exp(-D/var[0])
			elif typeK == 'Matern52': 
                        	dij = np.sqrt(5.)*D/var[0]
                        	return (1.+ dij + (dij**2)/3.)*np.exp(-dij)
#########################################################################################################################

def KernelCalc_withD(D,var=None,typeK='Gau',sqEuc=False):
##########################################################################################################################
#                                                         Kernel                                                         #
#                                                                                                                        #
#                           Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)                     #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#                                                                                                                        #
##################################################################################
#    var   :  An array containing the hyperparameters of the choosen kernel, its #
#             length will depend upon the specific kernel                        #
#    typeK :  Which kernel : polynomial 'Poly', Gaussian 'Gau', Exponential 'Exp'# 
#             or Matern 5/2 'Matern52' are possible                              #
#    sqEuc :  is the distance matrix the Euclidean distance squared?		 #
##################################################################################
	if (typeK == 'Gau') and (sqEuc==True):
        	return np.exp(-0.5*D/(var[0]**2))
	elif (typeK == 'Gau') and (sqEuc==False):
                return np.exp(-0.5*D**2/(var[0]**2))
        elif typeK == 'Exp':
                return np.exp(-D/var[0])
        elif typeK == 'Matern52':
                dij = np.sqrt(5.)*D/var[0]
                return (1.+ dij + (dij**2)/3.)*np.exp(-dij)
	else:
		print 'Kernel of type', typeK, ' is not implemented'
#########################################################################################################################

#Should add possibility to mix kenerls together like K1+K2, K1*K2 etc
