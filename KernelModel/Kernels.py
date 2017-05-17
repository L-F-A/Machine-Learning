import numpy as np
from MachineLearning.Distances import DistancesMatrix
import scipy as sp


#Need to add the possibility to mix kenerls together like K1+K2, K1*K2 etc

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
		elif typeK=='Gau_Diag':
                        D=DistancesMatrix(X/var,Xt/var,Nl,Nt,typeD='Euc',T=T,sqr=False)
                        return np.exp(-0.5*D)
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
			elif typeK == 'Gau_TriDiag':
                                #var represent a tridiagonal pxp matrix
                                #thus var[0:p] the diagonal, var[p:2*p-1] the lower diag and
                                #the upper diagonal = lower diagonal
                                p=Xt.shape[1]
                                d=var[0:p]
                                ld=var[p:2*p-1]
				#udd=np.zeros(p)
				#ldd=np.zeros(p)
				#udd[1:]=ld
				#ldd[:-1]=ld
				#ab=np.array([udd,d,ldd])
				#ab=np.array([udd,d])
                                U=sp.linalg.cholesky(np.diag(ld,-1)+np.diag(d,0)+np.diag(ld,1))
				#sigm1=sp.linalg.cho_solve((U,False),np.identity(p))
                                Ker=np.zeros((Nt,Nl))
                                for r in xrange(Nt):
					#print r
                                        DT=Xt[r,:]-X
					#eta=sp.linalg.solveh_banded(ab,DT.T)
					#eta=sp.linalg.solve_banded((1,1),ab,DT.T)	
                                        eta=sp.linalg.solve_triangular(U,DT.T)
					#Ker[r,:]=np.exp(-0.5*np.einsum('ij,ji->i',DT,eta))
					#Ker[r,:]=np.exp( -0.5*np.sum(eta**2,axis=0) )
                                        Ker[r,:]=np.exp(-0.5*np.einsum('ij,ji->i',eta.T,eta))
					#K[r,:]=np.exp(-0.5*np.einsum("ki,ij,kj->k",DT,sigm1,DT))

				return Ker
#########################################################################################################################

def KernelCalcDer(X,Xt,Nl,Nt,var=None,typeK='Gau',typeD='Euc',T=False,xinterval=None,dK_dlntheta=False):
#	Return K,dK/dsig where K is the Kernel matrix and dK/dsig is a matrix where each component is the derivative with
#	respect to sigma i.e. component ij is dK_ij/dsig 
#
#	dK_dlntheta is boolean. True if we are calculating the derivative with respect of log of hyperparameters

        if (typeD == 'Euc') and (typeK == 'Gau'):

                D = DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc',T=T,sqr=False)
                K=np.exp(-0.5*D/(var[0]**2))
		if dK_dlntheta==False:
                	return K,D*K/var[0]**3
		else:
			return K,D*K/var[0]**2

        elif (typeD == 'Euc_Cont') and (typeK == 'Gau'):

                D=DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc_Cont',T=T,sqr=False,xinterval=xinterval)
                K=np.exp(-0.5*D/(var[0]**2))
		if dK_dlntheta==False:
                	return K,D*K/var[0]**3
		else:
			return K,D*K/var[0]**2

	elif typeK=='Gau_Diag':

		D=DistancesMatrix(X/var,Xt/var,Nl,Nt,typeD='Euc',T=T,sqr=False)
		K=np.exp(-0.5*D)
		if dK_dlntheta==False:
			dk=(Xt[:,np.newaxis,:]-X[np.newaxis,:,:])**2/var**3*K[:,:,np.newaxis]
		else:
			dk=(Xt[:,np.newaxis,:]-X[np.newaxis,:,:])**2/var**2*K[:,:,np.newaxis]

		return K,dk
        else:
                if typeD=='Euc':
                        D=DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc',T=T,sqr=True)
                elif typeD=='Euc_Cont':
                        D=DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc_Cont',T=T,sqr=True,xinterval=xinterval)
		elif typeD=='Man':
			D=DistancesMatrix(X,Xt,Nl,Nt,typeD='Man',T=T)
		else:
                        print 'Distance of type', typeD, ' is not implemented'

                if typeK=='Gau':
                        K=np.exp(-0.5*D**2/(var[0]**2))
			if dK_dlntheta==False:
                        	return K,D**2*K/var[0]**3
			else:
				return K,D**2*K/var[0]**2
                elif typeK == 'Exp': 
                        K=np.exp(-D/var[0])
			if dK_dlntheta==False:	
                        	return K,D*K/var[0]**2
			else:
				return K,D*K/var[0]

                elif typeK == 'Matern52':
                        dij = np.sqrt(5.)*D/var[0]
                        K=(1.+ dij + (dij**2)/3.)*np.exp(-dij)
			if dK_dlntheta==False:
                        	return K, dij/var[0]*K - ( dij + 2.*dij**2/3.  )*np.exp(-dij)/var[0]
			else:
				return K, dij*K - ( dij + 2.*dij**2/3.  )*np.exp(-dij)
		else:
                	print 'Kernel of type', typeK, ' is not implemented'

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


def KernelCalcDer_withD(D,var=None,typeK='Gau',sqEuc=False,dK_dlntheta=False):
#	The first in return is the kernel, the second is the first derivative
#       dK_dlntheta is boolean. True if we are calculating the derivative with respect of log of hyperparameters

	if (typeK == 'Gau') and (sqEuc==True):
		K=np.exp(-0.5*D/(var[0]**2))
		if dK_dlntheta==False:
                	return K,D*K/var[0]**3
		else:
			return K,D*K/var[0]**2
        elif (typeK == 'Gau') and (sqEuc==False):
                K=np.exp(-0.5*D**2/(var[0]**2))
		if dK_dlntheta==False:
			return K,D**2*K/var[0]**3
		else:
			return K,D**2*K/var[0]**2
        elif typeK == 'Exp':
                K=np.exp(-D/var[0])
		if dK_dlntheta==False:
			return K,D*K/var[0]**2
		else:
			return K,D*K/var[0]
        elif typeK == 'Matern52':
		dij = np.sqrt(5.)*D/var[0]
                K=(1.+ dij + (dij**2)/3.)*np.exp(-dij)
		if dK_dlntheta==False:
                	return K, dij/var[0]*K - ( dij + 2.*dij**2/3.  )*np.exp(-dij)/var[0]
		else:
			return K, dij*K - ( dij + 2.*dij**2/3.  )*np.exp(-dij)
        else:
                print 'Kernel of type', typeK, ' is not implemented'

