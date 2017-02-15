import numpy as np
from scipy.spatial import distance

def DistancesMatrix(X,Xt,Nl,Nt,typeD='Euc',T=False,sqr=True,xinterval=None):
##############################################################################################
# Pairwise distances matrix between the points contained in a matrix X (training points) and #
#                           in a matrix Xt (testing points)				     #
#											     #
#          Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016)	     #
##############################################################################################
#   INPUT:										     #
# 											     #
#   X  	:   Matrix with the training inputs. The form is a numpy array containing	     # 
#	    the X's, where one X, X_i is a line vector with p dimensions. Must be in the form#
#	    numpy.array([[x11,x12,...,x1p],[x21,x22,...,x2p],...,[xn1,xn2,...,xnp]])	     #
#           so shape is (n,p)           !!!!!!VERY IMPORTANT!!!!!!			     #
# 					 						     #
#   Xt 	:   Matrix with the test inputs, same form as X				 	     #
#   Nl 	:   Number of learning examples						 	     #
#   Nt 	:   Number of test examples						 	     #
#   typeD : Distance metric. The possible metrics are Euclidean 'Euc' (default),	     #
# 	    Euclidean for continuous functions 'Euc_Cont' and Manhattan 'Man'. Any distance  #
#	    implemented in scipy cdist function can also be passed 	  	     	     #
#   T	:   If calculated for training purpose, T=True and when for testing T=False	     #
#   sq	:   If Euclidean distance, decide if return the square distance sqr=False 	     #
#           (faster) or the distance sqr=True						     #
#   xinterval: tuple containing the interval of integration for Euc_Cont: =(a,b)	     #
#											     #
#   OUTPUT:					 					     #
#											     #
#   D   :   Distances matrix. Can be sqaured distances if sqr=False			     #
##############################################################################################

	if typeD == 'Euc':
                Amat = X**2
                Bmat = Xt**2
                sqvec = np.sum(Amat,1)
                sqvec2 = np.sum(Bmat,1)
                Ainter = np.outer(sqvec,np.ones((1,Nt)))
                Binter = np.outer(sqvec2,np.ones((1,Nl)))
                #Euclidean distance square
                distSQ = Ainter.transpose() + Binter - 2.*np.dot(Xt,X.transpose())

		if T==True:
			#When training, Xt = X and the diagonal should be zero. 
			#However, the fast Euclidean calculation we use here gives 
			#something very small but not zero. So we just put 0
			np.fill_diagonal(distSQ,0.)
		if sqr==True:    #Return the distance
                	return np.sqrt(distSQ)
		elif sqr==False: #Return the square distance, faster since no sqrt 
				 #function to take
			return distSQ

	elif typeD == 'Euc_Cont':
	#Euclidean distance for continuous functions of one variable i.e. d_ij^2= int_a^b dx ( f(x_i)-f(x_j) )^2
	#Using Simpson 1/3
		xinit = xinterval[0]
		xfinal = xinterval[1]
		feat_N = X.shape[1]
		interval=float(feat_N-1)
		hstep  = (xfinal-xinit)/interval

		if interval % 2:
        		raise ValueError('The number of intervals n must be even')

		#Build the coefficients for Simpson 1/3
		indice_Simp=np.arange(0,feat_N)
		coeffs_simp_vec = 3.+(-1.)**(indice_Simp+1) #pattern of alternating 2's and 4'
		coeffs_simp_vec[0]=1.
		coeffs_simp_vec[-1]=1.
		coeffs_simp = np.diag(coeffs_simp_vec)
		   
		Amat = np.dot(X**2,coeffs_simp)
                Bmat = np.dot(Xt**2,coeffs_simp)
                sqvec = np.sum(Amat,1)
                sqvec2 = np.sum(Bmat,1)
                Ainter = np.outer(sqvec,np.ones((1,Nt)))
                Binter = np.outer(sqvec2,np.ones((1,Nl)))
                #Continuous Euclidean distance square
                distSQ = hstep/3*( Ainter.transpose() + Binter - 2.*np.dot(Xt.dot(coeffs_simp),X.transpose()) )

                if T==True:
                        #When training, Xt = X and the diagonal should be zero. 
                        #However, the fast Euclidean calculation we use here gives 
                        #for some entries something very small but not zero. So we just put 0
                        np.fill_diagonal(distSQ,0.)
                if sqr==True:    #Return the distance
                        return np.sqrt(distSQ)
                elif sqr==False: #Return the square distance, faster since no sqrt 
                                 #function to take
                        return distSQ
        elif typeD == 'Man': #Homemade one was too slow! Although scipy one not 
			    #that fast. Implementation in C not fast enough at the moment, even with using openmp, but will work on it 
		return distance.cdist(Xt,X,'cityblock')
	else:
		#Any distance implemented in cdist
		return distance.cdist(Xt,X,typeD)
	#else:
	#	raise ValueError('The distance metric %r is not included',typeD)
###############################################################################################















#vec = np.ones((Nl,1))
 #               Xtf = np.outer(vec,Xt)
  #              Y = X.copy()
   #             Np=X.shape[1]
    #            for rp in range(0,Np-1):
     #                   Y = np.concatenate((Y,X))









#distances_matrix = np.zeros((Nt,Nl))
                #for count1 in range(0,Nt):
                #       d_row = Xt[count1,:]
                 #       vec = np.ones((Nl,1))
                  #      d_row = np.outer(vec,d_row)
                   #     dist = np.absolute(d_row-X)
                    #    distances_matrix[count1,:] = np.sum(dist,1)
