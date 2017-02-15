import numpy as np

#Return the covariance matrix and its svd decomposition
def CovMat(f_exact,f):

	D=f_exact-f
	MCOV=np.cov(D)
	U, S, Vt = np.linalg.svd(M)
	V = Vt.T
	return MCOV,U,S,V


#########################################################################################
#             Covariance matrix from truncated svd with intact diagonal                 #
#                                                                                       # 
#               If the eigenvectors are h_k and eigenvalues lambda_k:                   #
#                                                                                       #
# SIGMA_hat = sum_k=1^p lambda_k h_kh_k^T + diag( sum_{k=p+1}^end lambda_k h_kh_k^T  )  #
#                                                                                       #
#########################################################################################
def truncatedCov(U,S,V,SIGMA,p)

        s=np.array(np.diag(S))
        s[p:]=0.
        Shat=np.diag(s)

        SIGMA_Trunc=U.dot( Shat.dot(V.T) )
        #We keep the diagonal intact
        np.fill_diagonal(SIGMA_Trunc,np.diag(SIGMA))
        #Make sure it is symmetric in case finite precision effect showed its 
        #ugly head
        return 0.5*(SIGMA_Trunc.T+SIGMA_Trunc)

#Return Lower Cholesky decomposition of a covariance matrix
def Lchol(SIGMA)

	return np.linalg.cholesky(SIGMA)



