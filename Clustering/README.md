Clustering methods

Only two implemented for the moment
  - k-means clustering
  - Gaussian mixture solved by EM.
  
        1) Covariance implemented: spherical, diagonal and full
        2) Need to improve probability calculation using log, to be done very shortly as for the Bayes 
           classifiers cases
        3) Need to replace taking explicitely full matrix-matrix multiplication then keeping only diagonal
           as np.diag(a.dot(b)) by Einstein summation np.einsum('ij,ji->i',a,b)). Trivial, just need to do it ...
           Will gain good performance
        4) Still needs some testing
