Probabilistic matrix factorization: LSMF_Explicit

        -Need to add possibility that the input matrix is already in a sparse format: Easy
        
Non negative matrix factorization

  - Working version
      
        -Need to implement better initial W and H matrices
        -Need to add regularization
        -Need to check again the algorithm. While W*H indeed gives something very close to X, 
         the matrix to be factorized, the elements of w and H are differents from Sklearn 
         with same starting W and H
