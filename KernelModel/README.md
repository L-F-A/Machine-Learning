Some kernel machine learning methods.

  1-Kernels: Few Kernels
  
  2-Kernel Ridge Regression
  
  3-Kernel SVM
  
  4-One class SVM with Kernels

The Kernel Ridge Regression class also offers a fit method (aka Gaussian process) to obtain the hyperparameters
by maximizing the log marginal likelihood of the training set. Two possible solvers, 'SLSQP' which does 
not use gradients while the other, 'L-BFGS-B', does. Both solvers accepts different possibilities of Kerels/distance metrics with different numbers of hyperparameters: 

-Gaussian kernel with diagonal covariance hyperparameters as 'Gau_Diag' where 
K_ij=exp( -0.5*sum_{d=1}^p[x_id-x_jd]^2/sig_d^2 ) with d the d^th dimension of the features vectors. 
A total of p+1 hyperparameters {sig_d}_1^p and $\lam$, the regularization.
      
-Gaussian 'Gau', Exponential 'Exp' or Matern 5/2 'Matern52' with one hyperparameter $\sigma$ and can consider distance metrics to          be chosen among Euclidean 'Euc', Manhattan 'Man' or Euclidean continuous 'Euc_Cont' (integral of the squared difference between           two functions). Thus a total of two hyperparameters; $\sigma$ and $\lambda$, the regularization.
      
Gaussian processes is still a work in progress, but tests done up to now (training set of size (1000,2049), (3000,2049, (5000,2049) and (10000,2049) and 5-20 outputs) indicates large efficiency, faster than Sklearn version for single sig cases and about the same for diagonal sig cases.


To add shortly: Kernel quantile regression (as used in https://arxiv.org/abs/1612.04895 ) and Kernel Logistic regression.
