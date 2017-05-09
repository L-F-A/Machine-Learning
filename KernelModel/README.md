Some kernel machine learning methods.

  1-Kernels: Few Kernels
  
  2-Kernel Ridge Regression
  
  3-Kernel SVM
  
  4-One class SVM with Kernels

The Kernel Ridge Regression class also offers a fit method (aka Gaussian process) to obtain the hyperparameters
by maximizing the log marginal likelihood of the training set. Three possible solvers, two of them 'SLSQP' and 'COBYLA' do 
not use gradients while 'L-BFGS-B' does. The latter is only implemented for the cases with Kernels: Gaussian, Exponential or Matern 5/2 with one hyperparameter $\sigma$ and distance metrics among Euclidean, Manhattan or Euclidean continuous (integral of the squared difference between two functions). Thus a total of two hyperparameters; $\sigma$ and $\lambda$, the regularization. Still a work in progress, but tests done up to now (Gaussian and Matern Kernels with Eucllidean distance with training set of size (1000,2049),(3000,2049) and (5000,2049) and 5-10 outputs) indicates large efficience, faster than Sklearn version.


To add: Kernel quantile regression and Kernel Logistic regression
