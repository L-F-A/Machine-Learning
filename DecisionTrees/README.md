Two ensemble methods involving decision trees

1- Quasi-Random Forest. For each tree, a sub-sample d of features is randomly choosen at the beginning and kept. I called the      class CentralParkForest as like in the park, there are something like random forests but started from a strict framework.

2- AdaBoost binary classification using decision tree stumps

Note that for the moment, an individual tree is created using Sklearn decision tree class. A home made decision tree to be added
