def trainingCV(dataObj,method,var,ErrMetric,typeD=None,typeK=None,typeRegu=None):
#This is Cross-Validation by doing a grid search and calculating a chosen error metric
	if dataObj.SplitType = 'Nfolds':
	#N-folds cross-validation
	elif dataObj.SplitType = 'LearnTest':
	#We use the test set for tuning
	elif dataObj.SplitType = 'LearnTuneTest':
	#We use the tuning set for tuning, no use of the test set here
