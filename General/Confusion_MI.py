import numpy as np

def confusion(ytrue1,ypred1):
#Confusion matrix for binary case
	ytrue=ytrue1.copy()
	ypred=ypred1.copy()
	#Could be optimized!!!!!
	if ytrue.ndim != 1:
		if ytrue.shape[0] == 1:
			s=ytrue.shape[1]
		else:
			s=ytrue.shape[0]
		ytrue=ytrue.reshape((s,))
	if ypred.ndim != 1:
		if ypred.shape[0] == 1:
		        s=ypred.shape[1]
		else:
		        s=ypred.shape[0]
		ypred=ypred.reshape((s,))

	if np.min(ytrue) < 0:
		ytrue[ytrue<0.] = 0.
		ypred[ypred<0.] = 0.

	#print ypred[0:11]
	#print ytrue[0:11]

	yd=ypred-ytrue
	FN = len(yd[yd==-1])
	FP = len(yd[yd==1])

	ind_pos = np.where(ytrue==1)[0]
	ind_neg = np.where(ytrue==0)[0]
	TP=np.sum(ypred[ind_pos])
	TN=len(ind_neg) - np.sum(ypred[ind_neg])

	return np.array([[TN,FP],[FN,TP]]).astype(float)

def MutInfo(C):
	#C is the confusion matrix
	Cp=C.copy()/(C[0,0]+C[0,1]+C[1,0]+C[1,1])
	#We also want the mutual information
	px=np.sum(Cp,axis=1)
	py=np.sum(Cp,axis=0)
	HX=-np.sum(px*np.log(px+1e-20))
	HY=-np.sum(py*np.log(py+1e-20))
	HXY=-np.sum(Cp*np.log(Cp+1e-20))
	return HX+HY-HXY

def acc_prec_rec_F1(C):

	TN=C[0,0]
	FP=C[0,1]
	FN=C[1,0]
	TP=C[1,1]
	prec=TP/(TP+FP)
	rec=TP/(TP+FN)
	acc=(TP+TN)/(TP+TN+FP+FN)
	F1=2.*prec*rec/(prec+rec)
	return acc,prec,rec,F1

