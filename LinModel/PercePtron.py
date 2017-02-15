import numpy as np
import warnings

class Percept:

	def __init__(self):
		pass
	
	def train(self,X,y,eta=1e-3,iteMax=1000):

		OneVec = np.ones((X.shape[0],1))
                XX = np.concatenate((X, OneVec), axis=1)

                #How many intances, dimensions of the inputs and how many outputs
                SizeXX = XX.shape
                self.pdim = SizeXX[1]
                self.Nlearn = SizeXX[0]
		w=np.zeros(self.pdim)
		

		for i in range(iteMax):
			#Find the indices where sign(x_train^Tw) is not the same as y_train 
			d=np.where(XX.dot(w)/y<0)[0]
			#Shuffle the indices to choose in the stochastic gradient
			np.random.shuffle(d)			

			if i==0:#Since we start with w vector 0, we choose a random instance
				#as starting for a stochastic gradient
				idx=np.random.randint(self.Nlearn, size=Nbatch)[0]
				w+=eta*y[idx]*XX[idx,:] 
			elif d.size == 0:
				self.w=w
				self.ite=i+1
				break
			elif i==iteMax-1:#If iteMax is too small of the data is not linearly
					 #separable, perceptron will not converge
				warnings.warn('Maximum number of iterations; did not converge')
				self.w
				self.ite=i+1
			else:#Update vector w with stochastic gradient
				w+=eta*y[d[0]]*XX[d[0],:]
			
	def query(self,Xt):

		OneVec = np.ones((Xt.shape[0],1))
                XXt = np.concatenate((Xt, OneVec), axis=1)
		return np.sign(XXt.dot(self.w))
