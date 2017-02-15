import numpy as np
from MachineLearning.Distances import DistancesMatrix
from scipy.optimize import minimize_scalar
import time
###########################################################################################################
#	1d kernel density estimation using Maximum likelihood cross-validation as given in Eq.12 of	  #
#													  #
#			https://cran.r-project.org/web/packages/kedd/vignettes/kedd.pdf			  #
#													  #
#		    Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2016) 		  #
###########################################################################################################
#   INPUT:												  #
#												  	  #
#   Constructor:											  #
#   x   :  a numpy array containing the x's. Must be in the form numpy.array([[x1],[x2],[x3],...,[xn]])   #
#	   so shpae is (n,1) !!!!!!VERY IMPORTANT!!!!!!							  #
#   Ex:													  #
#   model = Kerden(x)										          #
#													  #
#   Training:												  #
#   typeK :  which kernel to use. Gaussian ('Gau') or Epanechnikov ('Epan')				  #
#   Ex:													  #
#   model.train(typeK)											  #
#													  #
#   Query:												  #
#   xt  a numpy array with same form as x for the out of dtatabase prediction of the prob(xt)  		  #
#   Ex:													  #
#   ptest = model.query(xt)										  #
#													  #
#   OUTPUT: An object containing everything needed to define the model considering the data x_i's	  #
#													  #
#   model.x   :  the data										  #
#   model.n   :  the size of the database								  #
#   model.typeK  :  which kernel								          #
#   model.hSROT  :  Silverman's rule of thumb value of the bandwidth					  #
#   model.h      :  value of bandwidth from maximum likelihood CV					  #
###########################################################################################################
class KerDen:

	def __init__(self,x):
		self.x = x
		self.n = len(x)

	def train(self,typeK):
	#Finding the bandwidth h
		start = time.time()
		self.typeK = typeK
		#We choose a possible value as a Silverman's rule of thumb as in section 3.1 of
		#https://projecteuclid.org/download/pdfview_1/euclid.ss/1113832723
		#http://www.stat.washington.edu/courses/stat527/s13/readings/Sheather_StatSci_2004.pdf
		t = _SampStandDev_InterQuantile(self.x)
		h0 = 0.9*min([t[0],t[1]/1.34])*self.n**(-1./5.)
		#Keep it as part of the object for future used as well as the Max Likelihood 
		self.hSROT = h0
		#Distance between the x'i
		D2 = DistancesMatrix(self.x,self.x,self.n,self.n,T=True,sqr=False)#Eucl. dist. square
		tupARG = (D2)
		end = time.time()
		print 'Time before starting minimization = ', end-start
		#We assume that the bandwidth h should not be that differnt from h_SROT so we bound the method (with a somewhat 
		#large security factor) between hSROT/20 <= h <= 20*hSROT
		res = minimize_scalar(self.__LCV,bounds=(0.05*h0,20.*h0),args=tupARG, method='Brent',tol=1e-9)
		#res = minimize_scalar(self.__LCV,bounds=(0.01*h0,100.*h0),args=tupARG, method='Bounded',options={'xatol': 1e-09})
		self.h = res.x
		self.success = res.success
	def query(self,xt):
		d2 = DistancesMatrix(self.x,xt,self.n,xt.shape[0],sqr=False)
		d2 = d2/(self.h**2)
		KerT = _Kernel(d2,type=self.typeK)
		return np.sum(KerT,axis=1)/self.n/self.h
	def query_SROT(self,xt):
		d2 = DistancesMatrix(self.x,xt,self.n,xt.shape[0],sqr=False)
                d2 = d2/(self.hSROT**2)
                KerT = _Kernel(d2,type=self.typeK)
                return np.sum(KerT,axis=1)/self.n/self.hSROT

	def __LCV(self,h,D2):
		Ker = _Kernel(D2/(h**2),type=self.typeK)
		L = np.sum(Ker,axis=1).reshape((self.n,1))-np.diag(Ker).reshape((self.n,1))
		L1 = 1./self.n*np.sum(np.log(L))
		return -(L1-np.log( (self.n-1)*h ))
	
def _SampStandDev_InterQuantile(x):
	xm = np.mean(x)
	sstd = np.sqrt(np.sum((x-xm)**2)/(len(x)-1))
	#Calculating inter-quantile range following
	#http://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy/23229224
	iq = np.subtract(*np.percentile(x, [75, 25])) 
	return (sstd,iq)

def _Kernel(D2,type='Gau'):
#For the moment, the possible choices are Gaussain and Epanechnikov, Easy to add more
	if type=='Gau':
		return (1./np.sqrt(2.*np.pi))*np.exp(-D2/2.)
	elif type=='Epan':
		return (np.sqrt(D2)<=1)*3./4.*np.abs(1-D2)
		
		
