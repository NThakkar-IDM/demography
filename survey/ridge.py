""" ridge.py

Generic ridge regression and associated methods. """

## For matrix manipulation
import numpy as np

def gaussian_pdf(x,mu,var):
	return np.exp(-0.5*((x-mu)**2)/var)/np.sqrt(2.*np.pi*var)

def log_normal_density(x,mu,var):
	return np.exp(-((np.log(x)-mu)**2)/(2.*var))/(x*np.sqrt(2.*np.pi*var))

class RidgeRegression(object):

	""" Gaussian/ridge regression posterior class, for use in inference and regression
	on data distributed on the real line. Unlike many posterior classes, this one leverages the
	analytic solution to fit the model automatically. """

	def __init__(self,X,Y,lam=None):

		## Store the main regression problem geometry
		self.N, self.p = X.shape
		self.X = X
		self.Y = Y

		## If the model is unregularized, set the ridge penalty
		## accordingly
		if lam is None:
			self.lam = np.zeros((self.p,self.p))
		else:
			self.lam = lam

		## Solve the regression problem
		self.C = np.linalg.inv(np.dot(self.X.T,self.X)+self.lam)
		self.H = np.dot(self.C,self.X.T)

		## Compute key quantities
		## First the optimal parameter values
		self.beta_hat = np.dot(self.H,self.Y)

		## Then the residuals
		Y_hat = np.dot(self.X,self.beta_hat)
		self.resid = self.Y-Y_hat
		
		## And finally the covariance matrix
		RSS = np.sum(self.resid**2)
		self.var = RSS/(len(self.Y)-self.p)
		self.beta_cov = self.var*self.C