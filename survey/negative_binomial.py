""" negative_binomial.py

Functions and classes for negative binomial regression of count data. """

## For matrix manipulation
import numpy as np

## For model fitting
from scipy.optimize import minimize

## For evaluating the actual mass function
from scipy.special import gammaln

def pmf(k,mu,alpha):
	N = gammaln(k+(1./alpha))-gammaln(k+1)-gammaln((1./alpha))
	t1 = -(1./alpha)*np.log(1.+alpha*mu)
	t2 = k*np.log((alpha*mu)/(1.+alpha*mu))
	return np.exp(N+t1+t2)

class NegativeBinomialPosterior(object):

	""" Negative binomial count regression posterior class, for use in conjunction with 
	scipy.optimize to analyze over-dispersed count data. """

	def __init__(self,X,Y,alpha=None,lam=None,g=np.exp,gprime=np.exp):

		## Store the main regression problem geometry
		self.N, self.p = X.shape
		self.X = X
		self.Y = Y

		## And the link function
		self.g = g
		self.gprime = gprime

		## If alpha is none, set to a default value
		if alpha is None:
			emp_m = Y.mean()
			emp_v = Y.var()
			self.alpha = (emp_v/(emp_m**2)) - (1./emp_m)
		else:
			self.alpha = alpha

		## If the model is unregularized, set the ridge penalty
		## accordingly
		if lam is None:
			self.lam = np.zeros((self.p,self.p))
		else:
			self.lam = lam

		## Evaluate the constant prefactor
		self.const = gammaln(self.Y+(1./self.alpha))-gammaln(self.Y+1)-gammaln((1./self.alpha))
		self.const = self.const.sum()

	def __call__(self,beta):

		## Compute mu
		mu = self.g(np.dot(self.X,beta))

		## Collect the likelihood pieces
		ll = self.Y*np.log(self.alpha*mu)-(self.Y+(1./self.alpha))*np.log(1.+self.alpha*mu)

		## And the prior
		lp = -0.5*np.dot(beta.T,np.dot(self.lam,beta))

		return ll.sum() + lp + self.const

	def gradient(self,beta):

		## Compute mu and mu prime
		f_hat = np.dot(self.X,beta)
		mu = self.g(f_hat)
		muprime = self.gprime(f_hat)

		## Compute the prior component
		dlp = -np.dot(self.lam,beta)

		## And then the likelihood component
		dll = (self.Y*muprime)/mu -\
			  ((self.Y+(1./self.alpha))*(self.alpha*muprime))/(1.+self.alpha*mu)
		dll = np.dot(self.X.T,dll)

		return dll + dlp

def FitModel(log_post):

	""" A wrapper function that uses BFGS in scipy to fit the model. """

	result = minimize(lambda x: -log_post(x),
					  x0 = np.zeros((log_post.p,)),
					  method = "BFGS",
					  jac = lambda x: -log_post.gradient(x))

	return result