""" zero_inflated_nb.py

Objects and methods for fitting zero-inflated negative binomial regression models. Note, this
leverages the logistic regression implementation in logistic.py to handle the probability of
zeros. """
import sys

## For matrix manipulation
import numpy as np

## For model fitting
from scipy.optimize import minimize

## For evaluating the actual mass function
from scipy.special import gammaln

## For the logistic regression portions of the model
import survey.logistic as lr

def nb_pmf(k,mu,alpha):
	N = gammaln(k+(1./alpha))-gammaln(k+1)-gammaln((1./alpha))
	t1 = -(1./alpha)*np.log(1.+alpha*mu)
	t2 = k*np.log((alpha*mu)/(1.+alpha*mu))
	return np.exp(N+t1+t2)

def logistic_function(x):
	return 1./(1. + np.exp(-x))

def pmf(k,mu_lr,mu_nb,alpha):

	## Compute the components
	zero_pr = 1. - logistic_function(mu_lr)
	pr_k = nb_pmf(k,mu_nb,alpha)

	## Put them together
	pr_k = pr_k/(1.-pr_k[0])
	pr_k[0] = zero_pr
	pr_k[1:] *= 1.-zero_pr

	return pr_k

class ZeroInflatedNBPosterior(object):

	""" Zero inflated negative binomial regression, for count data with a seperate model associated with
	pr(zero). 

	This contains two sub-classes, a logistic regression posterior and a non-zero negative binomial posterior,
	- the call method leverages those. """

	def __init__(self,X,Y,alpha=None,lam=None,g=np.exp,gprime=np.exp):

		## Store the main regression problem geometry
		self.N, self.p = X.shape
		self.X = X
		self.Y = Y

		## Set up the logistic regression class
		zero_mask = (Y != 0)
		self.zero_or_not = zero_mask.copy().astype(float)
		self.lr_post = lr.LogisticRegressionPosterior(X,self.zero_or_not,lam=lam)

		## And the negative binomial class, for the non-zero
		## values.
		self.nb_post = NonZeroNegativeBinomial(X[zero_mask],
											   Y[zero_mask],
											   alpha=alpha,lam=lam,g=g,gprime=gprime)

	def __call__(self,lr_theta,nb_theta):

		return self.lr_post(lr_theta) + self.nb_post(nb_theta)

	def gradient(self,theta):

		raise NotImplementedError("""Gradient is purposefully not-implemented. Both the logistic
									 and NB portions of this model need to be fit independently""")


class NonZeroNegativeBinomial(object):

	""" Negative binomial count regression posterior class, for use in conjunction with 
	scipy.optimize to analyze over-dispersed count data. This is modified to only work with
	non-zero values, as a subclass to the zero-inflated NB class above. """

	def __init__(self,X,Y,alpha=None,lam=None,g=np.exp,gprime=np.exp):

		## Store the main regression problem geometry
		self.N, self.p = X.shape
		self.X = X
		self.Y = Y
		assert (Y != 0).all(), "0 values have to be handled seperately"

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

		## Add the zero-inflating
		ll += -np.log(1. - ((1.+self.alpha*mu)**(-1./self.alpha)))

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
			  ((self.Y+(1./self.alpha))*(self.alpha*muprime))/(1.+self.alpha*mu)-\
			  muprime*((1.+self.alpha*mu)**(-1.-(1./self.alpha)))/(1.-((1.+self.alpha*mu)**(-1./self.alpha)))
		dll = np.dot(self.X.T,dll)

		return dll + dlp

def FitModel(log_post):

	""" A wrapper function that uses BFGS in scipy to fit the model. """

	## Step 1, fit the logistic regression.
	lr_result = minimize(lambda x: -log_post.lr_post(x),
						 x0 = np.zeros((log_post.lr_post.p,)),
						 method = "BFGS",
						 jac = lambda x: -log_post.lr_post.gradient(x))

	## Step 2: fit the NB portion
	nb_result =  minimize(lambda x: -log_post.nb_post(x),
						  x0 = np.zeros((log_post.nb_post.p,)),
						  method = "BFGS",
						  jac = lambda x: -log_post.nb_post.gradient(x),
						  )

	return lr_result, nb_result