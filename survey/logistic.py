""" logistic.py

Class and fit functions, plus helpers to create ridge matrices. """

## For matrix manipulation
import numpy as np

## For model fitting
from scipy.optimize import minimize

## For regularization matrices
from scipy.linalg import block_diag

def logistic_function(x):
	return 1./(1. + np.exp(-x))

class LogisticRegressionPosterior(object):

	""" This class encapsulates the log posterior, it's derivative, and the
	hessian matrix as a function of the parameters. """

	def __init__(self,X,Y,p_i=None,lam=None):

		""" X = N x p array of covariates
			Y = N x 1 array of responses (must be 0 or 1)
			p_i = list of numbers of levels (i.e. columns) associated with
				  each feature class. First entry is assumed # of fixed effects
			lam = p x p precision matrix for L2/gaussian/ridge regularization 

			NB: I haven't implemented any exception catching, error handling, etc. so be 
			careful."""

		## Store problem geomtry parameters
		if p_i is None:
			self.N, self.p = X.shape
		else:
			self.N, _ = X.shape
			self.p_i = p_i
			self.p = sum(p_i)

		## Store the dataset
		self.X = X
		self.Y = Y

		## And set up the regularization matrix
		if lam is None:
			self.lam = np.zeros((self.p,self.p))
		else:
			self.lam = lam

	def __call__(self,beta):

		""" Evaluate the log posterior. """

		## Start by computing the linear component and 
		## activating it with the logistic funtion
		sigma = logistic_function(np.dot(self.X,beta))

		## Evaluate the full summand
		likelihood_components = self.Y*np.log(sigma) + (1.-self.Y)*np.log(1.-sigma)

		## Evaluate the prior components
		prior_components = -0.5*np.dot(beta.T,np.dot(self.lam,beta))

		return prior_components + np.sum(likelihood_components)

	def gradient(self,beta):

		""" Evaluate the gradient of the log posterior. """

		## Start with sigma
		sigma = logistic_function(np.dot(self.X,beta))

		## Evaluate the likelihood components
		likelihood_components = self.Y*(1.-sigma) - (1.-self.Y)*sigma

		## And the prior components
		prior_components = -np.dot(self.lam,beta)

		return prior_components + np.dot(self.X.T,likelihood_components)

	def hessian(self,beta):

		""" Evaluate the hessian of the log posterior. """

		## Start with sigma
		sigma = logistic_function(np.dot(self.X,beta))

		## Create a matrix of sigma derivatives
		d_sigma = np.diag(sigma*(1.-sigma))

		## Return the hessian
		return -self.lam - np.dot(self.X.T,np.dot(d_sigma,self.X))

def FitModel(log_post):

	""" A wrapper function that uses BFGS in scipy to fit the model. """

	result = minimize(lambda x: -log_post(x),
					  x0 = np.zeros((log_post.p,)),
					  method = "BFGS",
					  jac = lambda x: -log_post.gradient(x))

	return result

def HyperParameterFit(X,Y,p_i,effect_types,initial_guess=2.5):

	## Set up hyperparameter initial guess
	theta_0 = initial_guess*np.ones((len(p_i)-1,))

	## Define the over-arching cost function. This incorporates
	## the Jeffery's prior on theta
	def cost_function(theta):
		lam = RidgePenaltyMatrix(theta,p_i,effect_types)
		model = LogisticRegressionPosterior(X.values,Y.values,
												  lam=lam,p_i=p_i)
		result = FitModel(model)
		return -model(result.x)-np.sum(np.log(np.abs(theta)))

	## Minimize it w.r.t. hyper parameters
	result = minimize(cost_function,
					  x0 = theta_0,
					  method = "BFGS")

	return result

def RidgePenaltyMatrix(theta,p_i,effect_types=None):

	""" Function to create a ridge penalty matrix, i.e. the matrix for
	L2 regularization. theta is the sqrt(precision) for each block, p_i
	defines the length of each block, effect_types is a list with entries
	"fixed", "diagonal", or "random walk". 

	Note: p_i has assumed structure [# fixed effects, # re 1, ..., 3 re N] """

	## Start by adding the fixed effect theta to 
	## the input.
	adj_theta = np.hstack([[0],theta**2])

	## If effect types is not specified, random
	## intercepts are assumed.
	if effect_types is None:
		return np.diag(np.repeat(adj_theta,p_i))

	## Otherwise, we have to loop over effects
	ridge_matrix = []
	for i, effect in enumerate(effect_types):

		if effect == "fixed":
			ridge_matrix.append(np.zeros((p_i[i],p_i[i])))
		elif effect == "diagonal":
			ridge_matrix.append(adj_theta[i]*np.eye(p_i[i]))
		elif effect == "random walk":
			N = p_i[i]
			D2 = np.diag(N*[-2])+np.diag((N-1)*[1],k=1)+np.diag((N-1)*[1],k=-1)
			D2[0,2] = 1
			D2[-1,-3] = 1
			ridge_matrix.append(adj_theta[i]*np.dot(D2.T,D2))
		else:
			raise TypeError("Effect type must be fixed, diagonal, or random walk, not {}".format(effect))

	return block_diag(*ridge_matrix)