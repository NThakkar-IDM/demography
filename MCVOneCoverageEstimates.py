""" MCVOneCoverageEstimates.py

Using the distributions estimated in MCVOneProbability.py to estimate state
level coverage across birth cohorts.  """
import sys

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import pandas as pd

## For regression estimates
import survey.logistic as lr

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

if __name__ == "__main__":

	## Get the demographic cell populations
	dist = pd.read_pickle(os.path.join(
		"pickle_jar",
		"mom_distribution.pkl"))

	## Interpolate to the monthly scale
	time = pd.date_range(
		start=f"{sio.YEAR_MIN}-01-01",
		end=f"{sio.YEAR_MAX}-12-01",
		freq="MS",name="time")
	dist["year"] = pd.to_datetime({"year":dist["year"],
								   "month":6,"day":1})
	dist = dist.set_index([c for c in dist.columns if c != "weight"]).sort_index()["weight"]
	dist = dist.groupby([n for n in dist.index.names if n != "year"],
						observed=False).apply(
						lambda s: s.loc[s.name].reindex(time).interpolate(limit_direction="both")
						)
	dist = dist.reset_index()

	## Set the problem up in terms of state-time pairs, since we're going to
	## marginalize across the rest of the pieces
	dist = dist.set_index(["state","time"]).sort_index()
	
	## Get the logistic regression results
	lr_results = pd.read_pickle(os.path.join(
		"pickle_jar",
		"mcv1_logistic_regression_by_state.pkl"))
	lr_results["beta_var"] = lr_results["beta_err"]**2

	## And the associated covariance matrices
	covariances = pd.read_pickle(os.path.join(
		"pickle_jar",
		"mcv1_logistic_regression_covariances_by_state.pkl"))

	## Extract some details of the regression problem
	## from the serialized outputs
	dummy_state = lr_results.index.get_level_values(0)[0]
	time_covs = covariances.loc[dummy_state].index.str.startswith("time:")
	time_covs = covariances.loc[dummy_state].loc[time_covs].index
	num_fe = covariances.loc[dummy_state].shape[0]-len(time_covs)
	fixed_ef = lr_results.loc[dummy_state].index[:num_fe]
	reference = lr_results.loc[dummy_state].index[len(time_covs)+num_fe:]
	
	## Loop over cells to compute distribution parameters
	## for all possible combinations.
	print("\nComputing estimates by state/time...")
	output = []
	for st, s in dist.groupby(["state","time"]):

		## Unpack this subset
		state, dt = st
		t = f"time:{dt.year}-{dt.month:02}"

		## Normalize the weights
		weights = s.copy()
		weights["weight"] *= 1./(weights["weight"].sum())

		## Subset to the right state
		beta = lr_results.loc[state]
		beta_cov = covariances.loc[state]

		## Then compute the mom types
		mom_x = pd.get_dummies(weights[["area","mom_edu"]],
							   prefix="",prefix_sep="").astype(float)
		mom_x["intercept"] = np.ones((len(mom_x),))
		mom_x = mom_x[fixed_ef]

		## Add the time component
		X_time = pd.DataFrame(np.zeros((len(mom_x),len(time_covs))),
							  index=mom_x.index,
							  columns=time_covs)
		if t not in reference:
			X_time.loc[:,t] = 1.
		
		## Put it all together and compute
		mom_x = pd.concat([mom_x,X_time],axis=1).values
		lno = np.dot(mom_x,beta["beta"].values[:num_fe+len(time_covs)])
		var = np.diag(np.dot(np.dot(mom_x,beta_cov.values),mom_x.T))

		## Finally, compute probabilites and variances
		p_vax = lr.logistic_function(lno)
		p_var = var*(p_vax**2)*((1.-p_vax)**2)

		## And weight and store them
		p_vax = (weights["weight"]*p_vax).sum()
		p_var = ((weights["weight"]**2)*p_var).sum()

		## Store the results
		output.append((state,
					   dt,
					   p_vax,
					   p_var))

	## Put it all together
	print("\nFinal output...")
	ri = pd.DataFrame(output,
					  columns=["state","time","mcv","var"])
	print(ri)	
	ri.to_pickle(os.path.join(
		"pickle_jar",
		"mcv1_lr_estimate_by_state.pkl"))

