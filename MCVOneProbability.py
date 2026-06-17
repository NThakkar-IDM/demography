""" MCVOneProbability.py

Estimating the probability of getting an MCV one dose by birthdate and location
in Nigeria. """
import sys
import survey

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

## For regression estimates
import survey.logistic as lr

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

## For reference
colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return


if __name__ == "__main__":

    # Get the relevant DHS data
    kr_columns = ["caseid", "bord", "child_DoB", "live_child", 
                  "mom_DoB", "interview_date", "mcv1"]
    krs = {sio.infer_survey_name(path): sio.load_survey(path, False, True, kr_columns)
            for path in sio.dhs_kr_paths}
    ir_columns = ["caseid", "mom_edu", "state", "area", "region", "num_brs", "mom_age"]
    irs = {sio.infer_survey_name(path): sio.load_survey(path, True, True, ir_columns)
            for path in sio.dhs_ir_paths}
    
    # Merge them and put it all together
    dhs = []
    for k in krs.keys():
        this_dhs = krs[k].merge(irs[k],
                    on="caseid",
                    how="left",
                    validate="m:1",
                    )
        dhs.append(this_dhs)
    dhs = pd.concat(dhs,axis=0)
    dhs = dhs.sort_values(["survey","caseid","bord"]).reset_index(drop=True)
    print("\nThe DHS data for this analysis:")
    print(dhs)

    # Now shift to the MICS data
    ch_columns = ["cluster", "hh", "line_num", 
                  "child_birth_day", "child_birth_mon","child_birth_year",
                  "child_age",
                  "mcv1"]
    chs = {sio.infer_survey_name(path): sio.load_survey(path, False, True, ch_columns)
            for path in sio.mics_ch_paths}
    wm_columns = ["cluster", "hh", "line_num",
                  "interview_day","interview_mon","interview_year",
                  "mom_birth_mon","mom_birth_year", 
                  "mom_edu", "area", "state","region"]
    wms = {sio.infer_survey_name(path): sio.load_survey(path, True, True, wm_columns)
            for path in sio.mics_wm_paths}
    
    # Merge them and put it all together 
    mics = []
    for k in chs.keys():
        this_mics = chs[k].merge(wms[k],
                    on=["cluster","hh","line_num"],
                    how="left",
                    validate="m:1",
                    )
        mics.append(this_mics)
    mics = pd.concat(mics,axis=0).reset_index(drop=True)
    print("\nThe MICS data for this analysis:")
    print(mics)
    
    ## Subset to children alive and within the 12 to 35 month
    ## age range
    dhs["age"] = dhs["interview_date"] - dhs["child_DoB"]
    dhs = dhs.loc[(dhs["live_child"] == "yes") &\
                  (dhs["age"] >= 12) &\
                  (dhs["age"] < 24)]

    ## Set up the MICS birth dates and ages
    mics["child_DoB"] = pd.to_datetime({"month":mics["child_birth_mon"],
                                      "year":mics["child_birth_year"],
                                      "day":mics["child_birth_day"]},errors="coerce")
    mics["interview_date"] = pd.to_datetime({"month":mics["interview_mon"],
                                      "year":mics["interview_year"],
                                      "day":mics["interview_day"]})
    mics["age"] = 12*((mics["interview_date"] - mics["child_DoB"]).dt.days/365.).fillna(mics["child_age"])
    mics = mics.loc[(mics["age"] >= 12) &\
                    (mics["age"] < 24)]

    ## month-year stamps
    dhs["bday"] = sio.cmc_to_datetime(dhs["child_DoB"])
    dhs["time"] = dhs["bday"].apply(lambda d: f"{d.year}-{d.month:02}")
    mics["time"] = mics["child_DoB"].apply(lambda d: f"{d.year}-{d.month:02}")\
                        .replace("nan-nan",np.nan)

    ## Merge the two data sets
    variables = ["survey","area","region","state","mom_edu","time","mcv1"]
    df = pd.concat([dhs[variables],
                    mics[variables]],axis=0)
    df = df.loc[(df.notnull().all(axis=1))].reset_index(drop=True)
    print("\nCleaned and compiled dataset...")
    print(df)
    
    ## Standardize values across surveys:
    print("\nVariable values...")
    for c in variables:
        values = sorted(df[c].unique())
        print("{} values ({} of them) = {}".format(c,len(values),values))
    
    ## Set up the specifics of the regression problem
    response = "mcv1"
    features = ["area","mom_edu","time"]
    full_time = [f"{y}-{m:02}" for y in np.arange(sio.YEAR_MIN,sio.YEAR_MAX+1) \
                                      for m in np.arange(1,13)]
    #full_time = full_time[7:]
    full_time = set(full_time)
    general_correlation_time = (24**4)/8.
    corr_time = {"borno":(12**4)/8,"ebonyi":(12**4)/8,"gombe":(12**4)/8,
                 "lagos":(12**4)/8,"oyo":(12**4)/8,"zamfara":(12**4)/8,
                 "osun":(12**4)/8}
    reference = ["urban","primary",f"time:{sio.YEAR_MIN}-01"]

    ## Set up the GP correlation matrix
    T = len(full_time)-1
    D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
    #D2[0,2] = 1
    D2[-1,-3] = 1
    RW2 = np.dot(D2.T,D2)#*correlation_time

    ## Loop over states and compile models
    print("\nStarting the loop over states...")
    output = {}
    covariances = {}
    for name, sf in df.groupby("state"):
        
        ## Set up the regression problem's design matrix and 
        ## response vector
        Y = sf[response].astype(float)
        X = []
        for f in features:
            this_f = pd.get_dummies(sf[f],dtype=float)
            this_f.columns = [str(c) for c in this_f.columns]
            if f == "time":
                missing_times = full_time-set(this_f.columns)
                for c in missing_times:
                    this_f[c] = np.zeros((len(this_f),))
                this_f = this_f[sorted(this_f.columns)]
                this_f.columns = "time:"+this_f.columns
            X.append(this_f)
        X = pd.concat(X,axis=1)
        #X["bord"] = sf["bord"].copy().astype(float)

        ## Define the intercept
        X["intercept"] = np.ones((len(X),))
        X = X[["intercept"] + X.columns[:-1].tolist()]
        X = X.drop(columns=reference)
        p = len(X.columns)

        ## Specify the full regularization matrix
        lam = np.zeros((p,p))
        #lam[-T:,-T:] = RW2*corr_time.get(name,
        #                   general_correlation_time)
        lam[-T:,-T:] = RW2*general_correlation_time

        ## Set up the regression posterior and solve the problem
        log_post = lr.LogisticRegressionPosterior(X.values,
                                                  Y.values,
                                                  lam=lam,
                                                  )
        result = lr.FitModel(log_post)

        ## Output a status
        print("...in {}, success = {}".format(name,result.success))

        ## Report on the results
        beta_hat = result["x"]
        beta_cov = result["hess_inv"]
        beta_err = np.sqrt(np.diag(beta_cov))
        beta = pd.DataFrame(np.array([beta_hat,beta_err]).T,
                            columns=["beta","beta_err"],
                            index=X.columns)

        ## Add reference category placeholders and save the
        ## result
        beta = pd.concat([beta,
                          pd.DataFrame(np.zeros((len(reference),2)),
                                       index=reference,columns=beta.columns)],
                          axis=0)
        output[name] = beta

        ## Also store the covariance matrix
        covariances[name] = pd.DataFrame(beta_cov,
                                         index=X.columns,
                                         columns=X.columns)

    ## Put it all together
    output = pd.concat(output.values(),keys=output.keys())
    print("\nFinal output:")
    print(output)
    output.to_pickle(os.path.join(
        "pickle_jar",
        "mcv1_logistic_regression_by_state.pkl"))

    ## And the full covariance matrices, since sometimes you need
    ## the whole (gaussian approx) to the regression posterior
    covariances = pd.concat(covariances.values(),keys=covariances.keys())
    print("\nAssociated covariance matricies:")
    print(covariances)
    covariances.to_pickle(os.path.join(
        "pickle_jar",
        "mcv1_logistic_regression_covariances_by_state.pkl"))


