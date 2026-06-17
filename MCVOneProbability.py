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

def cms_to_datetime(a,d=15):
    assert (a >= 1).all(), "CMS format must be >= 1"
    years = 1900+((a-1)/12).astype(int)
    months = a - 12*(years-1900)
    dates = pd.to_datetime({"year":years,
                            "month":months,
                            "day":d})
    return dates

## Survey I/O
def get_a_dhs(path,columns,convert_categoricals=True,add_survey=False):
    df = pd.read_stata(path,
                       columns=columns.keys(),
                       convert_categoricals=convert_categoricals)
    df["caseid"] = df["caseid"].str.strip()

    ## Rename according to the schema
    df.columns = [columns[c] for c in df.columns]

    if add_survey:
        df["survey"] = path.split(os.path.sep)[1].lower()
    return df

def get_a_mics(path,columns,convert_categoricals=True,add_survey=False):
    df = pd.read_spss(path,
                      usecols=columns.keys(),
                      convert_categoricals=convert_categoricals)

    ## Rename according to the schema
    df.columns = [columns[c] for c in df.columns]

    if add_survey:
        df["survey"] = path.split(os.path.sep)[1].lower()
    return df

## DHS related schemas, which map a file path (recode) to the columns
## needed from the recode.
## Kid recode
kr_schema = {
    os.path.join("_surveys","DHS5_2008","NGKR53DT","NGKR53FL.DTA"):
        {
        "caseid":"caseid", ## mom id
        "bord":"bord", ## child birth order
        "b3":"child_DoB", #CMC DoB of child
        "b5":"live_child", # is the childe alive?
        "v011":"mom_DoB", #CMC DoB of mom
        "v008":"interview_date", #Interview CMC date
        "h9":"mcv1", # measles vaccine 1
        "h1":"has_card", # has a vaccine card
        "h9d":"mcv1_day", ## Recieved Measles 1
        "h9m":"mcv1_mon", ## Recieved Measles 1
        "h9y":"mcv1_yr", ## Recieved Measles 1
        },
    os.path.join("_surveys","DHS6_2013","NGKR6ADT","NGKR6AFL.DTA"):
        {
        "caseid":"caseid", ## mom id
        "bord":"bord", ## child birth order
        "b3":"child_DoB", #CMC DoB of child
        "b5":"live_child", # is the childe alive?
        "v011":"mom_DoB", #CMC DoB of mom
        "v008":"interview_date", #Interview CMC date
        "h9":"mcv1", # measles vaccine 1
        "h1":"has_card", # has a vaccine card
        "h9d":"mcv1_day", ## Recieved Measles 1
        "h9m":"mcv1_mon", ## Recieved Measles 1
        "h9y":"mcv1_yr", ## Recieved Measles 1
        },
    os.path.join("_surveys","DHS7_2018","NGKR7ADT","NGKR7AFL.DTA"):
        {
        "caseid":"caseid", ## mom id
        "bord":"bord", ## child birth order
        "b3":"child_DoB", #CMC DoB of child
        "b5":"live_child", # is the childe alive?
        "v011":"mom_DoB", #CMC DoB of mom
        "v008":"interview_date", #Interview CMC date
        "h9":"mcv1", # measles vaccine 1
        "h1":"has_card", # has a vaccine card
        "h9d":"mcv1_day", ## Recieved Measles 1
        "h9m":"mcv1_mon", ## Recieved Measles 1
        "h9y":"mcv1_yr", ## Recieved Measles 1
        },
}

## Individual recode
ir_schema = {
    os.path.join("_surveys","DHS5_2008","NGIR53DT","NGIR53FL.DTA"):
        {
         "caseid":"caseid", ## mom id
         "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
         "v023":"strata", ## sample strata (state, or region+state+U/R)
         "v024":"region", ## region
         "v025":"area", ## urban/rural
         "v224":"num_brs", ## number of entries in br recode
         "v013":"age_bin", ## age in 5 year bins
         "v005":"weight", #Mom's sample weight
        },
    os.path.join("_surveys","DHS6_2013","NGIR6ADT","NGIR6AFL.DTA"):
        {
         "caseid":"caseid", ## mom id
         "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
         "v023":"strata", ## sample strata (state, or region+state+U/R)
         "v024":"region", ## region
         "v025":"area", ## urban/rural
         "v224":"num_brs", ## number of entries in br recode
         "v013":"age_bin", ## age in 5 year bins
         "v005":"weight", #Mom's sample weight
        },
    os.path.join("_surveys","DHS7_2018","NGIR7ADT","NGIR7AFL.DTA"):
        {
         "caseid":"caseid", ## mom id
         "v106":"mom_edu", ## mom's educational attainment (v106 is low res, 149 is high)
         "v023":"strata", ## sample strata (state, or region+state+U/R)
         "v024":"region", ## region
         "v025":"area", ## urban/rural
         "v224":"num_brs", ## number of entries in br recode
         "v013":"age_bin", ## age in 5 year bins
         "v005":"weight", #Mom's sample weight
        },
}

## MICS related schemas, which map a file path (recode) to the columns
## needed from the recode.
## child recode
ch_schema = {
    os.path.join("_surveys","MICS5_2016","ch.sav"):
        {
         "HH1":"cluster", ## woman cluster
         "HH2":"hh", ## woman hh
         "UF6":"line_num", ## line number for mom
         "UF9":"complete_interview", ## interview completeness
         "AG1D":"child_birth_day", ## child birth day
         "AG1M":"child_birth_mon", ## child birth mon
         "AG1Y":"child_birth_year", ## child birth year
         "AG2":"child_age", ## Age to nearest year
         "IM3MD":"mcv_day", ## card based day
         "IM3MM":"mcv_mon", ## card based mon
         "IM3MY":"mcv_year", ## card based day
         "IM16":"mcv1", ## Ever recieved, recall based
        },
    os.path.join("_surveys","MICS6_2021","ch.sav"):
        {
         "HH1":"cluster", ## woman cluster
         "HH2":"hh", ## woman hh
         "UF4":"line_num", ## line number for mom
         "UF17":"complete_interview", ## interview completeness
         "UB1D":"child_birth_day", ## child birth day
         "UB1M":"child_birth_mon", ## child birth mon
         "UB1Y":"child_birth_year", ## child birth year
         "UB2":"child_age", ## Age to nearest year
         "IM6N1D":"mcv_day", ## card based day
         "IM6N1M":"mcv_mon", ## card based mon
         "IM6N1Y":"mcv_year", ## card based day
         "IM26":"mcv1", ## Ever recieved, recall based
        },
}

## women's recode
wm_schema = {
    os.path.join("_surveys","MICS5_2016","wm.sav"):
        {
         "HH1":"cluster", ## woman cluster
         "HH2":"hh", ## woman hh
         "LN":"line_num", ## line number
         "WM6D":"interview_day", ## interview day
         "WM6M":"interview_mon", ## interview month
         "WM6Y":"interview_year", ## interview year
         "WB1M":"mom_birth_mon", ## mom's birth month
         "WB1Y":"mom_birth_year", ## mom's birth year
         "WB4":"mom_edu", ## mom's education
         "HH6":"area", ## urban/rural
         "HH7":"state", ## state 
         "Zone":"region", ## region
         "wmweight":"weight", ## mom's sample weight
        },
    os.path.join("_surveys","MICS6_2021","wm.sav"):
        {
         "HH1":"cluster", ## woman cluster
         "HH2":"hh", ## woman hh
         "LN":"line_num", ## line number
         "WM6D":"interview_day", ## interview day
         "WM6M":"interview_mon", ## interview month
         "WM6Y":"interview_year", ## interview year
         "WB3M":"mom_birth_mon", ## mom's birth month
         "WB3Y":"mom_birth_year", ## mom's birth year
         "WB6A":"mom_edu", ## mom's education
         "HH6":"area", ## urban/rural
         "HH7":"state", ## state 
         "zone":"region", ## region
         "wmweight":"weight", ## mom's sample weight
        },
}

if __name__ == "__main__":

    ## Get the data via pandas, brs then irs
    krs = {path.split(os.path.sep)[1].lower()\
            :get_a_dhs(path,columns,True,True) 
            for path, columns in kr_schema.items()}
    irs = {path.split(os.path.sep)[1].lower():\
            get_a_dhs(path,columns,True,False)
            for path, columns in ir_schema.items()}

    ## Merge them and put it all together
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

    ## Now shift over to the MICS datasets, starting with the
    ## birth recodes
    chs = {path.split(os.path.sep)[1].lower()\
            :get_a_mics(path,columns,True,True) 
            for path, columns in ch_schema.items()}
    wms = {path.split(os.path.sep)[1].lower()\
            :get_a_mics(path,columns,True,False) 
            for path, columns in wm_schema.items()}

    ## Merge them and put it all together (why do 7k kids not
    ## have caregivers i.e. clusters without surveyed moms ??)
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
    dhs["live_child"] = dhs["live_child"].str.lower()
    dhs["age"] = dhs["interview_date"] - dhs["child_DoB"]
    dhs = dhs.loc[(dhs["live_child"] == "yes") &\
                  (dhs["age"] >= 12) &\
                  (dhs["age"] < 24)]

    ## Set up the MICS birth dates and ages
    mics["child_birth_mon"] = mics["child_birth_mon"].str.lower()\
                                .replace("no response",np.nan)\
                                .replace("missing",np.nan)\
                                .replace("inconsistent",np.nan)\
                                .map({"january":1,"february":2,"march":3,"april":4,
                                      "may":5,"june":6,"july":7,"august":8,
                                      "september":9,"october":10,"november":11,"december":12})
    mics["child_DoB"] = pd.to_datetime({"month":mics["child_birth_mon"],
                                      "year":mics["child_birth_year"],
                                      "day":mics["child_birth_day"]},errors="coerce")
    mics["interview_mon"] = mics["interview_mon"].str.lower()\
                            .map({"january":1,"february":2,"march":3,"april":4,
                                  "may":5,"june":6,"july":7,"august":8,
                                  "september":9,"october":10,"november":11,"december":12})
    mics["interview_date"] = pd.to_datetime({"month":mics["interview_mon"],
                                      "year":mics["interview_year"],
                                      "day":mics["interview_day"]})
    mics["age"] = 12*((mics["interview_date"] - mics["child_DoB"]).dt.days/365.).fillna(mics["child_age"])
    mics = mics.loc[(mics["age"] >= 12) &\
                    (mics["age"] < 24)]
    
    ## Process the vaccination history into a simpler column
    ## for use in modeling.
    dhs["mcv1"] = dhs["mcv1"].str.lower()
    interpretation = { # based on krs["h9"].value_counts(dropna=False)
                      "no":0,
                      "reported by mother":1,
                      "vaccination date on card":1,
                      "vacc. date on card":1,
                      "vaccination marked on card":1,
                      "dk":0,
                      np.nan:0,
                      "vacc. marked on card":1,
                      "don't know":0
                      }
    dhs["mcv"] = dhs["mcv1"].apply(interpretation.get)

    ## Process vaccination history for the mics
    mics["mcv1"] = mics["mcv1"].str.lower()\
                        .replace("dk",np.nan)\
                        .replace("no response",np.nan)\
                        .replace("missing",np.nan)
    mics["mcv_with_card"] = pd.to_numeric(mics["mcv_year"],errors="coerce").notnull()
    mics["mcv"] = (mics["mcv_with_card"] | (mics["mcv1"] == "yes")).astype(float)

    ## Create a column for the state
    state_regex = r"^[nsNS][ewcsEWCS]\s(.*)\s\b(?:urban|rural|Urban|Rural)\b"
    dhs["state"] = dhs["strata"].str.extract(state_regex)[0].str.lower()
    dhs["state"] = dhs["state"].fillna(dhs["strata"]).str.replace("fct abuja","abuja")
    mics["state"] = mics["state"].str.lower().str.replace("fct abuja","abuja").str.replace("fct","abuja")

    ## Clean up the area covariate
    mics["area"] = mics["area"].str.lower()
    mics["region"] = mics["region"].str.lower()

    ## And the education covariate
    mics["mom_edu"] = mics["mom_edu"].str.lower()
    edu_cats = {"secondary / secondary-technical":"secondary",
                np.nan:"no education",
                "non-formal":"no education",
                "preschool":"primary",
                "higher/tertiary":"higher",
                "senior secondary":"secondary",
                "junior secondary":"secondary",
                "secondary technical":"secondary",
                "eccde":"no education",
                "vei/iei":"no education",
                "no response":"no education",
                }
    mics["mom_edu"] = mics["mom_edu"].apply(lambda c: edu_cats.get(c,c))

    ## month-year stamps
    dhs["bday"] = cms_to_datetime(dhs["child_DoB"])
    dhs["time"] = dhs["bday"].apply(lambda d: f"{d.year}-{d.month:02}")
    mics["time"] = mics["child_DoB"].apply(lambda d: f"{d.year}-{d.month:02}")\
                        .replace("nan-nan",np.nan)

    ## Merge the two data sets
    variables = ["survey","area","region","state","mom_edu","time","mcv"]
    df = pd.concat([dhs[variables],
                    mics[variables]],axis=0)
    df = df.loc[(df.notnull().all(axis=1))].reset_index(drop=True)
    #df = dhs[variables].copy()
    print("\nCleaned and compiled dataset...")
    print(df)

    ## Standardize values across surveys:
    print("\nVariable values...")
    for c in variables:
        values = sorted(df[c].unique())
        print("{} values ({} of them) = {}".format(c,len(values),values))
    
    ## Set up the specifics of the regression problem
    response = "mcv"
    features = ["area","mom_edu","time"]
    full_time = [f"{y}-{m:02}" for y in np.arange(2006,2024) \
                                      for m in np.arange(1,13)]
    #full_time = full_time[7:]
    full_time = set(full_time)
    general_correlation_time = (24**4)/8.
    corr_time = {"borno":(12**4)/8,"ebonyi":(12**4)/8,"gombe":(12**4)/8,
                 "lagos":(12**4)/8,"oyo":(12**4)/8,"zamfara":(12**4)/8,
                 "osun":(12**4)/8}
    reference = ["urban","primary","time:2006-01"]

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
    output.to_pickle("pickle_jar\\mcv1_logistic_regression_by_state.pkl")

    ## And the full covariance matrices, since sometimes you need
    ## the whole (gaussian approx) to the regression posterior
    covariances = pd.concat(covariances.values(),keys=covariances.keys())
    print("\nAssociated covariance matricies:")
    print(covariances)
    covariances.to_pickle("pickle_jar\\mcv1_logistic_regression_covariances_by_state.pkl")


