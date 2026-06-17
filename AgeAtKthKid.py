""" AgeAtKthKid.py

Log-normal regression to estimate a Mom's age at the time of their
child's birth. """
import sys

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For model fitting
import survey.ridge as rr

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

if __name__ == "__main__":

    ## Get the data via pandas, brs then irs
    br_columns = ["caseid","bord","child_DoB","mom_DoB",
                  "interview_date","weight"]
    brs = {sio.infer_survey_name(path): sio.load_survey(path, False, True, br_columns)
            for path in sio.dhs_br_paths}
    ir_columns = ["caseid","mom_edu","state","region","area"]
    irs = {sio.infer_survey_name(path): sio.load_survey(path, True, True, ir_columns)
            for path in sio.dhs_ir_paths}
    
    ## Merge them and put it all together
    dhs = []
    for k in brs.keys():
        this_dhs = brs[k].merge(irs[k],
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
    bh_columns = ["cluster","hh","line_num","birth_ln",
                  "child_birth_mon","child_birth_year"]
    bhs = {sio.infer_survey_name(path): sio.load_survey(path, False, True, bh_columns)
            for path in sio.mics_bh_paths}
    wm_columns = ["cluster","hh","line_num","interview_day","interview_mon",
                  "interview_year","mom_birth_mon","mom_birth_year",
                  "mom_edu","area","state","region","weight"]
    wms = {sio.infer_survey_name(path): sio.load_survey(path, True, True, wm_columns)
            for path in sio.mics_wm_paths}

    ## Merge them and put it all together
    mics = []
    for k in bhs.keys():
        this_mics = bhs[k].merge(wms[k],
                    on=["cluster","hh","line_num"],
                    how="left",
                    validate="m:1",
                    )
        mics.append(this_mics)
    mics = pd.concat(mics,axis=0).reset_index(drop=True)
    print("\nThe MICS data for this analysis:")
    print(mics)

    ## Create a year covariate from the interview date for both
    ## The dhs and the mics
    dhs["year"] = sio.cmc_to_datetime(dhs["interview_date"]).dt.year.astype(str)
    mics["year"] = mics["interview_year"].astype(int).astype(str)

    ## Correct the birth order column, which in the MICS is binned, with the
    ## sequential order as the birth_ln.
    mics["bord"] = mics["birth_ln"].astype(int)

    ## Correct for twins, keeping the one with the lower birth order
    mics = mics.loc[~mics[
                ["survey","cluster","hh","line_num",
                "child_birth_mon","child_birth_year"]].duplicated(keep="first")]
    dhs = dhs.loc[~dhs[["survey","caseid","child_DoB"]].duplicated(keep="first")]
    
    ## Make a Mom's age covariate
    mics["mom_DoB"] = pd.to_datetime({"month":mics["mom_birth_mon"],
                                      "year":mics["mom_birth_year"],
                                      "day":1})
    mics["child_DoB"] = pd.to_datetime({"month":mics["child_birth_mon"],
                                      "year":mics["child_birth_year"],
                                      "day":1})
    mics["mom_age"] = (mics["child_DoB"]-mics["mom_DoB"]).dt.days/365.
    dhs["mom_age"] = (dhs["child_DoB"]-dhs["mom_DoB"])/12.

    ## Choose covariates
    variables = ["survey","area","region","state","year","mom_edu","bord","mom_age"]
    df = pd.concat([dhs[variables],
                    mics[variables]],axis=0)
    df = df.loc[(df.notnull().all(axis=1)) &\
                (df["mom_age"] > 5)].reset_index(drop=True)
    print("\nCleaned and compiled dataset...")
    print(df)

    ## Standardize values across surveys:
    print("\nVariable values...")
    for c in variables[:-1]:
        values = sorted(df[c].unique())
        print("{} values ({} of them) = {}".format(c,len(values),values))

    ## Add mom's age at birth, and compute the intended
    ## response variables
    df["ln_mom_age"] = np.log(df["mom_age"])
    
    ## Set up the specifics of the regression problem
    response = "ln_mom_age"
    features = ["area","mom_edu","year"]
    correlation_time = (3**4)/8.
    reference = ["urban","primary",f"year:{sio.YEAR_MIN}"]

    ## Loop over states and compile models
    print("\nStarting the loop over states...")
    output = {}
    for name, sf in df.groupby("state"):

        ## Update the states
        print("...fitting in {}".format(name))
        
        ## Set up the regression problem's design matrix and 
        ## response vector
        Y = sf[response].astype(float)
        X = []
        missing_year_strings = sorted(list(set(
            np.arange(sio.YEAR_MIN,sio.YEAR_MAX+1).astype(str))\
                                -set(sf["year"].unique())))
        for f in features:
            this_f = pd.get_dummies(sf[f],dtype=float)
            this_f.columns = [str(c) for c in this_f.columns]
            if f == "year":
                for c in missing_year_strings:
                    this_f[c] = np.zeros((len(this_f),))
                this_f = this_f[sorted(this_f.columns)]
                this_f.columns = "year:"+this_f.columns
            X.append(this_f)
        X = pd.concat(X,axis=1)
        X["bord"] = sf["bord"].copy().astype(float)

        ## Define the intercept
        X["intercept"] = np.ones((len(X),))
        X = X[["intercept","bord"] + X.columns[:-2].tolist()]
        X = X.drop(columns=reference)
        p = len(X.columns)

        ## Create the regulatization matrix
        years = [c for c in X.columns if c.startswith("year:")]
        T = len(years)
        D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
        #D2[0,2] = 1
        D2[-1,-3] = 1
        RW2 = np.dot(D2.T,D2)*correlation_time
        lam = np.zeros((p,p))
        lam[-T:,-T:] = RW2

        ## Fit the model
        lp = rr.RidgeRegression(X,Y,lam)
        beta_err = np.sqrt(np.diag(lp.beta_cov))
        beta = pd.DataFrame(np.array([lp.beta_hat,beta_err]).T,
                            columns=["beta","std_err"],
                            index=X.columns)
        
        ## Add reference category placeholders and save the
        ## result
        beta = pd.concat([beta,
                          pd.DataFrame(np.zeros((len(reference),2)),
                                       index=reference,columns=beta.columns)],
                          axis=0)
        beta["var"] = lp.var*np.ones((len(beta),))
        output[name] = beta

    ## Compile the output
    output = pd.concat(output.values(),keys=output.keys())
    print("\nFinal output:")
    print(output)
    output.to_pickle(os.path.join(
        "pickle_jar",
        "lognormal_age_at_k_by_state.pkl"))

    ## Make a pdf of fits across states
    cmap = plt.get_cmap("magma")
    bord_values = np.arange(1,11)
    colors = [cmap(i) for i in np.linspace(0.4,0.95,len(bord_values))]
    with PdfPages(os.path.join("_plots","age_at_kth_kid_by_state.pdf")) as book:

        ## Loop over states, making a page for each
        print("\nMaking a book of plots...")
        for state, sf in df.groupby("state"):

            ## Get the relevant features
            beta = output.loc[state]

            ## Compute histograms across bords
            by_ord = sf[["bord","mom_age"]].copy()
            by_ord["bord"] = np.clip(by_ord["bord"],0,10)
            by_ord["freq"] = np.ones((len(by_ord),),dtype=int)
            by_ord = by_ord.groupby(["bord","mom_age"]).count().sort_index()["freq"]

            ## Make a big plot
            fig, axes = plt.subplots(2,5,sharex=True,sharey=True,figsize=(15,6))
            axes = axes.reshape(-1)
            for ax in axes:
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(color="grey",alpha=0.2)

            ## Loop over k, making a panel for each
            for i in bord_values:

                ## Get the data
                hist = by_ord.loc[i]
                total = hist.sum()

                ## Make the histogram monthly
                hist.index = (12*hist.index).astype(int)/12.
                hist = hist.groupby(level=0).sum()
                
                ## Compute relevant averages, first by
                ## subsetting to this view of the data
                kf = sf.loc[sf["bord"] == i]

                ## Compute the average across other factors
                edu = kf["mom_edu"].value_counts()
                edu *= (1./(edu.sum()))
                edu = edu.loc[~edu.index.isin(reference)]
                edu = (edu*(beta.loc[edu.index,"beta"])).sum()

                ## Compute the average across other factors
                time = kf["year"].value_counts()
                time.index = "year:"+time.index
                time *= (1./(time.sum()))
                time = time.loc[~time.index.isin(reference)]
                time = (time*(beta.loc[time.index,"beta"])).sum()

                ## Compute the average across other factors
                ur = kf["area"].value_counts()
                ur *= (1./(ur.sum()))
                ur = ur.loc[~ur.index.isin(reference)]
                ur = (ur*(beta.loc[ur.index,"beta"])).sum()
                
                ## Compute the model-based estimate
                mu = beta.loc["intercept","beta"]+i*beta.loc["bord","beta"]+ur+time+edu
                pdf = rr.log_normal_density(hist.index.to_numpy(),mu,lp.var)
                pdf *= total/(pdf.sum())
            
                ## Plot it all
                axes[i-1].plot(hist,lw=2,color="k")
                axes[i-1].plot(hist.index,pdf,lw=4,color=colors[i-1])
                axes[i-1].text(0.01,0.99,"Child {}".format(i),
                                fontsize=22,color="k",#"#bf209f",
                                horizontalalignment="left",verticalalignment="top",
                                transform=axes[i-1].transAxes)
                axes[i-1].set_ylim((0,None))
                axes[i-1].set_xlim((-1,51))
                axes[i-1].set_yticks([])
                if i >= 6:
                    axes[i-1].set_xlabel("Mom's age at birth")
                    axes[i-1].set_xticks(np.arange(0,6)*10)

            ## Finish up
            fig.suptitle("Mom's age at kth childbirth in "+state.title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            plt.close(fig)

        ## Set up metadata
        d = book.infodict()
        d['Title'] = "The age of Nigerian mom's at their kth childbirth"
        d['Author'] = "Niket"

    ## Done
    print("...done!")