""" ZeroInflatedNumKids.py

Regressing mother's characteristics to better understand drivers in the number of 
children. This uses a negative-binomial regression with annual-level smoothing in time, but
has a additional logisitic regression step to handle zeros. """
import sys

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Get the regression model
import survey.zero_inflated_nb as znb

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return

if __name__ == "__main__":

    ## Get the DHS data via pandas
    ir_columns = ["caseid","mom_DoB","interview_date","mom_age",
                  "state","region","area","mom_edu","num_brs","weight"]
    irs = {sio.infer_survey_name(path): sio.load_survey(path, True, True, ir_columns)
            for path in sio.dhs_ir_paths}
    dhs = pd.concat(irs.values(),axis=0).reset_index(drop=True)

    ## And the MICS
    wm_columns = ["cluster","hh","line_num","interview_day","interview_mon",
              "interview_year","mom_birth_mon","mom_birth_year","mom_edu",
              "given_birth","num_brs","area","state","region","weight"]
    wms = {sio.infer_survey_name(path): sio.load_survey(path, True, True, wm_columns)
            for path in sio.mics_wm_paths}
    mics = pd.concat(wms.values(),axis=0)
    mics = mics.loc[mics["weight"] > 0].reset_index(drop=True)

    ## Create a year covariate from the interview date
    dhs["year"] = sio.cmc_to_datetime(dhs["interview_date"]).dt.year.astype(str)
    mics["year"] = mics["interview_year"].astype(int).astype(str)

    ## Bin mom's age in the MICS according to the 5 year
    ## DHS bins, first by computing the age
    mics["mom_DoB"] = pd.to_datetime({"month":mics["mom_birth_mon"],
                                      "year":mics["mom_birth_year"],
                                      "day":1})
    mics["interview_date"] = pd.to_datetime({"month":mics["interview_mon"],
                                      "year":mics["interview_year"],
                                      "day":mics["interview_day"]})
    mics["mom_age"] = (mics["interview_date"] - mics["mom_DoB"]).dt.days/365.

    ## Then by cutting
    mics["mom_age"] = pd.cut(mics["mom_age"],
                             bins=np.arange(15,55,5),
                             labels=dhs["mom_age"].cat.categories)

    ## Now put the surveys together
    variables = ["survey","mom_age","region","state","area","mom_edu","year","num_brs"]
    df = pd.concat([dhs[variables],
                    mics[variables]],axis=0)
    df = df.loc[df.notnull().all(axis=1)].reset_index(drop=True)
    print("\nCleaned and compiled dataset...")
    print(df)

    ## Standardize values across surveys:
    print("\nVariable values...")
    for c in variables[:-1]:
        values = sorted(df[c].unique())
        print("{} values ({} of them) = {}".format(c,len(values),values))

    ## Set up the specifics of the regression problem
    response = "num_brs"
    features = ["mom_age","area","mom_edu","year"]
    correlation_time = (5**4)/8.
    reference = ["20-24","urban","primary",f"year:{sio.YEAR_MIN}"]

    ## Loop over states and compile models
    print("\nStarting the loop over states...")
    output = {}
    for name, sf in df.groupby("state"):

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

        ## Define the intercept
        X["intercept"] = np.ones((len(X),))
        X = X[["intercept"] + X.columns[:-1].tolist()]
        X = X.drop(columns=reference)
        p = len(X.columns)

        ## Set alpha
        emp_m = Y.mean()
        emp_v = Y.var()
        alpha = ((emp_v/(emp_m**2)) - (1./emp_m))/9.
        
        ## Create the regulatization matrix
        years = [c for c in X.columns if c.startswith("year:")]
        T = len(years)
        D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
        #D2[0,2] = 1
        D2[-1,-3] = 1
        RW2 = np.dot(D2.T,D2)*correlation_time
        lam = np.zeros((p,p))
        lam[-T:,-T:] = RW2

        ## Set up the regression posterior
        log_post = znb.ZeroInflatedNBPosterior(X.values,
                                               Y.values,
                                               lam=lam,
                                               alpha=alpha,
                                               )
        lr_result, nb_result = znb.FitModel(log_post)
        print("...for {}, logistic success = {},"
              " negative binomial success = {}".format(name,
                                                       lr_result.success,
                                                       nb_result.success))
        
        ## Report on the results
        lr_beta_hat = lr_result["x"]
        lr_beta_cov = lr_result["hess_inv"]
        lr_beta_err = np.sqrt(np.diag(lr_beta_cov))
        lr_beta = pd.DataFrame(np.array([lr_beta_hat,lr_beta_err]).T,
                            columns=["lr_beta","lr_beta_err"],
                            index=X.columns)
        
        ## Step 2 is the NB step...
        nb_beta_hat = nb_result["x"]
        nb_beta_cov = nb_result["hess_inv"]
        nb_beta_err = np.sqrt(np.diag(nb_beta_cov))
        nb_beta = pd.DataFrame(np.array([nb_beta_hat,nb_beta_err]).T,
                            columns=["nb_beta","nb_beta_err"],
                            index=X.columns)

        ## Add reference category placeholders and save the
        ## result
        lr_beta = pd.concat([lr_beta,
                            pd.DataFrame(np.zeros((len(reference),2)),
                                         index=reference,columns=lr_beta.columns)],
                            axis=0)
        nb_beta = pd.concat([nb_beta,
                            pd.DataFrame(np.zeros((len(reference),2)),
                                          index=reference,columns=nb_beta.columns)],
                            axis=0)
        nb_beta["alpha"] = log_post.nb_post.alpha*np.ones((len(nb_beta),))
        beta = pd.concat([lr_beta,nb_beta],axis=1)
        output[name] = beta

    ## Compile the output
    output = pd.concat(output.values(),keys=output.keys())
    print("\nFinal output:")
    print(output)
    output.to_pickle(os.path.join(
        "pickle_jar",
        "zero_inf_neg_bin_num_kids_by_state.pkl"))

    ## Make a pdf of fits across states
    values = sorted(df["mom_age"].unique())
    k = np.arange(15)
    cmap = plt.get_cmap("magma")
    colors = [cmap(i) for i in np.linspace(0.1,0.9,len(values))]
    with PdfPages(os.path.join("_plots","num_kids_by_state.pdf")) as book:

        ## Loop over states, making a page for each
        print("\nMaking a book of plots...")
        for state, sf in df.groupby("state"):

            ## Get the relevant features
            beta = output.loc[state]

            ## Make a big plot
            fig, axes = plt.subplots(2,4,
                             sharex=True,sharey=False,
                             figsize=(16,6))
            axes = axes.reshape(-1)
            for ax in axes:
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            axes[-1].axis("off")

            ## Loop over age bins and compute estimates
            for i, ab in enumerate(values):

                ## Get the data for this age 
                af = sf.loc[(sf["mom_age"] == ab)]
                hist = af["num_brs"].value_counts().sort_index()
                hist *= 1/(hist.sum())

                ## Compute the average across other factors
                edu = af["mom_edu"].value_counts()
                edu *= (1./(edu.sum()))
                edu = edu.loc[~edu.index.isin(reference)]
                edu_lr = (edu*(beta.loc[edu.index,"lr_beta"])).sum()
                edu_nb = (edu*(beta.loc[edu.index,"nb_beta"])).sum()

                ## And same for time
                time = af["year"].value_counts()
                time.index = "year:"+time.index
                time *= (1./(time.sum()))
                time = time.loc[~time.index.isin(reference)]
                time_lr = (time*(beta.loc[time.index,"lr_beta"])).sum()
                time_nb = (time*(beta.loc[time.index,"nb_beta"])).sum()

                ## Compute the average across other factors
                ur = af["area"].value_counts()
                ur *= (1./(ur.sum()))
                ur = ur.loc[~ur.index.isin(reference)]
                ur_lr = (ur*(beta.loc[ur.index,"lr_beta"])).sum()
                ur_nb = (ur*(beta.loc[ur.index,"nb_beta"])).sum()

                ## Compute the model distribution 
                features = ["intercept",ab]
                alpha = beta["alpha"].values[0]
                mu_lr = beta.loc[features,"lr_beta"].sum()+edu_lr+time_lr+ur_lr
                mu_nb = log_post.nb_post.g(beta.loc[features,"nb_beta"].sum()+edu_nb+time_nb+ur_nb)
                pmf = znb.pmf(k,mu_lr,mu_nb,alpha)

                ## Plot the fit
                axes[i].plot(k,pmf,
                             color=colors[i],lw=6,
                             zorder=4,
                             )

                ## Plot the data
                axes[i].bar(hist.index,hist.values,
                             width=0.666,
                             color="grey",lw=5,ls="dashed",
                             zorder=1,
                             )

                ## Set up some text
                axes[i].text(0.99,0.99,"{} years old".format(ab),
                             fontsize=22,color="k",
                             horizontalalignment="right",verticalalignment="top",
                             transform=axes[i].transAxes)

                ## Details
                axes[i].set_yticks([])
                axes[i].set_ylim((0,None))
                if i >= 4:
                    axes[i].set_xlabel("Number of kids")

            ## Add a legend
            axes[-1].plot([],lw=5,color=colors[3],label="Model fit")
            axes[-1].fill_between([],[],[],lw=5,
                         color="grey",facecolor="grey",edgecolor="grey",
                         label="Survey data")
            axes[-1].legend(loc="center",frameon=False)

            ## Finish up
            fig.suptitle("Number of kids for moms by age in "+state.title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            plt.close(fig)

    ## Done
    print("...done!")