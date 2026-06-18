""" BirthSeasonality.py

State-by-state seasonality estimates. """
import sys

## For filepaths
import os

## I/O functionality is built on top
## of pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For evaluating the posterior and
## fitting the model.
import survey.buckets as sb

## For making PDFs
from matplotlib.backends.backend_pdf import PdfPages

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return

## For reference on the plots
num_to_month = {1:"January",2:"February",3:"March",4:"April",
                5:"May",6:"June",7:"July",8:"August",9:"September",
                10:"October",11:"November",12:"December"}

if __name__ == "__main__":

    ## Get the data via pandas, first the DHS
    br_columns = ["caseid","child_DoB","mom_DoB","interview_date","state","weight"]
    brs = {sio.infer_survey_name(path): sio.load_survey(path, True, True, br_columns)
            for path in sio.dhs_br_paths}
    dhs = pd.concat(brs.values(),axis=0)\
            .reset_index(drop=True)
    
    ## Now shift over to the MICS datasets
    bh_columns = ["cluster","hh","line_num","birth_ln","state","child_birth_year","child_birth_mon"]
    bhs = {sio.infer_survey_name(path): sio.load_survey(path, True, True, bh_columns)
            for path in sio.mics_bh_paths}
    mics = pd.concat(bhs.values(),axis=0)\
            .reset_index(drop=True)

    ## Get a birthdate column, and subset to recent
    ## births within the survey period.
    dhs["birth_date"] = sio.cmc_to_datetime(dhs["child_DoB"])
    dhs = dhs.loc[(dhs["birth_date"] >= f"{sio.YEAR_MIN-1}-01-01")\
                & (dhs["birth_date"] <= f"{sio.YEAR_MAX+1}-12-31")].reset_index(drop=True)
    dhs["birth_year"] = dhs["birth_date"].dt.year
    dhs["birth_month"] = dhs["birth_date"].dt.month

    ## And same for the mics
    mics["birth_date"] = pd.to_datetime({"month":mics["child_birth_mon"],
                                      "year":mics["child_birth_year"],
                                      "day":1})
    mics = mics.loc[(mics["birth_date"] >= f"{sio.YEAR_MIN-1}-01-01")\
                & (mics["birth_date"] <= f"{sio.YEAR_MAX+1}-12-31")].reset_index(drop=True)
    mics["birth_year"] = mics["birth_date"].dt.year
    mics["birth_month"] = mics["birth_date"].dt.month

    ## Put the full dataset together
    columns = ["survey","state",
               "birth_date","birth_year","birth_month"]
    df = pd.concat([dhs[columns],
                    mics[columns]],axis=0).reset_index(drop=True)
    print("\nFull dataset:")
    print(df)

    ## Start a pdf document, with one page per state.
    output = {}
    with PdfPages(os.path.join("_plots","birth_seasonality_by_state.pdf")) as pdf:

        ## Starting the loop through states
        print("\nStarting the loop through states...")
        for state, sf in df.groupby("state"):

            ## Compute the monthly fraction
            monthly = sf[["birth_year","birth_month"]].copy()
            monthly["frac"] = np.ones((len(sf),),dtype=np.int32)
            monthly = monthly.groupby(["birth_year","birth_month"]).sum()["frac"]
            monthly = monthly.unstack(level=1).fillna(0)
            total_by_year = monthly.sum(axis=1)
            monthly_frac = monthly.div(total_by_year,axis=0)

            ## For plotting, exclude small sample sizes
            monthly_frac = monthly_frac.loc[total_by_year > 50]

            ## Set up the buckets posterior
            buckets = sb.BinomialPosterior(monthly,
                                           correlation_time=10.,
                                           g2g_correlation=4.)
            result = sb.FitModel(buckets)
            print("for {}, success = {}".format(state,result.success))

            ## Unpack the result
            samples = sb.SampleBuckets(result,buckets)
            low = np.percentile(samples,2.5,axis=0)
            high = np.percentile(samples,97.5,axis=0)
            mid = samples.mean(axis=0)
            var = samples.var(axis=0)

            ## Create outputs for use elsewhere
            mid_df = pd.DataFrame(mid,
                                  columns=monthly.columns,
                                  index=monthly.index).stack().rename("avg")
            var_df = pd.DataFrame(var,
                                  columns=monthly.columns,
                                  index=monthly.index).stack().rename("var")
            this_output = pd.concat([mid_df,var_df],axis=1)
            output[state] = this_output

            ## Make a plot
            fig, axes = plt.subplots(3,4,sharex=True,sharey=True,figsize=(16,9))
            axes = axes.reshape(-1)
            for ax in axes:
                axes_setup(ax)
                ax.grid(color="grey",alpha=0.2)
            for i in monthly.columns:

                ## Plot the model
                axes[i-1].fill_between(monthly.index,
                                       low[:,i-1],high[:,i-1],
                                       facecolor="grey",edgecolor="None",alpha=0.4,label="Model")
                axes[i-1].plot(monthly.index,mid[:,i-1],
                               lw=2,color="grey")

                ## Plot the data
                axes[i-1].plot(monthly_frac[i],
                               ls="dashed",lw=3,color="k",
                               markersize=10,marker="o",label="Surveyed")
                axes[i-1].text(0.01,0.99,num_to_month[i],
                               horizontalalignment="left",verticalalignment="top",
                               fontsize=22,color="xkcd:red wine",
                               transform=axes[i-1].transAxes)
                if (i-1)%4 == 0:
                    axes[i-1].set_ylabel("Probability")
                if i == 4:
                    axes[i-1].legend(frameon=False,loc=1,fontsize=16)

            ## Finish up
            fig.suptitle("Seasonality in "+state.title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            pdf.savefig(fig)
            plt.close(fig)

        ## Set up metadata
        d = pdf.infodict()
        d['Title'] = "Birth seasonality in Nigeria"
        d['Author'] = "Niket"

    ## Finish up
    print("...finished.")

    ## Create the full output
    output = pd.concat(output.values(),keys=output.keys())
    print("\nFinal output:")
    print(output)
    output.to_pickle(os.path.join(
        "pickle_jar",
        "birth_seasonality_by_state.pkl"))


        