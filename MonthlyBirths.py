""" MonthlyBirths.py

Combining the output from YearlyBirths.py + population estimates via the WorldBank +
estimates of seasonality over time to estimate a gaussian process for monthly births. """
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

# For loading, renaming, and unifying DHS and MICS data
import survey_io as sio

def axes_setup(axes):
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return

## For the world bank data
def get_raw_spreadsheet(root,fname,
                        years=(2010,2019)):

    ## I/O based on pandas with specific dtypes
    ## for speed.
    columns = ["Country Name"]+[str(c) for c in range(*years)]
    dtypes = {c:np.float64 for c in columns[1:]}
    dtypes["Country Name"] = str
    df = pd.read_csv(os.path.join(root,fname),
                     skiprows=4,header=0,
                     usecols=columns,
                     dtype=dtypes)

    ## Adjust country names to the WHO spreadsheet values
    ## done manually here.
    adjustments = {"Yemen, Rep.":"yemen",
                   "Vietnam":"viet nam",
                   "Tanzania":"united republic of tanzania",
                   "Korea, Dem. People’s Rep.":"democratic people's republic of korea",
                   "Gambia, The":"gambia",
                   "Congo, Dem. Rep.":"democratic republic of the congo"}
    df["Country Name"] = df["Country Name"].apply(lambda x: adjustments.get(x,x))

    ## Basic formatting
    df.columns = [c.lower().replace(" name","") for c in df.columns]
    df["country"] = df["country"].str.lower()

    return df

def to_spacetime(df,name):
    df = df.set_index("country").stack(
        future_stack=True).reset_index()
    df.columns = ["country","year",name]
    return df

def GetWBPopulation(root,
                    years=(2010,2019),
                    countries=None):

    """ Subroutine to get the raw WB spreadsheets (located in root) and to
        reshape it into a space-time dataframe, interpolate, and compute
        births. """

    ## Get the raw data
    population = get_raw_spreadsheet(root,fname="wb_total_population.csv",
                                     years=years)

    ## Subset to specific countries
    if countries is not None:
        population = population.loc[population["country"].isin(countries)]

    ## Reshape both into space-time series
    population = to_spacetime(population,"population")

    ## Convert to timestamps
    population["time"] = pd.to_datetime({"year":population["year"],
                                         "month":6,
                                         "day":15})

    return population

def weighted_mean_var(x):
    avg = (x["pop_avg"]*x["pr_birth_last_year"]).sum()
    var = (x["pop_avg"]*x["pr_birth_last_year"]*(1.-x["pr_birth_last_year"])+\
           x["pop_var"]*(x["pr_birth_last_year"]**2)).sum()
    out = pd.Series([avg,var],index=["avg","var"])
    return out

if __name__ == "__main__":

    ## Get the population data, with 1 year buffer on the front,
    ## since YearlyBirths.py shifts years by 1.
    population = GetWBPopulation("_open_data",
                                 countries=["nigeria"],
                                 years=(sio.YEAR_MIN-1,sio.YEAR_MAX+1))
    population["year"] = population["year"].astype(np.int64)
    population = population[["year","population"]].set_index("year")["population"]
    
    ## Compute yearly births via the output from YearlyBirths.py
    dist = pd.read_pickle(os.path.join(
        "pickle_jar",
        "mom_distribution_inf.pkl"))
    
    ## Align the population values in time
    years = set((dist["year"].unique())) | set(population.index)
    years = sorted(list(years))
    population = population.reindex(years).interpolate(limit_direction="both")

    ## Adjust the weight by population
    dist["pop_avg"] = dist["weight"]*(population.loc[dist["year"].values].values)
    dist["pop_var"] = dist["weight"]*(1.-dist["weight"])*(population.loc[dist["year"].values].values)

    ## Compute yearly births
    births = dist[["state","weight","pop_avg","pop_var","year","pr_birth_last_year"]].copy()
    births = births.groupby(["state","year"]).apply(weighted_mean_var,include_groups=False)
    
    ## Set up the timing more intentionally, i.e. remove the
    ## extra early year, forward fill the final year.
    years = np.arange(sio.YEAR_MIN,sio.YEAR_MAX+1,dtype=np.int64)
    births = births.groupby("state").apply(
             lambda s: s.loc[s.name].reindex(years).ffill()#.bfill()
             )
    births = births.sort_index()
    
    ## Get the output from BirthSeasonality.py to move to a monthly
    ## estimate, taking care to align it to the years in births above
    seasonality = pd.read_pickle(os.path.join(
        "pickle_jar",
        "birth_seasonality_by_state.pkl"))
    seasonality = seasonality.groupby(level=(0,2)).apply(
                  lambda s: s.reset_index(level=(0,2),drop=True).reindex(years).ffill()
                  )
    seasonality = seasonality.swaplevel("birth_month","birth_year").sort_index()
    seasonality.index.rename(["state","year","month"],inplace=True)

    ## Then use it to distribute births
    ExpN = np.repeat(births["avg"].values,12)
    VarN = np.repeat(births["var"].values,12)
    exp_monthly_births = ExpN*seasonality["avg"]
    var_monthly_births = ExpN*seasonality["avg"]*(1.-seasonality["avg"])+\
                         seasonality["var"]*(ExpN**2)+\
                         VarN*(seasonality["avg"]**2)

    ## Put it all together
    monthly_births = pd.concat([exp_monthly_births.rename("avg"),
                                var_monthly_births.rename("var")],
                                axis=1)
    monthly_births["std"] = np.sqrt(monthly_births["var"])

    ## Set up the result as a timeseries
    monthly_births = monthly_births.reset_index()
    monthly_births["time"] = pd.to_datetime({"year":monthly_births["year"],
                                             "month":monthly_births["month"],
                                             "day":1})
    monthly_births = monthly_births[["state","time","avg","var","std"]]
    monthly_births = monthly_births.set_index(["state","time"])
    print("\nOverall result:")
    print(monthly_births)

    ## Save the result
    monthly_births.to_pickle(os.path.join(
        "pickle_jar",
        "monthly_births_by_state.pkl"))

    ## Make a book of plots
    with PdfPages(os.path.join("_plots","monthly_births_by_state.pdf")) as book:

        ## Loop over states, making a page for each
        print("\nMaking a book of plots...")
        for state, sf in monthly_births.groupby("state"):

            ## Get the time series
            ts = sf.loc[state]

            ## And the DHS weight trend
            weight_trend = dist.loc[dist["state"] == state,["year","pop_avg"]]
            weight_trend["year"] = pd.to_datetime({"year":weight_trend["year"],
                                                 "month":6,"day":15})
            weight_trend = weight_trend.groupby("year").sum()
            weight_trend *= (ts["avg"].mean())/(weight_trend.mean())
            
            ## Make the plot
            fig, axes = plt.subplots(figsize=(12,5))
            axes_setup(axes)
            axes.grid(color="grey",alpha=0.3)
            axes.fill_between(ts.index,
                              (ts["avg"]-2.*ts["std"]).values,
                              (ts["avg"]+2.*ts["std"]).values,
                              facecolor="#4FB0C6",edgecolor="None",alpha=0.6)
            axes.plot(ts.index,
                      ts["avg"].values,
                      lw=3,color="#4F86C6")
            axes.plot(weight_trend,
                      lw=4,ls="dashed",color="k",label="Est. population trend\n(driven by the DHS weights)")
            axes.set_ylabel("Monthly births")

            ## Add a legend to the first one
            if state == "abia":
                axes.legend(frameon=False)

            ## Set up the page
            fig.suptitle("Monthly births in "+state.title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            plt.close(fig)

    ## Done
    print("...done!")

