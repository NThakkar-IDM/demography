""" MomDistribution.py

Uses DHS weights to coarsely estimate the fraction of individuals in each
demographic cell (i.e., mom-characteristic combination). 

Dependancies: survey_io
Outputs:
    - ../pickle_jar/mom_distribution.pkl
    
Debugging tips:
    - If you encounter issues with data types, missing data, or if
        you want to add covariates, check the survey_io file, which
        sets up the DHS data.
    - Inspect the output(s) for NaN weight values, which should only
        occur in years prior to available data for that province, or 
        for uncommon age bins (e.g., 15-19).
    - Check the printed normalizations; sum of weights should equal 1.0
        across all strata each year in the main distribution.
"""

# For filepaths
import os
import sys

# Input/output functionality is built on top of pandas
import pandas as pd
import numpy as np

# For loading, renaming, and unifying DHS and MICS data
from survey_io import load_survey, infer_survey_name, dhs_ir_paths

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
YEAR_MAX = 2024

if __name__ == "__main__":

    # Get the relevant DHS data, individual recode (IR) has rows
    # associated with moms
    ir_columns = ["mom_DoB", "interview_date", "mom_age", "state",
              "area", "mom_edu", "num_brs", "weight"]
    irs = {
        infer_survey_name(path): load_survey(path, True, True, ir_columns)
        for path in dhs_ir_paths
    }
    irs = pd.concat(irs,axis=0).reset_index(drop=True)
    print("\nFull dataset:")
    print(irs)

    # Create a year column associated with the survey year,
    # which is when the weights were estimated
    irs["year"] = irs["survey"].str.slice(start=5).astype(np.int64)

    # Create a table of demographic cells
    df = irs[["state","area","mom_edu","mom_age","weight","year"]].copy()
    df = df.groupby([c for c in df.columns if c != "weight"],
            observed=False).sum().fillna(0)["weight"]
    
    # Iterpolate the table annually
    years = np.arange(irs["year"].min(),YEAR_MAX+1,dtype=np.int64)
    df = df.groupby(["state","area","mom_edu","mom_age"],
            observed=False).apply(
                lambda s: s.loc[s.name].reindex(years).interpolate()
                )
    df = df.reset_index()

    # Print and save
    print("\nTotal weight per cell:")
    print(df)
    df.to_pickle(os.path.join(
        "pickle_jar","mom_distribution.pkl")
        )

    # Check the normalization
    print("\nNormalization check:")
    print(df[["year","weight"]].groupby("year").sum())
    