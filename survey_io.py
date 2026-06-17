""" survey_io.py

Unified survey-loading utilities for DHS and MICS (Pakistan).

1. Set your survey directory:
   Edit SURVEYS at the top of this file so that SURVEYS points
   to the folder containing your DHS and MICS survey subfolders
   (e.g. DHS5_2006/, MICS6_2012/, etc.).

2. Run this file directly to inspect your data: 
   This prints a debug report 
   showing the unique values in every column for all survey files listed in 
   the path lists below. Use this to spot unexpected or inconsistent values.

3. Update the fix maps as needed:
   If the debug report reveals new inconsistencies (e.g. a province name
   you haven't seen before, a new spelling of "missing"), update the
   relevant map: PROVINCE_MAP, EDU_CATS, MCV_CATS, MCV_YEAR_FIX,
   MONTH_FIX, or DOSES_FIX.

4. Import from other scripts:
   Once the data is clean, use load_survey() in your analysis scripts:
       from survey_io import load_survey
       df = load_survey(path, add_survey=True, columns=["province", "mcv1"])

Note: the loader infers survey type (DHS/MICS), MICS wave (4/5/6),
   and recode type (IR/BR/KR/WM/BH/CH) from the folder and file names.
   Survey folders must be formatted like DHS7_2017/, MICS6_2018/, etc.
   Recode files must contain the recode code in the filename (e.g. PKIR52FL.DTA, 
   wm.sav, ch.sav).
"""

import os
import sys
import re
import traceback
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Time configuration across all the different estimates
# -------------------------------------------------------------------
YEAR_MIN = 2006
YEAR_MAX = 2024 # inclusive, i.e. through YEAR_MAX

# -------------------------------------------------------------------
# Paths to survey data
# -------------------------------------------------------------------
SURVEY_DIR = os.path.join(os.path.sep,
    "Users","niketth",
    "OneDrive - Bill & Melinda Gates Foundation",
    "Work","demographics","nigeria","_surveys")

dhs_ir_paths = [
    os.path.join(SURVEY_DIR,"DHS5_2008","NGIR53DT","NGIR53FL.DTA"),
    os.path.join(SURVEY_DIR,"DHS6_2013","NGIR6ADT","NGIR6AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS7_2018","NGIR7ADT","NGIR7AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS8_2023","NGIR8BFL","NGIR8BFL.DTA")
]

dhs_br_paths = [
    os.path.join(SURVEY_DIR,"DHS5_2008","NGBR53DT","NGBR53FL.DTA"),
    os.path.join(SURVEY_DIR,"DHS6_2013","NGBR6ADT","NGBR6AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS7_2018","NGBR7ADT","NGBR7AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS8_2023","NGBR8BFL","NGBR8BFL.DTA")
]

dhs_kr_paths = [
    os.path.join(SURVEY_DIR,"DHS5_2008","NGKR53DT","NGKR53FL.DTA"),
    os.path.join(SURVEY_DIR,"DHS6_2013","NGKR6ADT","NGKR6AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS7_2018","NGKR7ADT","NGKR7AFL.DTA"),
    os.path.join(SURVEY_DIR,"DHS8_2023","NGKR8BFL","NGKR8BFL.DTA")
]

mics_wm_paths = [
    os.path.join(SURVEY_DIR,"MICS5_2016","wm.sav"),
    os.path.join(SURVEY_DIR,"MICS6_2021","wm.sav"),
]

mics_bh_paths = [
    os.path.join(SURVEY_DIR,"MICS5_2016","bh.sav"),
    os.path.join(SURVEY_DIR,"MICS6_2021","bh.sav"),
]

mics_ch_paths = [
    os.path.join(SURVEY_DIR,"MICS5_2016","ch.sav"),
    os.path.join(SURVEY_DIR,"MICS6_2021","ch.sav"),
]

# -------------------------------------------------------------------------------
# Schemas (which map survey specific column names to interpretable, harmonized names)
# -------------------------------------------------------------------------------

# For the DHS
# Raw variable codes follow the DHS recode manual naming convention.
# See https://dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm

DHS_IR_SCHEMA = {
    "caseid": "caseid",        # unique respondent ID
    "awfactt": "awfactt",      # all-women factor (for ever-married surveys)
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v013": "mom_age",         # age in 5-year groups
    "v106": "mom_edu",         # highest education level
    "v224": "num_brs",         # number of entries in birth recode
    "v008": "interview_date",  # CMC date of interview
    "v023": "strata",          # sample strata (state or region+state+U/R)"
    "v024": "region",          # region/province
    "v025": "area",            # urban/rural
    "v005": "weight",          # sample weight (÷1e6)
}

DHS_BR_SCHEMA = {
    "caseid": "caseid",        # mother's unique ID
    "bord": "bord",            # birth order number
    "b3": "child_DoB",         # CMC date of birth of child
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v008": "interview_date",  # CMC date of interview
    "v005": "weight",          # sample weight (÷1e6)
    "v023": "strata",          # sample strata (state or region+state+U/R)"
    "v024": "region",          # region/province
}

DHS_KR_SCHEMA = {
    "caseid": "caseid",        # mother's unique ID
    "bord": "bord",            # birth order number
    "b3": "child_DoB",         # CMC date of birth of child
    "b5": "live_child",        # is the child alive? (yes/no)
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v008": "interview_date",  # CMC date of interview
    "h9": "mcv1",              # measles/MCV1 vaccination status
    "v005": "weight",          # sample weight (÷1e6)
    "h9d": "mcv1_day",         # day of MCV1 vaccination (from card)
    "h9m": "mcv1_mon",         # month of MCV1 vaccination (from card)
    "h9y": "mcv1_yr",          # year of MCV1 vaccination (from card)
}

## DHS7 and 8 has mcv2
DHS7_or_8_KR_SCHEMA = {
    "caseid": "caseid",        # mother's unique ID
    "bord": "bord",            # birth order number
    "b3": "child_DoB",         # CMC date of birth of child
    "b5": "live_child",        # is the child alive? (yes/no)
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v008": "interview_date",  # CMC date of interview
    "h9": "mcv1",              # measles/MCV1 vaccination status
    "h9a": "mcv2",             # MCV2 status
    "v005": "weight",          # sample weight (÷1e6)
    "h9d": "mcv1_day",         # day of MCV1 vaccination (from card)
    "h9m": "mcv1_mon",         # month of MCV1 vaccination (from card)
    "h9y": "mcv1_yr",          # year of MCV1 vaccination (from card)
}

# For the MICS
# Raw variable codes follow MICS naming conventions, which vary by wave.

MICS4_WM_SCHEMA = MICS5_WM_SCHEMA = {
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
}

MICS6_WM_SCHEMA = {
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
}

MICS4_BH_SCHEMA = MICS5_BH_SCHEMA = {
     "HH1":"cluster", ## woman cluster
     "HH2":"hh", ## woman hh
     "LN":"line_num", ## line number
     "BHLN":"birth_ln", ## birth history ln
     "birthord":"bord", ## child birth order
     "BH4M":"child_birth_mon", ## child birth month
     "BH4Y":"child_birth_year", ## child birth year
}

MICS6_BH_SCHEMA = {
     "HH1":"cluster", ## woman cluster
     "HH2":"hh", ## woman hh
     "LN":"line_num", ## line number
     "BHLN":"birth_ln", ## birth history ln
     "brthord":"bord", ## child birth order
     "BH4M":"child_birth_mon", ## child birth month
     "BH4Y":"child_birth_year", ## child birth year
}

MICS4_CH_SCHEMA = MICS5_CH_SCHEMA = {
     "HH1":"cluster", ## woman cluster
     "HH2":"hh", ## woman hh
     "UF6":"line_num", ## line number for mom
     "UF9":"complete_interview", ## interview completeness
     "AG1D":"child_birth_day", ## child birth day
     "AG1M":"child_birth_mon", ## child birth mon
     "AG1Y":"child_birth_year", ## child birth year
     "AG2":"child_age", ## Age to nearest year
     "IM3MD":"mcv1_day", ## card based day
     "IM3MM":"mcv1_mon", ## card based mon
     "IM3MY":"mcv1_year", ## card based day
     "IM16":"mcv1", ## Ever recieved, recall based
}

MICS6_CH_SCHEMA = {
     "HH1":"cluster", ## woman cluster
     "HH2":"hh", ## woman hh
     "UF4":"line_num", ## line number for mom
     "UF17":"complete_interview", ## interview completeness
     "UB1D":"child_birth_day", ## child birth day
     "UB1M":"child_birth_mon", ## child birth mon
     "UB1Y":"child_birth_year", ## child birth year
     "UB2":"child_age", ## Age to nearest year
     "IM6N1D":"mcv1_day", ## card based day
     "IM6N1M":"mcv1_mon", ## card based mon
     "IM6N1Y":"mcv1_year", ## card based day
     "IM26":"mcv1", ## Ever recieved, recall based
}

# -------------------------------------------------------------------
# Functions to map file names to survey type, recode type, and schema
# -------------------------------------------------------------------

# This regex parses paths to extract folder names like 
# DHS8_2023\ or MICS5_2016\ 
SURVEY_FROM_PATH = re.escape(os.path.sep)\
        +r"((?:DHS|MICS)\d_(?:19|20)\d{2})"\
        +re.escape(os.path.sep)
def infer_survey_name(path):
    m = re.search(SURVEY_FROM_PATH,path)
    if m:
        return m.group(1).lower()
    else:
        raise ValueError(f"Cannot detect survey name from path: {path}"
                "\nMake sure the survey data files are organized correctly!")

def infer_recode_type(path):
    base = os.path.basename(path).lower()
    if "ir" in base or "iq" in base: return "ir"
    if "br" in base: return "br"
    if "kr" in base: return "kr"
    if base == "wm.sav": return "wm"
    if base == "bh.sav": return "bh"
    if base == "ch.sav": return "ch"
    raise ValueError(f"Cannot infer recode type from filename: {path}")

def get_md_and_schema(path):

    # Get the survey info from the path, including
    # the recode type, by first pulling the survey name
    # from the folder structure (i.e. DHS8_2023) and then
    # parsing the 3 pieces with the regex below [i.e., 
    # (DHS,8,2023)].
    survey_name = infer_survey_name(path)
    survey, wave, year = re.search(
        r"^(.*)(\d)_(\d{4})$",
        survey_name).groups()
    recode = infer_recode_type(path)

    ## Then map survey type, wave, and recode
    ## to the appropriate schema above.
    if survey == "dhs":
        if int(wave) < 7:
            return (survey_name, recode,
                {"ir":DHS_IR_SCHEMA,"br":DHS_BR_SCHEMA,"kr":DHS_KR_SCHEMA}[recode])
        else:
            return (survey_name, recode,
                {"ir":DHS_IR_SCHEMA,"br":DHS_BR_SCHEMA,"kr":DHS7_or_8_KR_SCHEMA}[recode])

    elif survey == "mics":
        if recode == "wm":
            return (survey_name, recode,
                {4:MICS4_WM_SCHEMA, 5:MICS5_WM_SCHEMA, 6:MICS6_WM_SCHEMA}[int(wave)])
        if recode == "bh":
            return (survey_name, recode,
                {4:MICS4_BH_SCHEMA, 5:MICS5_BH_SCHEMA, 6:MICS6_BH_SCHEMA}[int(wave)])
        if recode == "ch":
            return (survey_name, recode,
                {4:MICS4_CH_SCHEMA, 5:MICS5_CH_SCHEMA, 6:MICS6_CH_SCHEMA}[int(wave)])

    raise ValueError(f"No schema for path: {path}")

# -------------------------------------------------------------------
# Data maps and cleaning functions (to harmonize values across surveys)
# -------------------------------------------------------------------

## Regex to extract state from the DHS strata
STATE_FROM_STRATA = r"^(?:[ns][ewcs]\s+)?(.*?)(?:\s+(?:urban|rural))?$"

# Data maps are dictionaries to more unified values,
# To make a data map, use df[col].value_counts(dropna=False) in the cleaning functions
# to see what types of values show up, and then construct the dictionary.
MICS_EDU_CATS = {
    "secondary / secondary-technical":"secondary",
    np.nan:"no education",
    "primary":"primary",
    "non-formal":"no education",
    "higher":"higher",
    "preschool":"primary",
    "higher/tertiary":"higher",
    "senior secondary":"secondary",
    "junior secondary":"secondary",
    "secondary technical":"secondary",
    "eccde":"no education",
    "vei/iei":"no education",
    "no response":"no education",
}

MCV_CATS = {
    "no":0,
    "yes":1,
    "reported by mother":1,
    "vaccination date on card":1,
    "vacc. date on card":1,
    "vaccination marked on card":1,
    "dk":0,
    np.nan:0,
    "vacc. marked on card":1,
    "don't know":0,
    "missing":0,
    "no response":0,
}

MONTH_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "dk": np.nan, "missing": np.nan, "no response": np.nan,
    "inconsistent": np.nan, np.nan:np.nan,
}

def cmc_to_datetime(a, d=15):
    years = 1900 + ((a - 1) // 12)
    months = a - 12*(years-1900)
    return pd.to_datetime({"year": years, "month": months, "day": d})

def compute_mics_mom_age(df):
    birth_year = fix_year_col(df["mom_birth_year"])
    birth_mon = fix_month_col(df["mom_birth_mon"])
    int_mon = fix_month_col(df["interview_mon"])
    int_year = fix_year_col(df["interview_year"])

    ## Make date times
    df["mom_DoB"] = pd.to_datetime(
        {"year": birth_year, "month": birth_mon, "day": 1},
        errors="coerce"
    )
    df["interview_date"] = pd.to_datetime(
        {"year": int_year, "month": int_mon, "day": df["interview_day"]},
        errors="coerce"
    )

    ## Appoximate age in years
    df["mom_age"] = (df["interview_date"] - df["mom_DoB"]).dt.days / 365.25

    return df

def clean_dhs(df):
    
    if "caseid" in df:
        df["caseid"] = df["caseid"].astype(str).str.strip()
    
    if "region" in df:
        df["region"] = df["region"].astype(str).str.lower()
    
    if "area" in df:
        df["area"] = df["area"].astype(str).str.lower()
    
    if "mom_edu" in df:
        df["mom_edu"] = df["mom_edu"].astype(str).str.lower()
    
    if "mcv1" in df:
        df["mcv1"] = df["mcv1"].str.lower().map(lambda x: MCV_CATS[x])
    
    if "mcv1_day" in df:
        df["mcv1_day"] = pd.to_numeric(df["mcv1_day"], errors="coerce")
    
    if "mcv1_mon" in df:
        df["mcv1_mon"] = pd.to_numeric(df["mcv1_mon"], errors="coerce")
    
    if "mcv1_yr" in df:
        df["mcv1_yr"] = pd.to_numeric(df["mcv1_yr"], errors="coerce")
    
    if "mcv2" in df:
        df["mcv2"] = df["mcv2"].str.lower().map(lambda x: MCV_CATS[x])
    
    if "live_child" in df:
        df["live_child"] = df["live_child"].str.lower()
    
    if "strata" in df:
        ## Strata is used to make a state column!
        df["state"] = df["strata"].astype(str).str.lower()\
                        .str.extract(STATE_FROM_STRATA)[0]
        df["state"] = df["state"].str.replace("fct abuja","abuja")\
                            .str.replace("fct","abuja")

    return df

def clean_mics(df):
    
    if "area" in df:
        df["area"] = df["area"].str.lower()

    if "mom_edu" in df:
        df["mom_edu"] = df["mom_edu"].str.lower().map(lambda x: MICS_EDU_CATS[x])        

    for col in ["mom_birth_mon", "child_birth_mon", "interview_mon"]:
        if col in df:
            df[col] = df[col].str.lower().map(lambda x: MONTH_TO_NUM[x])

    for col in ["mom_birth_year", "child_birth_year"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mcv1" in df:
        df["mcv1"] = df["mcv1"].str.lower().map(lambda x: MCV_CATS[x])
        df["mcv1_with_card"] = pd.to_numeric(df["mcv1_year"], errors="coerce").notnull()
        df["mcv1"] = (df["mcv1_with_card"] | df["mcv1"]).astype(int)

    if "region" in df:
        df["region"] = df["region"].str.lower()

    if "state" in df:
        df["state"] = df["state"].str.lower()\
            .str.replace("fct abuja","abuja").str.replace("fct","abuja")

    return df

# -------------------------------------------------------------------
# Main survey loader
# -------------------------------------------------------------------

def load_survey(path, add_survey=False, convert_categoricals=True, columns=None):
    
    ## get survey info
    survey_name, recode, schema = get_md_and_schema(path)
    survey, wave, year = re.search(
        r"^(.*)(\d)_(\d{4})$",
        survey_name).groups()
    
    # Always load all raw schema columns
    raw_cols = list(schema.keys())

    ## Load the data
    if survey == "dhs":
        df = pd.read_stata(path, columns=raw_cols,
                           convert_categoricals=convert_categoricals)
    elif survey == "mics":
        df = pd.read_spss(path, usecols=raw_cols,
                          convert_categoricals=convert_categoricals)
    else:
        raise TypeError(f"Pandas i/o not specified for survey type {survey}!")

    # Apply renaming
    df = df.rename(columns=schema)

    # Clean
    if survey == "dhs":
        df = clean_dhs(df)
    elif survey == "mics":
        df = clean_mics(df)

    # Add survey label
    if add_survey:
        df["survey"] = survey_name

    # Normalize weights
    if "weight" in df:
        df["weight"] *= 1e-6
        df["weight"] *= 1. / (df["weight"].sum())

    # Restrict to requested columns
    if columns is not None:    
        df = df[columns + int(add_survey)*["survey"]]

    return df

# -------------------------------------------------------------------
# DEBUG: run this file directly to inspect all survey data
# -------------------------------------------------------------------

def debug_print_unique_values(path, columns=None, max_unique=20):
    """
    Load each survey file via load_survey() and print all unique values
    per column. Use this to verify that province names, education
    categories, area codes, vaccination fields, and year extraction
    are all behaving as expected. Update the fix maps above if you
    spot anything unexpected.
    """
    print("\n--------------------------------------------------------------")
    print(f"FILE: {path[len(SURVEY_DIR):]}")
    print("--------------------------------------------------------------")

    ## Load survey files, catching exceptions
    ## so you can see which fail and which succeed without 
    ## exiting.
    try:
        df = load_survey(path, add_survey=True, convert_categoricals=True, columns=columns)
    except Exception as e:
        traceback.print_exception(e)
        return

    ## Print out a summary
    print(f"Loaded columns: {list(df.columns)}\n")
    for col in df.columns:
        values = df[col].unique()
        print(f"  {col} ({len(values)} unique, dtype={df[col].dtype}):")
        if len(values) > max_unique:
            shown = values[:max_unique]
            print(f"    {shown} ... ({len(values) - max_unique} more)")
        else:
            print(f"    {values}")
    print()


if __name__ == "__main__":
    all_paths = (
        dhs_ir_paths +
        dhs_br_paths +
        dhs_kr_paths +
        mics_wm_paths +
        mics_bh_paths +
        mics_ch_paths
    )
    print("\n==================== SURVEY DEBUG REPORT =====================")
    for path in all_paths:
        debug_print_unique_values(
            path,
            # columns = ['province', 'mcv1', 'mcv1_day', 'mcv1_mon', 'mcv1_yr'],
            max_unique=10,
        )