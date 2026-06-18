# Nigeria's measles demographics 

This repository is intended to serve as a partial update or complement to https://github.com/NThakkar-IDM/intensification/. That repository contained a set of methods for working with survey data to generate key demographic inputs for measles transmission models. This repository updates those methods to be more interpretable, robust, and adapatable to new survey data as it comes. 

Work through the scripts like so:
1. `survey_io.py` contains the user interface difference pieces of survey data, where column names are harmonized, the data is cleaned, and finally checked for consistency. Functions and objects from this script are called throughout. Critically, the time-window for all estimates downstream is configured in this script.
2. `MomDistribution.py` pulls post-stratification weights from only the DHS surveys in the dataset, and then interpolates them for use estimates throughout.
3. Then estimate routine vaccination coverage by creating a regression model in `MCVOneProbability.py` and post-stratifying in `MCVOneCoverageEstimates.py`.
4. Move on to estimating yearly births. Start by building the regression models in `AgeAtKthKid.py` and `ZeroInflatedNumKids.py`. Then post-stratify in `YearlyBirths.py`.
5. Finally upsample to monthly births. Start by estimating birth-seasonality in `BirthSeasonality.py`, and then use those estimates in `MonthlyBirths.py`.


This repository uses the same `environment.yml` as the intensification codebase, and assumes the user already has the following:
1. DHS, MICS, and/or other survey data in an organized directory. Note that `survey_io.py` requires specific file names and structures (see the comments for details).
2. Some national population estimate over time. Here, we use the World Bank's, included in `_open_data\`.