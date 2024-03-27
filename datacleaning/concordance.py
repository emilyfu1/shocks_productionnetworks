import pandas as pd
from functions import create_crosswalk, filter_by_granularity
from dotenv import dotenv_values, find_dotenv
import os

config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'
path_figures = os.path.abspath(config["FIGURES"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput = pd.read_pickle(path_cleandata + 'use_naics6.pkl')
# goods at 6th level of granularity
bea = filter_by_granularity(bea_products, target_granularity=6)
# make sure bea products actually have 2017 data
products_2017 = bea[(bea['date'].dt.year == 2017) & (bea['expenditures'].notnull())]['product'].unique()
bea = bea[bea['product'].isin(products_2017)]
bea['product'] = bea['product'].str.lstrip()

# run concordance function
concordance = create_crosswalk(inputoutput=inputoutput, bea=bea)
concordance = concordance[['product', 'NAICS_desc']]

concordance.to_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')

# PROPORTIONS (SPLITTING UP I-O VALUE)

# use 2017 expenditures to calculate proportions of sorts
bea2017 = bea[bea['date'].dt.year == 2017][['product', 'expenditures']].groupby('product').sum(min_count=1).reset_index()

# merge 2017 together with concordance 
concordance_calculateproportion = pd.merge(left=concordance, right=bea2017, on='product', how='inner')

# calculate total expenditures for each NAICS sector description
sector_totals = concordance_calculateproportion.groupby('NAICS_desc')['expenditures'].transform('sum')

# calculate I-O proportions
concordance_calculateproportion['IO_proportions'] = concordance_calculateproportion['expenditures'] / sector_totals

# use for merging
concordance_calculateproportion = concordance_calculateproportion[['product', 'NAICS_desc', 'IO_proportions']]

concordance_calculateproportion.to_pickle(path_cleandata + 'concordance//concordance6_naics6_addproportions.pkl')
