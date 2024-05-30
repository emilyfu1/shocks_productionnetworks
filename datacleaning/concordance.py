import pandas as pd
from functions import filter_by_granularity
from dotenv import dotenv_values, find_dotenv
import os

config = dotenv_values(find_dotenv())
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput = pd.read_pickle(path_cleandata + 'use_naics6.pkl')
# goods at 6th level of granularity
bea = filter_by_granularity(bea_products, target_granularity=6)
# make sure bea products actually have 2017 data
products_2017 = bea[(bea['date'].dt.year == 2017) & (bea['expenditures'].notnull())]['product'].unique()

# load raw concordance file
concordance = pd.read_excel(path_rawdata + 'PCEBridge_2017_DET.xlsx', sheet_name='2017')
concordance = concordance[4:].rename(columns={'Unnamed: 1': 'product', 'Unnamed: 3': 'NAICS_desc'})[['product', 'NAICS_desc']]

concordance.to_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')

# PROPORTIONS (split up I-O value based on the many-to-many matches in the concordance)

# calculate proportions of expenditures based on 2017 values
bea2017 = bea[bea['date'].dt.year == 2017][['product', 'expenditures']].groupby('product').sum(min_count=1).reset_index()

# merge 2017 together with concordance 
concordance_calculateproportion = pd.merge(left=concordance, right=bea2017, on='product', how='inner')

# calculate total expenditures for each NAICS sector description
sector_totals = concordance_calculateproportion.groupby('NAICS_desc')['expenditures'].transform('sum')

# calculate I-O proportions
concordance_calculateproportion['IO_proportions'] = concordance_calculateproportion['expenditures'] / sector_totals

# use for merging
concordance_calculateproportion = concordance_calculateproportion[['product', 'NAICS_desc', 'IO_proportions']]

# add line for final demand
concordance_calculateproportion.loc[len(concordance_calculateproportion)] = ['Personal consumption expenditures','fd_all',1]

concordance_calculateproportion.to_pickle(path_cleandata + 'concordance//concordance6_naics6_addproportions.pkl')
