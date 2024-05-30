import pandas as pd
import numpy as np
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

requirementstable = pd.read_pickle(path_cleandata + 'requirements_naics6.pkl')

iorequirements_wide_fillna = requirementstable.fillna(value=0)

# next, i need sectors that show up as both buyers and sellers
iosectors = list(set(iorequirements_wide_fillna.index) & set(iorequirements_wide_fillna.columns))
iorequirements_wide_fillna = iorequirements_wide_fillna[iosectors].loc[iosectors]
# list of buyers
inputsectors = [index for index in iorequirements_wide_fillna.index if iorequirements_wide_fillna.sum(axis=1)[index] != 0 and iorequirements_wide_fillna.sum(axis=0)[index] != 0]
# list of sellers
outputsectors = [column for column in iorequirements_wide_fillna.columns if iorequirements_wide_fillna.sum(axis=1)[column] != 0 and iorequirements_wide_fillna.sum(axis=0)[column] != 0]

# get the intersection of these lists
sectors_to_include = list(set(inputsectors) & set(outputsectors))

# filter I-O table and residuals
iorequirements_wide_fillna.drop(columns=[col for col in iorequirements_wide_fillna.columns if col not in sectors_to_include], inplace=True)
iorequirements_wide_fillna.drop(index=[idx for idx in iorequirements_wide_fillna.index if idx not in sectors_to_include], inplace=True)

# 1. Invert the data matrix to calculate Xi^(-1)
intermediate_salesshares = np.identity(len(iorequirements_wide_fillna)) - np.linalg.inv(iorequirements_wide_fillna)
# 2. Subtract this inverse from the identity matrix. This is Omega'*diag(gamma)
intermediate_salesshares = pd.DataFrame(intermediate_salesshares, index=iorequirements_wide_fillna.index, columns=iorequirements_wide_fillna.columns)
intermediate_salesshares = intermediate_salesshares.sum(axis=0)

# merge with concordance
intermediate_salesshares = intermediate_salesshares.reset_index().rename(columns={'desc_O': 'NAICS_desc', 0: 'intermediate_salesshare'})
concordance_naics6 = 'concordance6_naics6_addproportions'
concordance_calculateproportion = pd.read_pickle(path_cleandata + 'concordance//' + concordance_naics6 + '.pkl')
merge_with_concordance = pd.merge(left=intermediate_salesshares, right=concordance_calculateproportion, how='inner', on='NAICS_desc')
intermediate_salesshares_byproduct = merge_with_concordance[['product', 'intermediate_salesshare']].groupby('product').mean()

intermediate_salesshares_byproduct.to_pickle(path_cleandata + 'firstinversion//intermediate_salesshares.pkl')