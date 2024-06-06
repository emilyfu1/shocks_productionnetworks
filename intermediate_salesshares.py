import pandas as pd
import numpy as np
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import the industry-by-industry I-O requirements table
requirementstable = pd.read_pickle(path_cleandata + 'requirements_naics6.pkl')
iorequirements_wide_fillna = requirementstable.fillna(value=0)

# next, i need sectors that show up as both buyers and sellers
iosectors = list(set(iorequirements_wide_fillna.index) & set(iorequirements_wide_fillna.columns))
iorequirements_wide_fillna = iorequirements_wide_fillna[iosectors].loc[iosectors]
# list of buyers and sellers (remove redundant sectors for which total expenses/sales are zero)
inputsectors = [index for index in iorequirements_wide_fillna.index if iorequirements_wide_fillna.sum(axis=1)[index] != 0 and iorequirements_wide_fillna.sum(axis=0)[index] != 0]
outputsectors = [column for column in iorequirements_wide_fillna.columns if iorequirements_wide_fillna.sum(axis=1)[column] != 0 and iorequirements_wide_fillna.sum(axis=0)[column] != 0]
# get the intersection of these lists
sectors_to_include = list(set(inputsectors) & set(outputsectors))

# filter I-O table and residuals based on this list of sectors
iorequirements_wide_fillna.drop(columns=[col for col in iorequirements_wide_fillna.columns if col not in sectors_to_include], inplace=True)
iorequirements_wide_fillna.drop(index=[idx for idx in iorequirements_wide_fillna.index if idx not in sectors_to_include], inplace=True)

# this basically undoes the inverse to get (I_n − A) and subtracts I_n - (I_n − A) to get the industry-by-industry I-O matrix
iomatrix_wide = np.identity(len(iorequirements_wide_fillna)) - np.linalg.inv(iorequirements_wide_fillna)
iomatrix_wide = pd.DataFrame(iomatrix_wide, index=iorequirements_wide_fillna.index, columns=iorequirements_wide_fillna.columns)
# then this sum gives the share, for each sector, of what is used as intermediates
intermediate_salesshares = iomatrix_wide.sum(axis=0)

# merge with concordance to get product-level intermediate shares of sales

intermediate_salesshares = intermediate_salesshares.reset_index().rename(columns={'desc_O': 'NAICS_desc', 0: 'intermediate_salesshare'})
# import concordance
concordance_naics6 = 'concordance6_naics6'
concordance_calculateproportion = pd.read_pickle(path_cleandata + 'concordance//' + concordance_naics6 + '.pkl')
# perform merge
merge_with_concordance = pd.merge(left=intermediate_salesshares, right=concordance_calculateproportion, how='inner', on='NAICS_desc')
# just get a simple mean of each intermediate share by product
intermediate_salesshares_byproduct = merge_with_concordance[['product', 'intermediate_salesshare']].groupby('product').mean()
# save
intermediate_salesshares_byproduct.to_pickle(path_cleandata + 'inversions//intermediate_salesshares.pkl')