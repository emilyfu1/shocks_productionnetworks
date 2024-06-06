import pandas as pd
from functions import pce_tables_clean, inputoutput_clean, requirements_clean
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'

# import BEA tables (price indexes, quantity indexes, expenditures by product)
# indexes
pce_quantityindex = pd.read_excel(path_rawdata + '2_4_3U.xlsx')
pce_priceindex = pd.read_excel(path_rawdata + '2_4_4U.xlsx')
# expenditures
pce = pd.read_excel(path_rawdata + '2_4_5U.xlsx')
# setup for merge (and data cleaning)
pce_quantityindex = pce_tables_clean(pce_quantityindex)
pce_quantityindex = pce_quantityindex.rename(columns={'index': 'quantityindex'})
pce_priceindex = pce_tables_clean(pce_priceindex)
pce_priceindex = pce_priceindex.rename(columns={'index': 'priceindex'})
pce = pce_tables_clean(pce)
pce = pce.rename(columns={'index': 'expenditures'})
# merge these together
pce_clean = pd.merge(left=pce_quantityindex, right=pce_priceindex, on=['product', 'date'], how='outer')
pce_clean = pd.merge(left=pce_clean, right=pce, on=['product', 'date'], how='outer')
# save
pce_clean.to_pickle(path_cleandata + 'BEA_PCE.pkl')



# import input-output USE matrix
'''
the make matrix is a I x C matrix where C is a bunch of commodities and I is a bunch of industries that use these commodities
the use matrix is a C x (I + N) matrix where N is the number of final demand categories. 
the input-output matrix is either I x I or C x C that shows either uses of industries in other industries or uses of commodities in other commodities

commodities can be used by industries or used in final demand categories. 
we will be interested in final demand (i.e. all final demand categories), i.e. total use - total intermediates

for example, 
1. you can buy a wheel of cheese and eat the whole thing (personal consumption)
2. a restaurant owner can also buy it to make a meal to then sell to people (used as an intermediate good by the restaurant industry)
3. CSIS can buy the cheese to stick a camera into and spy on someone (government expenditure)

if M is the make matrix, then the entry M_i,c denotes how much of commodity c is made by industry i.
If the use matrix is denoted by U, then the entry U_c,i denotes the amount of commodity c used by industry i or final demand purpose i.
'''

IO_usetable = pd.read_excel(path_rawdata + 'Use_SUT_Framework_2017_DET.xlsx', sheet_name='2017')
# convert into my format
IO_usetable = inputoutput_clean(IO_usetable)
# save
IO_usetable.to_pickle(path_cleandata + 'use_naics6.pkl')


# import input-output REQUIREMENTS matrix
'''
this is the leontief inverse of the I-O matrix, which you get as (I_n − A)^(−1) for some n-sector I-O matrix A and a same-dimension identity matrix
# this represents the amount of gross output from industry i that is produced to satisfy a unit of final demand y from industry j

(n.b. around the code, i am referring to industries as sectors interchangeably. 
however, when i refer to products that is a different thing, as we are merging industries to products later on in the code)
'''

industry_requirements = pd.read_excel(path_rawdata + 'IxI_TR_2017_PRO_Det.xlsx', sheet_name='2017')
# convert into my format
industry_requirements = requirements_clean(industry_requirements, wide=True)
# save
industry_requirements.to_pickle(path_cleandata + 'requirements_naics6.pkl')