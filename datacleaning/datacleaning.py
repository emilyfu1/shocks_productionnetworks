import pandas as pd
from functions import pce_tables_clean, inputoutput_clean, requirements_clean
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'

# import BEA tables
# indexes
pce_quantityindex = pd.read_excel(path_rawdata + '2_4_3U.xlsx')
pce_priceindex = pd.read_excel(path_rawdata + '2_4_4U.xlsx')
# expenditures
pce = pd.read_excel(path_rawdata + '2_4_5U.xlsx')
# setup for merge
pce_quantityindex = pce_tables_clean(pce_quantityindex)
pce_quantityindex = pce_quantityindex.rename(columns={'index': 'quantityindex'})
pce_priceindex = pce_tables_clean(pce_priceindex)
pce_priceindex = pce_priceindex.rename(columns={'index': 'priceindex'})
pce = pce_tables_clean(pce)
pce = pce.rename(columns={'index': 'expenditures'})
pce_clean = pd.merge(left=pce_quantityindex, right=pce_priceindex, on=['product', 'date'], how='outer')
pce_clean = pd.merge(left=pce_clean, right=pce, on=['product', 'date'], how='outer')
# save
pce_clean.to_pickle(path_cleandata + 'BEA_PCE.pkl')

# import input-output tables
IO_usetable = pd.read_excel(path_rawdata + 'Use_SUT_Framework_2017_DET.xlsx', sheet_name='2017')
# convert into my format
IO_usetable = inputoutput_clean(IO_usetable)
# save
IO_usetable.to_pickle(path_cleandata + 'use_naics6.pkl')

# import I/O requirements table
industry_requirements = pd.read_excel(path_rawdata + 'IxI_TR_2017_PRO_Det.xlsx', sheet_name='2017')
industry_requirements = requirements_clean(industry_requirements, wide=True)
industry_requirements.to_pickle(path_cleandata + 'requirements_naics6.pkl')