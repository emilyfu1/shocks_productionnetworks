import pandas as pd
from functions import pce_tables_clean, inputoutput_clean
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
IO_supplytable = pd.read_excel(path_rawdata + 'Supply_2017_DET.xlsx', sheet_name='2017')
IO_usetable = pd.read_excel(path_rawdata + 'Use_SUT_Framework_2017_DET.xlsx', sheet_name='2017')

# convert into my format
IO_supplytable = inputoutput_clean(IO_supplytable)
IO_usetable = inputoutput_clean(IO_usetable)

# save
IO_supplytable.to_pickle(path_cleandata + 'supply.pkl')
IO_usetable.to_pickle(path_cleandata + 'use.pkl')