import pandas as pd
from functions import filter_by_granularity, merge_IO_BEA
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput_U = pd.read_pickle(path_cleandata + 'use.pkl')
inputoutput_S = pd.read_pickle(path_cleandata + 'supply.pkl')

# two different versions of the pce index tables:
bea4 = filter_by_granularity(bea_products, target_granularity=4)
bea6 = filter_by_granularity(bea_products, target_granularity=6)

bea6_IO = merge_IO_BEA(inputoutput=inputoutput_S, bea=bea6)
bea6_IO.to_pickle(path_cleandata + 'BEA6_IOsupply_merged.pkl')
