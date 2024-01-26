import pandas as pd
from functions import filter_by_granularity, merge_IO_BEA
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput_U = pd.read_pickle(path_cleandata + 'use.pkl')

bea4 = filter_by_granularity(bea_products, target_granularity=4)

bea6_IO_S = merge_IO_BEA(inputoutput=inputoutput_U, bea=bea4)
bea6_IO_S.to_pickle(path_cleandata + 'BEA4_IOuse_merged.pkl')
