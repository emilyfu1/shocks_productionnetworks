import pandas as pd
from functions import filter_by_granularity, merge_IO_BEA
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput_U = pd.read_pickle(path_cleandata + 'use.pkl')
# goods at 6th level of granularity
bea6 = filter_by_granularity(bea_products, target_granularity=6)

# get crosswalk
concordance_naics6 = 'concordance6_naics6'
concordance_calculateproportion = pd.read_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')[['product', 'NAICS_desc']]

# merge data
bea6_IO_U = merge_IO_BEA(inputoutput=inputoutput_U, bea=bea6, crosswalk_filename=concordance_naics6)
bea6_IO_U.to_pickle(path_cleandata + 'BEA6_naics6_merged.pkl')
