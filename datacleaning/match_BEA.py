import pandas as pd
from functions import filter_by_granularity, merge_IO_BEA, create_crosswalk
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
crosswalk_naics6 = 'concordance6_naics6'
crosswalk = create_crosswalk(inputoutput=inputoutput_U, bea=bea6, crosswalk_filename=crosswalk_naics6)
crosswalk.to_pickle(path_cleandata + 'concordance//' + crosswalk_naics6 + '.pkl')

# merge data
bea6_IO_S = merge_IO_BEA(inputoutput=inputoutput_U, bea=bea6, crosswalk_filename=crosswalk_naics6)
bea6_IO_S.to_pickle(path_cleandata + 'BEA6_IOuse_merged.pkl')
