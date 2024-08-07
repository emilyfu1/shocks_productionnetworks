import pandas as pd
from functions import filter_by_granularity, merge_IO_BEA
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput_U = pd.read_pickle(path_cleandata + 'use_naics6.pkl')
# goods at 6th level of granularity
bea = filter_by_granularity(bea_products, target_granularity=6)
# find the products that actually have data for 2017, which is the year the I-O data comes from (which is most of them)
products_2017 = bea[(bea['date'].dt.year == 2017) & (bea['expenditures'].notnull())]['product'].unique()
bea = bea[bea['product'].isin(products_2017)]

# merge data
concordance_naics6 = 'concordance6_naics6_addproportions'
bea6_IO_U = merge_IO_BEA(inputoutput=inputoutput_U, bea=bea, crosswalk_filename=concordance_naics6)
# save
bea6_IO_U.to_pickle(path_cleandata + 'BEA6_naics6_merged.pkl')
