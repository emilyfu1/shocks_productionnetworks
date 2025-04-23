import pandas as pd
from functions import filter_by_granularity
from dotenv import dotenv_values, find_dotenv
import os

config = dotenv_values(find_dotenv())
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# import PCE data (we will be using the EXPENDITURES data below)
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
# import use table
inputoutput = pd.read_pickle(path_cleandata + 'use_naics6.pkl')
# only look at goods at 6th level of granularity wherever possible (this function filters for it)
bea = filter_by_granularity(bea_products, target_granularity=6)
# find the products that actually have data for 2017, which is the year the I-O data comes from (which is most of them)
products_2017 = bea[(bea['date'].dt.year == 2017) & (bea['expenditures'].notnull())]['product'].unique()

# load raw concordance file provided by the BEA that maps sectors to products
concordance = pd.read_excel(path_rawdata + 'PCEBridge_2017_DET.xlsx', sheet_name='2017')
# clean up column names
concordance = concordance[4:].rename(columns={'Unnamed: 1': 'product', 'Unnamed: 3': 'NAICS_desc'})[['product', 'NAICS_desc']]
# save this version 
concordance.to_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')


# CALCULATE PROPORTIONS
'''
let's say there are only 3 products (cars, boats, motorcycles) and only 5 sectors (auto parts, vehicle insurance, airbag manufacturing, outdoor equipment, and lumber)
expenditure on cars is 100$ in 2017
expenditure on boats is 200$ in 2017
expenditure on motorcycles is 50$ in 2017

when we merge the PCE data with the I-O use table, we are going to convert the use table to a product-by-product format
the values in the use table will be aggregated based on the products in the PCE data
the proportions calculated below will help us account for the fact that some sectors get matched to more than one product

product      sector                 expenditure on product   expenditure on sector   proportions
car          auto parts             $100                     $100 + $50              $100 / ($100 + $50)
car          vehicle insurance      $100                     $100 + $200 + $50       $100 / ($100 + $200 + $50)  
car          airbag manufacturing   $100                     $100                    $100 / $100

boat         outdoor equipment      $200                     $200                    $200 / $200
boat         lumber                 $200                     $200                    $200 / $200
boat         vehicle insurance      $200                     $100 + $200 + $50       $200 / ($100 + $200 + $50)  

motorcycle   vehicle insurance      $50                      $100 + $200 + $50       $50 / ($100 + $200 + $50)  
motorcycle   auto parts             $50                      $100 + $50              $50 / ($100 + $50)

'''

# calculate total expenditures for each product for all of 2017
bea2017 = bea[bea['date'].dt.year == 2017][['product', 'expenditures']].groupby('product').sum(min_count=1).reset_index()

# merge 2017 expenditures together with concordance 
concordance_calculateproportion = pd.merge(left=concordance, right=bea2017, on='product', how='inner')

# calculate total expenditures for each corresponding NAICS sector description
sector_totals = concordance_calculateproportion.groupby('NAICS_desc')['expenditures'].transform('sum')

# calculate I-O proportions
concordance_calculateproportion['IO_proportions'] = concordance_calculateproportion['expenditures'] / sector_totals

# the columns that we will need
concordance_calculateproportion = concordance_calculateproportion[['product', 'NAICS_desc', 'IO_proportions']]

# add line for final demand
concordance_calculateproportion.loc[len(concordance_calculateproportion)] = ['Personal consumption expenditures','fd_all',1]

# save
concordance_calculateproportion.to_pickle(path_cleandata + 'concordance//concordance6_naics6_addproportions.pkl')
