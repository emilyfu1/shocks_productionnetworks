import pandas as pd
from functions import create_crosswalk, filter_by_granularity
from dotenv import dotenv_values, find_dotenv
import os

config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'
path_figures = os.path.abspath(config["FIGURES"]) + '\\'

# import data
bea_products = pd.read_pickle(path_cleandata + 'BEA_PCE.pkl')
inputoutput = pd.read_pickle(path_cleandata + 'use_naics6.pkl')
# goods at 6th level of granularity
bea = filter_by_granularity(bea_products, target_granularity=6)
# make sure bea products actually have 2017 data
products_2017 = bea[(bea['date'].dt.year == 2017) & (bea['expenditures'].notnull())]['product'].unique()
bea = bea[bea['product'].isin(products_2017)]
bea['product'] = bea['product'].str.lstrip()

# run concordance function
concordance = create_crosswalk(inputoutput=inputoutput, bea=bea)
concordance = concordance[['product', 'NAICS_desc']]

concordance.to_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')

# some (or a lot) manual fixes for the concordance file for persistent issues that the matching algorithm doesn't catch
# not sure how better to address these

# remove personal consumption expenditures as a match for any products that aren't personal consumption expenditures
# if personal consumption expenditures isn't in the product column, it shouldn't be the matching MAICS_desc either
concordance = concordance[~((concordance['product'].str.contains('Personal consumption expenditures') & ~concordance['NAICS_desc'].str.contains('Personal consumption expenditures')) |
                            (~concordance['product'].str.contains('Personal consumption expenditures') & concordance['NAICS_desc'].str.contains('Personal consumption expenditures')))]

# remove "Other animal food manufacturing" for anything that isn't about pets
concordance = concordance[~(~concordance['product'].str.contains('Pets and related products') & concordance['NAICS_desc'].str.contains('Other animal food manufacturing'))]

# remove the current "lotteries" matches because they aren't good
concordance = concordance[~(concordance['product'].str.contains('Lotteries'))]

# remove the current "Food products, not elsewhere classified" matches because they aren't good
concordance = concordance[~(concordance['product'].str.contains('Food products, not elsewhere classified'))]

# remove bad match for computer software
concordance = concordance[~(concordance['product'].str.contains('Computer software and accessories') & concordance['NAICS_desc'].str.contains('Professional and commercial equipment and supplies'))]

# remove bad match for other purchased meals
concordance = concordance[~(concordance['product'].str.contains('Other purchased meals') & concordance['NAICS_desc'].str.contains('Snack food manufacturing'))]

# remove current matches for other fuels
concordance = concordance[~(concordance['product'].str.contains('Other fuels'))]

# remove "Gross operating surplus"
concordance = concordance[~(concordance['NAICS_desc'].str.contains('Gross operating surplus'))]

# remove bad matches for hotels and motels
concordance = concordance[~(concordance['product'].str.contains('Hotels and motels') & concordance['NAICS_desc'].str.contains('Services to buildings and dwellings'))]
concordance = concordance[~(concordance['product'].str.contains('Hotels and motels') & concordance['NAICS_desc'].str.contains('Nursing and community care facilities'))]

# remove bad match for insurance
concordance = concordance[~(concordance['product'].str.contains('Net motor vehicle and other transportation insurance') & concordance['NAICS_desc'].str.contains('Motor vehicle and parts dealers'))]
concordance = concordance[~(concordance['product'].str.contains('Net motor vehicle and other transportation insurance') & concordance['NAICS_desc'].str.contains('Direct life insurance carriers'))]
concordance = concordance[~(concordance['product'].str.contains('Household insurance premiums') & concordance['NAICS_desc'].str.contains('Health and personal care stores'))]

# remove bad match for social services
concordance = concordance[~(concordance['product'].str.contains('Social services, gross output') & concordance['NAICS_desc'].str.contains('Accounting, tax preparation, bookkeeping, and payroll services'))]

# remove bad match for outpatient services
concordance = concordance[~(concordance['product'].str.contains('Outpatient services, gross output') & ~concordance['NAICS_desc'].str.contains('Outpatient care centers'))]

# remove current matches for "Hair, dental, shaving, and miscellaneous personal care products except electrical products"
concordance = concordance[~(concordance['product'].str.contains('Hair, dental, shaving, and miscellaneous personal care products except electrical products'))]

# remove bad match for drinks
concordance = concordance[~(concordance['product'].str.contains('Mineral waters, soft drinks, and vegetable juices') & concordance['NAICS_desc'].str.contains('Fruit and vegetable canning, pickling, and drying'))]

# ADDING ROWS
new_rows = [{'product': 'Hotels and motels', 'NAICS_desc': 'Travel arrangement and reservation services'},
            {'product': 'Furniture', 'NAICS_desc': 'Upholstered household furniture manufacturing'},
            {'product': 'Furniture', 'NAICS_desc': 'Nonupholstered wood household furniture manufacturing'},
            {'product': 'Furniture', 'NAICS_desc': 'Institutional furniture manufacturing'},
            {'product': 'Furniture', 'NAICS_desc': 'Office furniture and custom architectural woodwork and millwork manufacturing'},
            {'product': 'Furniture', 'NAICS_desc': 'Other furniture related product manufacturing'},
            {'product': 'Lotteries', 'NAICS_desc': 'Gambling industries (except casino hotels)'},
            {'product': 'Gasoline and other motor fuel', 'NAICS_desc': 'Oil and gas extraction'},
            {'product': 'Gasoline and other motor fuel', 'NAICS_desc': 'Drilling oil and gas wells'},
            {'product': 'Food products, not elsewhere classified', 'NAICS_desc': 'All other food manufacturing'},
            {'product': 'Computer software and accessories', 'NAICS_desc': 'Software publishers'},
            {'product': 'Cellular telephone services', 'NAICS_desc': 'Satellite, telecommunications resellers, and all other telecommunications'},
            {'product': 'Other purchased meals', 'NAICS_desc': 'Full-service restaurants'},
            {'product': 'Other purchased meals', 'NAICS_desc': 'Limited-service restaurants'},
            {'product': 'Alcohol in purchased meals', 'NAICS_desc': 'Full-service restaurants'},
            {'product': 'Alcohol in purchased meals', 'NAICS_desc': 'Limited-service restaurants'},
            {'product': 'Other fuels', 'NAICS_desc': 'Coal mining'},
            {'product': 'Other fuels', 'NAICS_desc': 'Natural gas distribution'},
            {'product': 'Social services, gross output', 'NAICS_desc': 'Residential mental health, substance abuse, and other residential care facilities'},
            {'product': 'Social services, gross output', 'NAICS_desc': 'Individual and family services'},
            {'product': 'Social services, gross output', 'NAICS_desc': 'Community food, housing, and other relief services, including rehabilitation services'},
            {'product': 'Hair, dental, shaving, and miscellaneous personal care products except electrical products', 'NAICS_desc': 'Health and personal care stores'},
            {'product': 'Hair, dental, shaving, and miscellaneous personal care products except electrical products', 'NAICS_desc': 'Soap and cleaning compound manufacturing'},
            {'product': 'Hair, dental, shaving, and miscellaneous personal care products except electrical products', 'NAICS_desc': 'Medicinal and botanical manufacturing'},
            {'product': 'Hair, dental, shaving, and miscellaneous personal care products except electrical products', 'NAICS_desc': 'Sanitary paper product manufacturing'},
            {'product': 'Mineral waters, soft drinks, and vegetable juices', 'NAICS_desc': 'Soft drink and ice manufacturing'},]

# PROPORTIONS (SPLITTING UP I-O VALUE)

# use 2017 expenditures to calculate proportions of sorts
bea2017 = bea[bea['date'].dt.year == 2017][['product', 'expenditures']].groupby('product').sum(min_count=1).reset_index()

# merge 2017 together with concordance 
concordance_calculateproportion = pd.merge(left=concordance, right=bea2017, on='product', how='inner')

# calculate total expenditures for each NAICS sector description
sector_totals = concordance_calculateproportion.groupby('NAICS_desc')['expenditures'].transform('sum')

# calculate I-O proportions
concordance_calculateproportion['IO_proportions'] = concordance_calculateproportion['expenditures'] / sector_totals

# use for merging
concordance_calculateproportion = concordance_calculateproportion[['product', 'NAICS_desc', 'IO_proportions']]

concordance_calculateproportion.to_pickle(path_cleandata + 'concordance//concordance6_naics6_addproportions.pkl')
