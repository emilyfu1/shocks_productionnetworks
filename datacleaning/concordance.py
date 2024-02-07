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

# filter bea data
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

# for any tenant-occupied housing services, change NAICS_desc to Tenant-occupied housing
concordance.loc[concordance['product'].str.contains('Tenant-occupied'), 'NAICS_desc'] = 'Tenant-occupied housing'

# only keep tobacco manufacturing
concordance = concordance[~(concordance['product'].str.contains('Tobacco') & ~concordance['NAICS_desc'].str.contains('Tobacco manufacturing'))]

# only keep outpatient services
concordance = concordance[~(concordance['product'].str.contains('Outpatient services, gross output') & ~concordance['NAICS_desc'].str.contains('Outpatient care centers'))]

# only keep accomodations
concordance = concordance[~(concordance['product'].str.contains('Hotels and motels') & ~concordance['NAICS_desc'].str.contains('Accommodation'))]

# only keep electricity generation and utilities
concordance = concordance[~((concordance['product'].str.contains('Electricity') & ~concordance['NAICS_desc'].str.contains('Electric power generation')) |
                            (concordance['product'].str.contains('Electricity') & ~concordance['NAICS_desc'].str.contains('Federal electric utilities')))]

# only keep wireless communications
concordance = concordance[~(concordance['product'].str.contains('Cellular telephone services') & concordance['NAICS_desc'].str.contains('Wired telecommunications carriers'))]

# remove amusement parks from toys
concordance = concordance[~(concordance['product'].str.contains('Games, toys, and hobbies') & concordance['NAICS_desc'].str.contains('Amusement parks and arcades'))]

# remove "Professional and commercial equipment and supplies"
concordance = concordance[~(concordance['product'].str.contains('Personal computers/tablets and peripheral equipment') & concordance['NAICS_desc'].str.contains('Professional and commercial equipment and supplies'))]

# remove "Other animal food manufacturing"
concordance = concordance[~(concordance['product'].str.contains('Beef and veal') & concordance['NAICS_desc'].str.contains('Other animal food manufacturing'))]

# only keep breweries
concordance = concordance[~(concordance['product'].str.contains('Beer') & ~concordance['NAICS_desc'].str.contains('Breweries'))]

# add "Motor vehicle and parts dealers"
concordance.loc[len(concordance)] = ['Used auto margin','Motor vehicle and parts dealers']

# only keep sugar and confectionery product manufacturing
concordance = concordance[~(concordance['product'].str.contains('Sugar and sweets') & ~concordance['NAICS_desc'].str.contains('Sugar and confectionery product manufacturing'))]

# remove "Motor home manufacturing"
concordance = concordance[~(concordance['product'].str.contains('New domestic autos') & concordance['NAICS_desc'].str.contains('Motor home manufacturing'))]

# remove "Sporting and athletic goods manufacturing"
concordance = concordance[~(concordance['product'].str.contains('Membership clubs') & concordance['NAICS_desc'].str.contains('Sporting and athletic goods manufacturing'))]

# remove household appliances
concordance = concordance[~(concordance['product'].str.contains('Clocks, lamps, lighting') & concordance['NAICS_desc'].str.contains('Household appliances and electrical and electronic goods'))]

# remove veterinary services
concordance = concordance[~(concordance['product'].str.contains('Pets and related products') & concordance['NAICS_desc'].str.contains('Veterinary services'))]

# include only civic, social, professional, and similar organizations
concordance = concordance[~(concordance['product'].str.contains('Professional advocacy, gross output') & ~concordance['NAICS_desc'].str.contains('Civic, social, professional, and similar organizations'))]

# include only coal (remove all others)
concordance = concordance[~(concordance['product'].str.contains('Other fuels'))]

# include only computer systems design services
concordance = concordance[~(concordance['product'].str.contains('Computer software and accessories') & ~concordance['NAICS_desc'].str.contains('Computer systems design services'))]

# remove "Glass and glass product manufacturing"
concordance = concordance[~(concordance['product'].str.contains('Corrective eyeglasses') & concordance['NAICS_desc'].str.contains('Glass and glass product manufacturing'))]

# include only distilleries
concordance = concordance[~(concordance['product'].str.contains('Spirits') & ~concordance['NAICS_desc'].str.contains('Distilleries'))]

# include only greenhouse, nursery, and floriculture production
concordance = concordance[~(concordance['product'].str.contains('Flowers, seeds, and potted plants') & ~concordance['NAICS_desc'].str.contains('Greenhouse, nursery, and floriculture production'))]

# remove "Commercial and industrial machinery and equipment repair and maintenance"
concordance = concordance[~(concordance['product'].str.contains('Moving, storage, and freight') & concordance['NAICS_desc'].str.contains('Commercial and industrial machinery and equipment repair and maintenance'))]

# remove "Warehousing and storage"
concordance = concordance[~(concordance['product'].str.contains('Garbage and trash collection') & concordance['NAICS_desc'].str.contains('Warehousing and storage'))]

# remove "Personal and household goods repair and maintenance"
concordance = concordance[~(concordance['product'].str.contains('Household cleaning products') & concordance['NAICS_desc'].str.contains('Personal and household goods repair and maintenance'))]

# remove "Periodical Publishers"
concordance = concordance[~(concordance['product'].str.contains('Recreational books') & concordance['NAICS_desc'].str.contains('Periodical Publishers'))]

# remove "Support activities for printing"
concordance = concordance[~(concordance['product'].str.contains('Stationery and miscellaneous') & concordance['NAICS_desc'].str.contains('Support activities for printing'))]

# remove "Hospitals"
concordance = concordance[~(concordance['product'].str.contains('Funeral and burial services') & concordance['NAICS_desc'].str.contains('Hospitals'))]

# remove "Household appliances and electrical and electronic goods"
concordance = concordance[~(concordance['product'].str.contains('Household cleaning products') & concordance['NAICS_desc'].str.contains('Hospitals'))]

# remove "State and local government consumption expenditures"
concordance = concordance[~(concordance['product'].str.contains('Foreign travel in the United States') & concordance['NAICS_desc'].str.contains('State and local government consumption expenditures'))]

# remove "Motor vehicle and parts dealers"
concordance = concordance[~(concordance['product'].str.contains('Net motor vehicle and other transportation insurance (116)') & concordance['NAICS_desc'].str.contains('Motor vehicle and parts dealers'))]

# add gambling and remove everything else
concordance = concordance[~(concordance['product'].str.contains('Lotteries'))]

# remove "Motor vehicle gasoline engine and engine parts"
concordance = concordance[~(concordance['product'].str.contains('Gasoline and other motor fuel') & concordance['NAICS_desc'].str.contains('Motor vehicle gasoline engine and engine parts'))]

# remove "Accounting, tax preparation, bookkeeping, and payroll services"
concordance = concordance[~(concordance['product'].str.contains('Social services, gross output') & concordance['NAICS_desc'].str.contains('Accounting, tax preparation, bookkeeping, and payroll services'))]

# remove "Snack food manufacturing"
concordance = concordance[~(concordance['product'].str.contains('Other purchased meals') & concordance['NAICS_desc'].str.contains('Snack food manufacturing'))]

# remove "Motor vehicle and parts dealers"
concordance = concordance[~(concordance['product'].str.contains('Net motor vehicle and other transportation insurance (116)') & concordance['NAICS_desc'].str.contains('Motor vehicle and parts dealers'))]

# remove "Gross operating surplus"
concordance = concordance[~(concordance['NAICS_desc'].str.contains('Gross operating surplus'))]

# ADDING ROWS
new_rows = [{'product': 'Hotels and motels', 'NAICS_desc': 'Travel arrangement and reservation services'},
            {'product': 'Foreign travel in the United States', 'NAICS_desc': 'Accommodation'},
            {'product': 'Personal computers/tablets and peripheral equipment', 'NAICS_desc': 'Household appliances and electrical and electronic goods'},
            {'product': 'Other fuels', 'NAICS_desc': 'Coal mining'},
            {'product': 'Moving, storage, and freight', 'NAICS_desc': 'Warehousing and storage'},
            {'product': 'Household paper products', 'NAICS_desc': 'Paperboard container manufacturing'},
            {'product': 'Household paper products', 'NAICS_desc': 'Printing'},
            {'product': 'Household paper products', 'NAICS_desc': 'Stationery product manufacturing'},
            {'product': 'Lotteries', 'NAICS_desc': 'Gambling industries (except casino hotels)'},
            {'product': 'Gasoline and other motor fuel', 'NAICS_desc': 'Gasoline stations'},
            {'product': 'Gasoline and other motor fuel', 'NAICS_desc': 'OIl and gas extraction'},
            {'product': 'Gasoline and other motor fuel', 'NAICS_desc': 'Drilling oil and gas wells'}]
concordance = pd.concat([concordance, pd.DataFrame(new_rows)], ignore_index=True)


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
