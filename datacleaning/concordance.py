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
concordance = concordance[~(concordance['product'].str.contains('Hotels and motels') & ~concordance['NAICS_desc'].str.contains('Accomodation'))]

# only keep electricity generation and utilities
concordance = concordance[~((concordance['product'].str.contains('Electricity') & ~concordance['NAICS_desc'].str.contains('Electric power generation')) |
                            (concordance['product'].str.contains('Electricity') & ~concordance['NAICS_desc'].str.contains('Federal electric utilities')))]

# only keep wireless communications
concordance = concordance[~(concordance['product'].str.contains('Cellular telephone services') & concordance['NAICS_desc'].str.contains('Wired telecommunications carriers'))]

# remove amusement parks from toys
concordance = concordance[~(concordance['product'].str.contains('Games, toys, and hobbies') & concordance['NAICS_desc'].str.contains('Amusement parks and arcades'))]

# remove "Professional and commercial equipment and supplies"
concordance = concordance[~(concordance['product'].str.contains('Personal computers/tablets and peripheral equipment') & concordance['NAICS_desc'].str.contains('Professional and commercial equipment and supplies'))]
# add "Household appliances and electrical and electronic goods"
concordance.loc[len(concordance)] = ['Personal computers/tablets and peripheral equipment','Household appliances and electrical and electronic goods']

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

# include only petroleum refineries
concordance = concordance[~(concordance['product'].str.contains('Other fuels') & ~concordance['NAICS_desc'].str.contains('Petroleum refineries'))]

# include only computer systems design services
concordance = concordance[~(concordance['product'].str.contains('Computer software and accessories') & ~concordance['NAICS_desc'].str.contains('Computer systems design services'))]
# add "Household appliances and electrical and electronic goods"
concordance.loc[len(concordance)] = ['Computer software and accessories','Household appliances and electrical and electronic goods']

# remove "Glass and glass product manufacturing"
concordance = concordance[~(concordance['product'].str.contains('Corrective eyeglasses') & concordance['NAICS_desc'].str.contains('Glass and glass product manufacturing'))]

# include only distilleries
concordance = concordance[~(concordance['product'].str.contains('Spirits') & ~concordance['NAICS_desc'].str.contains('Distilleries'))]

# include only greenhouse, nursery, and floriculture production
concordance = concordance[~(concordance['product'].str.contains('Flowers, seeds, and potted plants') & ~concordance['NAICS_desc'].str.contains('Greenhouse, nursery, and floriculture production'))]

# remove "Commercial and industrial machinery and equipment repair and maintenance"
concordance = concordance[~(concordance['product'].str.contains('Moving, storage, and freight') & concordance['NAICS_desc'].str.contains('Commercial and industrial machinery and equipment repair and maintenance'))]
# add warehousing and storage
concordance.loc[len(concordance)] = ['Moving, storage, and freight','Warehousing and storage']

# add paperboard container manufacturing	
concordance.loc[len(concordance)] = ['Household paper products','Paperboard container manufacturing']
# add printing
concordance.loc[len(concordance)] = ['Household paper products','Printing']
# add stationery product manufacturing
concordance.loc[len(concordance)] = ['Household paper products','Stationery product manufacturing']

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

concordance.to_pickle(path_cleandata + 'concordance//concordance6_naics6.pkl')

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
