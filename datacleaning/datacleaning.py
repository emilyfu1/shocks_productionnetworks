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
IO_usetable = pd.read_excel(path_rawdata + 'Use_SUT_Framework_2017_DET.xlsx', sheet_name='2017')
# convert into my format
IO_usetable = inputoutput_clean(IO_usetable)
# quick remove rest-of-world adjustment
IO_usetable = IO_usetable[IO_usetable['desc_I'] != 'Rest of the world adjustment']
# save
IO_usetable.to_pickle(path_cleandata + 'use.pkl')

# another version of the input-output table using 4-digit naics

# full list of naics codes that i found online
naicstonaics = pd.read_excel(path_rawdata + '2022_NAICS_Structure.xlsx', skiprows=2)[['2022 NAICS Code', '2022 NAICS Title']]
# trim footnote thing
naicstonaics['2022 NAICS Title'] = naicstonaics['2022 NAICS Title'].str.rstrip('T ')
# 4-digit naics
naicstonaics = naicstonaics[naicstonaics['2022 NAICS Code'].apply(lambda x: len(str(x))==4)]
# convert code to string
naicstonaics['2022 NAICS Code'] = naicstonaics['2022 NAICS Code'].astype('str')
# append personal consumption expenditures
naicstonaics.loc[len(naicstonaics)] = ['F010', 'Personal consumption expenditures']
# dict
naicstonaics = dict(zip(naicstonaics['2022 NAICS Code'], naicstonaics['2022 NAICS Title']))

# get 4 digit naics
IO_usetable4 = IO_usetable.copy()
IO_usetable4['NAICS4_I'] = IO_usetable4['NAICS_I'].apply(lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 4 else str(x))
IO_usetable4['NAICS4_O'] = IO_usetable4['NAICS_O'].apply(lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 4 else str(x))
IO_usetable4 = IO_usetable4[['NAICS4_I', 'desc_I', 'NAICS4_O', 'desc_O', 'value']]
# get 4 digit naics descriptions
IO_usetable4['desc_I'] = IO_usetable4['NAICS4_I'].map(naicstonaics)
IO_usetable4['desc_O'] = IO_usetable4['NAICS4_O'].map(naicstonaics)
# group by 4 digit naics
IO_usetable4 = IO_usetable4.groupby(['NAICS4_I', 'NAICS4_O', 'desc_I', 'desc_O'], as_index=False)['value'].sum(min_count=1)
IO_usetable4['value'] = pd.to_numeric(IO_usetable4['value'], errors='coerce')
# save
IO_usetable4.to_pickle(path_cleandata + 'use_naics4.pkl')