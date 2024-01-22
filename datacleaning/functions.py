import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import dotenv_values, find_dotenv
import os
config = dotenv_values(find_dotenv())
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'

# formats the quantity/price index tables 
def pce_tables_clean(df):
    # put rows 2 and 3 together to get a real date row and make that the set of column names instead
    new_row = df.iloc[2:4].astype(str).apply(''.join, axis=0)

    # replace column names with the concatenated row
    df.columns = new_row

    # drop the empty rows
    df = df.iloc[4:]

    # reset the index
    df = df.reset_index(drop=True)
    
    # assorted data cleaning stuff
    df = df.drop('LineLine', axis=1)
    df = df.rename(columns={'nannan': 'product'})

    # get rid of the weird aggregates that we dont need
    index_to_remove = df.index[df['product'] == 'Additional aggregates:']
    df = df.iloc[:index_to_remove[0]]
    
    # wide to long
    df_long = pd.melt(df, id_vars=['product'], var_name='date', value_name='index')

    # convert to numeric
    df_long['index'] = pd.to_numeric(df_long['index'], errors='coerce')
    
    # convert to datetime
    df_long['date'] = pd.to_datetime(df_long['date'], format='mixed')
    df_long['date'] = df_long['date'] + pd.offsets.MonthEnd(0)

    return df_long

# formats the I-O tables
def inputoutput_clean(df, wide=False):
    # temporarily join rows 3 and 4 (convenient for merging)
    new_row = df.iloc[3:5].astype(str).apply('-- '.join, axis=0)

    # replace column names with the concatenated row
    df.columns = new_row

    # drop the empty rows
    df = df.iloc[5:]

    # reset index
    df = df.reset_index(drop=True)

    # assorted data cleaning stuff
    df = df.rename(columns={'Commodities/Industries-- Code': 'NAICS_I', 'nan-- Commodity Description': 'desc_I'})

    if wide == True:
        # regex pattern to identify descriptions and codes
        to_remove = r'.*--\s(\w+)'

        # Apply the regex pattern to columns with the specified format
        df.columns = df.columns.to_series().replace(to_remove, r'\1', regex=True)

        df.drop('desc_I', axis=1, inplace=True)

        # removing rows with no naics (different for wide format)
        # get rid of the last row since its a footnote in the raw data
        df = df.iloc[:-1]
        df = df.dropna(subset=['NAICS_I'])

        return df
    
    else:
        # wide to long
        df_long = pd.melt(df, id_vars=['NAICS_I', 'desc_I'], var_name='NAICS_desc_O', value_name='value')

        # split the NAICS code and descriptions back up
        df_long[['desc_O', 'NAICS_O']] = df_long['NAICS_desc_O'].str.split('-- ', expand=True)

        # reorder columns
        df_long = df_long [['NAICS_I', 'desc_I', 'NAICS_O', 'desc_O', 'value']]

        # removing rows with no naics
        # specify the value column since i dont want to get rid of nans there
        exclude_columns = ['value']
        df_long = df_long.dropna(subset=df_long.columns.difference(exclude_columns))

        return df_long

''' function for filtering data in BEA data tables (2.4.3U and 2.4.4U)
example:
- first level: goods
- second level: durable goods
- third level: motor vehicles and parts
- fourth level: new motor vehicles
- fifth level: new autos
- sixth level: new domestic autos
output: products of specified granularity and of lower granularity if no further granularity available'''

def filter_by_granularity(df, target_granularity):
    if target_granularity not in [1, 2, 3, 4, 5, 6]:
        raise Exception("Select an appropriate level of granularity")
    
    # collection of rows
    filtered_rows = []

    # loop through every row in the passed dataframe
    for index, row in df.iterrows():
        # find how many spaces are in front of the product name in that ropw
        current_indent = len(row['product']) - len(row['product'].lstrip())
        
        # find out how many indents are in front of the product name (4 spaces per indent)
        # each indent represents products that fall under the previous one
        current_granularity = current_indent // 4
        # append the row with the product if it's the correct granularity
        if current_granularity == target_granularity:
            filtered_rows.append(index)
            # print(current_indent)
            # print(row)
        # if theres a less granular product in that row, investigate it further
        elif current_granularity < target_granularity:
            # check the rows below
            for i in range(index + 1, len(df)):
                # find how many spaces are in front of the product name in that ropw
                below_indent = len(df.loc[i, 'product']) - len(df.loc[i, 'product'].lstrip())
                # find out how many indents are in front of the product name (4 spaces per indent)
                below_granularity = below_indent // 4
                # if the row below is less granular, add the current row
                if below_granularity <= current_granularity:
                    filtered_rows.append(index)
                    break
                else:
                    break
        # ignore the products that are too granular
        else:
            pass

    result_df = df.loc[filtered_rows]
    return result_df

def merge_IO_BEA(inputoutput, bea):
    # all the products included in these versions
    products_bea = list(set(bea['product']))
    # all the NAICS descriptions
    naicsdescriptions = list(set(list(inputoutput['desc_O']) + list(inputoutput['desc_I'])))

    # load the NLP model
    bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # create the crosswalk
    crosswalk = pd.DataFrame(columns=['product', 'NAICS_desc', 'similarity'])
    for product in products_bea:
        print(product)

        # get embeddings for the product category and NAICS sectors
        category_embedding = bert.encode(product, convert_to_tensor=True).reshape(1, -1)
        sector_embeddings = [bert.encode(sector, convert_to_tensor=True).reshape(1, -1) for sector in naicsdescriptions]

        # calculate cosine similarity
        similarities = [cosine_similarity(category_embedding, sector_embedding).item() for sector_embedding in sector_embeddings]

        # filter matches based on the similarity threshold
        # first look for a near-perfect-ish match
        # im taking anything with above 0.7 cosine similarity or the highest 3 matches if none above 70 exist
        matching_indices = [i for i, sim in enumerate(similarities) if sim > 0.95]
        if not matching_indices:
            matching_indices = [i for i, sim in enumerate(similarities) if sim > 0.7]
            if not matching_indices:
                matching_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]

        # append the new matches to the dataframe
        rows = pd.DataFrame({'product': [product] * len(matching_indices),
                            'NAICS_desc': [naicsdescriptions[i] for i in matching_indices],
                            'similarity': [similarities[i] for i in matching_indices]})
        crosswalk = pd.concat([crosswalk, rows], ignore_index=True)
    
    crosswalk.to_pickle(path_cleandata + 'cleandata//condordance.pkl')

    # merging with NAICS I-O table 

    # merging with crosswalk
    crosswalk_I = crosswalk[['product', 'NAICS_desc']].rename(columns={'product': 'product_I', 'NAICS_desc': 'desc_I'})
    crosswalk_O = crosswalk[['product', 'NAICS_desc']].rename(columns={'product': 'product_O', 'NAICS_desc': 'desc_O'})

    # merge each side together
    add_naics_I = pd.merge(left=crosswalk_I, right=inputoutput, on='desc_I', how='right')
    add_naics_O = pd.merge(left=crosswalk_O, right=inputoutput, on='desc_O', how='right')
    IO_naics = pd.merge(left=add_naics_I, right=add_naics_O, on=['NAICS_I', 'desc_I', 'NAICS_O', 'desc_O', 'value'], how='inner')

    # sum all values in the value column of the I-O matrix with the same product_I and product_O
    IO_naics = IO_naics.sort_values(by=['product_I', 'product_O'])
    IO_naics = IO_naics[['product_I', 'product_O', 'value']]
    IO_naics['value'] = pd.to_numeric(IO_naics['value'])
    IO_naics_grouped = IO_naics.groupby(['product_I', 'product_O'], as_index=False)['value'].sum(min_count=1)

    # left merge (keep everything in I-O)

    # merge with BEA table (I)
    IO_naics_I = pd.merge(left=IO_naics_grouped, right=bea, left_on='product_I', right_on='product', how='left')
    IO_naics_I.drop(columns=['product'], inplace=True)
    IO_naics_I.rename(columns={
        'value': 'IO_value',
        'quantityindex': 'quantityindex_I',
        'priceindex': 'priceindex_I',
        'expenditures': 'expenditures_I'
    }, inplace=True)

    # merge with BEA table (O)
    IO_naics_O = pd.merge(left=IO_naics_grouped, right=bea, left_on='product_O', right_on='product', how='left')
    IO_naics_O.drop(columns=['product'], inplace=True)
    IO_naics_O.rename(columns={
        'value': 'IO_value',
        'quantityindex': 'quantityindex_O',
        'priceindex': 'priceindex_O',
        'expenditures': 'expenditures_O'
    }, inplace=True)

    IO_naics = pd.merge(left=IO_naics_I, right=IO_naics_O, on=['product_I', 'product_O', 'IO_value', 'date'], how='outer')

    return IO_naics
