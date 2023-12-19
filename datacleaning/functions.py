import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # convert to datetime
    df_long['date'] = pd.to_datetime(df_long['date'], format='mixed')
    df_long['date'] = df_long['date'] + pd.offsets.MonthEnd(0)

    return df_long

# formats the I-O tables
def inputoutput_clean(df):
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
