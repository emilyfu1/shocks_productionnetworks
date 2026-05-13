import pandas as pd
import numpy as np
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt


def clean_bea_PQE_table(df, data_type, long=False, frequency="auto"):
    """Clean BEA P/Q/E tables and infer monthly vs quarterly date columns."""

    year_row = 4
    period_row = 5
    first_data_row = 6
    footer_rows = 5
    product_col = 1
    first_value_col = 2

    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    value_positions = list(range(first_value_col, df.shape[1]))
    years = pd.Series(df.iloc[year_row, first_value_col:].to_numpy(), index=value_positions).ffill()
    periods = (
        pd.Series(df.iloc[period_row, first_value_col:].to_numpy(), index=value_positions)
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_month = periods.isin(month_map)
    valid_quarter = periods.str.fullmatch(r"Q[1-4]", na=False)

    if frequency == "auto":
        if valid_month.sum() and not valid_quarter.sum():
            frequency = "monthly"
        elif valid_quarter.sum() and not valid_month.sum():
            frequency = "quarterly"
        elif valid_month.sum() >= valid_quarter.sum():
            frequency = "monthly"
        elif valid_quarter.sum():
            frequency = "quarterly"
        else:
            raise ValueError("Could not infer BEA table frequency from header row.")

    if frequency == "monthly":
        valid_periods = valid_month
        valid_positions = valid_periods[valid_periods].index.tolist()
        date_labels = [
            pd.Timestamp(year=int(float(y)), month=month_map[p], day=1)
            for y, p in zip(years.loc[valid_positions], periods.loc[valid_positions])
        ]
    elif frequency == "quarterly":
        valid_periods = valid_quarter
        valid_positions = valid_periods[valid_periods].index.tolist()
        date_labels = [
            pd.Period(f"{int(float(y))}{p}", freq="Q").to_timestamp(how="start")
            for y, p in zip(years.loc[valid_positions], periods.loc[valid_positions])
        ]
    else:
        raise ValueError("frequency must be one of 'auto', 'monthly', or 'quarterly'.")

    cleaned = df.iloc[first_data_row:-footer_rows, [product_col] + valid_positions].copy()
    cleaned.columns = ["products"] + date_labels

    cleaned.loc[
        cleaned["products"] == "    Final consumption expenditures of nonprofit institutions serving households (NPISHs) (132)",
        "products",
    ] = "final consumption expenditures of nonprofit institutions serving households"

    cleaned["products"] = cleaned["products"].str.replace(r"\s*\((?=[^)]*\d)[^)]*\)", "", regex=True)
    cleaned.set_index("products", inplace=True)
    cleaned.index = cleaned.index.str.lower()
    cleaned.index = cleaned.index.str.strip()

    df_long = cleaned.reset_index()
    df_long = pd.melt(df_long, id_vars="products", var_name="date", value_name=f"{data_type}")

    if long:
        return df_long
    else:
        return cleaned


def clean_bea_peq_table(df_raw, data_type, long=False):
    """Clean PEQ quarterly tables that use explicit year and quarter header rows."""

    product_col = 1

    quarter_row = None
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).str.strip()
        if row.str.fullmatch(r"Q[1-4]", na=False).sum() >= 4:
            quarter_row = i
            break

    if quarter_row is None or quarter_row == 0:
        raise ValueError("Could not locate PEQ quarter header row.")

    year_row = quarter_row - 1
    first_data_row = quarter_row + 1

    quarters_full = df_raw.iloc[quarter_row].astype(str).str.strip().str.upper()
    valid_quarter = quarters_full.str.fullmatch(r"Q[1-4]", na=False)
    value_cols = valid_quarter[valid_quarter].index.tolist()

    years = pd.Series(df_raw.iloc[year_row, value_cols].to_numpy(), index=value_cols).ffill()
    quarters = quarters_full.loc[value_cols]

    date_labels = [
        pd.Period(f"{int(float(y))}{q}", freq="Q").to_timestamp(how="start")
        for y, q in zip(years, quarters)
    ]

    cleaned = df_raw.iloc[first_data_row:, [product_col] + value_cols].copy()
    cleaned.columns = ["products"] + date_labels

    cleaned["products"] = (
        cleaned["products"]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )
    cleaned = cleaned[cleaned["products"].notna()]

    stop_mask = cleaned["products"].str.contains(
        r"legend|footnote|footnotes|includes net purchases",
        case=False,
        na=False,
    )
    if stop_mask.any():
        stop_pos = np.where(stop_mask)[0][0]
        cleaned = cleaned.iloc[:stop_pos]

    cleaned["products"] = (
        cleaned["products"]
        .str.replace(r"\d+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    cleaned.set_index("products", inplace=True)

    for c in cleaned.columns:
        cleaned[c] = (
            cleaned[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"---": np.nan, "--": np.nan, "—": np.nan, "nan": np.nan, "": np.nan})
        )
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

    df_long = cleaned.reset_index()
    df_long = pd.melt(df_long, id_vars="products", var_name="date", value_name=f"{data_type}")

    if long:
        return df_long
    else:
        return cleaned

def requirements_clean(requirements):
    """Cleans the Industry by Industry Requirments table"""

    requirements = requirements.iloc[2:-1, 1:]
    requirements = requirements.drop(requirements.index[1])
    requirements.columns = requirements.iloc[0]
    requirements = requirements[1:].reset_index(drop=True)
    requirements = requirements.set_index(requirements.columns[0])
    requirements = requirements.rename(columns={'Automotive repair and maintenance (including car washes)': 'Automotive repair and maintenance'})
    requirements = requirements.rename(columns={"Drugs and druggists' sundries": "Drugs and druggists sundries"})
    requirements = requirements.rename(index={"Drugs and druggists' sundries": "Drugs and druggists sundries"})
    requirements.index = requirements.index.str.lower()
    requirements.columns = requirements.columns.str.lower()
    requirements.columns = requirements.columns.str.strip()
    requirements.index = requirements.index.str.strip()

    return requirements


def concordance_PCE_clean(pce_bridge): 
    """Cleans the PCE Concordance Table to Return only the PCE products bridged to industries"""
    pce_bridge = pce_bridge.iloc[4:]
    pce_bridge = pce_bridge.iloc[:, [1,3]]
    pce_bridge.rename(columns={'Unnamed: 1': 'PCE Bridge Products' , 'Unnamed: 3': 'PCE Bridge Industries'}, inplace=True)
    pce_bridge.loc[pce_bridge['PCE Bridge Industries'] == 'Tobacco product manufacturing', 'PCE Bridge Industries'] = 'Tobacco manufacturing'
    pce_bridge.loc[pce_bridge['PCE Bridge Industries'] == 'Insurance Carriers, except Direct Life Insurance', 'PCE Bridge Industries'] = 'Insurance carriers, except direct life'
    # Manually added rows so taht total PCE is correct
    new_rows = pd.DataFrame([
    ["Tenant-occupied stationary homes", "Tenant-occupied housing"],
    ["Tenant-occupied, including landlord durables", "Tenant-occupied housing"]], columns=["PCE Bridge Products", "PCE Bridge Industries"])
    pce_bridge = pd.concat([pce_bridge, new_rows], ignore_index=True)

    pce_bridge["PCE Bridge Products"] = pce_bridge["PCE Bridge Products"].str.lower()
    pce_bridge["PCE Bridge Industries"] = pce_bridge["PCE Bridge Industries"].str.lower()
    pce_bridge.loc[pce_bridge['PCE Bridge Products'] == 'community food and housing/emergency/other relief services', 'PCE Bridge Products'] = 'community food and housing / emergency / other relief services'
    pce_bridge.loc[pce_bridge['PCE Bridge Products'] == 'cosmetic/perfumes/bath/nail preparations and implements', 'PCE Bridge Products'] = 'cosmetic / perfumes / bath / nail preparations and implements'
    pce_bridge.loc[pce_bridge['PCE Bridge Products'] == 'final consumption expenditures of nonprofit institutions serving households (npish)', 'PCE Bridge Products'] = 'final consumption expenditures of nonprofit institutions serving households'

    return pce_bridge


def concordance_PCQ_clean(peq_bridge): 
    """Cleans the PEQ Concordance Table to Return only the investment products bridged to industries"""
    peq_bridge = peq_bridge.iloc[4:]
    peq_bridge = peq_bridge.iloc[:, [1,3]]
    peq_bridge.rename(columns={'Unnamed: 1': 'PEQ Investment Products' , 'Unnamed: 3': 'Industries'}, inplace=True)
    peq_bridge["PEQ Investment Products"] = peq_bridge["PEQ Investment Products"].str.lower()
    peq_bridge["Industries"] = peq_bridge["Industries"].str.lower()
    return peq_bridge


def find_intermediate_industries(use_table):
    """Takes BEA Use table as input, returns industires with zero PCE expenditures"""
    use_table = use_table.iloc[4:-11]
    use_table = use_table.loc[:, use_table.iloc[0].isin(['Commodity Description', 'F01000'])]
    use_table = use_table.iloc[1:]
    use_table.rename(columns={'Unnamed: 1': 'Industry' , 'Unnamed: 405': 'PCE Expenditure'}, inplace=True)
    use_table.loc[use_table['Industry'] == 'Drugs and druggists’ sundries', 'Industry'] = 'Drugs and druggists sundries'
    use_table.loc[use_table['Industry'] == 'Insurance Carriers, except Direct Life Insurance', 'Industry'] = 'Insurance carriers, except direct life'
    use_table.loc[use_table['Industry'] == 'Tobacco product manufacturing', 'Industry'] = 'Tobacco manufacturing'
    use_table.loc[use_table['Industry'] == 'Scenic and sightseeing transportation and support activities for transportatio', 'Industry'] = 'scenic and sightseeing transportation and support activities'
    use_table.loc[use_table['Industry'] == 'Community food, housing, and other relief services, including rehabilitation services', 'Industry'] = 'community food, housing, and other relief services, including vocational rehabilitation services'
    use_table["Industry"] = use_table["Industry"].str.lower()
    use_table["Industry"] = use_table["Industry"].str.strip()
    use_table = use_table.dropna(subset=['Industry'])
    use_table = use_table[use_table['PCE Expenditure'].isna()]
    return use_table

def get_final_demand_from_use_table(use_table):
    """Takes BEA Use table as input, returns industires with zero PCE expenditures"""
    use_table = use_table.iloc[4:-11]
    use_table = use_table.loc[:, use_table.iloc[0].isin(['Commodity Description', 'F01000'])]
    use_table = use_table.iloc[1:]
    use_table.rename(columns={'Unnamed: 1': 'Industry' , 'Unnamed: 405': 'PCE Expenditure'}, inplace=True)
    use_table.loc[use_table['Industry'] == 'Drugs and druggists’ sundries', 'Industry'] = 'Drugs and druggists sundries'
    use_table.loc[use_table['Industry'] == 'Insurance Carriers, except Direct Life Insurance', 'Industry'] = 'Insurance carriers, except direct life'
    use_table.loc[use_table['Industry'] == 'Tobacco product manufacturing', 'Industry'] = 'Tobacco manufacturing'
    use_table.loc[use_table['Industry'] == 'Scenic and sightseeing transportation and support activities for transportatio', 'Industry'] = 'scenic and sightseeing transportation and support activities'
    use_table.loc[use_table['Industry'] == 'Community food, housing, and other relief services, including rehabilitation services', 'Industry'] = 'community food, housing, and other relief services, including vocational rehabilitation services'
    use_table["Industry"] = use_table["Industry"].str.lower()
    use_table["Industry"] = use_table["Industry"].str.strip()
    use_table = use_table.dropna(subset=['Industry'])
    use_table.loc[use_table['PCE Expenditure'].isna(),'PCE Expenditure'] = 0  # Make nans 0 (we will remove these sectors anyways)
    use_table = use_table[['Industry','PCE Expenditure']]
    
    return use_table  



def get_sales_from_make_matrix(make_matrix):
    """Cleans and gets sales in dollars from BEA Make Matrix ie sums the columns of make matrix"""
    make_matrix = make_matrix.iloc[3:,1:]
    make_matrix = make_matrix.drop(make_matrix.index[1])
    make_matrix.columns = make_matrix.iloc[0]
    make_matrix = make_matrix.drop(make_matrix.index[0])
    make_matrix.reset_index(drop = True, inplace=True)
    make_matrix = make_matrix[[make_matrix.columns[0], 'Total Commodity Output']]
    make_matrix.rename(columns={make_matrix.columns[0]: 'Industries', 'Total Commodity Output': "Sales"}, inplace=True)
    make_matrix.loc[make_matrix['Industries'] == 'Drugs and druggists’ sundries', 'Industries'] = 'Drugs and druggists sundries'
    make_matrix.loc[make_matrix['Industries'] == 'Insurance Carriers, except Direct Life Insurance', 'Industries'] = 'Insurance carriers, except direct life'
    make_matrix.loc[make_matrix['Industries'] == 'Automotive repair and maintenance (including car washes)', 'Industries'] = 'automotive repair and maintenance'
    make_matrix.loc[make_matrix['Industries'] == 'Scenic and sightseeing transportation and support activities for transportation', 'Industries'] = 'scenic and sightseeing transportation and support activities'
    make_matrix.loc[make_matrix['Industries'] == 'Community food, housing, and other relief services, including rehabilitation services', 'Industries'] = 'community food, housing, and other relief services, including vocational rehabilitation services'
    make_matrix["Industries"] = make_matrix["Industries"].str.lower()
    make_matrix['Industries'] = make_matrix['Industries'].str.strip()
    with pd.option_context("future.no_silent_downcasting", True):
        make_matrix = make_matrix.fillna(0).infer_objects(copy=False)
    
    sales = make_matrix.iloc[:-3]
    return sales


def clean_make_matrix(make_matrix):
    """Cleans BEA Make Matrix"""
    make_matrix = make_matrix.iloc[3:,1:]
    make_matrix = make_matrix.drop(make_matrix.index[1])
    make_matrix.columns = make_matrix.iloc[0]
    make_matrix = make_matrix.drop(make_matrix.index[0])
    make_matrix.reset_index(drop = True, inplace=True)
    make_matrix = make_matrix.iloc[:-3,:-12]
    make_matrix_rows = make_matrix[[make_matrix.columns[0]]]
    make_matrix_columns = pd.DataFrame(make_matrix.columns).iloc[1:]
    make_matrix_columns.columns= ["Column Industries"]
    make_matrix_rows.columns= ["Row Industries"]
    make_matrix.set_index(make_matrix.columns[0], inplace=True)

    industries_not_in_make_matrix_row = ["Secondary smelting and alloying of aluminum", "Federal electric utilities",\
                        "State and local government passenger transit", "State and local government electric utilities"]
                        
    industries_not_in_make_columns = ["scrap", "used and secondhand goods", "rest of the world adjustment"]

    # new_rows = pd.DataFrame([[None] * make_matrix.shape[1]] * len(industries_not_in_make_matrix_row), columns=make_matrix.columns, index=industries_not_in_make_matrix_row)

    make_matrix = pd.concat([make_matrix, pd.DataFrame(0, index=make_matrix.index, columns=industries_not_in_make_columns)], axis=1)

    # Add new rows by concatenating with a DataFrame of zeros
    make_matrix = pd.concat([make_matrix, pd.DataFrame(0, index=industries_not_in_make_matrix_row, columns=make_matrix.columns)])

    # make_matrix = pd.concat([make_matrix, new_rows])

    make_matrix.rename(index={"Drugs and druggists’ sundries": 'Drugs and druggists sundries'}, inplace=True)
    make_matrix.rename(columns={"Drugs and druggists’ sundries": 'Drugs and druggists sundries'}, inplace=True)

    make_matrix.rename(index={'Insurance Carriers, except Direct Life Insurance': 'Insurance carriers, except direct life'}, inplace=True)
    make_matrix.rename(columns={'Insurance Carriers, except Direct Life Insurance': 'Insurance carriers, except direct life'}, inplace=True)

    make_matrix.rename(index={'Tobacco product manufacturing': 'Tobacco manufacturing'}, inplace=True)
    make_matrix.rename(columns={'Tobacco product manufacturing': 'Tobacco manufacturing'}, inplace=True)

    make_matrix.rename(index={'Scenic and sightseeing transportation and support activities for transportation': \
                              'scenic and sightseeing transportation and support activities'}, inplace=True)
    make_matrix.rename(columns={'Scenic and sightseeing transportation and support activities for transportation': \
                                'scenic and sightseeing transportation and support activities'}, inplace=True)

    make_matrix.rename(index={'Community food, housing, and other relief services, including rehabilitation services': \
                            'community food, housing, and other relief services, including vocational rehabilitation services'}, inplace=True)
    make_matrix.rename(columns={'Community food, housing, and other relief services, including rehabilitation services': \
                            'community food, housing, and other relief services, including vocational rehabilitation services'}, inplace=True)


    make_matrix.rename(index={'Automotive repair and maintenance (including car washes)': \
                            'automotive repair and maintenance'}, inplace=True)
    make_matrix.rename(columns={'Automotive repair and maintenance (including car washes)': \
                            'automotive repair and maintenance'}, inplace=True)

    make_matrix.index = make_matrix.index.str.lower()
    make_matrix.columns = make_matrix.columns.str.lower()
    make_matrix.columns = make_matrix.columns.str.strip()
    make_matrix.index = make_matrix.index.str.strip()

    with pd.option_context("future.no_silent_downcasting", True):
        make_matrix = make_matrix.fillna(0).infer_objects(copy=False)

    return make_matrix


def get_demand_shock_from_shaipro_output(df, haver_product_map):
    """Returns a df that indicates whether a product is labelled as a demand shock for a particular time period. Only works 
    if the input df is in the same format as the results producted by Shapiro's Stata code"""

    df = df[['time_month'] + [col for col in df.columns if col.startswith('demdum_')]]
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.index = df.index.str.replace('^demdum_', '', regex=True)
    df.index = df.index.map(haver_product_map)

    return df

def get_expenditure_weights_from_shapiro_outputs(df, haver_product_map):

    df = df[['time_month', "s"] + [col for col in df.columns if col.startswith('weight_')]]
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.index = df.index.str.replace('^weight_', '', regex=True)
    df.index = df.index.map(lambda x: x[:-1] if len(x) > 1 else x)
    df.index = df.index.map(haver_product_map)

    return df


def plot_shapiro_graph_from_shapiro_ouput(shapiro_code_output, plot_title):
    inflation_decomped = shapiro_code_output[['time_month', "dem_contr_y", "sup_contr_y"]]
    inflation_decomped.set_index('time_month', inplace=True)

    supply_inflation = inflation_decomped[["sup_contr_y"]].copy()
    supply_inflation.rename(columns={'sup_contr_y': 'Supply Inflation'}, inplace=True)

    demand_inflation = inflation_decomped[["dem_contr_y"]].copy()
    demand_inflation.rename(columns={'dem_contr_y': 'Demand Inflation'}, inplace=True)

    supply_inflation['supply_pos'] = supply_inflation['Supply Inflation'].apply(lambda x: x if x > 0 else 0)
    demand_inflation['demand_pos'] = demand_inflation['Demand Inflation'].apply(lambda x: x if x > 0 else 0)
    supply_inflation['supply_neg'] = supply_inflation['Supply Inflation'].apply(lambda x: x if x < 0 else 0)
    demand_inflation['demand_neg'] = demand_inflation['Demand Inflation'].apply(lambda x: x if x < 0 else 0)

    demand_inflation = demand_inflation.iloc[:-1]
    supply_inflation = supply_inflation.iloc[:-1]

    plt.figure(figsize=(26, 12))

    plt.stackplot(supply_inflation.index, demand_inflation['demand_pos'], supply_inflation['supply_pos'], colors= ["#008000", "#FF0000"], labels = ["Deamnd", "Supply"])
    plt.stackplot(supply_inflation.index, demand_inflation['demand_neg'], supply_inflation['supply_neg'], colors= ["#008000", "#FF0000"])
    plt.xlabel('Date')
    plt.ylabel('Inflation Percent')
    plt.title(f'{plot_title}')
    plt.legend()

    return plt
