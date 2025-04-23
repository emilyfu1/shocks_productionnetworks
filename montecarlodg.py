import numpy as np
import pandas as pd
import random
from dotenv import dotenv_values, find_dotenv
import os

# set path parameters
config = dotenv_values(find_dotenv())
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_cleandata = os.path.abspath(config["CLEANDATA"]) + '\\'
path_figures = os.path.abspath(config["FIGURES"]) + '\\'

# import I-O shares
intermediate_costshares = pd.read_pickle(path_cleandata + 'inversions//intermediate_costshares.pkl')
intermediate_salesshares = pd.read_pickle(path_cleandata + 'inversions//intermediate_salesshares.pkl')

# filter for products
products_to_include = list(intermediate_costshares.index)
products_to_include.sort()
intermediate_salesshares = intermediate_salesshares[intermediate_salesshares.index.isin(products_to_include)]['intermediate_salesshare']

# get list of dates to use (just comes from whatevers available in the BEA data)
prices = pd.read_pickle(path_cleandata + 'inversions//prices.pkl')
quantities = pd.read_pickle(path_cleandata + 'inversions//quantities.pkl')
expenditures = pd.read_pickle(path_cleandata + 'inversions//expenditures.pkl')
dates = list(set(prices['date'].unique()) & set(quantities['date'].unique()) & set(expenditures['date'].unique()))
dates.sort()

# monte carlo parameters

random.seed(420)

# initial shock parameters
sd = 0.01
supplyshock_params = np.random.uniform(-0.02, 0.02, size=len(products_to_include))
demandshock_params = np.random.uniform(-0.02, 0.02, size=len(products_to_include))

# supply curve
alpha = 1.2

# price generation convergence
convergence_threshold = 1e-7
max_change = float('inf')

# initial values

# supply and demand shocks
shocks_generated = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'product': pd.Series(dtype='str'),
                   'supply_shock': pd.Series(dtype='float'),
                   'demand_shock': pd.Series(dtype='float')})
for product in products_to_include:
    initial_shock = pd.DataFrame([[dates[0], product, 0.0, 0.0]], columns=shocks_generated.columns)
    shocks_generated = pd.concat([shocks_generated, initial_shock])

# random walk
store_randomwalk_supply = [0]
store_randomwalk_demand = [0]

# fill in fake supply and demand
for date in dates[1:]:
    # create new random
    randomwalk_supply = np.random.normal(loc=0, scale=sd, size=len(products_to_include))
    store_randomwalk_supply.append(randomwalk_supply)
    randomwalk_demand = np.random.normal(loc=0, scale=sd, size=len(products_to_include))
    store_randomwalk_demand.append(randomwalk_demand)

    # initial shock
    shocks_date = shocks_generated.loc[shocks_generated['date'] == dates[dates.index(date)-1]][['product', 'supply_shock', 'demand_shock']]

    # current shock = previous shock + (parameter * previous random + current random)
    shocks_date['supply_shock'] = shocks_date['supply_shock'] + (supplyshock_params * store_randomwalk_supply[dates.index(date)-1] + randomwalk_supply)
    shocks_date['demand_shock'] = shocks_date['demand_shock'] + (demandshock_params * store_randomwalk_demand[dates.index(date)-1] + randomwalk_demand)
    
    shocks_date['date'] = date

    shocks_generated = pd.concat([shocks_generated, shocks_date])

# so i had the wrong order of operations here before (i was taking e^shock every time which is wrong)
shocks_generated['supply_shock'] = np.exp(shocks_generated['supply_shock'])
shocks_generated['demand_shock'] = np.exp(shocks_generated['demand_shock'])

# repeatedly get a guess for prices

iteration = 1

# starting point for generated prices
montecarlo_prices = prices.copy()
montecarlo_prices['priceindex'] = 1

# create diagonal matrix from intermediate_salesshares
diag_matrix = np.diag(intermediate_salesshares)

# goes until convergence threshold is met
while max_change >= convergence_threshold:

    prev_prices = montecarlo_prices.copy()

    montecarlo_intermediates = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'product': pd.Series(dtype='str'),
                   'intermediates': pd.Series(dtype='float')})
    montecarlo_sales = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'product': pd.Series(dtype='str'),
                   'sales': pd.Series(dtype='float')})
    montecarlo_output = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'product': pd.Series(dtype='str'),
                   'real_output': pd.Series(dtype='float')})
    montecarlo_valueadded = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                   'product': pd.Series(dtype='str'),
                   'value_added': pd.Series(dtype='float')})

    # generate real output and value added
    for date in dates:
        # filter for current date
        demandshock_date = shocks_generated[shocks_generated['date'] == date][['product', 'demand_shock']]
        demandshock_date = demandshock_date.sort_values('product')
        supplyshock_date = shocks_generated[shocks_generated['date'] == date][['product', 'supply_shock']]
        supplyshock_date = supplyshock_date.sort_values('product')
        prices_date = montecarlo_prices[montecarlo_prices['date'] == date][['product', 'priceindex']].set_index('product')
        prices_date = prices_date.sort_index()

        # generate intermediates
        # the problem with this intermediates stuff is that it runs in O(n^2)
        # but i have not come up with a better way to fill out the values
        for i in products_to_include:
            montecarlo_intermediates.loc[len(montecarlo_intermediates)] = [date, i , np.exp(intermediate_costshares.loc[i] @ np.log(prices_date['priceindex']))]

        # calculate sales (use generated demand shock as per demand-side assumption)

        # calculate sales in each sector
        sales_date = np.linalg.inv(np.identity(len(intermediate_costshares)) - (intermediate_costshares.T @ diag_matrix)) @ demandshock_date[['demand_shock']]
        # set some columns to append
        sales_date['product'] = products_to_include
        sales_date['date'] = date
        sales_date.rename(columns={'demand_shock': 'sales'}, inplace=True)

        # append
        montecarlo_sales = pd.concat([montecarlo_sales, sales_date], ignore_index=True)

        # calculate real output
        real_output_date = pd.merge(left=sales_date, right=prices_date, on=['product'], how='inner')
        real_output_date['real_output'] = real_output_date['sales'] / real_output_date['priceindex']
        real_output_date['date'] = date
        # append
        montecarlo_output = pd.concat([montecarlo_output, real_output_date[['product', 'date', 'real_output']]], ignore_index=True)

        # calculate price of value added
        valueadded_date = pd.merge(left=real_output_date, right=supplyshock_date, on=['product'], how='inner')
        valueadded_date['value_added'] = np.power(valueadded_date['real_output'], alpha) * valueadded_date['supply_shock']
        valueadded_date['date'] = date
        # append
        montecarlo_valueadded = pd.concat([montecarlo_valueadded, valueadded_date[['product', 'date', 'value_added']]], ignore_index=True)

    print('iteration ' + str(iteration))

    # Merge and calculate new prices
    montecarlo_prices = pd.merge(left=montecarlo_intermediates, right=montecarlo_valueadded, on=['product', 'date'], how='inner')
    montecarlo_prices = pd.merge(left=montecarlo_prices, right=intermediate_salesshares.reset_index(), on='product', how='inner')
    montecarlo_prices['priceindex'] = (np.power(montecarlo_prices['value_added'], (1 - montecarlo_prices['intermediate_salesshare']))) * (np.power(montecarlo_prices['intermediates'], montecarlo_prices['intermediate_salesshare']))
    montecarlo_prices = montecarlo_prices.sort_values(['date', 'product'])

    # diff between current and previous guess
    max_change = np.max(np.abs(montecarlo_prices['priceindex'] - prev_prices['priceindex']))

    print(montecarlo_prices[['date', 'product', 'priceindex']].tail())

    iteration += 1

    print('\n')

# save everything
shocks_generated.to_pickle(path_cleandata + 'montecarlogenerated\\shocks.pkl')
montecarlo_intermediates.to_pickle(path_cleandata + 'montecarlogenerated\\intermediateprices.pkl')
montecarlo_sales.to_pickle(path_cleandata + 'montecarlogenerated\\sales.pkl')
montecarlo_output.to_pickle(path_cleandata + 'montecarlogenerated\\realoutput.pkl')
montecarlo_valueadded.to_pickle(path_cleandata + 'montecarlogenerated\\valueaddedprices.pkl')
montecarlo_prices.to_pickle(path_cleandata + 'montecarlogenerated\\prices.pkl')