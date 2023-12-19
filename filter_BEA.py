import pandas as pd

'''
function for filtering data in BEA data tables (2.4.3U and 2.4.4U)

example:
- first level: goods
- second level: durable goods
- third level: motor vehicles and parts
- fourth level: new motor vehicles
- fifth level: new autos
- sixth level: new domestic autos

output: products of specified granularity and of lower granularity if no further granularity available

'''


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