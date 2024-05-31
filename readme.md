# Readme

# folder structure
We have a shared Dropbox folder that stores all our data and notes explaining the methodology that we're pursuing

## Raw data: our downloaded data from various sources!
1. [NIPA personal consumption expenditures (PCE) tables](https://www.bea.gov/itable/national-gdp-and-personal-income) from the US Bureau of Economic Analysis. 

    __2.4.3U:__ product-level quantity indexes for real personal consumption expenditures, quarterly
    __2.4.4U:__ product-level price indexes for personal consumption expenditures, quarterly
    __2.4.5U:__ product-level personal consumption expenditures in millions of dollars, quarterly

    The NIPA tables have products at various types of disaggregations. More specific products have a number of indents by their relative level of disaggregation. For instance:
    
    ```
    (1) services → 
        (2) transportation services → 
            (3) public transportation → 
                (4) air transportation → 
                    (5) ground transportation → 
                        (6) railway transportation
    ```

    We are using the sixth level of disaggregation wherever possible. If a type of product doesn't have this level of specificity, we just take the most specific category available. 

2. [I-O use table and requirements tables for NAICS sectors](https://www.bea.gov/itable/input-output) from the US Bureau of Economic Analysis, 2017 data
3. [Concordance between PCE products and NAICS sectors](https://www.bea.gov/industry/industry-underlying-estimates) from the US Bureau of Economic Analysis
4. The _\canada_ folder stores the Canadian versions of prices and expenditures data (and metadata) from Statistics Canada. we're not currently using these
5. The _I-O\_old_ folder stores the I-O tables for older sector definitions from the US Bureau of Economic Analysis. We're not currently using these

## Clean data: any output data is saved here
1. Cleaned PCE tables all merged together: BEA_PCE.pkl
2. Merge on all the cleaned PCE tables (at the 6-digit NAICS and what we're calling the 6th level of product granularity): BEA6_NAICS6_merged
3. Cleaned I-O requirements table for 6-digit NAICS sectors: requirements_naics6.pkl
4. Cleaned I-O use table for 6-digit NAICS sectors: use_naics6.pkl
5. The _\concordance\\_ folder has the cleaned version of the product-to-sector concordance 
6. The _\inversions\\_ folder is where the I-O adjusted data and VAR residuals are saved
7. The _\montecarlogenerated\\_ folder is where all the simulation data are saved

# code

__datacleaning:__ Works with all the raw data to make it useable for our transformations and calculations

__concordance:__ Filters the raw concordance for the products we want and calculates proportions by sector that are used to convert I-O values (handles the many-to-many product to sector matches we have)

__match_BEA:__ Merges the PCE data with the I-O using the proportion-calculation concordance

__mergechecks:__ Visualizing how the I-O and PCE data line up within the merged data

__intermediate_salesshares:__ uses the cleaned I-O requirements matrix to calculate intermediate shares of sales (this saves into the _\inversions\\_ folder)

__inversions:__ I-O adjustment code and VAR results on I-O adjusted prices and quantities with classifications of shocks (and some nice visuals). Uses the merged PCE/I-O data and the calculated intermediate shares of sales

__montecarlo:__ generates fake data using a Monte Carlo method that we can use to test our estimation work

# other stuff

This project use Jupyter Notebook. It's a live format Python editor that can present visualizations and output.

We store data in pickle files. It's a nice fast way to store various data structures.

In your local repository, you should have a file called ```.env``` with relevant path names. We all want to use the code to deal with data and it's being hosted on Dropbox, and this will make the code universal. Whatever you put in it will look something like this:

```
MY_PATH="C:\Users\\you\Dropbox (Bank of Canada)\whatever"
```

The BEA _does_ offer an API so it'd be nice to switch to that eventually. It would make the data loading much cleaner but it wouldn't affect any of the results. Getting the data through here would make retrieving monthly PCE data a lot easier since the BEA's online data tables can't load that large amount of data.