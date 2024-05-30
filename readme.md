# Readme

# folder structure
We have a shared Dropbox folder that stores all our data and notes explaining the methodology that we're pursuing

## \rawdata: our downloaded data from various sources!
1. [NIPA personal consumption expenditures (PCE) tables](https://www.bea.gov/itable/national-gdp-and-personal-income) from the US Bureau of Economic Analysis
    2.4.3U: product-level quantity indexes for real personal consumption expenditures
    2.4.4U: product-level price indexes for personal consumption expenditures
    2.4.5U: product-level personal consumption expenditures in millions of dollars
2. [I-O use table and requirements tables for NAICS sectors](https://www.bea.gov/itable/input-output) from the US Bureau of Economic Analysis
3. [Concordance between PCE products and NAICS sectors](https://www.bea.gov/industry/industry-underlying-estimates) from the US Bureau of Economic Analysis
4. The _\canada_ folder stores the Canadian versions of prices and expenditures data (and metadata). we're not currently using these.
5. The _I-O\_old_ folder stores the I-O tables for older sector definitions. We're not currently using these.

## \cleandata: where we save our transformed data, calculations, simulation results, etc.
1. Cleaned PCE tables: BEA_PCE.pkl
2. Merge on cleaned PCE tables (at the 6-digit NAICS and what we're calling the 6th level of product granularity): BEA6_NAICS6_merged
3. Cleaned I-O requirements table for 6-digit NAICS sectors: requirements_naics6
4. Cleaned I-O use table for 6-digit NAICS sectors: use_naics6.pkl
5. The _\concordance_ folder has the cleaned version of the product-to-sector concordance 
6. The _\inversions_ folder is where the I-O adjusted data and VAR residuals get saved
7. The _\montecarlogenerated_ folder is where all the simulation data gets saved

# code

__functions:__ All the data cleaning and transformations functions are here

__datacleaning:__ Works with all the raw data to make it look nice(?)

__concordance:__ Filters the raw concordance for the products we want and calculates proportions by sector that are used to convert I-O values (handles the many-to-many product to sector matches we have)

__match_BEA:__ Merges the PCE data with the I-O using the proportion-calculation concordance

__intermediate_salesshares:__ uses the cleaned I-O requirements matrix to calculate intermediate shares of sales (this saves into the _\inversions_ folder)

__inversions:__ I-O adjustment code and VAR results on I-O adjusted prices and quantities with classifications of shocks (and some nice visuals). Uses the merged PCE/I-O data and the calculated intermediate shares of sales

__montecarlo:__ generates fake data using a Monte Carlo method that we can use to test our estimation work

# other stuff

This project (and many others here) use Jupyter Notebook. It's a live format Python editor that can present visualizations and output.

We store data in pickle files. It's a nice fast way to store various data structures.

In your local repository, you should have a file called ```.env``` with relevant path names. We all want to use the code to deal with data and it's being hosted on Dropbox, and this will make the code universal. Whatever you put in it will look something like this:

```
MY_PATH="C:\Users\\you\Dropbox (Bank of Canada)\whatever"
```

The BEA _does_ offer an API so it'd might be nice to switch to that eventually.