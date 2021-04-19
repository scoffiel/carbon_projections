# carbon_projections

<h2>Data and code to accompany 
“Climate-driven limits to future carbon storage in California’s wildland ecosystems”</h2>
AGU Advances, 2021

Corresponding author: Shane Coffield scoffiel@uci.edu 

All data are available at at Dryad https://doi.org/10.7280/D1568Z 

Google Earth Engine scripts are available at

https://code.earthengine.google.com/?accept_repo=users/scoffiel/carbon_projections
<br>git clone https://earthengine.googlesource.com/users/scoffiel/carbon_projections 

Python scripts are available on GitHub:
https://github.com/scoffiel/carbon_projections 

<h2>Data overview:</h2>

<b>input_data</b>: contains all files needed to run Python scripts. All were derived from public sources:
<ul>
    <li>CARB aboveground wildland carbon  https://ww2.arb.ca.gov/nwl-inventory
    <li>LEMMA species biomass https://lemma.forestry.oregonstate.edu/data
    <li>NLCD land coverhttps://www.mrlc.gov/data/nlcd-2016-land-cover-conus
    <li>ESA CCI biomass https://catalogue.ceda.ac.uk/uuid/bedc59f37c9545c981a839eb552e4084
    <li>BCSD CMIP5 downscaled climate projectionsftp://gdo-dcp.ucllnl.org/pub/dcp/archive/cmip5/bcsd
</ul>

<br><b>model_output</b>: corresponds to the four approaches discussed in the manuscript. For all approaches, we provide projections for 6 scenarios: RCP4.5 & RCP8.5 x dry/mean/wet
<ul>
    <li>Random forest regression of carbon density
    <li>Random forest classification of dominant vegetation type
    <li>Climate analogues
    <li>Random forest regression of individual species’ carbon density
</ul>

<h2>Google Earth Engine code overview</h2>
<ol type="1">
<li>Carbon_data: rescales 30m CARB carbon data layer (available upon request from CARB) to 1/8-degree to match the BCSD climate dataset, including masking out water/ag/urban landcover
<li>Valid_land_fraction: calculates the fraction of sub-gridcell area (of 1/8 degree cells) that is allowed to support aboveground carbon (excludes water/ag/urban/barren cover)
<li>Elevation: rescales 30m USGS elevation data to 1/8-degree to match the BCSD climate dataset
<li>Landcover: rescales 30m NLCD land cover data to 1/8 degree (forest, shrub/grass, null)
<li>Landcover_mask: creates a 1/8-degree layer masking out any areas of the western US that are not 50% wildland cover (for climate analogues analysis)
<li>Cci_biomass: rescales 100m CCI biomass data to 1/8-degree for US and Mexico
<li>Lemma_spp: reformats LEMMA species-level data into one raster layer with one band for each species’ density at 1/8 degree
</ol>

<h2>Python code overview</h2>
<ol type="1">
<li>Process_climate.py: Process raw BCSD monthly climate data into combined netcdf files
<li>Process_climate_10yrs.py: Duplicate of script 1 to process raw BCSD climate data, but modified slightly to maintain all 10 years of data in the present. This is needed for calculating the interannual variability in the climate analogues approach.
<li>Plot_clim_change.py: Generate maps of mean annual T & P change for RCP4.5 & RCP8.5 (Fig 1)
<li>Plot_clim_spread.py: Generate maps of spread of precipitation across 32 models for RCP8.5 (FigS1) and dry vs. wet models averages for RCP8.5 (FigS2)
<li>RF_carbon_model.py: Approach #1. Model present-day distribution of CARB carbon layer based on climate data. Project future carbon and change
<li>RF_model_spread.py: Rebuild RF regression models from script 5, for each of the 32 climate models. Compare 3 different runs: T & P both change, T only (P constant), and P only (T constant).
<li>Offsets_change.py: Compare RF regression model results from script 5 for all forests vs. carbon offset projects
<li>RF_veg_class_model.py: Approach #2. Model present-day distribution of NLCD forest-vs-shrub layer based on climate data. Project future land cover, change, and associated carbon change
<li>Climate_analogues.py. Approach #3. Match future climate pixels with their present analogue using Mahalanobis distance. Project future carbon density by assigning that of the present analogue. 3 runs: full domain (25-49 lat and -125 - -100 lon), 500 km radius, 100 km radius
<li>Analogue_whittaker_plots.py: Approach #3 supplementary figure - Whittaker scatter plots of mean annual P vs. T, showing how CA's gridcells shift
<li>Species_models.py: Approach #4. Fit RF regression models to each of the top 20 tree spp in California. Project future carbon and change. Apply restrictions on distance between spp present and future locations (migration scenarios).
</ol>
