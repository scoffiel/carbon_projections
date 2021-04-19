#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Process raw BCSD monthly climate data into combined netcdf files

Inputs:
    BCSD downscaled CMIP5 monthly CONUS climate data from https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html
    One file for each model (32) and variable (T, P). All are r1i1p1 ensemble. 
    
Outputs: 4 netcdf files (2 for RCP4.5 and 2 for RCP8.5)
    climate_present
    climate_future
with dimensions for 32 models (in order of most drying to wetting), 8 variables (4 seasons of T & P), and lat/lon
"""

import numpy as np
import xarray as xr
import datetime as dt
import os
import regionmask

root = '/Users/scoffiel/california/'

scenarios = ['rcp45','rcp85']

for scenario in scenarios:
    
    #gather all 32 BCSD files for T and P (monthly 2006-2099)
    files = os.listdir(root + 'bcsd/{}/'.format(scenario))
    t_files = sorted([f for f in files if f[0]=='B' and f[14]=='t']) #sort alphabetically to ensure consistent order
    p_files = sorted([f for f in files if f[0]=='B' and f[14]=='p'])
    
    #Temperature first
    t_present = [] #list to store all 32 models
    t_future = []
    
    for f in t_files:
        model = xr.open_dataset(root + 'bcsd/{}/{}'.format(scenario, f))
        model = model.sel(longitude=slice(235,260), latitude=slice(25,49)) #crop down to Western N.America
        
        present = model.sel(time=slice(dt.datetime(2006,1,1), dt.datetime(2015,12,31)))
        months = []    
        for i in range(12):
            month = present.isel(time=np.arange(i, 120, 12)).mean(dim='time')
            months.append(month)
        months = xr.concat(months, dim='months')   
        t_present.append(months)
        
        #future
        future = model.sel(time=slice(dt.datetime(2090,1,1), dt.datetime(2099,12,31)))
        months = []    
        for i in range(12):
            month = future.isel(time=np.arange(i, 120, 12)).mean(dim='time')
            months.append(month)
        months = xr.concat(months, dim='months')   
        t_future.append(months)
        
    t_present = xr.concat(t_present, dim='models') #32 models, 12 months each
    t_future = xr.concat(t_future, dim='models') 
    
    #Repeat for precipitation
    p_present = [] #list to store all 32 models
    p_future = []
    
    for f in p_files:
        model = xr.open_dataset(root + 'bcsd/{}/{}'.format(scenario, f))
        model = model.sel(longitude=slice(235,260), latitude=slice(25,49)) 
        
        present = model.sel(time=slice(dt.datetime(2006,1,1), dt.datetime(2015,12,31)))
        months = []    
        for i in range(12):
            month = present.isel(time=np.arange(i, 120, 12)).mean(dim='time')
            months.append(month)
        months = xr.concat(months, dim='months')   
        p_present.append(months)
        
        #future
        future = model.sel(time=slice(dt.datetime(2090,1,1), dt.datetime(2099,12,31)))
        months = []    
        for i in range(12):
            month = future.isel(time=np.arange(i, 120, 12)).mean(dim='time')
            months.append(month)
        months = xr.concat(months, dim='months')   
        p_future.append(months)
        
    p_present = xr.concat(p_present, dim='models') #32 models, 12 months each
    p_future = xr.concat(p_future, dim='models')     
    
    
    #append T and P together into one file for present climate and one file for future climate
    climate_present = xr.concat([t_present.tas, p_present.pr], dim='months')
    climate_future = xr.concat([t_future.tas, p_future.pr], dim='months')
    
    #regroup from months into seasons
    climate_present = climate_present.assign_coords(months=['t_winter','t_winter','t_spring','t_spring','t_spring','t_summer','t_summer','t_summer','t_fall','t_fall','t_fall','t_winter',
                                                        'p_winter','p_winter','p_spring','p_spring','p_spring','p_summer','p_summer','p_summer','p_fall','p_fall','p_fall','p_winter']).groupby('months').mean(dim='months')
    climate_present = climate_present.rename({'months':'variables'})
    
    climate_future = climate_future.assign_coords(months=['t_winter','t_winter','t_spring','t_spring','t_spring','t_summer','t_summer','t_summer','t_fall','t_fall','t_fall','t_winter',
                                                        'p_winter','p_winter','p_spring','p_spring','p_spring','p_summer','p_summer','p_summer','p_fall','p_fall','p_fall','p_winter']).groupby('months').mean(dim='months')
    climate_future = climate_future.rename({'months':'variables'})
    

    
    #order 'models' dimension by most drying to most wetting for California ------------------------------
    change = climate_future - climate_present
        
    mask = regionmask.defined_regions.natural_earth.us_states_50.mask(change.longitude, change.latitude, wrap_lon=True)
    cali = mask==4
    cali = cali.rename({'lon':'longitude', 'lat':'latitude'})
    
    precip = change.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables'])
    precip = precip.where(cali).mean(dim=['latitude','longitude'])
    
    order = list(precip.argsort().values) #the indices of models ranked driest to wettest (first value =21 is the index of driest model). Use in Script 4 to match to names in p_files
    model_order = [order.index(i) for i in range(32)] #indicates the final order of models. (first value =5 indicates that the model in position 0 now goes to position 5). Use in Script 2
    
    print('order',order)
    print('model order', model_order)
    
    climate_present_ordered = climate_present.assign_coords(models=model_order).sortby('models')
    climate_future_ordered = climate_future.assign_coords(models=model_order).sortby('models')
     
    #save as netcdf
    climate_present_ordered.to_netcdf(root + 'bcsd/{}/climate_present.nc4'.format(scenario)) #32 models, 8 variables (4 seasons of T & P), and lat/lon
    climate_future_ordered.to_netcdf(root + 'bcsd/{}/climate_future.nc4'.format(scenario))
