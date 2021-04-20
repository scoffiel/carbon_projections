#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Duplicate of script 1 to process raw BCSD climate data, but modified
slightly to maintain all 10 years of data in the present. This is needed for 
calculating the interannual variability in the climate analogs approach. 

Inputs:
    BCSD downscaled CMIP5 monthly CONUS climate data from https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html
    One file for each model (32) and variable (T, P). All are r1i1p1 ensemble. 
    
Outputs: 2 netcdf files
    climate_present_10yrs (RCP4.5)
    climate_present_10yrs (RCP8.5)
with dimensions for 2 variables, 32 models (in order of drying to wetting), 40 timesteps (4 seasons x 10 yrs), and lat/lon
"""

import xarray as xr
import datetime as dt
import os

root = '/Users/scoffiel/california/'

scenarios = ['rcp45','rcp85']

#rearranging models in order of most drying to most wetting for california (from script 1)
model_order = {'rcp45':[16,2,31,28,17,11,19,25,10,14,20,12,26,22,13,24,18,7,9,30,27,6,1,0,29,23,15,4,21,8,5,3],
             'rcp85':[5,31,18,28,26,2,25,21,27,12,11,6,29,3,8,24,22,20,30,14,17,0,1,4,9,23,19,13,16,15,10,7]}

for scenario in scenarios:
    
    #gather all 32 BCSD files for T and P (monthly 2006-2099)
    files = os.listdir(root + 'bcsd/{}/'.format(scenario))
    t_files = sorted([f for f in files if f[0]=='B' and f[14]=='t']) #sort alphabetically to ensure consistent order
    p_files = sorted([f for f in files if f[0]=='B' and f[14]=='p'])
    
    #Temperature first
    t_present = [] #list to store all 32 models
    
    for f in t_files:
        model = xr.open_dataset(root + 'bcsd/{}/{}'.format(scenario, f))
        model = model.sel(longitude=slice(235,260), latitude=slice(25,49)) #crop down to Western N.America
        
        present = model.sel(time=slice(dt.datetime(2006,1,1), dt.datetime(2015,12,31))).assign_coords(time=range(120))
        t_present.append(present)
            
    t_present = xr.concat(t_present, dim='models') #32 models, 120 months each
    
    #Repeat for precipitation
    p_present = [] #list to store all 32 models
    p_future = []
    
    for f in p_files:
        model = xr.open_dataset(root + 'bcsd/{}/{}'.format(scenario, f))
        model = model.sel(longitude=slice(235,260), latitude=slice(25,49)) 
        
        present = model.sel(time=slice(dt.datetime(2006,1,1), dt.datetime(2015,12,31))).assign_coords(time=range(120))
        p_present.append(present)
             
    p_present = xr.concat(p_present, dim='models') #32 models, 120 months each
  
    
    #append T and P together into one file for present climate and one file for future climate
    present_10yrs = xr.concat([t_present.tas, p_present.pr], dim='var').assign_coords(var=['t','p']) #2 vars, 32 models, 120 months, lat & lon
    
    #group 120 months into seasons. first group by seasons within each year and average for each year
    labels = ['0_winter','0_winter','0_spring','0_spring','0_spring','0_summer','0_summer','0_summer','0_fall','0_fall','0_fall','0_winter',
              '1_winter','1_winter','1_spring','1_spring','1_spring','1_summer','1_summer','1_summer','1_fall','1_fall','1_fall','1_winter',
              '2_winter','2_winter','2_spring','2_spring','2_spring','2_summer','2_summer','2_summer','2_fall','2_fall','2_fall','2_winter',
              '3_winter','3_winter','3_spring','3_spring','3_spring','3_summer','3_summer','3_summer','3_fall','3_fall','3_fall','3_winter',
              '4_winter','4_winter','4_spring','4_spring','4_spring','4_summer','4_summer','4_summer','4_fall','4_fall','4_fall','4_winter',
              '5_winter','5_winter','5_spring','5_spring','5_spring','5_summer','5_summer','5_summer','5_fall','5_fall','5_fall','5_winter',
              '6_winter','6_winter','6_spring','6_spring','6_spring','6_summer','6_summer','6_summer','6_fall','6_fall','6_fall','6_winter',
              '7_winter','7_winter','7_spring','7_spring','7_spring','7_summer','7_summer','7_summer','7_fall','7_fall','7_fall','7_winter',
              '8_winter','8_winter','8_spring','8_spring','8_spring','8_summer','8_summer','8_summer','8_fall','8_fall','8_fall','8_winter',
              '9_winter','9_winter','9_spring','9_spring','9_spring','9_summer','9_summer','9_summer','9_fall','9_fall','9_fall','9_winter']
    present_10yrs = present_10yrs.assign_coords(time=labels).groupby('time').mean(dim='time')
    
    #then group across all years
    labels = present_10yrs.time.values
    labels = [l[2:] for l in labels] #chopping off the year number
    present_10yrs = present_10yrs.assign_coords(time=labels)
    
    #re-order from most drying to most wetting like in Script 1
    present_10yrs_ordered = present_10yrs.assign_coords(models=model_order[scenario]).sortby('models')
     
    #save as netcdf
    present_10yrs_ordered.to_netcdf(root + 'bcsd/{}/climate_present_10yrs.nc4'.format(scenario)) #2 variables, 32 models, 40 timesteps (4 seasons x 10 yrs), and lat/lon


