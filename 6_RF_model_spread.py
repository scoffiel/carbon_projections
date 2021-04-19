#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose:
- Rebuild RF regression models from script 5, for each of the 32 climate models
- Compare 3 different runs: T & P both change, T only (P constant), and P only (T constant)
- Generate histogram + boxplot figure of spread across 32 models for 3 different runs (Fig 5)

Inputs: (similar to script 5)
- climate_present and climate_future nc files generated from script 1
- carbon_eighth.tif  (generated from Google Earth Engine script 1)
- valid_fraction.tif (generated from Google Earth Engine script 2)

Outputs:
- Figure 5 histogram + boxplots of model spread and T vs P effect
- optional pkl file to store results

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import regionmask
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

root = '/Users/scoffiel/california/'
SCENARIO = 'rcp85'

#read in climate data ---------------------------------------------------------
present_climate_all = xr.open_dataset(root + 'bcsd/{}/climate_present.nc4'.format(SCENARIO)).tas 
future_climate_all = xr.open_dataset(root + 'bcsd/{}/climate_future.nc4'.format(SCENARIO)).tas

#read in carbon data layer and valid land fraction data layer (generated from GEE scripts 1 and 2)
carb = xr.open_rasterio(root + 'carb_carbon/carbon_eighth.tif')[0,:,:]
valid = xr.open_rasterio(root + 'land_cover/valid_fraction.tif')[0,:,:]
valid = valid.where(carb > -9.999)
carb = carb.where(carb > -9.999)

carb_total = carb*valid


#3 main runs: T & P change, T only (P constant), and P only (T constant)
runs = [0,1,2]
percent_changes = [[],[],[]]

for run in runs: #takes several minutes
    future_climate_run = future_climate_all.copy()
    
    if run==1:
        future_climate_run[:, :4, :, :] = present_climate_all[:, :4, :, :] #T only, P held constant
    if run==2:
        future_climate_run[:, 4:, :, :] = present_climate_all[:, 4:, :, :] #P only, T held constant


    for m in range(32):
        print('model', m)
            
        present_climate = present_climate_all[m,:,:,:]
        future_climate = future_climate_run[m,:,:,:]
            
        #build table and join to BCSD climate --------------------------------------------------   
        mask = regionmask.defined_regions.natural_earth.us_states_50.mask(present_climate.longitude, present_climate.latitude, wrap_lon=True)
        cali = mask==4
        cali = cali.rename({'lon':'longitude', 'lat':'latitude'})
        present_climate = present_climate.where(cali)
        
        table = present_climate.sel(variables='t_winter').to_dataframe('t_winter').dropna().reset_index()
        del table['variables'], table['models']
        table['t_spring'] = present_climate.sel(variables='t_spring').to_dataframe('t_spring').dropna().reset_index()['t_spring']
        table['t_summer'] = present_climate.sel(variables='t_summer').to_dataframe('t_summer').dropna().reset_index()['t_summer']
        table['t_fall']   = present_climate.sel(variables='t_fall').to_dataframe('t_fall').dropna().reset_index()['t_fall']
        table['p_winter'] = present_climate.sel(variables='p_winter').to_dataframe('p_winter').dropna().reset_index()['p_winter']
        table['p_spring'] = present_climate.sel(variables='p_spring').to_dataframe('p_spring').dropna().reset_index()['p_spring']
        table['p_summer'] = present_climate.sel(variables='p_summer').to_dataframe('p_summer').dropna().reset_index()['p_summer']
        table['p_fall']   = present_climate.sel(variables='p_fall').to_dataframe('p_fall').dropna().reset_index()['p_fall']
        cvars = table.columns[2:10] #climate variables
        cvar_names = ['T winter','T spring','T summer','T fall','P winter','P spring','P summer','P fall']
        
        #join to carbon data (new column)
        x = table.longitude.to_xarray() - 360
        y = table.latitude.to_xarray()
        
        table['carb'] = carb.sel(x=x, y=y, method='nearest').data
        table['carb_total'] = carb_total.sel(x=x, y=y, method='nearest').data #accounting for bare fraction, use for calc total C
        table['valid'] = valid.sel(x=x, y=y, method='nearest').data
        table = table.dropna().reset_index(drop=True) #added reset index later, check
        
        #Random forest regression modeling -----------------------------------------------------------------
        x = table[cvars]
        y = table.carb
        
        #cross validation with 10 random groups
        kf = KFold(n_splits=10, shuffle=True, random_state=0) 
        
        for train, test in kf.split(x):
            xtrain, xtest, ytrain, ytest = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
            
            rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=25, random_state=0)
            rfr.fit(xtrain, ytrain) 
            y_pred = rfr.predict(xtest)
              
        #build single RFR
        rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=25, random_state=0)
        rfr.fit(x, y)
        table['carb_pred'] = rfr.predict(x)
        
        #apply future climate -------------------------------------------------
        x = table.longitude.to_xarray()
        y = table.latitude.to_xarray()
        
        table_future = pd.DataFrame()
        for cvar in cvars:
            table_future[cvar] = future_climate.sel(variables=cvar, longitude=x, latitude=y).data
    
        table_future['carb_pred'] = rfr.predict(table_future[cvars])
        
        change = ((table_future['carb_pred']*table.valid).sum() - (table['carb_pred']*table.valid).sum())/ (table['carb_pred']*table.valid).sum()*100
        print(change)
        
        percent_changes[run].append(change)


'''
#optional save data to avoid rerunning entire script

with open(root + 'model_output/1_RF_regression/percent_changes_all.pkl', 'wb') as f_out:
    pickle.dump(percent_changes, f_out)
    
with open(root + 'model_output/1_RF_regression/percent_changes_all.pkl', 'rb') as f_in:
    percent_changes = pickle.load(f_in)
'''


fig, (ax1,ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,2]}, figsize=(5,6))

#3 overlapping histograms
counts, bins = np.histogram(percent_changes[0], bins=range(-50,46,5))
ax1.bar(bins[:-1], counts, width=5, alpha=0.5, label='T and P change', align='edge', color='k', ec='k')

counts, bins = np.histogram(percent_changes[1], bins=range(-50,46,5))
ax1.bar(bins[:-1], counts, width=5, alpha=0.5, label='T only, P constant', align='edge', color='orange')

counts, bins = np.histogram(percent_changes[2], bins=range(-50,46,5))
ax1.bar(bins[:-1], counts, width=5, alpha=0.5, label='P only, T constant', align='edge', color='deepskyblue')

ax1.text(0.01,0.94,'(a)', fontweight='bold', transform=ax1.transAxes)
ax1.set_ylabel('Number of CMIP5 models')
ax1.set_yticks(range(0,20,4))
ax1.set_xticklabels([])
ax1.set_xlim((-54,47))
ax1.legend()

#3 boxplots
data = np.vstack((percent_changes[2], percent_changes[1], percent_changes[0])).T
bp = ax2.boxplot(data, vert=False, widths=0.7, patch_artist=True)

colors = ['lightskyblue', 'orange', 'gray']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax2.text(0.01,0.92,'(b)', fontweight='bold', transform=ax2.transAxes)
ax2.set_xlabel('Total AGL carbon change (%)')
ax2.set_yticks([1,2,3])
ax2.set_yticklabels(['P only,\nT constant','T only,\nP constant','T and P\nchange'])
ax2.set_xlim((-54,47))

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(root + 'figures/fig5_TPspread.pdf') #eps doesn't work with semi-opacity