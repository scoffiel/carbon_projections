#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Approach #4
- Fit RF regression models to each of the top 20 tree spp in California
- Project future carbon and change
- Apply restrictions on distance between spp present and future locations (migration scenarios)

Inputs:
- climate_present and climate_future nc files (generated from script 1)
- valid_fraction (generated from GEE script 1)
- lemma_39spp_eighth.tif (generated from GEE script 7)
- spp_groups.csv (generated by hand in Excel)

Outputs:
- netcdf raster layer of projected carbon change (one for each RCP+moisture scenario)
- Figures: maps of present-day carbon density, change, analogue arrows, novelty of future climate
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import regionmask
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

SCENARIO = 'rcp85'
MODEL = 'mean' #wet, dry, mean

root = '/Users/scoffiel/california/'

#read in climate data ---------------------------------------------------------
present_climate = xr.open_dataset(root + 'bcsd/{}/climate_present.nc4'.format(SCENARIO)).tas 
future_climate = xr.open_dataset(root + 'bcsd/{}/climate_future.nc4'.format(SCENARIO)).tas

if MODEL=='dry': #get first 8 models only
    present_climate = present_climate.sel(models=slice(0,7))
    future_climate = future_climate.sel(models=slice(0,7))

if MODEL=='wet': #get last 8 models only
    present_climate = present_climate.sel(models=slice(24,31))
    future_climate = future_climate.sel(models=slice(24,31))

present_climate = present_climate.mean(dim='models')
future_climate = future_climate.mean(dim='models')

#read in lemma species data 
#used GEE script 7 to remap species densities from FID grid and reproject to 1/8 degree
lemma = xr.open_rasterio(root + 'lemma_species/lemma_39spp_eighth.tif') 
spps = lemma.sel(band=1).descriptions
lemma = lemma/1000 #from kg biomass per hectare to ton biomass per hectare

valid = xr.open_rasterio(root + 'land_cover/valid_fraction.tif')[0,:,:]
valid = valid.where(lemma[0,:,:] > -9.999)

lemma = lemma.where(lemma > -9.999)
lemma = lemma * 0.47 #convert biomass to carbon based on Gonzalez 2015

lemma_total = lemma*valid

groups = pd.read_csv(root + 'lemma_species/spp_groups.csv')
groups['name'] = spps


#build table and join to BCSD climate --------------------------------------------------
#use BCSD coordinates as base for x, y

mask = regionmask.defined_regions.natural_earth.us_states_50.mask(present_climate.longitude, present_climate.latitude, wrap_lon=True)
cali = mask==4
cali = cali.rename({'lon':'longitude', 'lat':'latitude'})
present_climate = present_climate.where(cali)

table = present_climate.sel(variables='t_winter').to_dataframe('t_winter').dropna().reset_index()
del table['variables']
table['t_spring'] = present_climate.sel(variables='t_spring').to_dataframe('t_spring').dropna().reset_index()['t_spring']
table['t_summer'] = present_climate.sel(variables='t_summer').to_dataframe('t_summer').dropna().reset_index()['t_summer']
table['t_fall']   = present_climate.sel(variables='t_fall').to_dataframe('t_fall').dropna().reset_index()['t_fall']
table['p_winter'] = present_climate.sel(variables='p_winter').to_dataframe('p_winter').dropna().reset_index()['p_winter']
table['p_spring'] = present_climate.sel(variables='p_spring').to_dataframe('p_spring').dropna().reset_index()['p_spring']
table['p_summer'] = present_climate.sel(variables='p_summer').to_dataframe('p_summer').dropna().reset_index()['p_summer']
table['p_fall']   = present_climate.sel(variables='p_fall').to_dataframe('p_fall').dropna().reset_index()['p_fall']
cvars = table.columns[2:10] #climate variables

x = table.longitude.to_xarray() - 360
y = table.latitude.to_xarray()

for i in range(len(spps)):
    spp = spps[i]
    z = lemma[i,:,:]
    table[spp] = z.sel(x=x, y=y, method='nearest').data
table['valid'] = valid.sel(x=x, y=y, method='nearest').data
table = table.dropna().reset_index(drop=True) #added reset index later, check


#add columns for different groups ---------------------------------
conifers = groups[groups.conifer_hardwood=='Conifer'].name
table['conifer'] = table[conifers].sum(axis=1)
hardwoods = groups[groups.conifer_hardwood=='Hardwood'].name
table['hardwood'] = table[hardwoods].sum(axis=1)

pines = groups[groups.pine_oak_other=='Pine'].name
table['pine'] = table[pines].sum(axis=1)
oaks = groups[groups.pine_oak_other=='Oak'].name
table['oak'] = table[oaks].sum(axis=1)
other = groups[groups.pine_oak_other=='Other'].name
table['other'] = table[other].sum(axis=1)

group_names = ['conifer','hardwood','pine','oak','other']

#focus on top 20 species by carbon
spps = spps[:20]

'''
#Quick plots of carbon density by group --------------------------------------------------------------
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
fig = plt.figure(figsize=(12,4))
count = 0
for g in group_names:
    count+=1
    ax = fig.add_subplot(1,5,count, projection=ccrs.Miller())
    ax.set_extent([235,246,33,45], crs=ccrs.Miller())
    plot = ax.scatter(table.longitude, table.latitude, c=table[g], s=3, transform=ccrs.PlateCarree(), cmap='YlGn', vmin=0, vmax=100, marker='s')
    ax.add_feature(states, edgecolor='0.2')
    ax.set_title(g)
plt.colorbar(plot, label='Carbon density (ton/ha)', orientation='vertical', shrink=0.8, ax=ax, extend='max')   
'''

#Model: RF regression on carbon density ---------------------------------------------------------------------
#dictionaries to store one average/representative model trained on entire set, for making figures & projections
rfrs = {} #dictionary of RF models for 

for spp in list(spps) + ['conifer','hardwood','pine','oak','other']:

    x = table[cvars]     #climate predictors
    y = table[spp]       #carbon density
    
    #cross validation with 8 random groups (using this to report metrics and select hyperparameters) ----- 
    kf = KFold(n_splits=10, shuffle=True, random_state=0) 
    rmses = []
    
    for train, test in kf.split(x):
        xtrain, xtest, ytrain, ytest = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]

        #random forest regressor
        ytrain, ytest = y.iloc[train], y.iloc[test]
        rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=20, random_state=0)
        rfr.fit(xtrain, ytrain) 
        y_pred = rfr.predict(xtest)
        rmses.append(np.sqrt(mean_squared_error(ytest, y_pred)))
        table.loc[test, spp+'_density_pred_cv'] = y_pred #cross validation predictions for scatterplots
       
    print('{} mean RFR error {:.2f} +/- {:.2f}, R2={:.2f}'.format(spp, np.mean(rmses), np.std(rmses), r2_score(table[spp], table[spp+'_density_pred_cv'])))
    
    #build single model for figures and projections -------
    rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=20, random_state=0)
    rfr.fit(x, y) 
    rfrs[spp] = rfr
    table[spp+'_density_pred'] = rfr.predict(x)
  

#apply future climate ---------------------------------------------------------------
x = table.longitude.to_xarray()
y = table.latitude.to_xarray()

table_future = pd.DataFrame()

for cvar in cvars:
    table_future[cvar] = future_climate.sel(variables=cvar, longitude=x, latitude=y).data

for spp in list(spps) + ['conifer','hardwood','pine','oak','other']:
    rfr = rfrs[spp]
    table_future[spp+'_density_pred'] = rfr.predict(table_future[cvars])


#calculate total carbon change (goes into Table 3 ----------------------------------------------
table['total_c'] = table[list(spps)].sum(axis=1) * table.valid
table['total_c_pred'] = table[[spp+'_density_pred' for spp in spps]].sum(axis=1) * table.valid
table_future['total_c_pred'] = table_future[[spp+'_density_pred' for spp in spps]].sum(axis=1) * table.valid
print('20 spps', (table_future.total_c_pred.sum() - table.total_c_pred.sum() )/ table.total_c_pred.sum())

#calculate total carbon change - groupings
grp = ['conifer','hardwood']
table['total_c_groups'] = table[grp].sum(axis=1) * table.valid
table['total_c_pred_groups'] = table[[g+'_density_pred' for g in grp]].sum(axis=1) * table.valid
table_future['total_c_pred_groups'] = table_future[[g+'_density_pred' for g in grp]].sum(axis=1) * table.valid
print('conifer/hardw', (table_future.total_c_pred_groups.sum() - table.total_c_pred_groups.sum() )/ table.total_c_pred_groups.sum())

grp = ['pine','oak','other']
table['total_c_groups'] = table[grp].sum(axis=1) * table.valid
table['total_c_pred_groups'] = table[[g+'_density_pred' for g in grp]].sum(axis=1) * table.valid
table_future['total_c_pred_groups'] = table_future[[g+'_density_pred' for g in grp]].sum(axis=1) * table.valid
print('pine/oak/other', (table_future.total_c_pred_groups.sum() - table.total_c_pred_groups.sum() )/ table.total_c_pred_groups.sum())



'''
#map predicted, observed, and error (spatial residuals for Fig S3) -------------------------------------------------
fig = plt.figure(figsize=(12,34))
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
vmaxs = {'DouglasFir':70,'PonderosaPine':30,'CanyonLiveOak':30,'Redwood':130,'conifer':120,'hardwood':70}
letters = ['(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)',]
count = 0
for spp in ['DouglasFir','PonderosaPine','CanyonLiveOak','Redwood','conifer','hardwood']:

    count+=1
    ax = fig.add_subplot(6,3,count, projection=ccrs.Miller())
    ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
    plot = ax.scatter(table.longitude, table.latitude, c=table[spp], s=4, transform=ccrs.PlateCarree(), cmap='YlGn', marker='s', vmin=0, vmax=vmaxs[spp])
    ax.add_feature(states, edgecolor='0.2')
    ax.set_title('Observed '+spp, fontsize=14)
    ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
    ax.add_feature(states, edgecolor='0.2')
    ax.text(-124.2,33.5,letters[count-1],fontsize=16, fontweight='bold')
    cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.07)
    cbar.set_label('ton C/ha', size=13)
    cbar.ax.tick_params(labelsize=13)
    ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
    ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-124,-122,-120,-118,-116,''])
    ax.set_yticklabels([32,34,36,38,40,42])
    ax.tick_params(top=True, right=True)
    
    count+=1
    ax = fig.add_subplot(6,3,count, projection=ccrs.Miller())
    ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
    plot = ax.scatter(table.longitude, table.latitude, c=table[spp+'_density_pred'], s=4, transform=ccrs.PlateCarree(), cmap='YlGn', marker='s', vmin=0, vmax=vmaxs[spp])
    ax.add_feature(states, edgecolor='0.2')
    ax.set_title('Predicted '+spp, fontsize=14)
    ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
    ax.add_feature(states, edgecolor='0.2')
    ax.text(-124.2,33.5,letters[count-1],fontsize=16, fontweight='bold')
    cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.07)
    cbar.set_label('ton C/ha', size=13)
    cbar.ax.tick_params(labelsize=13)
    ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
    ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-124,-122,-120,-118,-116,''])
    ax.set_yticklabels([32,34,36,38,40,42])
    ax.tick_params(top=True, right=True)
    
    count+=1
    ax = fig.add_subplot(6,3,count, projection=ccrs.Miller())
    ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
    plot = ax.scatter(table.longitude, table.latitude, c=table[spp+'_density_pred']-table[spp], s=4, transform=ccrs.PlateCarree(), cmap='PiYG', marker='s', vmin=-vmaxs[spp]/2, vmax=vmaxs[spp]/2)
    ax.add_feature(states, edgecolor='0.2')
    ax.text(-119, 43, 'Underpredict', fontsize=10, color='violet')
    ax.text(-119, 42, 'Overpredict', fontsize=10, color='green')
    ax.set_title('Error for '+spp, fontsize=14)
    ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
    ax.add_feature(states, edgecolor='0.2')
    ax.text(-124.2,33.5,letters[count-1],fontsize=16, fontweight='bold')
    cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.07)
    cbar.set_label('ton C/ha', size=13)
    cbar.ax.tick_params(labelsize=13)
    ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
    ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-124,-122,-120,-118,-116,''])
    ax.set_yticklabels([32,34,36,38,40,42])
    ax.tick_params(top=True, right=True)

plt.savefig(root + 'figures/figS3gx_species.eps')
'''
stop

#Make maps of change --------------------------------------------------------------
#6 subplots for 4 species and 2 types

ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')

names = ['DouglasFir','PonderosaPine','CanyonLiveOak','Redwood','conifer','hardwood']
titles = ['Douglas Fir','Ponderosa Pine','Canyon Live Oak','Redwood','All Conifers','All Hardwoods']
letters = ['(a)','(b)','(c)','(d)','(e)','(f)',]
vmaxs = [35,15,15,70,60,40]

fig = plt.figure(figsize=(14,20))
for i in range(6):
    ax = fig.add_subplot(3,2,i+1, projection=ccrs.Miller())
    ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
    ax.add_feature(states, edgecolor='0.2')
    ax.add_feature(ecoregions, edgecolor='0.2', facecolor='none', linewidth=0.2)
    plot = ax.scatter(table.longitude, table.latitude, c=table_future[names[i]+'_density_pred']-table[names[i]+'_density_pred'], vmin=-vmaxs[i], vmax=vmaxs[i], s=14, transform=ccrs.PlateCarree(), cmap='PRGn', marker='s')
    cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8, pad=0.01, extend='both')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('ton C/ha', size=15)
    ax.set_title(titles[i] + ' Change', fontsize=18)
    present = (table[names[i]+'_density_pred'] * table.valid).sum()
    future = (table_future[names[i]+'_density_pred'] * table.valid).sum()
    change = (future - present) / present * 100
    ax.text(0.55,0.81,'{:.1f}%'.format(change), fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.text(-124.2,33.5,letters[i],fontsize=18, fontweight='bold')
    ax.text(0.55,0.7,'total AGL\ncarbon change', fontsize=15, transform=ax.transAxes)
    ax.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
    ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
    ax.set_xticklabels([-124,-122,-120,-118,-116,''])
    ax.set_yticklabels([32,34,36,38,40,42])
    ax.tick_params(top=True, right=True)
plt.subplots_adjust(wspace=0, hspace=0.15)
#plt.savefig(root + 'figures/fig4_sppchange.eps')

#save changse as netcdfs, one for each spp----------------------------------------------------
#use climate dataset as a template
export = present_climate.sel(latitude=slice(32.5, 42.3), longitude=slice(235.1,246.3), variables='p_fall')
export_array = np.zeros(export.shape) - np.nan
export = xr.DataArray(export_array, coords=[export.latitude, export.longitude], dims=["latitude", "longitude"])

for spp in list(spps) + ['conifer','hardwood']:
    export_spp= export.copy()
    for i in table.index:
        export_spp.loc[{'latitude':table.loc[i,'latitude'], 'longitude':table.loc[i,'longitude']}] = table_future.loc[i,spp+'_density_pred'] - table.loc[i,spp+'_density_pred']
        
    export_spp.attrs["units"] = "tonC-per-ha"
    export_spp = export_spp.rename('carbon_change')
    export_spp.to_dataset(name='carbon_change').to_netcdf(root + 'model_output/4_species_models/{}_{}/{}.nc4'.format(SCENARIO,MODEL,spp))

'''
#get total carbon by species for Table S1 --------------------------------------
totals = table.loc[:,spps] * 111*.125 * 88*.125 * 100 #tonC/ha * 100ha/km2 * km2/pixel -> tonC/pixel
totals = totals.sum()/1e6 #ton to Mt

for spp in spps:
    present = (table[spp+'_density_pred'] * table.valid).sum()
    future = (table_future[spp+'_density_pred'] * table.valid).sum()
    change = (future - present) / present * 100
    print(spp, change)
'''

#MIGRATION COMPONENT -----------------------------------------------------------------
#force anywhere in the future to "zero" if it's not geographically close enough to somewhere that currently has at least 1 ton observed

table2 = table[[spp for spp in spps]] 
table_future2 = table_future[[spp + '_density_pred' for spp in spps]]

for spp in spps:
    table2[spp+'_presence'] = (table[spp] > 1) + 0 #binary. anywhere with > 1 Mt/ha carbon
    
for spp in spps:
    present_presence = table2[spp+'_presence']
    
    for i in table_future2.index: #takes a min
        #within slow? (up to 1 pixel (0.125 deg) in any direction)
        lat1 = table.loc[i, 'latitude']
        lon1 = table.loc[i, 'longitude'] 
        
        lat2 = table.latitude 
        lon2 = table.longitude
        
        within_slow = (np.abs(lat1-lat2) < 0.13) & (np.abs(lon1-lon2) < 0.13) #for a given point, here are all the locations within 1 pixel (buffer)
        table_future2.loc[i, spp+ '_possible_slow'] = np.sum(within_slow & present_presence) > 1 #this pixel could have future presence if there is any overlap between its buffer and the present-day range 
        
        #within fast? (up to 500*85 m in any direction)
        lat1 = table.loc[i, 'latitude'] * np.pi/180
        lon1 = table.loc[i, 'longitude'] * np.pi/180
        
        lat2 = table.latitude * np.pi/180
        lon2 = table.longitude * np.pi/180
        
        dlat = lat2 - lat1 #vector
        dlon = lon2 - lon1 #vector
        
        d = np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlon)) * 6371000 #distance to every other point
        within_fast = (d < 500*85)
        table_future2.loc[i, spp+ '_possible_fast'] = np.sum(within_fast & present_presence) > 1
        
for spp in spps:
    table_future2[spp+'_density_slow'] = table_future2[spp+'_density_fast'] = table_future2[spp+'_density_pred']
    table_future2.loc[table_future2[spp+'_possible_slow']==False, spp+'_density_slow'] = table_future2[spp+'_density_pred'].min()
    table_future2.loc[table_future2[spp+'_possible_fast']==False, spp+'_density_fast'] = table_future2[spp+'_density_pred'].min() #not zero
    
    
#calculate total carbon change
equil_names = [spp + '_density_pred' for spp in spps]
slow_names = [spp + '_density_slow' for spp in spps]
fast_names = [spp + '_density_fast' for spp in spps]
table_future2['total_c_equil'] = table_future2[equil_names].sum(axis=1) * table.valid
table_future2['total_c_slow'] = table_future2[slow_names].sum(axis=1) * table.valid
table_future2['total_c_fast'] = table_future2[fast_names].sum(axis=1) * table.valid
table2['total_c_present_pred'] = table[equil_names].sum(axis=1) * table.valid

print('equil',(table_future2.total_c_equil.sum() - table2.total_c_present_pred.sum() )/ table2.total_c_present_pred.sum())
print('fast',(table_future2.total_c_fast.sum() - table2.total_c_present_pred.sum() )/ table2.total_c_present_pred.sum())
print('slow',(table_future2.total_c_slow.sum() - table2.total_c_present_pred.sum() )/ table2.total_c_present_pred.sum())    