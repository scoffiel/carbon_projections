#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Approach #2
- Model present-day distribution of NLCD forest-vs-shrub layer based on climate data
- Project future land cover, change, and associated carbon change

Inputs:
- climate_present and climate_future nc files generated from script 1
- landcover_eighth.tif  (generated from Google Earth Engine script 4)
- carbon_eighth.tif & valid_fraction.tif
- ecoregion_carbon_densities.tiff (generated from Google Earth Engine script 4)

Outputs:
- netcdf raster layer of projected veg type change (one for each RCP+moisture scenario)
- Figures: maps of present-day veg cover, model residuals, veg type change
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import regionmask
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

SCENARIO = 'rcp45'
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


#read in vegetation data from GEE
#Use GEE script 7 to simplify land cover classes and reproject to 1/8 degree
#fill in NaN for anything
veg = xr.open_rasterio(root + 'land_cover/landcover_eighth.tif')[0,:,:]
veg = veg.where(veg > 0)


#Make map of dom veg type --------------------------------------------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.contourf(veg.x, veg.y,veg, transform=ccrs.PlateCarree(), levels=[0,1,2], cmap='YlGn') #PiYG
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax.add_feature(states, edgecolor='0.2')
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.set_title('Dominant vegetation type', fontsize=18)
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.65, pad=0.05, ticks=[0.5,1.5])
cbar.ax.set_xticklabels(['Shrub/grass','Forest'], fontsize=18) 
ax.text(-124.2,33.5,'(a)',fontsize=18, fontweight='bold')
#cbar.ax.set_xticklabels(['Forest loss','Forest gain'], fontsize=20) #hack - just needed this for a PiYG colorbar in final figure
ax.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

#plt.savefig(root + 'figures/fig3a_vegtype.eps')

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
cvar_names = ['T winter','T spring','T summer','T fall','P winter','P spring','P summer','P fall']

x = table.longitude.to_xarray() - 360
y = table.latitude.to_xarray()

table['veg'] = veg.sel(x=x, y=y, method='nearest').data
table = table.dropna().reset_index(drop=True) #added reset index later, check


#Random forest classification modeling ---------------------------------------------------------------------
x = table[cvars]
y = table.veg

#cross validation with 8 random groups (using this to report metrics and select hyperparameters) ----- 
kf = KFold(n_splits=10, shuffle=True, random_state=0) 
accuracies = []

for train, test in kf.split(x):
    xtrain, xtest, ytrain, ytest = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
    
    rfc = RandomForestClassifier(n_estimators=100, max_leaf_nodes=30, random_state=0)
    rfc.fit(xtrain, ytrain) 
    y_pred = rfc.predict(xtest)
    accuracies.append(accuracy_score(ytest, y_pred)*100)
    table.loc[test, 'veg_pred_cv'] = y_pred #cross validation predictions for scatterplots
   
print('Mean RFC accuracy {:.2f} +/- {:.2f}%, R2={:.2f}'.format(np.mean(accuracies), np.std(accuracies), r2_score(table.veg, table['veg_pred_cv'])))
print('Mean confusion matrix %', confusion_matrix(table.veg, table.veg_pred_cv)/len(table)*100)

#build single RFC for figures and projections -------
rfc = RandomForestClassifier(n_estimators=100, max_leaf_nodes=30, random_state=0)
rfc.fit(x, y)
table['veg_pred'] = rfc.predict(x)
total_accuracy = accuracy_score(table.veg, table.veg_pred)*100
print('non-CV total accuracy', total_accuracy)
#tree.export_graphviz(clf, out_file=root+'model_output/decision_trees/'+spp+'.dot', feature_names=cvars, filled=True)

fig4 = plt.figure()
plt.barh(range(len(cvars)), rfc.feature_importances_, tick_label=cvar_names)
plt.xlim((0,0.4))
plt.title('Variable Importance')



#apply future climate ---------------------------------------------------------------
x = table.longitude.to_xarray()
y = table.latitude.to_xarray()

table_future = pd.DataFrame()

for cvar in cvars:
    table_future[cvar] = future_climate.sel(variables=cvar, longitude=x, latitude=y).data

table_future['veg_pred'] = rfc.predict(table_future[cvars])


#calculate carbon change, accounting for ecoregions -------------------------------------
carb = xr.open_rasterio(root + 'carb_carbon/carbon_eighth.tif')[0,:,:]
carb = carb.where(carb > -9.999)
table['carb'] = carb.sel(x=x-360, y=y, method='nearest').data
valid = xr.open_rasterio(root + 'land_cover/valid_fraction.tif')[0,:,:]
table['valid'] = valid.sel(x=x-360, y=y, method='nearest').data
table_future['valid'] = table.valid

#add column for ecoregion-wide mean forest carbon 
ecoregion_forest_carbon = xr.open_rasterio(root + 'epa_ecoregions3/ecoregion_carbon_densities.tiff')[0,:,:]
table['ecoreg_forest_carbon'] = ecoregion_forest_carbon.sel(x=x-360, y=y, method='nearest').data
#plt.scatter(table.longitude, table.latitude, c=table.ecoreg_forest_carbon, s=0.5)

shrub_carb = table[table.veg==1]['carb'].mean() #one statewide estimate for shrubs is sufficient (about 10 ton/ha)

#go through and overwrite 'carb' column based on model prediction, as either shrub average or ecoregion forest average
for i in table.index:
    if table.loc[i, 'veg_pred']==1: #shrub
        table.loc[i, 'carb'] = shrub_carb
    else: #forest
        table.loc[i, 'carb'] = table.loc[i, 'ecoreg_forest_carbon']

for i in table_future.index:
    if table_future.loc[i, 'veg_pred']==1: #shrub
        table_future.loc[i, 'carb'] = shrub_carb
    else:
        table_future.loc[i, 'carb'] = table.loc[i, 'ecoreg_forest_carbon']

change = ( (table.veg_pred==2).sum() - (table_future.veg_pred==2).sum() ) / (table.veg_pred==2).sum() *100 #change in forest cover area
change2 = ((table_future.carb * table_future.valid).sum() - (table.carb * table.valid).sum()) / (table.carb*table.valid).sum() * 100 #change in calculated total carbon density


#map predicted vs observed CLASSIFICATION ----------------------------------------------------
fig = plt.figure(figsize=(12,6))

import matplotlib as mpl
cmap = plt.cm.YlGn  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[100:220]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds = np.arange(1,4)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

ax = fig.add_subplot(131, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table.veg, s=4, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, marker='s')
ax.add_feature(states, edgecolor='0.2')
ax.set_title('Observed', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(d)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.8, pad=0.06, ticks=[1.5,2.5])
cbar.ax.set_xticklabels(['Shrub/grass','Forest'], fontsize=13) 
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

ax = fig.add_subplot(132, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table.veg_pred, s=4, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, marker='s')
ax.add_feature(states, edgecolor='0.2')
ax.set_title('RF classification predicted', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(e)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.8, pad=0.06, ticks=[1.5,2.5])
cbar.ax.set_xticklabels(['Shrub/grass','Forest'], fontsize=13) 
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

cmap = plt.cm.PiYG  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[50:-50]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds = np.arange(-1.5,2.5)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

ax = fig.add_subplot(133, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table.veg_pred-table.veg, s=4, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, marker='s')
ax.add_feature(states, edgecolor='0.2')
ax.set_title('RF classification error', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(f)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.9, pad=0.06, ticks=[-1,1])
cbar.ax.set_xticklabels(['Underpredict\nforest','Overpredict\nforest'], fontsize=13)
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

#plt.savefig(root + 'figures/figS9def_classif.eps')

#save change as netcdf  -------------------------------------------------------
#use climate dataset as a template
export = present_climate.sel(latitude=slice(32.5, 42.3), longitude=slice(235.1,246.3), variables='p_fall')
export_array = np.zeros(export.shape) - np.nan
export = xr.DataArray(export_array, coords=[export.latitude, export.longitude], dims=["latitude", "longitude"])

for i in table.index:
    export.loc[{'latitude':table.loc[i,'latitude'], 'longitude':table.loc[i,'longitude']}] = table_future.loc[i, 'veg_pred'] - table.loc[i,'veg_pred']
    
export.attrs["units"] = "0nochange_-1forestloss_1forestgain"
export = export.rename('veg_change')
export.to_dataset(name='veg_change').to_netcdf(root + 'model_output/2_RF_veg_class/{}_{}.nc4'.format(SCENARIO,MODEL))

#Make map of dom veg type change --------------------------------------------------------------
fig7 = plt.figure(figsize=(8,8))
ax = fig7.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table_future.veg_pred-table.veg_pred, s=12, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, marker='s')
ax.add_feature(states, edgecolor='0.2')
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.text(0.55,0.9,'{:.1f}%'.format(change), fontsize=15, transform=ax.transAxes)
ax.text(0.55,0.86,'loss of forest area', transform=ax.transAxes, fontsize=15)
ax.text(0.55,0.75,'{:.1f}%'.format(change2), fontsize=15, fontweight='bold', transform=ax.transAxes)
ax.text(0.55,0.66,'total AGL\ncarbon change', transform=ax.transAxes, fontsize=15)
ax.set_title('RCP4.5 Mean Change', fontsize=18)
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.65, pad=0.05, ticks=[-1,1])
cbar.ax.set_xticklabels(['Forest loss','Forest gain'], fontsize=18) 
ax.text(-124.2,33.5,'(b)',fontsize=18, fontweight='bold')
ax.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

#plt.savefig(root + 'figures/fig3b_veg45change.eps')