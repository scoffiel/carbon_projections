#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Approach #1
- Model present-day distribution of CARB carbon layer based on climate data
- Project future carbon and change

Inputs:
- climate_present and climate_future nc files generated from script 1
- carbon_eighth.tif  (generated from Google Earth Engine script 1)
- valid_fraction.tif (generated from Google Earth Engine script 2)
- elev_eighth.tif    (generated from Google Earth Engine script 3)

Outputs:
- netcdf raster layer of projected carbon change (one for each RCP+moisture scenario)
- Figures: maps of present-day carbon density, residuals, variable correlations, model performance, and projected change
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import regionmask
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

SCENARIO = 'rcp85'
MODEL = 'wet' #_wet, _dry, mean
title = 'RCP8.5 Wet'
letter = '(f)'

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


#read in carbon data layer and valid land fraction data layer (generated from GEE scripts 1 and 2)
carb = xr.open_rasterio(root + 'carb_carbon/carbon_eighth.tif')[0,:,:]
valid = xr.open_rasterio(root + 'land_cover/valid_fraction.tif')[0,:,:]
valid = valid.where(carb > -9.999)
carb = carb.where(carb > -9.999)

carb_total = carb*valid

#Make map of carbon density --------------------------------------------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.contourf(carb.x, carb.y,carb, transform=ccrs.PlateCarree(), levels=np.arange(15)*10, extend='max', cmap='YlGn')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.set_title('Present-day AGL carbon density', fontsize=18)
ax.text(-124.2,33.5,'(a)',fontsize=18, fontweight='bold')
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8, pad=0.01)
cbar.set_label('ton C/ha', size=15)
cbar.ax.tick_params(labelsize=15)

ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

#plt.savefig(root + 'figures/fig2a_carbonmap.eps')
#plt.savefig(root + 'figures/figS10a_carbonunfiltered.eps')


'''
#optional map of valid land fraction --------------------------------------------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.contourf(carb.x, carb.y,valid, transform=ccrs.PlateCarree(), levels=np.arange(11)/10, cmap='magma')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax.add_feature(states, edgecolor='0.2')
ax.set_title('Valid land fraction')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.01)
'''

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

'''
#correlation matrix for climate data (Fig S3c)
corr = table.iloc[:, 2:10].corr() #correlation matrix of 8 climate vars
corr = corr.rename({'t_winter':'T win','t_spring':'T spr','t_summer':'T sum','t_fall':'T fall','p_winter':'P win','p_spring':'P spr','p_summer':'P sum','p_fall':'P fall'})
corr = corr.rename({'t_winter':'T win','t_spring':'T spr','t_summer':'T sum','t_fall':'T fall','p_winter':'P win','p_spring':'P spr','p_summer':'P sum','p_fall':'P fall'}, axis=1)
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(6,6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True,
            square=True, linewidths=.5, cbar=False)
ax.text(7.2,0.6,'(c)', fontsize=16, fontweight='bold')
[ax.spines[i].set_visible(True) for i in ['top','bottom','left','right']]
ax.set_title('Correlation matrix', fontsize=16)
#plt.savefig(root + 'figures/figS3c_correlation.eps')
'''

#join to carbon data (new column)

x = table.longitude.to_xarray() - 360
y = table.latitude.to_xarray()

table['carb'] = carb.sel(x=x, y=y, method='nearest').data
table['carb_total'] = carb_total.sel(x=x, y=y, method='nearest').data #accounting for bare fraction, use for calc total C
table['valid'] = valid.sel(x=x, y=y, method='nearest').data
table = table.dropna().reset_index(drop=True) #added reset index later, check

carb_total = table.carb_total.mean() * 15263 * len(table) #hectares in 1/8 degree box, sample size

'''
#optional data exploration: scatterplots with carbon vs. each of the 8 climate variables
fig2 = plt.figure(figsize=(15,8), tight_layout=True)
count = 0
for var in cvars: #8 climate vars
    count += 1
    ax = fig2.add_subplot(2,4,count)
    label = cvar_names[count-1]
    if var[0]=='t':
        color='red'
        label = label + ' (\u00B0C)'
        xrange = (-14,35)
    else:
        color='blue'
        label = label + ' (mm/day)'
        xrange = (0.015, 18)
    plot = ax.scatter(table[var], table.carb, s=0.5, c=color)
    #ax.set_yscale('log')
    #if var[0]=='p':ax.set_xscale('log')
    ax.set_xlabel(label, fontsize=12)
    ax.set_xlim(xrange)
    if count in [1,5]: ax.set_ylabel('Carbon density (ton/ha)', fontsize=12)
'''

#Random forest regression modeling ---------------------------------------------------------------------
x = table[cvars]
y = table.carb

#cross validation with 10 random groups (using this to report metrics and select hyperparameters) ----- 
kf = KFold(n_splits=10, shuffle=True, random_state=0) 
rmses = []

for train, test in kf.split(x):
    xtrain, xtest, ytrain, ytest = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
    
    rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=25, random_state=0)
    rfr.fit(xtrain, ytrain) 
    y_pred = rfr.predict(xtest)
    rmses.append(np.sqrt(mean_squared_error(ytest, y_pred)))
    table.loc[test, 'carb_pred_cv'] = y_pred #cross validation predictions for scatterplots
   
print('Mean RFR error {:.2f} +/- {:.2f}, R2={:.2f}'.format(np.mean(rmses), np.std(rmses), r2_score(table.carb, table['carb_pred_cv'])))


#build single RFR for figures and projections 
rfr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=25, random_state=0)
rfr.fit(x, y)
table['carb_pred'] = rfr.predict(x)


#model details plot with variable importance and model performance -----------------------
fig = plt.figure(figsize=(8,4), tight_layout=True)
order = np.argsort(rfr.feature_importances_)
ax = fig.add_subplot(121)
ax.barh(range(len(cvars)), rfr.feature_importances_[order], tick_label=[cvar_names[i] for i in order])
ax.set_xlim((0,0.5))
ax.set_title('Variable Importance')
ax.set_xlabel('Relative importance')
ax.text(0.85, 0.05, '(a)',fontsize=12, fontweight='bold', transform=ax.transAxes)
ax2 = fig.add_subplot(122)
ax2.scatter(table.carb, table.carb_pred_cv, s=1, c='gray')
ax2.set_xlabel('Observed AGL carbon (ton C/ha)')
ax2.set_ylabel('Predicted AGL carbon (ton C/ha)')
ax2.set_xlim((0,200))
ax2.set_ylim((0,200))
slope, inter, r, _, _ = stats.linregress(table.carb, table.carb_pred_cv)
ax2.plot([0,200], [0,200], c='black', linestyle='--')
ax2.plot([0, 200], [inter, slope*200+inter], c='black') #fixed
ax2.text(75, 160, 'R$^2$ = {:.3f}'.format(r**2))
ax2.set_title('Model Performance')
ax2.text(0.85, 0.05, '(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
#plt.savefig(root + 'figures/figS4ab_performance.eps')


#map predicted, observed, error for the present (fig S3abc) ----------------------------------------------------
'''
fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(131, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table.carb, s=4, transform=ccrs.PlateCarree(), cmap='YlGn', marker='s', vmin=0, vmax=160)
ax.add_feature(states, edgecolor='0.2')
ax.set_title('Observed', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(a)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.8, pad=0.06)
cbar.set_label('ton C/ha', size=13)
cbar.ax.tick_params(labelsize=13)
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

ax = fig.add_subplot(132, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table['carb_pred'], s=4, transform=ccrs.PlateCarree(), cmap='YlGn', marker='s', vmin=0, vmax=160)
ax.add_feature(states, edgecolor='0.2')
ax.set_title('RF regression predicted', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(b)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.8, pad=0.06)
cbar.set_label('ton C/ha', size=13)
cbar.ax.tick_params(labelsize=13)
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

ax = fig.add_subplot(133, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax.scatter(table.longitude, table.latitude, c=table['carb_pred']-table.carb, s=4, transform=ccrs.PlateCarree(), cmap='PiYG', marker='s', vmin=-100, vmax=100)
ax.add_feature(states, edgecolor='0.2')
ax.text(-119, 43, 'Underpredict', fontsize=10, color='violet')
ax.text(-119, 42, 'Overpredict', fontsize=10, color='green')
ax.set_title('RF error (pred.-obs.)', fontsize=15)
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.text(-124.2,33.5,'(c)',fontsize=16, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.8, pad=0.06)
cbar.set_label('ton C/ha', size=13)
cbar.ax.tick_params(labelsize=13)
ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

plt.savefig(root + 'figures/figS9abc_regression.eps')

'''


#apply future climate ---------------------------------------------------------------
x = table.longitude.to_xarray()
y = table.latitude.to_xarray()

table_future = pd.DataFrame()

for cvar in cvars:
    table_future[cvar] = future_climate.sel(variables=cvar, longitude=x, latitude=y).data


'''
#optional model exploration, toggling holding precip or temp constant
seasons = ['winter','spring','summer','fall']
seasons= ['t_' + season for season in seasons]
for season in seasons:
    table_future[season] = table[season]
'''

table_future['carb_pred'] = rfr.predict(table_future[cvars])



#save change as netcdf  -------------------------------------------------------
#use climate dataset as a template
export = present_climate.sel(latitude=slice(32.5, 42.3), longitude=slice(235.1,246.3), variables='p_fall')
export_array = np.zeros(export.shape) - np.nan
export = xr.DataArray(export_array, coords=[export.latitude, export.longitude], dims=["latitude", "longitude"])

for i in table.index:
    export.loc[{'latitude':table.loc[i,'latitude'], 'longitude':table.loc[i,'longitude']}] = table_future.loc[i, 'carb_pred'] - table.loc[i,'carb_pred']
    
export.attrs["units"] = "tons-per-ha"
export = export.rename('carbon_change')
#export.to_dataset(name='carbon_change').to_netcdf(root + 'model_output/1_RF_regression/{}_{}.nc4'.format(SCENARIO,MODEL))


#carbon change plot (Fig 2 and S5) -------------------------------------------
change = ((table_future['carb_pred']*table.valid).sum() - (table['carb_pred']*table.valid).sum())/ (table['carb_pred']*table.valid).sum()*100

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,246,33,45], crs=ccrs.Miller())
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
plot = ax.scatter(table.longitude, table.latitude, c=table_future['carb_pred']-table['carb_pred'], s=20, vmin=-75, vmax=75, marker='s', cmap='PRGn', transform=ccrs.PlateCarree()) #change scale as needed
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.65, pad=0.05, extend='both') #.65 and .05 for horizontal; 0.8 and .01 for vert 
cbar.ax.tick_params(labelsize=14)
cbar.set_label('ton C/ha', size=16)
ax.text(0.51,0.82,'{:.1f}%'.format(change), fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.51,0.7,'total AGL\ncarbon change', fontsize=16, transform=ax.transAxes)
ax.text(-124.2,33.5,letter,fontsize=18, fontweight='bold')
ax.set_title(title + ' Change', fontsize=18)

ax.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax.set_xticklabels([-124,-122,-120,-118,-116,''])
ax.set_yticklabels([32,34,36,38,40,42])
ax.tick_params(top=True, right=True)

plt.savefig(root + 'figures/figS5f_85wet.eps')
#plt.savefig(root + 'figures/fig2c_85change.eps')

total_area = 5600*2258 #hectares per pixel times number of pixels


'''
#additional plots: change in carbon as a function of T&P, elevation (Fig 6) -----------------------
change = table_future.carb_pred - table.carb_pred
t = table[['t_winter','t_spring','t_summer','t_fall']].mean(axis=1)
p = table[['p_winter','p_spring','p_summer','p_fall']].mean(axis=1)*365.25

fig = plt.figure(figsize=(7,10))
ax1 = fig.add_subplot(211)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cbaxes = inset_axes(ax1, width="20%", height="75%", loc='center right') 
[s.set_visible(False) for s in cbaxes.spines.values()]
[t.set_visible(False) for t in cbaxes.get_xticklines()]
[t.set_visible(False) for t in cbaxes.get_yticklines()]
cbaxes.axes.get_xaxis().set_visible(False)
cbaxes.axes.get_yaxis().set_visible(False)

cbar = ax1.scatter(t, p, c=change, s=10, vmin=-50, vmax=50, cmap='PRGn', edgecolors='gray', linewidths=0.5)
cbar = plt.colorbar(cbar, ax=cbaxes, extend='both', pad=0.05, shrink=0.8, anchor=(-0.2,0.7))
cbar.ax.tick_params(labelsize=13)
cbar.set_label('AGL carbon change (ton C/ha)', size=13)

ax1.set_xlim((-2,28))
ax1.set_xlabel('Mean annual temperature (C)', fontsize=15)
ax1.set_ylabel('Mean annual precipitation (mm/y)', fontsize=15)
ax1.tick_params(labelsize=13)
ax1.text(-1.1,2850,'(a)',fontsize=15, fontweight='bold')


elev = xr.open_rasterio(root + 'topo/elev_eighth.tif')[0,:,:]
x = table.longitude.to_xarray() - 360
y = table.latitude.to_xarray()
table['elev'] = elev.sel(x=x, y=y, method='nearest').data

ax2 = fig.add_subplot(212)
ax2.scatter(table[table.carb > 15].elev, change[table.carb > 15], s=2)

ax2.set_xlabel('Elevation (m)', fontsize=15)
ax2.set_ylabel('AGL carbon change (ton C/ha)', fontsize=15)
ax2.tick_params(labelsize=13)
ax2.text(-20,48,'(b)',fontsize=15, fontweight='bold')

#plt.savefig(root + 'figures/fig6_tpelev.eps')
'''