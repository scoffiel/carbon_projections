#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Approach #3
- Match future climate pixels with their present analog using Mahalanobis distance
- Project future carbon density by assigning that of the present analog
- 3 runs: full domain (25-49 lat and -125 - -100 lon), 500 km radius, 100 km radius

Inputs:
- climate_present and climate_future nc files (generated from script 1)
- present_10yrs climate (generated from script 2)
- landcover_mask_eighth (generated from GEE script 5)
- valid_fraction (generated from GEE script 1)
- cci_eighth.tif (generated from GEE script 6)

Outputs:
- netcdf raster layer of projected carbon change (one for each RCP+moisture+restriction scenario)
- Figures: maps of present-day carbon density, change, analog arrows, novelty of future climate
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import regionmask
from scipy.spatial.distance import cdist
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from sklearn.metrics.pairwise import haversine_distances
from scipy import stats
import matplotlib

#global variables to set before running
SCENARIO = 'rcp85'
MODEL = 'mean' #wet, dry, mean 
RESTRICTION = 1e99 #100 or 500 km search radius restriction. Default 1e99

if MODEL == 'mean': N_MODELS = 32
else: N_MODELS = 8

root = '/Users/scoffiel/california/'

#read in data
present_climate = xr.open_dataset(root + 'bcsd/{}/climate_present.nc4'.format(SCENARIO)).tas
future_climate = xr.open_dataset(root + 'bcsd/{}/climate_future.nc4'.format(SCENARIO)).tas
present_10yrs = xr.open_dataset(root + 'bcsd/{}/climate_present_10yrs.nc4'.format(SCENARIO)).tas #needed for ICV and correlation

if MODEL=='dry': #get first 8 models only
    present_climate = present_climate.sel(models=slice(0,7))
    future_climate = future_climate.sel(models=slice(0,7))
    present_10yrs = present_10yrs.sel(models=slice(0,7))

if MODEL=='wet': #get last 8 models only
    present_climate = present_climate.sel(models=slice(24,31))
    future_climate = future_climate.sel(models=slice(24,31))
    present_10yrs = present_10yrs.sel(models=slice(24,31))

#average across the different models
present_climate = present_climate.mean(dim='models')
future_climate = future_climate.mean(dim='models')

#calculate ICV and correlation using the full 10 yrs of data from the present in California ----------------------------------------
present_10yrs = present_10yrs.sel(latitude=slice(32.5, 42.3), longitude=slice(235.1,246.3))
mask = regionmask.defined_regions.natural_earth.us_states_50.mask(present_climate.longitude, present_climate.latitude, wrap_lon=True)
cali = mask==4
cali = cali.rename({'lon':'longitude', 'lat':'latitude'})  #is this necessary? yes
present_10yrs_cali = present_10yrs.where(cali)


#Build a table of coordinates from climate data, with columns indicating California and Land Cover masks -------------
table = present_climate[0,:,:].to_dataframe('foo').dropna().reset_index()
del table['foo'], table['variables']

x = table.longitude.to_xarray()
y = table.latitude.to_xarray()

table['cali'] = cali.sel(longitude=x, latitude=y).data

lclu = xr.open_rasterio(root + 'land_cover/landcover_mask_eighth.tif')[0,:,:]
table['lclu'] = lclu.sel(x=x-360, y=y, method='nearest').data #exact land cover code
table['land'] = ( (table.lclu==-9999) | (table.lclu==1) ) #True/False valid or invalid
table = table[table.land].reset_index(drop=True) #drop rows of urban/ag/water

#plt.scatter(table.longitude, table.latitude, s=0.01) #quick map to verify mask

#Build matrices of the present and future climate, now with invalid land excluded (n x 8)
x = table.longitude.to_xarray()
y = table.latitude.to_xarray()

future_array = future_climate.sel(longitude=x, latitude=y).T.values
present_array = present_climate.sel(longitude=x, latitude=y).T.values

#variable order is p fall, spring, summer, winter, t fall spring summer winter


#Calculate the Mahalanobis distance ------------------------------------------
#For each point in California, calculate dist to all other points and find minimum

table['latitude_rad'] = np.radians(table.latitude)
table['longitude_rad'] = np.radians(table.longitude)

table_cali = table[table.cali]

for i in table_cali.index:
    b = future_array[i,:] #(1 x 8)
    a = present_array     #(n x 8)
       
    icv = present_10yrs.sel(longitude=table.loc[i,'longitude'], latitude=table.loc[i,'latitude'])
    icv = np.concatenate((icv.sel(var='p').values.reshape(N_MODELS,10,4), icv.sel(var='t').values.reshape(N_MODELS,10,4)), axis=2) #(32 x 10 x 8)
        
    mean = np.mean(icv, axis=(0,1))    
    sigma = np.std(icv, axis=1).mean(axis=0)
    
    #standardize 
    bprime = (b - mean)/sigma
    aprime = (a - mean)/sigma
     
    corr = np.ones((N_MODELS,8,8))
    for m in range(N_MODELS):
        corr[m,:,:] = np.corrcoef(icv[m,:,:].T)
    corr = corr.mean(axis=0)
    corrinv = np.linalg.inv(corr)
    
    d = cdist(aprime, bprime.reshape(1,8), 'mahalanobis', VI=corrinv) #mahalanobis distance of climate
    
    #calculate geographical (haversine) distances in case of restricting search radius
    x1y1 = np.array([table.loc[i, 'latitude_rad'], table.loc[i, 'longitude_rad']])
    d_h = haversine_distances(table[['latitude_rad','longitude_rad']].values, x1y1.reshape((1,2)) ) * 6371
    d = np.where(d_h < RESTRICTION, d, np.nan)
    
    imin = np.nanargmin(d) #locations of minimum values
    table_cali.loc[i, 'analog_dist'] = float(d[imin])
    table_cali.loc[i, 'analog_x'] = table.loc[imin, 'longitude']
    table_cali.loc[i, 'analog_y'] = table.loc[imin, 'latitude']

    
#calculate present and future biomass for California and change -------------------------- 

x = table_cali.longitude.to_xarray()
y = table_cali.latitude.to_xarray()
analx = table_cali.analog_x.to_xarray()
analy = table_cali.analog_y.to_xarray()

bio = xr.open_rasterio(root + 'cci_biomass/cci_eighth.tif')[0,:,:]
bio = bio.where(bio>0)

table_cali['present_bio'] = bio.sel(x=x-360, y=y, method='nearest')
table_cali['future_bio'] = bio.sel(x=analx-360, y=analy, method='nearest')
table_cali['change'] = table_cali.future_bio-table_cali.present_bio #not yet weighted by bare

#calculate total percent change of biomass for California, accounting for bare areas
valid = xr.open_rasterio(root + 'land_cover/valid_fraction.tif')[0,:,:]
table_cali['valid'] = valid.sel(x=x-360, y=y, method='nearest').data
percent_change = ((table_cali.future_bio*table_cali.valid).sum() - (table_cali.present_bio*table_cali.valid).sum() )/ (table_cali.present_bio*table_cali.valid).sum() *100


#save change as netcdf  -------------------------------------------------------
#use climate dataset as a template
export = present_climate.sel(latitude=slice(32.5, 42.3), longitude=slice(235.1,246.3), variables='p_fall')
export_array = np.zeros(export.shape) - np.nan
export = xr.DataArray(export_array, coords=[export.latitude, export.longitude], dims=["latitude", "longitude"])

for i in table_cali.index:
    export.loc[{'latitude':table_cali.loc[i,'latitude'], 'longitude':table_cali.loc[i,'longitude']}] = table_cali.loc[i,'change']*0.47
    
export.attrs["units"] = "tonC-per-ha"
export = export.rename('carbon_change')
export.to_dataset(name='carbon_change').to_netcdf(root + 'model_output/3_climate_analogs/{}_{}.nc4'.format(SCENARIO,MODEL))

#save table as csv (at least for RCP85 mean scenario) for script 10 plots
#table_cali.to_csv(root + 'model_output/3_climate_analogs/table_rcp85_mean.csv')


#plotting ----------------------------------------------------------------------------------
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')

#change plot
fig = plt.figure(figsize=(7,7))
ax2 = fig.add_subplot(111, projection=ccrs.Miller())
ax2.set_extent([235.5,246,33,45], crs=ccrs.Miller())
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
ax2.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax2.add_feature(states, edgecolor='0.2')
plot = ax2.scatter(table_cali.longitude-360, table_cali.latitude, c=table_cali.change*0.47, s=10, vmin=-75, vmax=75, marker='s', cmap='PRGn', transform=ccrs.PlateCarree()) #change scale as needed
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.6, pad=0.05, extend='both')
cbar.ax.tick_params(labelsize=13)
cbar.set_label('ton C/ha', size=13)
ax2.text(0.55,0.8,'{:.1f}%'.format(percent_change), fontsize=15, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.55,0.7,'total AGL\ncarbon change', fontsize=13, transform=ax2.transAxes)
ax2.set_title('RCP4.5 Mean Change', fontsize=15)
ax2.text(-124.2,33.5,'(b)',fontsize=18, fontweight='bold')
ax2.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax2.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax2.set_xticklabels([-124,-122,-120,-118,-116,''])
ax2.set_yticklabels([32,34,36,38,40,42])
ax2.tick_params(top=True, right=True)

#plt.savefig(root + 'figures/figS6b_45change.eps')


#Analog arrows and novelty maps (Fig S8) 
fig, axs = plt.subplots(2,2, subplot_kw=dict(projection=ccrs.Miller()), figsize=(14,14))
(ax1,ax2,ax3,ax4) = axs.flatten()

#Analog arrows
ax1.set_extent([235.5,246,33,45], crs=ccrs.Miller())
for i in table_cali.index[::2]:
    x2, y2, x1, y1 = table_cali.loc[i, ['longitude','latitude','analog_x','analog_y']]
    x2 = x2-360
    x1 = x1-360
    ax1.arrow(x2-(x2-x1)/15, y2-(y2-y1)/15, (x2-x1)/15, (y2-y1)/15,color='k', head_width=0.07, width=0, transform=ccrs.PlateCarree())
ax1.add_feature(states, edgecolor='0.2')
ax1.set_title('Climate Analog Pairs', fontsize=15)
ax1.text(-124.2,33.5,'(a)',fontsize=18, fontweight='bold')
ax1.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax1.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax1.set_yticklabels([32,34,36,38,40,42])
ax1.tick_params(top=True, right=True)

#Novelty in Mahalanobis distance units
ax3.set_extent([235.5,246,33,45], crs=ccrs.Miller())
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
ax3.add_feature(ecoregions, edgecolor='0', facecolor='none', linewidth=0.2)
ax3.add_feature(states, edgecolor='0.2')
plot = ax3.scatter(table_cali.longitude-360, table_cali.latitude, c=table_cali.analog_dist, s=14.5, vmin=0, vmax=5,marker='s', cmap='viridis', transform=ccrs.PlateCarree()) #change scale as needed
cbar = plt.colorbar(plot, shrink=0.8, orientation='vertical', pad=0.01, extend='max',ax=ax3)
cbar.ax.tick_params(labelsize=13)
cbar.set_label('Mahalanobis distance to best analog', size=13)
ax3.set_title('Novelty of future climates', fontsize=15)
ax3.text(-124.2,33.5,'(b)',fontsize=18, fontweight='bold')
ax3.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax3.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax3.set_yticklabels([32,34,36,38,40,42])
ax3.tick_params(top=True, right=True)

#Novelty interpretting distances as percentiles on chi2 distr with df = 8
table_cali['chi_dist_prob'] = stats.chi2.cdf(table_cali.analog_dist**2, 8) #percentile/probability 
table_cali['chi_dist_z'] = stats.norm.ppf(table_cali.chi_dist_prob) #z-score

ax4.set_extent([235.5,246,33,45], crs=ccrs.Miller())
ax4.add_feature(ecoregions, edgecolor='0', facecolor='none', linewidth=0.2)
ax4.add_feature(states, edgecolor='0.2')
plot = ax4.scatter(table_cali.longitude-360, table_cali.latitude, c=table_cali.chi_dist_prob*100, s=14.5, marker='s', norm=matplotlib.colors.BoundaryNorm([0,68,95,97.5,99,99.7,100], ncolors=300), cmap='viridis', transform=ccrs.PlateCarree()) #change scale as needed
cbar = plt.colorbar(plot, shrink=0.8, orientation='vertical', pad=0.01, extend='max', ax=ax4)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_yticklabels([r'$0\sigma$',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$',r'$6\sigma$'])
cbar.set_label('Standardized distance to best analog', size=13)
ax4.set_title('Novelty of future climates', fontsize=15)
ax4.text(-124.2,33.5,'(c)',fontsize=18, fontweight='bold')
ax4.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax4.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax4.set_yticklabels([32,34,36,38,40,42])
ax4.tick_params(top=True, right=True)

ax2.remove()
ax1.set_anchor('W')
ax3.set_anchor('W')
ax4.set_anchor('W')
plt.subplots_adjust(wspace=0.1, hspace=0.15)

plt.savefig(root + 'figures/figS8_arrows-novelty.eps')

'''
#present biomass plot
fig = plt.figure(figsize=(7,7))
ax2 = fig.add_subplot(111, projection=ccrs.Miller())
ax2.set_extent([235.5,246,33,45], crs=ccrs.Miller())
ax2.add_feature(states, edgecolor='0.2')
ax2.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
plot = ax2.scatter(table_cali.longitude-360, table_cali.latitude, c=table_cali.present_bio*0.47, s=10, vmin=0, vmax=140, marker='s', cmap='YlGn', transform=ccrs.PlateCarree()) #change scale as needed
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.6, pad=0.05, extend='max')
cbar.ax.tick_params(labelsize=13)
cbar.set_label('ton C/ha', size=13)
ax2.set_title('Present-day AGL carbon density', fontsize=15)
ax2.text(-124.2,33.5,'(a)',fontsize=18, fontweight='bold')
ax2.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax2.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax2.set_xticklabels([-124,-122,-120,-118,-116,''])
ax2.set_yticklabels([32,34,36,38,40,42])
ax2.tick_params(top=True, right=True)
#plt.savefig(root + 'figures/figS6a_present.eps')
'''