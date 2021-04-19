#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Generate maps of
- spread of precipitation across 32 models for RCP8.5 (FigS1)
- dry vs. wet models averages for RCP8.5 (FigS2)

Inputs:
- climate_present and climate_future nc files generated from script 1
- List "order" of indices of models from driest to wettest which comes from scrip 1

Outputs:
- Figures S1 & S2 showing spread of dry/wet models
"""

import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import regionmask

root = '/Users/scoffiel/california/'

climate_present = xr.open_dataset(root+ 'bcsd/rcp85/climate_present.nc4').tas
climate_future = xr.open_dataset(root + 'bcsd/rcp85/climate_future.nc4').tas
change = climate_future - climate_present


#Spread of precip change across all 32 models, ordered from most drying to wetting (Fig S1)

files = os.listdir(root + 'bcsd/rcp85/')
p_files = sorted([f for f in files if f[0]=='B' and f[14]=='p'])
order = [21,22,5,13,23,0,11,31,14,24,30,10,9,27,19,29,28,20,2,26,17,7,16,25,15,6,4,8,3,12,18,1] #from script 1

fig, axs = plt.subplots(4,8, gridspec_kw={'width_ratios':[1,1,1,1,1,1,1,1.3]}, figsize=(14,8), subplot_kw={'projection': ccrs.Miller()})
axs = axs.flatten()
for i in range(32):
    
    model_name = p_files[order[i]].split('_')[4]
    dp = change.sel(models=i, variables=['p_spring','p_fall','p_summer','p_winter']).mean(dim='variables') * 365.25
    axs[i].set_extent([235,246,33,45], crs=ccrs.Miller())
    contour_plot = axs.flatten()[i].contourf(dp.longitude, dp.latitude, dp, np.arange(-550,600,50), cmap='seismic_r', extend='both', transform=ccrs.PlateCarree())
    states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
    axs[i].add_feature(states, edgecolor='0.2')
    axs[i].set_title(model_name, fontsize=10)

plt.subplots_adjust(wspace=0, hspace=0.2)
cbar = plt.colorbar(contour_plot, orientation='vertical', ax=[axs[7],axs[15],axs[23],axs[31]], pad=0.1, shrink=0.6)
cbar.set_label(r'$\Delta$ mm/y', size=12)
cbar.ax.tick_params(labelsize=12) 

plt.savefig(root + 'figures/figS1_precip.eps')


#Dry models mean change and wet models mean change (Fig S2) -------------------
dries = change.isel(models=slice(0,8), variables=slice(0,4)).mean(dim=('models','variables'))*365.25
wets = change.isel(models=slice(24,32), variables=slice(0,4)).mean(dim=('models','variables'))*365.25
mean = change.isel(variables=slice(0,4)).mean(dim=('models','variables'))*365.25

fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(121, projection=ccrs.Miller())
ax1.set_extent([235,246,33,45], crs=ccrs.Miller())
plot = ax1.pcolor(dries.longitude, dries.latitude, dries, vmin=-800, vmax=800, cmap='seismic_r', transform=ccrs.PlateCarree())
ax1.add_feature(states, edgecolor='0.2')
ax1.set_title('Dry models', fontsize=15)
ax1.text(-124.5,33.5,'(a)',fontsize=15, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.05, extend='both')
cbar.set_label('Precipitation change (mm/y)', size=13)
cbar.ax.tick_params(labelsize=11) 
ax1.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax1.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax1.set_yticklabels([32,34,36,38,40,42])
ax1.tick_params(top=True, right=True, labelsize=8)     

ax2 = fig.add_subplot(122, projection=ccrs.Miller())
ax2.set_extent([235,246,33,45], crs=ccrs.Miller())
plot = ax2.pcolor(wets.longitude, wets.latitude, wets, vmin=-800, vmax=800, cmap='seismic_r', transform=ccrs.PlateCarree())
ax2.add_feature(states, edgecolor='0.2')
ax2.set_title('Wet models', fontsize=15)
ax2.text(-124.5,33.5,'(b)',fontsize=15, fontweight='bold')
cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.7, pad=0.05, extend='both')
cbar.set_label('Precipitation change (mm/y)', size=13)
cbar.ax.tick_params(labelsize=11) 
ax2.set_xticks([236,238,240,242,244,246], crs=ccrs.PlateCarree())
ax2.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax2.set_yticklabels([32,34,36,38,40,42])
ax2.tick_params(top=True, right=True, labelsize=8)

plt.savefig(root + 'figures/figS2_wetdry.eps')


#print out the dry, wet, and mean differences ---------------------------------
mask = regionmask.defined_regions.natural_earth.us_states_50.mask(change.longitude, change.latitude, wrap_lon=True)
cali = mask==4
cali = cali.rename({'lon':'longitude', 'lat':'latitude'})

print('Average change for dry models (mm/y)', dries.where(cali).mean())
print('Average change for wet models (mm/y)', wets.where(cali).mean())
print('Average change for all models (mm/y)', mean.where(cali).mean())

