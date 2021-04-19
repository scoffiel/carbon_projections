#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose:
- Compare RF regression model results from script 5 for all forests vs. carbon offset projects

Inputs:
- model output for RCP8.5 mean (generated from script 5)
- carbon_eighth.tif
- shapefile of offset projects in California (collected from https://webmaps.arb.ca.gov/ARBOCIssuanceMap/)
- model output for RCP8.5 mean clipped to offset projects (done in QGIS)

Outputs:
- Figure 7 map + histogram zooming in on forest carbon offset projects

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

root = '/Users/scoffiel/california/'
SCENARIO = 'rcp85'

results = xr.open_dataset(root + 'model_output/1_RF_regression/{}_mean.nc4'.format(SCENARIO)).carbon_change
table = results.to_dataframe('change').dropna().reset_index()

#first panel: map of change plot with offsets overlaid
fig = plt.figure(figsize=(7,9))
ax = fig.add_subplot(211, projection=ccrs.Miller())
ax.set_extent([235,245,39,45], crs=ccrs.Miller())
ecoregions = ShapelyFeature(Reader(root + "epa_ecoregions3/level3_cali.shp").geometries(), ccrs.PlateCarree())
projects = ShapelyFeature(Reader(root + "carb_shapefiles/offsets/offsets_cali.shp").geometries(), ccrs.PlateCarree())
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax.add_feature(ecoregions, edgecolor='0.3', facecolor='none', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2')
ax.add_feature(projects, edgecolor='0', facecolor='0')
plot = ax.scatter(table.longitude, table.latitude, c=table.change, s=24, vmin=-100, vmax=100, marker='s', cmap='PRGn', transform=ccrs.PlateCarree()) #change scale as needed

ax.text(-124.9,44.6,'(a)',fontsize=12, fontweight='bold')
ax.set_title('Carbon change in forest offset projects', fontsize=12)
ax.set_xticks([236,238,240,242,244], crs=ccrs.PlateCarree())
ax.set_yticks([38,40,42], crs=ccrs.PlateCarree())
ax.set_yticklabels([38,40,42], fontsize=8)
ax.tick_params(top=True, right=True, labelsize=8)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cbaxes = inset_axes(ax, width="20%", height="75%", loc='center right') 
[s.set_visible(False) for s in cbaxes.spines.values()]
[t.set_visible(False) for t in cbaxes.get_xticklines()]
[t.set_visible(False) for t in cbaxes.get_yticklines()]
cbaxes.axes.get_xaxis().set_visible(False)
cbaxes.axes.get_yaxis().set_visible(False)
cbar = plt.colorbar(plot, ax=cbaxes, orientation='vertical', shrink=0.8, pad=0.01, extend='both') #changed to both and 75 from 100 for FigS3 
cbar.ax.tick_params(labelsize=10)
cbar.set_label('ton C/ha', size=10)


#second panel: histogram of all forests' change vs. offsets change
table.loc[table.change==0, 'change'] = 0.001

carb = xr.open_rasterio(root + 'carb_carbon/carbon_eighth.tif')[0,:,:]
carb = carb.where(carb > -9.999)
x = table.longitude.to_xarray() - 360
y = table.latitude.to_xarray()
table['carb'] = carb.sel(x=x, y=y, method='nearest').data
table = table[table.carb > 15] #forested pixels only

state = table.change
counts, bins = np.histogram(state, bins=range(-80,80,10))
counts = counts/counts.sum()

ax2 = fig.add_subplot(212)
ax2.bar(bins[:-1], counts, width=10, alpha=0.5, label='All California forests')
print('mean for California', np.mean(state))

offsets = xr.open_rasterio(root + 'model_output/1_RF_regression/{}_mean_offsets.tif'.format(SCENARIO))[0,:,:]

offsets = np.array(offsets).flatten()
offsets = offsets[offsets>-9999]
counts, bins = np.histogram(offsets, bins=range(-80,80,10))
counts = counts/counts.sum()
ax2.bar(bins[:-1], counts, width=10, alpha=0.5, label='Carbon offset locations')
print('mean for offset projects', np.mean(offsets))
ax2.text(-91,0.345,'(b)',fontsize=12, fontweight='bold')
ax2.set_xlabel('AGL carbon change (ton C/ha)')
ax2.set_ylabel('Density')
ax2.legend()
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.06)

plt.savefig(root + 'figures/fig7_offsets.png', dpi=250)
