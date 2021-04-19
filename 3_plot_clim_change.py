#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Generate maps of mean annual T & P change for RCP4.5 & RCP8.5 (Fig 1)

Inputs: climate_present and climate_future nc files generated from script 1

Outputs: Fig 1 maps (6) of mean RCP4.5 & RCP8.5 climate change
    
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

root = '/Users/scoffiel/california/'

present_climate45 = xr.open_dataset(root + 'bcsd/rcp45/climate_present.nc4').tas 
future_climate45 = xr.open_dataset(root + 'bcsd/rcp45/climate_future.nc4').tas

present_climate85 = xr.open_dataset(root + 'bcsd/rcp85/climate_present.nc4').tas 
future_climate85 = xr.open_dataset(root + 'bcsd/rcp85/climate_future.nc4').tas


present_t = present_climate85.sel(variables=['t_fall','t_spring','t_summer','t_winter']).mean(dim=['variables','models'])
present_p = present_climate85.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables','models'])*365.25

t_change45 = future_climate45.sel(variables=['t_fall','t_spring','t_summer','t_winter']).mean(dim=['variables','models']) - present_climate45.sel(variables=['t_fall','t_spring','t_summer','t_winter']).mean(dim=['variables','models'])
p_change45 = (future_climate45.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables','models']) - present_climate45.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables','models'])) * 365.25

t_change85 = future_climate85.sel(variables=['t_fall','t_spring','t_summer','t_winter']).mean(dim=['variables','models']) - present_climate85.sel(variables=['t_fall','t_spring','t_summer','t_winter']).mean(dim=['variables','models'])
p_change85 = (future_climate85.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables','models']) - present_climate85.sel(variables=['p_fall','p_spring','p_summer','p_winter']).mean(dim=['variables','models'])) * 365.25


fig, axs = plt.subplots(2,3, gridspec_kw={'width_ratios':[6,5,6]}, figsize=(17,10), subplot_kw={'projection': ccrs.PlateCarree()})
(ax1,ax2,ax3),(ax4,ax5,ax6) =axs

#1 - present temp
ax1.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax1.pcolor(present_t.longitude, present_t.latitude, present_t, transform=ccrs.PlateCarree(), cmap='plasma')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax1.add_feature(states, edgecolor='0.2')
ax1.set_title('Present-day temperature', fontsize=18)
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8, pad=0.01, ax=ax1)
cbar.set_label('\u00B0C', size=18)
cbar.ax.tick_params(labelsize=15)
ax1.text(-124.2,32.5,'(a)',fontsize=18, fontweight='bold')
ax1.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax1.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax1.set_xticklabels([-124,-122,-120,-118,-116,''])
ax1.set_yticklabels([32,34,36,38,40,42])
ax1.tick_params(top=True, right=True)

#2 - 4.5 T change
ax2.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax2.pcolor(t_change45.longitude, t_change45.latitude, t_change45, transform=ccrs.PlateCarree(), vmin=1, vmax=5, cmap='Reds')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax2.add_feature(states, edgecolor='0.2')
ax2.set_title('RCP4.5 temperature change', fontsize=18)
ax2.text(-124.2,32.5,'(b)',fontsize=18, fontweight='bold')
ax2.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax2.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax2.set_xticklabels([-124,-122,-120,-118,-116,''])
ax2.set_yticklabels([32,34,36,38,40,42])
ax2.tick_params(top=True, right=True)

#3 - 8.5 T change
ax3.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax3.pcolor(t_change85.longitude, t_change85.latitude, t_change85, transform=ccrs.PlateCarree(), vmin=1, vmax=5, cmap='Reds')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax3.add_feature(states, edgecolor='0.2')
ax3.set_title('RCP8.5 temperature change', fontsize=18)
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8, pad=0.01, ax=ax3, extend='both')
cbar.set_label('\u00B0C', size=18)
cbar.ax.tick_params(labelsize=15)
ax3.text(-124.2,32.5,'(c)',fontsize=18, fontweight='bold')
ax3.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax3.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax3.set_xticklabels([-124,-122,-120,-118,-116,''])
ax3.set_yticklabels([32,34,36,38,40,42])
ax3.tick_params(top=True, right=True)

#4 - present precip
ax4.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax4.pcolor(present_p.longitude, present_p.latitude, present_p, transform=ccrs.PlateCarree(), vmax=3000, cmap='GnBu')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax4.add_feature(states, edgecolor='0.2')
ax4.set_title('Present-day precipitation', fontsize=18)
cbar = plt.colorbar(plot, label='mm/yr', orientation='vertical', shrink=0.8, pad=0.01, ax=ax4)
cbar.set_label('mm/y', size=18)
cbar.ax.tick_params(labelsize=15)
ax4.text(-124.2,32.5,'(d)',fontsize=18, fontweight='bold')
ax4.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax4.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax4.set_xticklabels([-124,-122,-120,-118,-116,''])
ax4.set_yticklabels([32,34,36,38,40,42])
ax4.tick_params(top=True, right=True)

#5 - 4.5 P change
ax5.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax5.pcolor(p_change45.longitude, p_change45.latitude, p_change45, transform=ccrs.PlateCarree(), vmin=-180, vmax=180, cmap='RdBu')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax5.add_feature(states, edgecolor='0.2')
ax5.set_title('RCP4.5 precipitation change', fontsize=18)
ax5.text(-124.2,32.5,'(e)',fontsize=18, fontweight='bold')
ax5.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax5.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax5.set_xticklabels([-124,-122,-120,-118,-116,''])
ax5.set_yticklabels([32,34,36,38,40,42])
ax5.tick_params(top=True, right=True)

#6 - 8.5 P change 
ax6.set_extent([235.5,246,33,45], crs=ccrs.Miller())
plot = ax6.pcolor(p_change85.longitude, p_change85.latitude, p_change85, transform=ccrs.PlateCarree(), vmin=-180, vmax=180, cmap='RdBu')
states = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
ax6.add_feature(states, edgecolor='0.2')
ax6.set_title('RCP8.5 precipitation change', fontsize=18)
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8, pad=0.01, ax=ax6, extend='both')
cbar.set_label('mm/y', size=18)
cbar.ax.tick_params(labelsize=15)
ax6.text(-124.2,32.5,'(f)',fontsize=18, fontweight='bold')
ax6.set_xticks([-124, -122, -120, -118, -116, -114], crs=ccrs.PlateCarree())
ax6.set_yticks([32,34,36,38,40,42], crs=ccrs.PlateCarree())
ax6.set_xticklabels([-124,-122,-120,-118,-116,''])
ax6.set_yticklabels([32,34,36,38,40,42])
ax6.tick_params(top=True, right=True)

plt.savefig(root + 'figures/fig1_climate.png', dpi=250) 
plt.savefig(root + 'figures/fig1_climate.eps') #eps doesn't seem to turn out as well
