#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Approach #3 supplementary figure
- Whittaker scatter plots of mean annual P vs. T, showing how CA's gridcells shift

Inputs:
- climate_present and climate_future nc files (generated from script 1)
- model 3 output CSV table (generated from script 9) for RCP8.5 mean scenario

Outputs:
- Fig S7 whittaker plots with arrows

"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

root = '/Users/scoffiel/california/'
SCENARIO = 'rcp85'

table = pd.read_csv(root + 'model_output/3_climate_analogues/table_rcp85_mean.csv')

present_climate = xr.open_dataset(root + 'bcsd/{}/climate_present.nc4'.format(SCENARIO)).tas
future_climate = xr.open_dataset(root + 'bcsd/{}/climate_future.nc4'.format(SCENARIO)).tas

present_t = present_climate[:,4:,:,:].mean(dim=('models','variables'))
present_p = present_climate[:,:4,:,:].mean(dim=('models','variables'))*365.25

future_t = future_climate[:,4:,:,:].mean(dim=('models','variables'))
future_p = future_climate[:,:4,:,:].mean(dim=('models','variables'))*365.25

x = table.longitude.to_xarray()
y = table.latitude.to_xarray()
table['present_t'] = present_t.sel(longitude=x, latitude=y).data
table['present_p'] = present_p.sel(longitude=x, latitude=y).data
table['future_t'] = future_t.sel(longitude=x, latitude=y).data
table['future_p'] = future_p.sel(longitude=x, latitude=y).data


#biomes boundaries. this has been digitally traced out of Chapin textbook
l1 = np.array([[30.692, 2.260],
[30.692, 11.299],
[30.173, 29.379],
[29.654, 49.718],
[29.135, 67.797],
[28.962, 85.876],
[28.788, 110.734],
[28.442, 140.113],
[28.096, 216.949],
[27.750, 314.124],
[27.750, 381.921],
[27.404, 463.277],
[27.231, 506.215],
[26.885, 560.452],
[26.712, 607.910],
[26.538, 644.068],
[26.365, 675.706],
[26.192, 714.124],
[26.019, 714.124],
[25.327, 666.667],
[24.462, 607.910],
[23.769, 560.452],
[22.904, 494.915],
[22.558, 456.497],
[21.692, 424.859],
[20.654, 388.701],
[20.481, 372.881],
[18.923, 354.802],
[17.538, 336.723],
[15.981, 323.164],
[14.596, 309.605],
[13.731, 309.605],
[12.519, 316.384],
[11.308, 332.203],
[10.442, 352.542],
[9.404, 372.881],
[8.365, 393.220],
[7.500, 406.780],
[6.462, 418.079],
[6.115, 418.079],
[5.769, 411.299],
[5.077, 395.480],
[4.558, 368.362],
[3.692, 338.983],
[3.000, 296.045],
[2.264, 261.017],
[2.048, 248.588],
[1.745, 239.548],
[1.183, 231.073],
[0.404, 214.124],
[-0.635, 194.915],
[-1.370, 183.051],
[-2.279, 172.316],
[-3.707, 160.452],
[-5.351, 148.023],
[-6.995, 149.153],
[-10.111, 150.282],
[-12.014, 150.282],
[-13.053, 146.328],
[-14.394, 141.243],
[-15.606, 136.158],
[-16.731, 131.073],
[-17.553, 127.684],
[-18.721, 116.949],
[-20.322, 101.130],
[-20.971, 95.480],
[-22.745, 77.966],
[-24.562, 64.972],
[-26.293, 52.542],
[-27.678, 42.373],
[-28.284, 37.288],
[-29.841, 29.944],
[-30.577, 24.294],
[-31.183, 17.514],
[-31.183, 9.605],
[-30.707, 3.390],
[-29.841, 1.130],
[-29.062, 0.000],
[-20.019, 0.000],
[-19.370, 6.780],
[-17.510, 22.599],
[-15.822, 35.028],
[-15.087, 39.548],
[-13.875, 48.588],
[-12.274, 59.887],
[-10.976, 70.056],
[-9.851, 79.096],
[-9.159, 93.785],
[-8.769, 99.435],
[-8.466, 106.780],
[-7.471, 122.034],
[-6.736, 132.768],
[-5.957, 141.808],
[-5.351, 148.023],
])

l2 = np.array([[2.048, 242.938],
[1.918, 221.469],
[1.875, 206.215],
[1.832, 193.220],
[1.702, 184.181],
[1.572, 168.362],
[1.442, 154.802],
[1.312, 136.158],
[1.139, 113.559],
[1.010, 100.000],
[0.837, 92.090],
[0.620, 79.661],
[0.404, 67.232],
[0.274, 59.887],
[-0.462, 55.367],
[-1.111, 51.412],
[-1.803, 47.458],
[-2.495, 43.503],
[-3.317, 40.113],
[-4.139, 37.853],
[-5.221, 33.898],
[-6.303, 29.944],
[-7.125, 25.989],
[-7.471, 24.294],
[-7.861, 33.898],
[-8.293, 46.328],
[-8.510, 52.542],
[-8.683, 75.141],
[-8.683, 89.831],
[-8.726, 101.130]])

l3 = np.array([[-14.611, 0.000],
[-11.495, 5.085],
[-10.024, 8.475],
[-8.337, 11.299],
[-7.644, 12.429],
[-6.779, 14.124],
[-4.788, 16.384],
[-3.663, 18.079],
[-2.019, 20.904],
[-0.332, 23.164],
[1.745, 26.554],
[3.692, 29.944],
[6.159, 33.333],
[8.798, 37.288],
[11.005, 40.678],
[13.558, 45.198],
[16.587, 49.153],
[20.178, 54.802],
[23.163, 59.322],
[24.764, 62.712],
[26.668, 64.972],
[29.135, 68.362]])

l4 = np.array([[28.442, 141.243],
[26.106, 141.243],
[23.639, 140.678],
[21.043, 139.548],
[18.880, 137.853],
[17.106, 134.463],
[15.505, 131.638],
[13.212, 125.424],
[11.784, 120.339],
[10.356, 113.559],
[8.755, 106.215],
[7.197, 98.305],
[5.769, 90.960],
[4.558, 84.181],
[2.957, 76.271],
[1.269, 66.667],
[0.274, 60.452]])

l5 = np.array([[27.923, 267.797],
[25.457, 263.277],
[22.947, 258.192],
[21.173, 254.802],
[19.572, 252.542],
[17.755, 249.153],
[15.332, 242.373],
[13.904, 238.418],
[12.433, 235.028],
[10.832, 231.638],
[8.625, 224.294],
[6.548, 217.514],
[4.990, 210.734],
[4.168, 205.650],
[2.870, 198.870],
[1.832, 193.785]])

l6 = np.array([[20.481, 372.881],
[20.394, 350.847],
[20.178, 324.294],
[19.875, 287.006],
[19.615, 252.542],
[19.399, 226.554],
[19.183, 196.045],
[18.880, 157.627],
[18.837, 137.853]])

l7 = np.array([[-7.428, 24.294],
[-7.125, 13.559]])

temps = [l1[:,0], l2[:,0], l3[:,0], l4[:,0], l5[:,0], l6[:,0], l7[:,0]]
precips = [l1[:,1], l2[:,1], l3[:,1], l4[:,1], l5[:,1], l6[:,1], l7[:,1]]

nrecs = 100
nlines = len(temps)
temps_array = np.ma.masked_all([nlines, nrecs])
precips_array = np.ma.masked_all([nlines, nrecs])
for i in range(nlines):
    temps_array[i,0:len(temps[i])] = temps[i]
    precips_array[i,0:len(temps[i])] = precips[i]


#bivariate whittaker scatterplots (Fig S6) -----------------------------------
fig, axs = plt.subplots(ncols=2, nrows=2, gridspec_kw={'height_ratios':[2,3]}, figsize=(10,12))
gs = axs[1, 0].get_gridspec() #locaitn of new big axis
# remove the underlying axes
for ax in axs[1, :]:
    ax.remove()
ax3 = fig.add_subplot(gs[1,:])
ax = axs[0,0]
ax2 = axs[0,1]

#present
for i in range(len(temps)):
    ax.plot(temps[i], precips[i]*10, color='k')
    
ax.set_xlabel('Mean annual temperature (\u00B0C)', fontsize=15)
ax.set_ylabel('Mean annual precipitation (mm/y)', fontsize=15)
ax.set_xlim((-15,30))
ax.set_ylim((0,6000))
ax.text(21,2800,'Tropical\nwet\nforest')
ax.text(20,1900, 'Tropical\ndry forest')
ax.text(20,800, 'Grassland/\nsavanna')
ax.text(23, 300, 'Desert')
ax.text(-14, 150, 'Tundra')
ax.text(-6,800, 'Boreal\nforest')
ax.text(3, 1500, 'Temperate\nforest')
ax.text(4,2500, 'Temperate wet\nforest')
    
ax.scatter(table.present_t, table.present_p, s=1, c='gray')
ax.set_title('Present-day climate', fontsize=15)
ax.text(-14,5600,'(a)',fontsize=15, fontweight='bold')
ax.tick_params(labelsize=12)


#future
ax2.scatter(table.future_t, table.future_p, s=1, c='gray')

for i in range(len(temps)):
    ax2.plot(temps[i], precips[i]*10, color='k')

ax2.set_xlabel('Mean annual temperature (\u00B0C)', fontsize=15)    
ax2.set_xlim((-15,30))
ax2.set_ylim((0,6000))
ax2.text(21,2800,'Tropical\nwet\nforest')
ax2.text(20,1900, 'Tropical\ndry forest')
ax2.text(20,800, 'Grassland/\nsavanna')
ax2.text(23, 300, 'Desert')
ax2.text(-14, 150, 'Tundra')
ax2.text(-6,800, 'Boreal\nforest')
ax2.text(3, 1500, 'Temperate\nforest')
ax2.text(4,2500, 'Temperate wet\nforest')
ax2.set_title('Future climate', fontsize=15)
ax2.text(-14, 5600,'(b)',fontsize=15, fontweight='bold')
ax2.tick_params(labelsize=12)


#change
for i in table.index:
    ax3.annotate("",
            xy=(table.loc[i,'future_t'], table.loc[i,'future_p']), xycoords='data',
            xytext=(table.loc[i,'present_t'], table.loc[i,'present_p']), textcoords='data',
            arrowprops=dict(arrowstyle="-|>",
                            connectionstyle="arc3", facecolor='gray', ec='gray'),
            )

for i in range(len(temps)):
    ax3.plot(temps[i], precips[i]*10, color='k', zorder=10000)
    
ax3.text(22,2900,'Tropical\nwet\nforest', fontsize=12)
ax3.text(23,2100, 'Tropical\ndry forest', fontsize=12)
ax3.text(23,1000, 'Grassland/\nsavanna', fontsize=12)
ax3.text(25, 300, 'Desert', fontsize=12)
ax3.text(-3,900, 'Boreal\nforest', fontsize=12)
ax3.text(3, 1700, 'Temperate\nforest', fontsize=12)
ax3.text(4,3000, 'Temperate wet\nforest', fontsize=12)

ax3.set_xlabel('Mean annual temperature (\u00B0C)', fontsize=15)
ax3.set_ylabel('Mean annual precipitation (mm/y)', fontsize=15)
ax3.text(-4.3, 3300,'(c)',fontsize=15, fontweight='bold')
ax3.set_xlim((-5,30))
ax3.set_ylim((0,3500))
ax3.tick_params(labelsize=12)

plt.savefig(root + 'figures/figS7_whittaker.eps')
