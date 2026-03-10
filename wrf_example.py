#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:15:23 2023

@author: carracedo
"""

import numpy as np
from netCDF4 import Dataset
np.float_ = np.float64
import xarray as xr
import glob
import sys, os
sys.path.append('/media/antia/Elements1/ROHP-PAZ/rohppaz_processing/rohppaz/lib/raytracer/lib')
import angle_functions as af
sys.path.append('/media/antia/Elements1/ROHP-PAZ/rohppaz_processing/rohppaz/lib/database')
from database_funcs import get_dphaseCal
from database_funcs import get_hflag
import scipy
# import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy
import pandas as pd
from wrf import (to_np, getvar, get_cartopy,
                  latlon_coords, pw, dbz,CoordPair, vertcross,
                  g_geoht, tk)
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime
sys.path.append('/media/antia/Elements1/ROHP-PAZ/rohppaz_processing/rohppaz/lib/occult')
import interpolation3D as int3d
from matplotlib.pyplot import cm
from sklearn.linear_model import LinearRegression
from collections import Counter
sys.path.append('/home/antia/backup/Carpetas Segons Email/Desktop/ICE/scripts/WRF')
from paper_wrf_arts_6 import watercontent_percen
from paper_wrf_arts_6 import scatter_mp

def linear_regresion(x,y):
    """ function that calculates the Linear Regression between two arrays and 
    returns the slope """
    
   
    x = np.reshape(x,(-1,1))
    model = LinearRegression().fit(x,y)
    slope = model.coef_
    
    y_new = model.predict(x)
    r2 = model.score(x,y)
    
    return y_new,slope,r2



def PW(wrfout,time_instant):
    """ function for calculating the precipitable water """
    
    ncfile = Dataset(wrfout) 
    
    total_pressure = getvar(ncfile,'PB',timeidx = time_instant) + getvar(ncfile,'P', timeidx=time_instant) #base-state pressure + perturbation pressure
    temperature = getvar(ncfile,'T',timeidx = time_instant)
    qv = getvar(ncfile,'QVAPOR',timeidx=time_instant) #water vapor mixing ratio
    gp = getvar(ncfile,'PHB',timeidx=time_instant) + getvar(ncfile,'PH',timeidx=time_instant) #base-state geopotential + perturbation geopotential
    
    PW = pw(total_pressure,temperature,qv,gp)
    
    return PW

def reflect(wrfout,time_instant):
    """ function that calculates the simulated radar reflectivity from 
    WRF """
    
    ncfile = Dataset(wrfout)
    
    total_pressure = getvar(ncfile,'PB',timeidx = time_instant) + getvar(ncfile,'P', timeidx=time_instant) #base-state pressure + perturbation pressure
    temperature = getvar(ncfile,'T',timeidx = time_instant)
    qv = getvar(ncfile,'QVAPOR',timeidx=time_instant) #water vapor mixing ratio
    qs = getvar(ncfile,'QSNOW',timeidx=time_instant)
    qr = getvar(ncfile,'QRAIN',timeidx=time_instant)
    qg = getvar(ncfile,'QGRAUP',timeidx=time_instant)
    
    Z = dbz(total_pressure, temperature, qv, qr, qs, qg)
    
    return Z

def figure_wrf(wrfout,variable,time_instant,roid,vmin,vmax):
    """ function that plots a figure of an specific variable from the wrf
    out file over the specific domain """
    srcpath = '/media/antia/Elements/data/collocations'
    figpath = '/media/antia/Elements/figures'
    
    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'ice*' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])

    domain = wrfout[-23:-20]
    mp_scheme = wrfout[-32:-29]
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                              ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    
    
    # Open the NetCDF file
    ncfile = Dataset(wrfout) 
    if variable == 'RAINT':
        var1 = getvar(ncfile,'RAINC',timeidx = time_instant)
        var2 = getvar(ncfile,'RAINNC',timeidx = time_instant)
        var = var1 + var2
        time = var1.Time.values
        units = 'mm'
        description = 'Total Surface Precipitation'
        
    elif variable == 'PW':
        var = PW(wrfout,time_instant)
        var2 = getvar(ncfile, 'QRAIN', timeidx = time_instant)
        time = var2.Time.values
        units = '$kg/m^{2}$'
        description = var.description
    elif variable == 'rh':
        var2 = getvar(ncfile,'rh')
        
        time = var2.Time.values
        units = var2.units 
        description = var2.description
        
    else:
        var = getvar(ncfile, variable, timeidx = time_instant)
        var2 = getvar(ncfile, 'QRAIN', timeidx = time_instant)
        time = var2.Time.values
        units = var.units
        description = var.description
        
    if len(var.shape) > 2:
        var = var.max('bottom_top')

    # Smooth the sea level pressure since it tends to be noisy near the
    # mountains
    # smooth_slp = smooth2d(var, 3, cenweight=4)

    # Get the latitude and longitude points
    lats, lons = latlon_coords(var2)
    raylon,raylat = ray['lon'][ray['h']<15], ray['lat'][ray['h']<15]

    # Get the cartopy mapping object
    cart_proj = get_cartopy(var2)

    # Create a figure
    fig = plt.figure(figsize=(8,8))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)
    levels = np.linspace(vmin,vmax,8)
    im = ax.contourf(lons.values, lats.values, var.values, 10,vmin=vmin,vmax=vmax,levels=levels,
             transform=crs.PlateCarree(),
             cmap=get_cmap("RdYlBu_r"))
    # Add a color bar
    cb=plt.colorbar(im,ax=ax, shrink=.98,extend='max')
    cb.set_label(description + ' ' + '(' + units + ')')
    ax.scatter(file.lon_occ,file.lat_occ,c='black',transform=crs.PlateCarree())
    
    ax.scatter(raylon,raylat,color='grey',alpha=0.05, transform=crs.PlateCarree()) 
  
    ax.add_feature(cartopy.feature.BORDERS)
    ax.coastlines('50m', linewidth=0.8)
    plt.title(time)
    # plt.xticks(np.linspace(file.lon_occ-6,file.lon_occ+6,4))
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp_scheme , exist_ok = True)
        plt.savefig(figpath + '/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp_scheme + '/' + str(variable) + '_' + domain + '_' + str(pd.Timestamp(time))[11:13] + '.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp_scheme , exist_ok = True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp_scheme + '/' + str(variable) + '_' + domain + '_' + str(pd.Timestamp(time))[11:13] + '.png')
 

    return None

def figure_vertical(wrfout,variable,time_instant,roid):
    """ function for calculating the vertical 2d image of a variable from 
    WRF simulation """

    srcpath_col = '/media/antia/Elements/data/collocations/PAZ'
    
    fname_col = glob.glob(srcpath_col + '/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file = xr.open_dataset(fname_col)

    domain = wrfout[61:64] 

    mp_scheme = wrfout[50:53]
    
    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                              ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    
    # Open the NetCDF file
    ncfile = Dataset(wrfout)
    if variable == 'reflectivity':
        var = reflect(wrfout, time_instant)
        var2 = getvar(ncfile, 'QRAIN', timeidx = time_instant)
        time = var2.Time.values
        units = 'dBZ'
        description = var.description
    
    # Set the start point and end point for the cross section
    start_point = CoordPair(lat=np.nanmin(ray['lat'][ray['h']<10]), lon=np.nanmin(ray['lon'][ray['h']<20]))
    end_point = CoordPair(lat=np.nanmax(ray['lat'][ray['h']<10]), lon=np.nanmax(ray['lon'][ray['h']<20]))
    Z = 10**(var/10.)
    z_cross = vertcross(Z, var, wrfin=ncfile, start_point=start_point,
                    end_point=end_point, latlon=True, meta=True)
    dbz_cross = 10.0 * np.log10(z_cross)
    # Make the contour plot for dbz
    levels = [5 + 5*n for n in range(9)]
    
    cart_proj = get_cartopy(var2)
    
    plt.figure(figsize=(8,4))
    
    dbz_contours = plt.contourf(to_np(dbz_cross), levels=levels,
                               cmap=get_cmap("jet"))
    cb_dbz = plt.colorbar(dbz_contours)
    cb_dbz.ax.tick_params(labelsize=13)
    cb_dbz.set_label(description + ' ' + '(' + units + ')')
    
    # coord_pairs = to_np(dbz_cross.coords["xy_loc"])
    # x_ticks = np.arange(coord_pairs.shape[0])
    # x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
    # plt.xticks(x_ticks[::20], x_labels[::20], rotation=45, fontsize=14)
    
    # vert_vals = to_np(dbz_cross.coords["vertical"])
    # v_ticks = np.arange(vert_vals.shape[0])
    # plt.yticks(v_ticks[::20],vert_vals[::20], fontsize=14)
    # plt.xlabel("Latitude, Longitude", fontsize=15)
    # plt.ylabel("Height (m)", fontsize=15)

    plt.title(time)
    plt.savefig('/media/antia/Elements/figures/WRF/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '/' + 'cross_section_Z' + '.png')
   
    
    return None

def figure_xarray(wrfout,variable,time_instant):
    """ function that creates a plot of the variable specified from the 
    wrfout file using xarray. Time instant represents the selected time
    if needed """
    
    srcpath_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations'

    fname_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/2019/2019.075/iceCol_PAZ1.2019.075.09.34.G15_2010.2640_V06.nc'
    file_col = fname_col[70:120]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    # domain = wrfout[57:60] # goddard
    domain = wrfout[62:65] # WDM6 and Morrison
    # domain = wrfout[63:66] # thompson 

    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                          ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    #open file
    ncfile = xr.open_dataset(wrfout)

    #select variable
    var = ncfile[variable]        
    
    #select time if needed
    if var.shape[0] > 1:
        var = var[time_instant]
    
    #select vertical
    if len(var.shape) > 2:
        var = var.isel(bottom_top = 1)

    
    # Get the latitude and longitude points
    lats, lons = var.XLAT.values, var.XLONG.values

    # Get the cartopy mapping object
    cart_proj = cartopy.crs.LambertConformal()

    # Create a figure
    fig = plt.figure(figsize=(12,6))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)


    # Make the contour outlines and filled contours for the smoothed sea level
    # pressure.
    plt.contourf(lons, lats, var, 10,
             transform=crs.PlateCarree(),
             cmap=get_cmap("RdYlBu_r"))
    # Add a color bar
    cb=plt.colorbar(ax=ax, shrink=.98)
    cb.set_label(var.description + ' ' + '(' + var.units + ')')
    plt.scatter(ray['lon'][ray['h']<20],ray['lat'][ray['h']<20],c=(0.75,0.75,0.75),transform = crs.PlateCarree())
    plt.scatter(file.lon_occ,file.lat_occ,c='black',transform = crs.PlateCarree())
    ax.add_feature(cartopy.feature.BORDERS)
    ax.coastlines('50m', linewidth=0.8)
    plt.title(var.XTIME.values)
    plt.savefig('/home/aliga/carracedo/Desktop/ICE/figures/WRF/Morrison/' + str(variable) + '_' + domain + '_' + str(pd.Timestamp(var.XTIME.values))[11:13] + '.png')

    return None


def mixing_ratio(wrfout,roid):
    """ function that makes a vertical profile of the mixing ratio for each 
    hydrometeor """
    srcpath_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations'
    figpath = '/media/antia/easystore/figures/WRF'
    
    fname_col = glob.glob(srcpath_col + '/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file = xr.open_dataset(fname_col)
   
    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                              ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    
    
    dphi_PAZ = get_dphaseCal(roid,db='rohppaz_db')
    
    #densities
    rho_rain = 0.997 #g/cm3
    rho_snow = 0.30 #g/cm3
    rho_graupel = 0.500 #g/cm3 0.050-0.89
    rho_ice = 0.9 #g/cm3
    rho_hail = 0.85 #g/cm3

    #axis ratio
    ar_rain = 0.9 #??? 
    ar_snow = 0.5
    ar_ice = 0.6
    ar_graup = 0.5
    ar_hail = 0.5
     
    
    C = 1.6 #(g cm⁻3)⁻2
    
    try:
        interp = np.load('/media/antia/easystore/data/interp' + '/' + 
              roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'interp_' + wrfout[54:57] + '.npy',allow_pickle=True).item()

    except:
        interp_wrf(roid,wrfout)
        interp = np.load('/media/antia/easystore/data/interp' + '/' + 
              roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'interp_' + wrfout[54:57] + '.npy',allow_pickle=True).item()

    #Calculate the Kdp
    kdp_rain = (1/2)*C*rho_rain*interp['rain']*(1-ar_rain)
    kdp_rain = np.ma.masked_invalid(kdp_rain)
    kdp_snow = (1/2)*C*rho_snow*interp['snow']*(1-ar_snow)
    kdp_snow = np.ma.masked_invalid(kdp_snow)
    kdp_graupel = (1/2)*C*rho_graupel*interp['graupel']*(1-ar_graup)
    kdp_graupel = np.ma.masked_invalid(kdp_graupel)
    kdp_ice = (1/2)*C*rho_ice*interp['ice']*(1-ar_ice)
    kdp_ice = np.ma.masked_invalid(kdp_ice)
    
    #calculation of dphi from NEXRAD
    ray['dist'][np.isnan(ray['dist'])] = 0
    #integrate the Kdp along the ray
    dphi_rain = np.trapz(kdp_rain,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    dphi_snow = np.trapz(kdp_snow,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    dphi_graupel = np.trapz(kdp_graupel,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    dphi_ice = np.trapz(kdp_ice,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    dphi_total = dphi_rain + dphi_snow + dphi_graupel + dphi_ice #total contribution
    ht = np.nanmin(ray['h'],axis=1)
    
    
    plt.figure(figsize=(8,8))
    plt.plot(dphi_rain,ht,label='rain'+ ' ' + r'$\rho$=' + str(rho_rain) + ' ' + 'ar = ' + str(ar_rain))
    plt.plot(dphi_snow,ht,label='snow'+ ' ' + r'$\rho$=' + str(rho_snow) + ' ' + 'ar = ' + str(ar_snow))
    plt.plot(dphi_graupel,ht,label='graupel'+ ' ' + r'$\rho$=' + str(rho_graupel) + ' ' + 'ar = ' + str(ar_graup))
    plt.plot(dphi_ice,ht,label='ice' + ' ' + r'$\rho$=' + str(rho_ice) + ' ' + 'ar = ' + str(ar_ice))
    plt.plot(dphi_total,ht,label = 'total contribution')
    plt.plot(dphi_PAZ[0:150],np.flip(ht)[0:150],color='black',label='PAZ')
    # plt.plot(dphi_hail,ht,label = 'hail')
    plt.legend()
    plt.ylim(0,15)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + wrfout[54:57] + '/' + '_' + wrfout[65:68] + '_dphi_wrfout.png')

    return None

def dphi_WRF(var,wrfout_tho,wrfout_mor,wrfout_god,wrfout_ws6,roid):
    """ function that plots a vertical profile of the differential 
    phase shift from WRF associated with a specific hydrometeor for
    4 mp schemes """
    srcpath_col = '/media/antia/easystore/data/collocations/PAZ'
  
    figpath = '/media/antia/easystore/figures'
    resPrf_path = '/home/dilong/carracedo/Desktop/ICE/data/PAZ/profiles'
    file_term = Dataset(resPrf_path + '/' + roid[5:13] + '/' + 
                        'resPrf_' + roid + '_2010.2640_V07.nc')

    T = file_term['profiles'].variables['temperature'][:]
    file_term.close()


    fname_col = glob.glob('/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file_col = fname_col[0][79:]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                          ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])


    #densities
    rho_rain = 0.997 #g/cm3
    rho_snow = 0.30 #g/cm3
    rho_graupel = 0.500 #g/cm3 0.050-0.89
    rho_ice = 0.9 #g/cm3
    rho_hail = 0.85 #g/cm3
  
    #axis ratio
    ar_rain = 0.9 #??? 
    ar_snow = 0.5
    ar_ice = 0.5
    ar_graup = 0.5
    ar_hail = 0.5
    
    C = 1.6 #(g cm⁻3)⁻2

    DPHI_rain = np.zeros((1,220))
    wrfout = [wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6]
    ht = np.nanmin(ray['h'],axis=1)
    
    #find height where T is T=0
    index_0 = np.nanargmin(T-273.15)
    h_0 = ht[index_0.astype(int)] 
    
    for i in range(len(wrfout)):
        try:
            interp = np.load('/home/aliga/carracedo/Desktop/ICE/data' + '/' + 'WRF' + '/' + 
                  roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'interp_' + wrfout[i][54:57] + '.npy',allow_pickle=True).item()

        except:
            interp_wrf(roid,wrfout[i])
            interp = np.load('/home/aliga/carracedo/Desktop/ICE/data' + '/' + 'WRF' + '/' + 
                  roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'interp_' + wrfout[i][54:57] + '.npy',allow_pickle=True).item()

        #Calculate the Kdp
        kdp_rain = (1/2)*C*rho_rain*interp['rain']*(1-ar_rain)
        kdp_rain = np.ma.masked_invalid(kdp_rain)
        
        kdp_snow = (1/2)*C*rho_snow*interp['snow']*(1-ar_snow)
        kdp_snow = np.ma.masked_invalid(kdp_snow)
        
        kdp_graupel = (1/2)*C*rho_graupel*interp['graupel']*(1-ar_graup)
        kdp_graupel = np.ma.masked_invalid(kdp_graupel)

        kdp_ice = (1/2)*C*rho_ice*interp['ice']*(1-ar_ice)
        kdp_ice = np.ma.masked_invalid(kdp_ice)

        #calculation of dphi from NEXRAD
        ray['dist'][np.isnan(ray['dist'])] = 0
        #integrate the Kdp along the ray
        dphi_rain = np.trapz(kdp_rain,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_snow = np.trapz(kdp_snow,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_ice = np.trapz(kdp_ice,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_graupel = np.trapz(kdp_graupel,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    
        dphi_total = dphi_rain + dphi_snow + dphi_graupel + dphi_ice  #total contribution
        
        DPHI_rain = np.append(DPHI_rain,dphi_snow.reshape(1,-1),axis=0)
       
    DPHI_rain = DPHI_rain[1:,:].T

    dphi_PAZ = get_dphaseCal(roid,db='rohppaz_db')
    
    #calculate the correlation coefficients
    cc_tho = np.ma.corrcoef(np.ma.masked_invalid(dphi_PAZ[0:150]),np.ma.masked_invalid(np.flip(DPHI_rain[:,0])[0:150]))[0,1]
    cc_god = np.ma.corrcoef(np.ma.masked_invalid(dphi_PAZ[0:150]),np.ma.masked_invalid(np.flip(DPHI_rain[:,1])[0:150]))[0,1]
    cc_mor = np.ma.corrcoef(np.ma.masked_invalid(dphi_PAZ[0:150]),np.ma.masked_invalid(np.flip(DPHI_rain[:,2])[0:150]))[0,1]
    cc_ws6 = np.ma.corrcoef(np.ma.masked_invalid(dphi_PAZ[0:150]),np.ma.masked_invalid(np.flip(DPHI_rain[:,3])[0:150]))[0,1]
   
    
    plt.figure(figsize=(7,10))
    # plt.axhline(h_0,color='grey')
    plt.plot(DPHI_rain[:,0],ht,color='red',label= 'Thompson ' + 'cc = ' + str(np.round(cc_tho,2)))
    plt.plot(DPHI_rain[:,1],ht,color='green',label='Goddard ' + 'cc = ' + str(np.round(cc_god,2)))
    plt.plot(DPHI_rain[:,2],ht,color='blue',label = 'Morrison ' + 'cc = ' + str(np.round(cc_mor,2)))
    plt.plot(DPHI_rain[:,3],ht,color='violet',label = 'WSM6 ' + 'cc = ' + str(np.round(cc_ws6,2)))
    # plt.plot([],[],color='white',label = 'rain ' + r'$\rho$=' + str(rho_rain) + ' ' + 'ar = ' + str(ar_rain))
    # plt.plot([],[],color='white',label = 'snow ' + r'$\rho$=' + str(rho_snow) + ' ' + 'ar = ' + str(ar_snow))
    # plt.plot([],[],color='white',label = 'graupel ' + r'$\rho$=' + str(rho_graupel) + ' ' + 'ar = ' + str(ar_graup))
    # plt.plot([],[],color='white',label = 'ice ' + r'$\rho$=' + str(rho_ice) + ' ' + 'ar = ' + str(ar_rain)) 
   
    plt.plot(dphi_PAZ[0:150],np.flip(ht)[0:150],color='black',label='PAZ')
    plt.xlabel('Differential phase shift (mm)',fontsize=15)
    plt.ylabel('Height (km)',fontsize=15)
    plt.legend(fontsize=15)
    plt.ylim(0,15)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + var + '_dphi_wrf.png')
    
    return None

def interp_wrf(roid,wrfout):
    """ function that calculates the interpolation values of the WC from 
    WRF into the PRO rays """
    
    figpath = '/media/antia/Elements1/figures'
    srcpath = '/media/antia/Elements1/data/collocations'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'ice*' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
        
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                  ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])


    #Calculate the Liquid Water Contents
    lon_wrf,lat_wrf,h_wrf,temp,wc,dt = read_WRF(wrfout) #wc is in g/m3

    new_h = np.linspace(0,19.9,200)

    lon_wrf_3d = np.tile(lon_wrf,
         (new_h.shape[0],1,1))
    lat_wrf_3d = np.tile(lat_wrf,
        (new_h.shape[0],1,1))
    new_h_3d = np.tile(new_h,
         (lon_wrf_3d.shape[2],
         lon_wrf_3d.shape[1],1)).T

    new_wc_wrf = regrid_in_height(h_wrf,wc,new_h)

    #Interpolation of the LWC into the PRO rays
    interp = interpolate_models_v2(ray, lon_wrf_3d, lat_wrf_3d, new_h_3d, new_wc_wrf)
    
    if 'PAZ1' in roid:
        os.makedirs('/media/antia/Elements1/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
        np.save('/media/antia/Elements1/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',interp)
    else:
        os.makedirs('/media/antia/Elements1/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
        np.save('/media/antia/Elements1/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',interp)

    return interp

def diff_wrf_paz(roid,wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6):
    """ function that calculates the difference of dphi between PAZ and WRF """
   
    srcpath_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations'
    figpath = '/home/aliga/carracedo/Desktop/ICE/figures/WRF'
    resPrf_path = '/home/dilong/carracedo/Desktop/ICE/data/PAZ/profiles'
    file_term = Dataset(resPrf_path + '/' + roid[5:13] + '/' + 
                            'resPrf_' + roid + '_2010.2640_V07.nc')

    T = file_term['profiles'].variables['temperature'][:]
    file_term.close()
    
    fname_col = glob.glob('/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file_col = fname_col[0][79:]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                      ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])


    #densities
    rho_rain = 0.997 #g/cm3
    rho_snow = 0.30 #g/cm3
    rho_graupel = 0.500 #g/cm3 0.050-0.89
    rho_ice = 0.9 #g/cm3
    rho_hail = 0.85 #g/cm3

    #axis ratio
    ar_rain = 0.8 #??? 
    ar_snow = 0.5
    ar_ice = 0.5
    ar_graup = 0.5
    ar_hail = 0.5

    C = 1.6 #(g cm⁻3)⁻2
    dphi_PAZ = get_dphaseCal(roid,db='rohppaz_db')

    DPHI = np.zeros((1,220))

    ht = np.nanmin(ray['h'],axis=1)   
    
    wrfout = [wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6]
    for i in range(4):
        
        interp = np.load('/home/aliga/carracedo/Desktop/ICE/data/WRF' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] +'/' + 
                     'interp_' + wrfout[i][54:57] + '.npy',allow_pickle=True).item()

        #Calculate the Kdp
        kdp_rain = (1/2)*C*rho_rain*interp['rain']*(1-ar_rain)
        kdp_rain = np.ma.masked_invalid(kdp_rain)
    
        kdp_snow = (1/2)*C*rho_snow*interp['snow']*(1-ar_snow)
        kdp_snow = np.ma.masked_invalid(kdp_snow)
    
        kdp_graupel = (1/2)*C*rho_graupel*interp['graupel']*(1-ar_graup)
        kdp_graupel = np.ma.masked_invalid(kdp_graupel)

        kdp_ice = (1/2)*C*rho_ice*interp['ice']*(1-ar_ice)
        kdp_ice = np.ma.masked_invalid(kdp_ice)

        #calculation of dphi from NEXRAD
        ray['dist'][np.isnan(ray['dist'])] = 0
        #integrate the Kdp along the ray
        dphi_rain = np.trapz(kdp_rain,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_snow = np.trapz(kdp_snow,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_ice = np.trapz(kdp_ice,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        dphi_graupel = np.trapz(kdp_graupel,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band

        dphi_total = dphi_rain + dphi_snow + dphi_graupel + dphi_ice  #total contribution
    
        DPHI = np.append(DPHI,dphi_snow.reshape(1,-1),axis=0)
   
    DPHI = DPHI[1:,:].T
    
    diff_tho = dphi_PAZ[0:150]-np.flip(DPHI[:,0])[0:150]
    diff_god = dphi_PAZ[0:150]-np.flip(DPHI[:,1])[0:150]
    diff_mor = dphi_PAZ[0:150]-np.flip(DPHI[:,2])[0:150]
    diff_ws6 = dphi_PAZ[0:150]-np.flip(DPHI[:,3])[0:150]
    
   
    plt.figure(figsize=(7,10))
    plt.axvline(0,color='grey')
    plt.plot(diff_tho,np.flip(ht)[0:150],color='red',label='Thompson')
    plt.plot(diff_god,np.flip(ht)[0:150],color='green',label='Goddard')
    plt.plot(diff_mor,np.flip(ht)[0:150],color='blue',label='Morrison')
    plt.plot(diff_ws6,np.flip(ht)[0:150],color='violet',label='WSM6')
    # plt.legend(fontsize=13)
    plt.ylabel('Height (km)',fontsize=15)
    # plt.title('Total contribution',fontsize=13)
    plt.xlabel('$\Delta\Phi_{PAZ}$-$\Delta\Phi_{WRF}$ (mm)',fontsize=15)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'diff_PAZ_WRF_total.png')
    
    return None

def dphi_wrf_aratio(wrfout,roid):
    """ function that calculates the dphi from a wrfout but plots different
    profiles depending on the value of the axis ratio """
    
    srcpath_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations'
    figpath = '/home/aliga/carracedo/Desktop/ICE/figures/WRF'

    fname_col = glob.glob('/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file_col = fname_col[0][79:]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                          ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    #Calculate the Liquid Water Contents
    lon_wrf,lat_wrf,h_wrf,temp,wc,dt = read_WRF(wrfout) #wc is in g/m3


    new_h = np.linspace(0,19.9,200)

    lon_wrf_3d = np.tile(lon_wrf,
             (new_h.shape[0],1,1))
    lat_wrf_3d = np.tile(lat_wrf,
            (new_h.shape[0],1,1))
    new_h_3d = np.tile(new_h,
             (lon_wrf_3d.shape[2],
             lon_wrf_3d.shape[1],1)).T

    new_wc_wrf = regrid_in_height(h_wrf,wc,new_h)

   
    #Interpolation of the LWC into the PRO rays
    interp = interpolate_models_v2(ray, lon_wrf_3d, lat_wrf_3d, new_h_3d, new_wc_wrf)

    #densities
    rho_rain = 1 #g/cm3
    rho_snow = 0.30 #g/cm3
    rho_graupel = 0.0500 #g/cm3
    rho_ice = 0.9 #g/cm3
    rho_hail = 0.85 #g/cm3
    
    #calculation of dphi from NEXRAD
    ray['dist'][np.isnan(ray['dist'])] = 0
    
    axis_ratio = np.arange(0,1,0.1)
    C = 1.6 #(g cm⁻3)⁻2
    #calculate for each hydrometeor various Kdp depending on the axis ratio
    DPHI_total = np.zeros((1,220))
    for i in range(len(axis_ratio)):
        kdp_rain = (1/2)*C*rho_rain*interp['rain']*(1-axis_ratio[i])
        kdp_rain = np.ma.masked_invalid(kdp_rain)
        dphi_rain = np.trapz(kdp_rain,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
 
        kdp_snow = (1/2)*C*rho_snow*interp['snow']*(1-axis_ratio[i])
        kdp_snow = np.ma.masked_invalid(kdp_snow)
        dphi_snow = np.trapz(kdp_snow,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
   
        kdp_graupel = (1/2)*C*rho_graupel*interp['graupel']*(1-axis_ratio[i])
        kdp_graupel = np.ma.masked_invalid(kdp_graupel)
        dphi_graupel = np.trapz(kdp_graupel,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
    
        kdp_ice = (1/2)*C*rho_ice*interp['ice']*(1-axis_ratio[i])
        kdp_ice = np.ma.masked_invalid(kdp_ice)
        dphi_ice = np.trapz(kdp_ice,dx=ray['dist'],axis=1) #integration of Kdp and conversion from deg to mm and S band to L band
        
        # kdp_hail = (1/2)*C*rho_ice*interp['hail']*(1-axis_ratio[i])
        # kdp_hail = np.ma.masked_invalid(kdp_hail)
        # dphi_hail = np.trapz(kdp_hail,dx=ray['dist'],axis=1) 
        
        dphi_total =  dphi_snow + dphi_graupel + dphi_ice #total contribution
        DPHI_total = np.append(DPHI_total,dphi_total.reshape(1,-1),axis=0)
    
    ht = np.nanmin(ray['h'],axis=1)
    color=cm.jet(np.linspace(0,1,len(axis_ratio)))
    plt.figure(figsize=(8,8))
    for i in range(len(axis_ratio)):
        plt.plot(DPHI_total[i],ht,color=color[i],label = 'axis ratio = ' + str(np.round(axis_ratio[i],2)))
    plt.ylim(0,15)
    plt.plot([],[],color='white',label = "$\u03C1_{rain}$ = " + str(rho_rain))
    plt.plot([],[],color='white',label = "$\u03C1_{snow}$ = " + str(rho_snow))
    plt.plot([],[],color='white',label = "$\u03C1_{graupel}$ = " + str(rho_graupel))
    plt.plot([],[],color='white',label = "$\u03C1_{ice}$ = " + str(rho_ice))
   
    plt.ylabel('Height (km)')
    plt.legend(ncol=3)
    plt.xlabel('Differential phase shift (mm)')
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + wrfout[54:57] + '/' + '_' + wrfout[65:68] + '_dphi_axisratio.png')
        
    return None

def scatter_dphi_mp(wrfout1,wrfout2,wrfout3,wrfout4,roid):
    """ function that plots a scatter plot of the deltaphi from PAZ vs 
    the deltaphi depending on the mp scheme and makes a linear regression """
    
    srcpath_col = '/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations'
    figpath = '/home/aliga/carracedo/Desktop/ICE/figures/WRF'

    fname_col = glob.glob('/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file_col = fname_col[0][79:]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    file.close()

    dphi_PAZ = get_dphaseCal(roid,db='rohppaz_db')

    # wrfout = [wrfout1,wrfout2,wrfout3,wrfout4]
    # DPHI_rain = np.zeros((1,220))
    # DPHI_snow = np.zeros((1,220))
    # DPHI_graupel = np.zeros((1,220))
    # DPHI_ice = np.zeros((1,220))
    DPHI_total = np.zeros((1,220))
    
    dphi_total_tho = mixing_ratio(wrfout1, roid)
    dphi_total_god = mixing_ratio(wrfout2, roid)
    dphi_total_ws6 = mixing_ratio(wrfout3, roid)
    dphi_total_mor = mixing_ratio(wrfout4, roid)
 
    # DPHI_rain = np.append(DPHI_rain,dphi_rain.reshape(1,-1),axis=0)
    # DPHI_snow = np.append(DPHI_snow,dphi_snow.reshape(1,-1),axis=0)
    # DPHI_graupel = np.append(DPHI_graupel,dphi_graupel.reshape(1,-1),axis=0)
    # DPHI_ice =  np.append(DPHI_ice,dphi_ice.reshape(1,-1),axis=0)
    DPHI_total =  np.append(DPHI_total,dphi_total_tho.reshape(1,-1),axis=0)
    DPHI_total =  np.append(DPHI_total,dphi_total_god.reshape(1,-1),axis=0)
    DPHI_total =  np.append(DPHI_total,dphi_total_ws6.reshape(1,-1),axis=0)
    DPHI_total =  np.append(DPHI_total,dphi_total_mor.reshape(1,-1),axis=0)
    DPHI_total = DPHI_total[1:,:]
        
        
    #Linear Regression
    y_new1, slope1, r21 = linear_regresion(np.flip(DPHI_total[0])[0:150],dphi_PAZ[~np.isnan(dphi_PAZ)][0:150])
    y_new2, slope2, r22 = linear_regresion(np.flip(DPHI_total[1])[0:150],dphi_PAZ[~np.isnan(dphi_PAZ)][0:150])
    y_new3, slope3, r23 = linear_regresion(np.flip(DPHI_total[2])[0:150],dphi_PAZ[~np.isnan(dphi_PAZ)][0:150])
    y_new4, slope4, r24 = linear_regresion(np.flip(DPHI_total[3])[0:150],dphi_PAZ[~np.isnan(dphi_PAZ)][0:150])
    
    
    plt.figure(figsize=(8,8))
    plt.plot(np.flip(DPHI_total[0])[0:150],dphi_PAZ[0:150],'o',color='red',label = 'Thompson')
    plt.plot(np.flip(DPHI_total[1])[0:150],dphi_PAZ[0:150],'o',color='green',label = 'Goddard')
    plt.plot(np.flip(DPHI_total[2])[0:150],dphi_PAZ[0:150],'o',color='violet',label = 'WS6')
    plt.plot(np.flip(DPHI_total[3])[0:150],dphi_PAZ[0:150],'o',color='blue',label = 'Morrison')
    
    plt.plot(np.flip(DPHI_total[0])[0:150],y_new1,color='red',label='slope = ' + str(np.round(slope1[0],2)) + ' ' + '$r^{2}$ = ' + str(np.round(r21,2)))
    plt.plot(np.flip(DPHI_total[1])[0:150],y_new2,color='green',label='slope = ' + str(np.round(slope2[0],2)) + ' ' + '$r^{2}$ = ' + str(np.round(r22,2)))
    plt.plot(np.flip(DPHI_total[2])[0:150],y_new3,color='violet',label='slope = ' + str(np.round(slope3[0],2)) + ' ' + '$r^{2}$ = ' + str(np.round(r23,2)))
    plt.plot(np.flip(DPHI_total[3])[0:150],y_new4,color='blue',label='slope = ' + str(np.round(slope4[0],2)) + ' ' + '$r^{2}$ = ' + str(np.round(r24,2)))
    
    plt.xlabel('Differential phase shift from WRF (mm)')
    plt.ylabel('Differential phase shift from PAZ (mm)')
    plt.legend()
    plt.xlim(-1,15)
    plt.ylim(-1,15)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + 'dphi_scatter_snow.png')
    
    return None

def RH(roid,wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6,var,var_wrf):
    """ function that makes a vertical profile of RH from PAZ and the ones
    obtained from different mp schemes from WRF """
    srcpath_col = '/media/antia/easystore/data/collocations/PAZ'
   
    resPrf_path = '/home/dilong/carracedo/Desktop/ICE/data/PAZ/profiles'
    figpath = '/home/aliga/carracedo/Desktop/ICE/figures/WRF'
    file_term = Dataset(resPrf_path + '/' + roid[5:13] + '/' + 
                                'resPrf_' + roid + '_2010.2640_V07.nc')
    
    hr = file_term['profiles'].variables[var][:]
    # var2 = file_term['profiles'].variables[var]
    file_term.close()
    
    fname_col = glob.glob('/home/aliga/carracedo/Desktop/ICE/data/PAZ/collocations/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file_col = fname_col[0][79:]
    file = xr.open_dataset(srcpath_col + '/' + file_col[12:16] + '/' + file_col[12:20] + '/' + file_col[12:20] + '/' + file_col)

    #variables of the rays
    lon = np.array(file.variables['longitude'][:],dtype=float)
    lat = np.array(file.variables['latitude'][:],dtype=float)
    hei = np.array(file.variables['height'][:],dtype=float)
    file.close()
     
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                   ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    ht = np.nanmin(ray['h'],axis=1)

    wrfout = [wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6]
    RH_wrf = np.zeros((1,220))
    for i in range(4):
        #Calculate the Liquid Water Contents
        ncfile = Dataset(wrfout[i])
        if var_wrf == 'T':
            pres = np.array(ncfile.variables['PB'][:][0,:,:,:],dtype=float) + np.array(ncfile.variables['P'][:][0,:,:,:],dtype=float) #Pa
            temp = np.array(ncfile.variables['T'][:][0,:,:,:],dtype=float) +  300 #Potential temperature
            # rh_wrf = temp*(pres/100000)**(2/7)
            rh_wrf = tk(pres=pres,theta=temp,units='K')
        if var_wrf == 'pressure':
            pres = np.array(ncfile.variables['PB'][:][0,:,:,:],dtype=float) + np.array(ncfile.variables['P'][:][0,:,:,:],dtype=float) #Pa
            rh_wrf = pres #hPa
        if var_wrf == 'rh':
            rh_wrf = getvar(ncfile,var_wrf)
            
        rh_dict = {var: np.array(rh_wrf[:,:,:],dtype=float)}
        lon_wrf,lat_wrf,h_wrf,temp,wc,dt = read_WRF(wrfout[i]) #wc is in g/m3

        new_h = np.linspace(0,19.9,200)

        lon_wrf_3d = np.tile(lon_wrf,
                (new_h.shape[0],1,1))
        lat_wrf_3d = np.tile(lat_wrf,
                (new_h.shape[0],1,1))
        new_h_3d = np.tile(new_h,
                (lon_wrf_3d.shape[2],
                 lon_wrf_3d.shape[1],1)).T

        new_rh_wrf = regrid_in_height(h_wrf,rh_dict,new_h)
        
        #Interpolation of the LWC into the PRO rays
        interp = interpolate_models_v2(ray, lon_wrf_3d, lat_wrf_3d, new_h_3d, new_rh_wrf)
        
        ind = []
        for j in range(220):
            try:
                index = np.nanargmin(ray['h'][j,:]-ht[j])
            except:
                index = np.nan
            ind = np.append(ind,index)
        
        rh_ = []
        for j in range(220):
            try:
                t = interp[var][j,ind[j].astype(int)]
            except:
                t = np.nan
            rh_ = np.append(rh_,t)
        
        RH_wrf = np.append(RH_wrf,rh_.reshape(1,-1),axis=0)
    
    RH_wrf = RH_wrf[1:,:].T
    
    plt.figure(figsize=(8,10))
    plt.plot(hr[0:150],np.flip(ht)[0:150],color='black',label='PAZ')
    plt.plot(np.flip(RH_wrf[:,0])[0:150],np.flip(ht)[0:150],color='red',label='Thompson')
    plt.plot(np.flip(RH_wrf[:,1])[0:150],np.flip(ht)[0:150],color='green',label='Goddard')
    plt.plot(np.flip(RH_wrf[:,2])[0:150],np.flip(ht)[0:150],color='blue',label='Morrison')
    plt.plot(np.flip(RH_wrf[:,3])[0:150],np.flip(ht)[0:150],color='violet',label='WSM6')
    # plt.legend(fontsize=13)
    plt.xlabel('Temperature' + ' ' + '(' + 'K'+ ')',fontsize=13)
    plt.ylabel('Height (km)',fontsize=13)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '/' + var + '.png')
    
    return None

def read_WRF(wrfout):
    """ function for calculating the water contents with WRF """

    fwrf = Dataset(wrfout)

    dt = datetime.strptime((('').join(list(np.array(fwrf.variables['Times'][0,:],dtype=str)))),
                           "%Y-%m-%d_%H:%M:%S")

    lon = np.array(fwrf.variables['XLONG'][:][0,:,:],dtype=float)
    # lon[lon>0]-=360.

    lat = np.array(fwrf.variables['XLAT'][:][0,:,:],dtype=float)
    # h = np.array(fwrf.variables['HGT'][:][0,:,:],dtype=float)*1e-3 #terrain height
    h = g_geoht.get_height(fwrf) *1e-3 #km
    h = np.array(h.values,dtype=float)
   
    pres = np.array(fwrf.variables['PB'][:][0,:,:,:],dtype=float) + np.array(fwrf.variables['P'][:][0,:,:,:],dtype=float) #Pa
    temp = np.array(fwrf.variables['T'][:][0,:,:,:],dtype=float) + np.array(fwrf.variables['T00'][:],dtype=float)[0] #Potential temperature
    T = tk(pres,temp,units='K')


    q={
    'snow': np.array(fwrf.variables['QSNOW'][:][0,:,:,:],dtype=float),
    'rain': np.array(fwrf.variables['QRAIN'][:][0,:,:,:],dtype=float),
    'ice' : np.array(fwrf.variables['QICE'][:][0,:,:,:],dtype=float),
    'cloud' : np.array(fwrf.variables['QCLOUD'][:][0,:,:,:],dtype=float),
    'graupel' : np.array(fwrf.variables['QGRAUP'][:][0,:,:,:],dtype=float)
    }

    R_dry = 287.1 * 1e-3  # J K-1 kg-1  --> J K-1 g-1

    wc={}
    for qi in q.keys():
        wc[qi] = q[qi] * pres / ( R_dry* T) #g/m3

    return lon,lat,h,T,wc,dt 

def interpolate_with_3dfields(rays,lat,lon,data,height):

    LAT=np.copy(lat)
    LON=np.copy(lon)
    
    avg=[]
    for ii in range(100):
        i1=np.random.random_integers(1,lat.shape[0]-1)
        i2=np.random.random_integers(1,lon.shape[1]-1)
        a_=np.sqrt( (LAT[i1,i2]-LAT[i1-1,i2])**2. + (LON[i1,i2]-LON[i1-1,i2])**2. )
        b_=np.sqrt( (LAT[i1,i2]-LAT[i1,i2-1])**2. + (LON[i1,i2]-LON[i1,i2-1])**2. )
        avg.append((a_+b_)/2.)
    scalefactor=np.array(avg).mean()
    scalefactor_occ=scalefactor/np.diff(height)[0]
    
    newLON=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))
    newLAT=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))
    newHEI=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))
    
    for j in range(height.shape[0]):
        newLAT[:,:,j]=np.copy(LAT[:,:])
        newLON[:,:,j]=np.copy(LON[:,:])
        newHEI[:,:,j]=np.ones(LAT[:,:].shape)*j
    
    newHEI=newHEI*scalefactor
    
    z={}
    
    z['R']=np.copy(data.ravel())    
    
    cond = np.isfinite(z['R'])
    
    known=np.array([newLON.ravel()[cond],newLAT.ravel()[cond],newHEI.ravel()[cond]]).T
    try:
        ask=np.array([rays['lon'].ravel(),rays['lat'].ravel(),
                  rays['h'].ravel()]).T
    except:
        ask=np.array([rays['lon'].ravel(),rays['lat'].ravel(),
                  rays['h'].ravel()]).T
    
    z['R'][np.isnan(z['R'])]=0.
    z['R'] = z['R'][cond]
    A=int3d.Interpolation3D(known,z)
    interpRay=A(ask,eps=0.3,dup=scalefactor*4)
    
    return interpRay['R'].reshape(-1,rays['lat'].shape[1])

def interpolate_models_v2(ray,lon,lat,height, variables):
    """ new version of the interpolation """

    # LAT = lat[0,:,:]*1.
    # LON = lon[0,:,:]*1.
    
    # newLON=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))
    # newLAT=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))
    # newHEI=np.zeros((LAT.shape[0],LAT.shape[1],height.shape[0]))

    # for j in range(height.shape[0]):
    #     newLAT[:,:,j]=np.copy(LAT[:,:])
    #     newLON[:,:,j]=np.copy(LON[:,:])
    #     newHEI[:,:,j]=np.ones(LAT[:,:].shape)*j
    
    known=np.array([lon.ravel(),lat.ravel(),
                    height.ravel()]).T

    ask = np.array([ray['lon'].ravel(),
                    ray['lat'].ravel(),
                    ray['h'].ravel()]).T
    
    interpRay = {}
    for v in variables.keys():
        var = variables[v].ravel()
        # ask_ = ask[ray['h'].ravel()<20.]
        # interp_ = np.ones_like(ray['h'].ravel())*np.nan
        A=int3d.Interpolation3D_v2(known,var)
        interpRay_=A(ask,nne=8, eps=0.3, p=1, dup=np.inf)
        # interp_[ray['h'].ravel()<20.] = interpRay_
        # interpRay[v] = interp_.reshape(ray['lon'].shape)
        interpRay[v] = interpRay_.reshape(ray['lon'].shape)
    

    return interpRay


def regrid_in_height(h, data_dict, new_h):

    """
    it may be slow
    """
    newdata = {}
    for k in data_dict.keys():

        data = np.array(data_dict[k])
        newdata[k] = np.ones((new_h.shape[0],data.shape[1],data.shape[2]))
        for ix in range(data.shape[1]):
            for ij in range(data.shape[2]):
                newdata[k][:,ix,ij] = np.interp(new_h,
                                             h[:,ix,ij],
                                             data[:,ix,ij],
                                             left=np.nan,right=np.nan)
    return newdata 

def fig_rh(roid,wrfout,wrfout1):
    """ function that plots the relative humidity in a 2D map """
    srcpath = '/media/antia/Elements1/data/collocations'
    figpath = '/media/antia/Elements1/figures/AR'
    import matplotlib.patches as mpatches
    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'ice*' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])

    domain = wrfout[-21:-18]

    mp_scheme = wrfout[-32:-29]
    
    
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                              ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    
    ncfile = Dataset(wrfout)
    ncfile1 = Dataset(wrfout1)
    rh = getvar(ncfile,'rh')
    rh1 = getvar(ncfile1,'rh')
    # Get the latitude and longitude points
    lats, lons = latlon_coords(rh)
    lats_d01, lons_d01 = latlon_coords(rh1)
    raylon,raylat = ray['lon'][ray['h']<20], ray['lat'][ray['h']<20]

    # Get the cartopy mapping object
    cart_proj = get_cartopy(rh)
    
    # Get the extent of the im2 data
    min_lon = np.min(lons.values)
    max_lon = np.max(lons.values)
    min_lat = np.min(lats.values)
    max_lat = np.max(lats.values)
    
    vmin = np.nanmin(np.nanmean(rh,axis=0))
    vmax = np.nanmax(np.nanmean(rh,axis=0))
    
    # Create a figure
    fig = plt.figure(figsize=(10,7))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=crs.PlateCarree())
    
    levels = np.linspace(0,100,10)
    im2 = ax.contourf(lons_d01.values, lats_d01.values, np.nanmean(rh1,axis=0), 10, vmin=vmin, vmax=vmax, levels=levels,
         transform=crs.PlateCarree(),
         cmap=get_cmap("RdYlBu_r"))
    im = ax.contourf(lons.values, lats.values, np.nanmean(rh,axis=0), 15,vmin=vmin,vmax=vmax,
         transform=crs.PlateCarree(),
         cmap=get_cmap("RdYlBu_r"))
    
   
    # Extract rh1 grid edges
    x = lons.values
    y = lats.values

    # Build border using edges of the grid
    edge_lon = np.concatenate([x[0, :], x[:, -1], x[-1, ::-1], x[::-1, 0]])
    edge_lat = np.concatenate([y[0, :], y[:, -1], y[-1, ::-1], y[::-1, 0]])

    # Plot the border
    ax.plot(edge_lon, edge_lat, color='black', linewidth=2, transform=crs.PlateCarree(), zorder=10)



    ax.coastlines(resolution='50m')
    # Add a color bar
    cb=plt.colorbar(im,ax=ax, shrink=.98,extend='max')
    cb.set_label('Mean relative humidity' + ' ' + '(' + rh.units + ')',fontsize=14)
    ax.scatter(file.lon_occ,file.lat_occ,c='black',transform=crs.PlateCarree())
    ax.scatter(raylon,raylat,color='grey', transform=crs.PlateCarree()) 
  
    # Set longitude and latitude ticks manually
    lon_ticks = np.arange(file.lon_occ-12, file.lon_occ+12, 5)
    lat_ticks = np.arange(file.lat_occ-12, file.lat_occ+12, 5)

    ax.set_xticks(lon_ticks, crs=crs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=crs.PlateCarree())
    plt.xlabel('Longitude (\N{DEGREE SIGN})',fontsize=14)
    plt.ylabel('Latitude (\N{DEGREE SIGN})',fontsize=14) 
    ax.add_feature(cartopy.feature.BORDERS)
    ax.coastlines('50m', linewidth=0.8)
    plt.title(rh.Time.values,fontsize=13)
    # plt.xticks(np.linspace(file.lon_occ-6,file.lon_occ+6,4))
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp_scheme , exist_ok = True)
        plt.savefig(figpath + '/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp_scheme + '/' + str('rh') + '_' + domain + '_' + '.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp_scheme , exist_ok = True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp_scheme + '/' + str('rh') + '_' + domain + '_' + '.png')
 
    
    return None

def rays(roid,wrfout):
    """ function for representing the PRO rays and a variable """
    figpath = '/media/antia/Elements/figures'
    
    srcpath_col = '/media/antia/Elements/data/collocations/PAZ'

    fname_col = glob.glob(srcpath_col + '/' + roid[5:9] + '/' + roid[5:13] + '/' + roid[5:13] + '/iceCol_' + roid + '*.nc')
    file = xr.open_dataset(fname_col)
    
    x,y=np.meshgrid(file.npoint.values,file.nray.values)
    p = file.precipitation.values
    h = file.height.values
    file.close()
    
    # f = np.load('/home/aliga/carracedo/Desktop/ICE/data/stats_NEXRAD/stats_' + 
    #            roid + '_' + str(4).zfill(2) + '.npz',allow_pickle=True)
    # kdp = f['arr_0']
    # ray = {}
    # ray['h'] = f['arr_8']
    
    ncfile = Dataset(wrfout)
    interp = np.load('/media/antia/easystore/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] +
           'interp_' + wrfout[54:57] + '.npy',allow_pickle=True).item()

    
    plt.figure(figsize=(12,8))
    for i in range(44):
        i=i*5
        plt.scatter(x[i,:],h[i,:],c=interp['snow'][i,:],marker='.',cmap='RdYlBu_r')
    cb = plt.colorbar()
    plt.ylim(0,15)
    cb.set_label('Snow water content ($g/m^{3}$)',fontsize=13)
    plt.ylabel('Hieght (km)',fontsize=13)
    plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16]+ '/' + 'rays.png' )
    
    return None


def intWC_profile(roid,wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6,var):
    """ function that plots the dphi from PAZ and the integrated WC 
    from WRF """
    
    figpath = '/media/antia/Elements/figures/AR'
    srcpath = '/media/antia/Elements/data/collocations'
    
    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])

    
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                   ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    ht = np.nanmin(ray['h'],axis=1)
    dphi_PAZ = get_dphaseCal(roid)
    WC_int = np.zeros((1,220))
    wrfout = [wrfout_tho,wrfout_god,wrfout_mor,wrfout_ws6]
    ray['dist'][np.isnan(ray['dist'])] = 0
    for i in range(len(wrfout)):
        if 'PAZ1' in roid:
            interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[i][-32:-29] + '.npy',allow_pickle=True).item()
        else:
            interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[i][-32:-29] + '.npy',allow_pickle=True).item()
    
        interp[var][np.isnan(interp[var])] = 0
        int_wc = np.trapz(interp[var],dx=ray['dist'],axis=1) #integrated WC for an specific WC
        WC_int = np.append(WC_int,int_wc.reshape(1,-1),axis=0)

    WC_int = WC_int[1:,:].T
    
    plt.figure(figsize=(8,10))
    im1 = plt.plot(dphi_PAZ[20:150],np.flip(ht)[20:150],color='black',label = 'PRO')
    
    plt.xlabel('Differntial Phase shift (mm)',fontsize=14)
    plt.ylabel('Height (km)',fontsize=14)
    ax = plt.twiny()
    im2 = ax.plot(np.flip(WC_int[:,0])[0:150],np.flip(ht)[0:150],color='red',label='Thompson')
    im3 = ax.plot(np.flip(WC_int[:,1])[0:150],np.flip(ht)[0:150],color='green',label='Goddard')
    im4 = ax.plot(np.flip(WC_int[:,2])[0:150],np.flip(ht)[0:150],color='blue',label='Morrison')
    im5 = ax.plot(np.flip(WC_int[:,3])[0:150],np.flip(ht)[0:150],color='violet',label='WSM6')
    lbls = [l.get_label() for l in (im1+im2+im3+im4+im5)]
    
    plt.legend(im1 + im2 + im3 + im4 + im5,lbls,fontsize=14)
    plt.xlabel('Integrated Water Content (kg/kg)',fontsize=14)
    plt.ylabel('Height (km)',fontsize=14)
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'integrated_WC' + '_' + var + '.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17]  + '/' + 'integrated_WC' + '_' + var + '.png')

    
    return None

def intWC_profile_hydrom(roid,wrfout):
    """ function that plots the dphi from PAZ and the integrated WC 
    from WRF """
    figpath = '/media/antia/Elements/figures/AR'
    srcpath = '/media/antia/Elements/data/collocations'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                   ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    ht = np.nanmin(ray['h'],axis=1)
    dphi_PAZ = get_dphaseCal(roid)
    
    ray['dist'][np.isnan(ray['dist'])] = 0
    if 'PAZ1' in roid:
        interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
    else:
        interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
        
    interp['snow'][np.isnan(interp['snow'])] = 0
    wc_snow = np.trapz(interp['snow'],dx=ray['dist'],axis=1) #integrated WC for an specific WC
    
    interp['rain'][np.isnan(interp['rain'])] = 0
    wc_rain = np.trapz(interp['rain'],dx=ray['dist'],axis=1)
    
    interp['graupel'][np.isnan(interp['graupel'])] = 0
    wc_graupel = np.trapz(interp['graupel'],dx=ray['dist'],axis=1)
    
    interp['ice'][np.isnan(interp['ice'])] = 0
    wc_ice = np.trapz(interp['ice'],dx=ray['dist'],axis=1)
    
    
    plt.figure(figsize=(6, 8))
    im1 = plt.plot(np.flip(wc_rain)[0:120],np.flip(ht)[0:120],color='royalblue',label='rain')
    im2 = plt.plot(np.flip(wc_snow)[0:120],np.flip(ht)[0:120],color='orange',label='snow')
    im3 = plt.plot(np.flip(wc_graupel)[0:120],np.flip(ht)[0:120],color='yellowgreen',label='graupel')
    im4 = plt.plot(np.flip(wc_ice)[0:120],np.flip(ht)[0:120],color='darkblue',label='ice')
    plt.ylabel('Height (km)',fontsize=14)
    plt.xlabel('Integrated Water Content (kg/kg)',fontsize=14)
    twin = plt.twiny()
    im5 = twin.plot(dphi_PAZ[15:120],np.flip(ht)[15:120],color='black',label = 'PRO')
  
    # Combine and create a single legend
    lbls = [l.get_label() for l in (im1+im2+im3+im4+im5)]
    plt.legend(im1 + im2 + im3 + im4 + im5, lbls, fontsize=14)
    plt.xlabel('Differential phase shift (mm)',fontsize=14)
    plt.ylabel('Height (km)',fontsize=14)
    
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'integrated_WC' + '_' + wrfout[-32:-29] + '.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17]  + '/' + 'integrated_WC' + '_' + wrfout[-32:-29] + '.png')

    return None

def hist_mp(ROID_done_AR):
    
    mp_schemes = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values[0]
        if mp_best == 'd_w':
            mp_best = 'Goddard'
        mp_schemes = np.append(mp_schemes,mp_best)
    
    #counts the occurrences of each scheme
    mp_counts = Counter(mp_schemes)
     
    #separate the keys and the values for plotting
    schemes = list(mp_counts.keys())
    frequencies = list(mp_counts.values())
     
    #Figure
    plt.figure(figsize=(8,8))
    plt.bar(schemes,frequencies,color='skyblue',alpha=0.7)
    plt.xlabel('Microphysics schemes',fontsize=14)
    plt.ylabel('number of AR',fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylim(0,18)
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_best_schmes.png')
     
    
    return None

def hist_combinations(cases_done):
    
    snow_type = []; ice_type = []; graupel_type = []
    for roid in cases_done:
        if 'PAZ1' in roid:
            try:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
            except:
                lstsq(roid)
        else:
            try:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
            except:
                lstsq(roid)
        
        snow = df[df['cost'] == np.nanmin(df['cost'])]['type_snow'].values[0]
        ice = df[df['cost'] == np.nanmin(df['cost'])]['type_ice'].values[0]
        graupel = df[df['cost'] == np.nanmin(df['cost'])]['type_graupel'].values[0]
        
        if snow == 'snow_plate':
            snow_type = np.append(snow_type,'p')
        elif snow == 'snow_agg':
            snow_type = np.append(snow_type,'a')
        else:
            snow_type = np.append(snow_type,snow[0])
            
        if ice == 'snow_plate':
            ice_type = np.append(ice_type,'p')
        elif ice == 'snow_agg':
            ice_type = np.append(ice_type,'a')
        else:
            ice_type = np.append(ice_type,ice[0])
            
        if graupel == 'snow_plate':    
            graupel_type = np.append(graupel_type,'p')
        elif graupel == 'snow_agg':
            graupel_type = np.append(graupel_type,'a')
        else:
            graupel_type = np.append(graupel_type,graupel[0])
    
    #combine the particle types
    combinations = [f"{s}-{i}-{g}" for s, i, g in zip(snow_type, ice_type, graupel_type)]
    
    #count the frequency of the combinations
    comb_count = Counter(combinations)
    
    #separate keys and values for plotting
    label = list(comb_count.keys())
    frequencies = list(comb_count.values())
    
    legend_text= ("a = snow_agg\n"
                  "p = snow_plate\n"
                  "i = ice\n"
                  "g = graupel\n"
                  "b = bullet_rosette\n"
                  "r = rosette\n"
                  "h = hail")
    
    #Figure
    plt.figure(figsize=(8,8))
    plt.bar(label,frequencies,color = 'lightgreen',alpha=0.8)
    plt.xticks(rotation = 60)
    plt.xlabel('Particle type combinations',fontsize=14)
    plt.ylabel('number of AR',fontsize=14)
    
    plt.tight_layout(rect = [0,0,0.8,1])
    plt.gcf().text(0.82,0.5,legend_text,fontsize=14,va='center',bbox = dict(boxstyle='round',facecolor='white',edgecolor='black'))
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_best_combinations_EP.png')
     
    
    return None

def std_dphi(roid):
    
    sigma = []
    for i in range(200):
        if i < 40:
            s =(2)
        else:
            s = (1.5)
        sigma = np.append(sigma,s)
    
    return sigma

def lstsq(roid):
    figpath = '/media/antia/Elements1/figures/AR'
    srcpath = '/media/antia/Elements1/data/collocations'
    path = '/media/antia/Elements1/data/collocations/iceCal'
    
    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        try:
            h_flag = get_hflag(roid)[0][0]
            if roid == 'PAZ1.2023.089.07.24.G01':
                h_flag = 1.5
            if roid == 'PAZ1.2023.363.10.52.G14':
                h_flag= 1.

            # h_flag = h_flag.values[0][0]+1
        except:
            h_flag = 0
            if roid == 'PAZ1.2020.356.05.00.G13':
                h_flag = 2.
            elif roid == 'PAZ1.2021.164.03.59.G04':
                h_flag = 2.2
            elif roid == 'PAZ1.2021.182.03.34.G27':
                h_flag = 2.
            elif roid ==  'PAZ1.2021.129.17.13.G16':
                h_flag= 2.
            elif roid == 'PAZ1.2020.355.18.18.G25':
                h_flag = 0.8
            # elif roid == 'PAZ1.2020.125.20.46.G30':
            #     h_flag = 2.5
        dphase = get_dphaseCal(roid)
        # dphase = np.array(file_cal.variables['dph_smooth_grid'][:],dtype=float)
        try:
            fname_col = glob.glob(path + '/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCal_' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_col[0])
            sigma =  np.array(file_cal.variables['dph_smooth_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
        except:
            sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
      
        try:
            h_flag = file.height_flag_smooth
            sigma =  np.array(file.variables['dph_smooth_L2_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            dphase = get_dphaseCal(roid)
        except:
            fname_cal = glob.glob(srcpath + '/' + 'Spire/' + 'iceCal*' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_cal[0])
            sigma = np.array(file_cal.variables["dph_smooth_std_lin"][:],dtype=float)/np.sqrt(50)
            h_flag = 1.
            # sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            # dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
            dphase = get_dphaseCal(roid)
            
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                          ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    h = np.nanmin(ray['h'],axis=1)
    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
    wrfouts = [wrfout_god,wrfout_mor,wrfout_tho,wrfout_ws6]

    #coeficients from the dict of ARTS
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])
    particle_habits = list(dict_coefs.keys())
    type_particles = np.array([value[1] for value in dict_coefs.values()])
    
    mp = ['Goddard','Morrison','Thompson','WSM6']
    color = ['green','blue','red','violet']
    x_s = []; x_i = []; x_g = []; dphi_simu = []; cost_ = []
    cc_ = []; std_stats = []; mean_ = []; particle_snow = []; type_snow = []
    particle_ice = []; type_ice = []; particle_graupel = []; type_graupel = []
    int_dphi_pro = []; int_snow = []; mp_s = []; percen_snow = []
    res = []; std_xsnow = []; std_xice = []; std_xgraupel = []
    #height flag
    dphase[0:220][np.flip(h) < h_flag] = np.nan
    dphase_fig = dphase[0:220]
    dphase_ = dphase.copy()
    dphase_[0:220][np.flip(h) < h_flag] = np.nan
    first_valid_index = np.argmax(~np.isnan(std[0:120]))
   
    std = std[0:120][first_valid_index:] 
    dphase_ = dphase_[0:120][first_valid_index:]
    sigma = sigma[0:120][first_valid_index:]
    std[np.isnan(std)] = 0
    sigma[np.isnan(sigma)] = 0
    dphase_[np.isnan(dphase_)] = 0
    for i in range(len(wrfouts)):
        wc = []
        df_params, i_iwc, sim_dphi_ = parameters(roid, wrfouts[i])
        mp_ = wrfouts[i][-32:-29] #microphysics scheme
        if roid == 'PAZ1.2019.028.11.45.G30':
            mp_ = wrfouts[i][-34:-31]
        if mp_ == 'god':
            mp__ = 'Goddard'
        elif mp_ == 'mor':
            mp__ = 'Morrison'
        elif mp_ == 'tho':
            mp__ = 'Thompson'
        elif mp_ == 'ws6':
            mp__ = 'WSM6'
        mp_s = np.append(mp_s,mp__)
        i_iwc['snow'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['snow'] = i_iwc['snow'][0:120][first_valid_index:]
        i_iwc['snow'][np.isnan(i_iwc['snow'])] = 0
        i_iwc['ice'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['ice'] = i_iwc['ice'][0:120][first_valid_index:]
        i_iwc['ice'][np.isnan(i_iwc['ice'])] = 0
        i_iwc['graupel'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['graupel'] = i_iwc['graupel'][0:120][first_valid_index:]
        i_iwc['graupel'][np.isnan(i_iwc['graupel'])] = 0
        wc = np.append(wc,i_iwc['snow'])
        wc = np.append(wc,i_iwc['ice'])
        wc = np.append(wc,i_iwc['graupel'])
        wc = np.reshape(wc,(3,len(i_iwc['snow'])))
        i_iwc['rain'][np.flip(h)[0:200]<h_flag] = np.nan
        i_iwc['rain'] = i_iwc['rain'][0:120][first_valid_index:]
        i_iwc['rain'][np.isnan(i_iwc['rain'])] = 0
        
        x = scipy.optimize.lsq_linear(np.diag(std) @ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['x']
        cost = scipy.optimize.lsq_linear( np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['cost']
        residuals = scipy.optimize.lsq_linear(np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['fun']
        # res = np.append(res,residuals)
        x_snow = x[0]
        x_ice = x[1]
        x_graupel = x[2]
        x_s = np.append(x_s,x_snow)
        x_i = np.append(x_i,x_ice)
        x_g = np.append(x_g,x_graupel)
        cost_ = np.append(cost_,cost/len(dphase_))
        
        #covariance of the solution
        R_inv = np.diag(1/sigma**2)
        R_inv[R_inv == np.inf] =0
        A = R_inv @ wc.T
        J = A
        #covariance matrix of x
        cov_x = np.linalg.inv(J.T @ J)
        #standard deviation of x (square root of diagonal elements)
        std_x = np.sqrt(np.diag(cov_x))
        std_xsnow = np.append(std_xsnow,std_x[0])
        std_xice = np.append(std_xice,std_x[1])
        std_xgraupel = np.append(std_xgraupel,std_x[2])
        
        sim_dphi = i_iwc['rain'] + i_iwc['snow']*x_snow + i_iwc['ice']*x_ice + i_iwc['graupel']*x_graupel
        
        int_dphi = np.trapz(dphase_[0:120])
        int_dphi_pro = np.append(int_dphi_pro,int_dphi)
        int_IWC_snow = np.trapz(i_iwc['snow'][0:120])
        int_snow = np.append(int_snow,int_IWC_snow)
        
        cc = np.corrcoef(dphase_[0:200],sim_dphi)
        cc_ = np.append(cc_,cc[0,1])
        
        diff_dphi = dphase_[0:200] - sim_dphi
        std_diff = np.nanstd(diff_dphi)
        std_stats = np.append(std_stats,std_diff)
        mean_ = np.append(mean_,np.nanmean(diff_dphi))
        
        index_s = np.argmin(np.abs(x_snow-b_param_coef))
        p_snow = particle_habits[index_s]
        t_snow = type_particles[index_s]
        particle_snow = np.append(particle_snow,p_snow)
        type_snow = np.append(type_snow,t_snow)
        
        index_i = np.argmin(np.abs(x_ice-b_param_coef))
        p_ice = particle_habits[index_i]
        t_ice = type_particles[index_i]
        particle_ice = np.append(particle_ice,p_ice)
        type_ice = np.append(type_ice,t_ice)
        
        index_g = np.argmin(np.abs(x_graupel-b_param_coef))
        p_graupel = particle_habits[index_g]
        t_graupel = type_particles[index_g]
        particle_graupel = np.append(particle_graupel,p_graupel)
        type_graupel = np.append(type_graupel,t_graupel)
        
        dphi_simu = np.append(dphi_simu,sim_dphi)
        
        percentages = WC_percen(roid,wrfouts[i])
        percen_snow = np.append(percen_snow, percentages['snow'])
        
        #Figure
        # dphase_fig = dphase_
        # dphase_fig[dphase_fig == 0] = np.nan
        plt.figure(figsize=(8,10))
        plt.plot(dphase_,np.flip(h)[0:120][first_valid_index:],color='black',label='PRO')
        plt.plot(sim_dphi,np.flip(h)[0:120][first_valid_index:],color=color[i],label=mp[i])
        plt.plot([],[],color = 'white',label = 'x_snow =' + str(np.round(x_snow,5)))
        plt.plot([],[],color = 'white',label = 'x_ice =' + str(np.round(x_ice,5)))
        plt.plot([],[],color = 'white',label = 'x_graupel =' + str(np.round(x_graupel,5)))
        plt.plot([],[],color = 'white', label = 'cost = ' + str(np.round(cost/len(dphase_),2)))
        plt.xlabel('Differential phase shift (mm)',fontsize=14)
        plt.ylabel('Height (km)',fontsize=14)
        plt.legend(fontsize=14)
        plt.ylim(0,12.4)
        plt.title(mp[i],fontsize=14)

        if 'PAZ1' in roid:
            os.makedirs(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
            plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' +  wrfouts[i][-32:-29]+ '_lstsq3.png')
        else:
            os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
            plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' +  wrfouts[i][-32:-29] + '_lstsq3.png')
     
    dphi_simu = np.reshape(dphi_simu,(4,len(dphase_)))
    #Figure
    plt.figure(figsize=(8,10))
    plt.plot(dphase_,np.flip(h)[0:120][first_valid_index:],color='black',label='PRO')
    plt.plot(dphi_simu[0],np.flip(h)[0:120][first_valid_index:],color='green',label='Goddard')
    plt.plot(dphi_simu[1],np.flip(h)[0:120][first_valid_index:],color='blue',label='Morrison')
    plt.plot(dphi_simu[2],np.flip(h)[0:120][first_valid_index:],color='red',label='Thompson')
    plt.plot(dphi_simu[3],np.flip(h)[0:120][first_valid_index:],color='violet',label='WSM6')
    plt.xlabel('Differential phase shift (mm)',fontsize=14)
    plt.ylabel('Height (km)',fontsize=14)
    plt.legend(fontsize=14)
    plt.ylim(0,12.4)
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'lstsq3.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'lstsq3.png')
  
    
    data = {'param_snow': x_s, 'param_ice': x_i, 'param_graupel': x_g,
          'std_diff': std_stats, 'mean_diff': mean_, 'cc': cc_, 'particle_graupel': particle_graupel,
          'particle_ice': particle_ice, 'particle_snow': particle_snow, 'type_snow': type_snow,
          'type_ice': type_ice, 'type_graupel': type_graupel, 'cost': cost_, 'int_dphi': int_dphi_pro,
          'int_snow': int_snow, 'mp': mp_s, 'percen_snow': percen_snow, 'std_x_snow': std_xsnow, 'std_x_ice': std_xice, 'std_xgraupel': std_xgraupel}
    df_lstsq = pd.DataFrame(data)
    
    if 'PAZ1' in roid:
        df_lstsq.to_pickle('/media/antia/Elements/data/interp/' + roid[0:4] + '_' +  roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl')
    else:
        df_lstsq.to_pickle('/media/antia/Elements/data/interp/' + roid[0:5] + '_' +  roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl')
 
    
    return df_lstsq

def parameters(roid,wrfout):
    
    figpath = '/media/antia/Elements1/figures'
    srcpath = '/media/antia/Elements1/data/collocations'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        dphase = get_dphaseCal(roid)
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        try:
            dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
        except:
            dphase = get_dphaseCal(roid)
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                      ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    if 'PAZ1' in roid:
        try:
            interpRay_wrf = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
        except:
            interpRay_wrf = interp_wrf(roid,wrfout)
    else:
        try:
            interpRay_wrf = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
        except:
            interpRay_wrf = interp_wrf(roid,wrfout)
    
    ice_param_list = 10**np.linspace(-3,0,25).reshape(1,1,-1,1,1)
    snow_param_list = 10**np.linspace(-3,0,26).reshape(1,-1,1,1,1)
    graupel_param_list = 10**np.linspace(-3,0,27).reshape(-1,1,1,1,1)


    i_iwc = {}
    for hy in ['ice', 'snow', 'graupel']:
        X = np.ma.masked_invalid(interpRay_wrf[hy])
        d = np.ma.masked_invalid(ray['dist'])
    
        i_iwc[hy] = np.trapz(X,dx=d,axis=1)

        i_iwc[hy] = np.interp( np.linspace(0,19.9,200),
                           np.nanmin(ray['h'],1)[::-1],
                           i_iwc[hy][::-1],
                           left = np.nan, right = np.nan)
        
    #Kdp calculation for rain
    A = 0.13; B=1.314
    X = np.ma.masked_invalid(interpRay_wrf['rain'])
    Kdp_rain = A*X**B
    dphi_rain = np.trapz(Kdp_rain,dx=d,axis=1)
    i_iwc['rain'] = np.interp( np.linspace(0,19.9,200),
                       np.nanmin(ray['h'],1)[::-1],
                       dphi_rain[::-1],
                       left = np.nan, right = np.nan)

    sim_dphi = (i_iwc['rain']+ i_iwc['ice']*ice_param_list +
            i_iwc['snow']*snow_param_list +
            i_iwc['graupel']*graupel_param_list)
    sim_dphi_norain = (i_iwc['ice']*ice_param_list +
            i_iwc['snow']*snow_param_list +
            i_iwc['graupel']*graupel_param_list)
    #simulated dphi
    dphi_sim = np.nanmean(sim_dphi,axis=3)
    #coeficients from the dict of ARTS
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])
    particle_habits = list(dict_coefs.keys())
    type_particles = np.array([value[1] for value in dict_coefs.values()])
    
    param_snow = []; param_ice = []; param_graupel = []; std = []; mean_ = []
    chi = []; particle_graupel =[]; particle_snow = []; particle_ice = []
    type_snow = []; type_graupel = []; type_ice = [];cc =[]
    for i in range(27):
        for j in range(26):
            for k in range(25):
                snow_param = snow_param_list[0,j,0,0,0]
                ice_param = ice_param_list[0,0,k,0,0]
                graupel_param = graupel_param_list[i,0,0,0,0]
                dphi_wrf = dphi_sim[i,j,k,:]
                
                #difference
                diff_dphi = dphase[0:200] - dphi_wrf
                std_diff = np.nanstd(diff_dphi)
                masked_dp = np.ma.masked_invalid(dphase[0:200])
                masked_wrf = np.ma.masked_invalid(dphi_wrf)
                
                joint_mask = ~np.logical_or(masked_dp.mask,masked_wrf.mask)
        
                cc_ = np.corrcoef(dphase[0:200][joint_mask],dphi_wrf[joint_mask])[0,1]
                mean = np.nanmean(diff_dphi)
                #chi squared
                # expect_norm = dphi_wrf*(np.sum(dphase[0:200])/np.sum(dphi_wrf))
                try:
                    chi2,p_value = scipy.stats.chisquare(dphase[0:200],dphi_wrf)
                except:
                    chi2,p_value = np.nan, np.nan
                #type of particle habit
                index_s = np.argmin(np.abs(snow_param-b_param_coef))
                p_snow = particle_habits[index_s]
                t_snow = type_particles[index_s]
                index_i = np.argmin(np.abs(ice_param-b_param_coef))
                p_ice = particle_habits[index_i]
                t_ice = type_particles[index_i]
                index_g = np.argmin(np.abs(graupel_param-b_param_coef))
                p_graupel = particle_habits[index_g]
                t_graupel = type_particles[index_g]
                
                param_snow = np.append(param_snow,snow_param)
                param_ice = np.append(param_ice,ice_param)
                param_graupel = np.append(param_graupel,graupel_param)
                std = np.append(std,std_diff)
                mean_ = np.append(mean_,mean)
                cc = np.append(cc,cc_)
                chi = np.append(chi,chi2)
                particle_graupel = np.append(particle_graupel,p_graupel)
                particle_ice = np.append(particle_ice,p_ice)
                particle_snow = np.append(particle_snow,p_snow)
                type_snow = np.append(type_snow,t_snow)
                type_ice = np.append(type_ice,t_ice)
                type_graupel = np.append(type_graupel,t_graupel)
            
    data = {'param_snow': param_snow, 'param_ice': param_ice, 'param_graupel': param_graupel,
            'std_diff': std, 'mean_diff': mean_, 'cc': cc, 'particle_graupel': particle_graupel,
            'particle_ice': particle_ice, 'particle_snow': particle_snow, 'type_snow': type_snow,
            'type_ice': type_ice, 'type_graupel': type_graupel, 'chi': chi}
    
    df_params = pd.DataFrame(data)
    # if 'PAZ1' in roid:
    #     df_params.to_pickle('/media/antia/Elements/data/interp/' + roid[0:4] + '_' +
    #                         roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_params_' + wrfout[])
    # else:
    try:
        np.save('/media/antia/Elements1/data/interp/' + roid[0:4] + '_' +
                             roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'i_iwc_' + wrfout[-32:-29],i_iwc)
    except:
        np.save('/media/antia/Elements1/data/interp/' + roid[0:5] + '_' +
                            roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'i_iwc_' + wrfout[-32:-29],i_iwc)
 
        
    return df_params, i_iwc, sim_dphi

def plot_p_snow(cases_done):
    """ function that makes 4 plots in one figure showing
    a histogram of the percentage of snow depending on the microphysics"""
    
    god_snow = []; mor_snow = []; tho_snow = []; ws6_snow = []
    for roid in cases_done:
        wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
        p_snow, p_rain, p_graupel, p_ice = watercontent_percen(roid,wrfout_god)
        god_snow = np.append(god_snow,p_snow)
        p_snow, p_rain, p_graupel, p_ice = watercontent_percen(roid,wrfout_mor)
        mor_snow = np.append(mor_snow,p_snow)
        p_snow, p_rain, p_graupel, p_ice = watercontent_percen(roid,wrfout_tho)
        tho_snow = np.append(tho_snow,p_snow)
        p_snow, p_rain, p_graupel, p_ice = watercontent_percen(roid,wrfout_ws6)
        ws6_snow = np.append(ws6_snow,p_snow)

    dict_snow = {'Goddard': god_snow, 'Morrison': mor_snow, 'Thompson': tho_snow, 'WSM6': ws6_snow}
   
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Define histogram parameters
    bins = len(cases_done)
    colors = ['green', 'blue', 'red', 'violet']
    labels = ['Goddard', 'Morrison', 'Thompson', 'WSM6']
    datasets = ['Goddard', 'Morrison', 'Thompson', 'WSM6']

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Loop through the datasets and plot each histogram
    for i, dataset in enumerate(datasets):
        axes[i].hist(dict_snow[dataset], bins=bins, color=colors[i], histtype='step', label=labels[i])
        axes[i].legend(fontsize=13)
        axes[i].set_xlabel('Snow percentage (%)',fontsize=13)
        axes[i].set_xlim(0,100)
        axes[i].set_ylim(0,7)
        axes[i].set_ylabel('nº AR',fontsize=13)
        # axes[i].set_title(f'Histogram of {labels[i]}')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/hist_p_snow.png')

    
    plt.figure(figsize=(8,8))
    plt.hist(dict_snow['Goddard'],bins=17,color='green',histtype='step',label='Goddard')
    plt.hist(dict_snow['Morrison'],bins=17,color='blue',histtype='step',label='Morrison')
    plt.hist(dict_snow['Thompson'],bins=17,color='red',histtype='step',label='Thompson')
    plt.hist(dict_snow['WSM6'],bins=17,color='violet',histtype='step',label='WSM6')
    plt.legend(fontsize=14)
    plt.xlabel('Snow percentage (%)', fontsize=14)
    plt.ylabel('AR cases', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,110)
    plt.savefig('/media/antia/Elements/figures/AR/hist_p_snow2.png')

    return None

def get_wrfout(roid):
    
    srcpath = '/media/antia/Elements1/data/wrfout/WRFout'
    srcpath2 = '/home/antia/Build_WRF/WRF/test/em_real'
    srcpath_col = '/media/antia/Elements1/data/collocations'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath_col + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        time = file.timeUTC
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        hour = time[11:13]
        minutes = time[14:16]
        if int(minutes) < 30:
            hour = time[11:13]
        else:
            hour = str(int(hour) +1).zfill(2)
             
    else:
        try:
            fname_col = glob.glob(srcpath_col + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath_col + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        try:
            time = file.startTimeReference
        except:
            time = file.timeUTC
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        hour = time[11:13]
        minutes = time[14:16]
        if int(minutes) < 30:
            hour = time[11:13]
        else:
            hour = str(int(hour) +1).zfill(2)
    
    wrfouts = ['god','mor','tho','ws6']
    wrf_files = []
    if 'PAZ1' in roid:
        for mp in wrfouts:
            wrfout_fn = glob.glob(srcpath + '/' + roid[0:4] + '_' +  roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp + '_wrfout_d02_' + year + '-' + month + '-' + day + '_' + hour + '*')
            if wrfout_fn == []:
                wrfout_fn = glob.glob(srcpath2 + '/' + roid[0:4] + '_' +  roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + mp + '_wrfout_d02_' + year + '-' + month + '-' + day + '_' + hour + '*')
            wrf_files = np.append(wrf_files, wrfout_fn[0])
        
        wrfout_god = wrf_files[0]
        wrfout_mor = wrf_files[1]
        wrfout_tho = wrf_files[2]
        wrfout_ws6 = wrf_files[3]
     
    else:
        for mp in wrfouts:
            wrfout_fn = glob.glob(srcpath + '/' + roid[0:5] + '_' +  roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp + '_wrfout_d02_' + year + '-' + month + '-' + day + '_' + hour + '*')
            if wrfout_fn == []:
                wrfout_fn = glob.glob(srcpath2 + '/' + roid[0:5] + '_' +  roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + mp + '_wrfout_d02_' + year + '-' + month + '-' + day + '_' + hour + '*')
            wrf_files = np.append(wrf_files, wrfout_fn[0])

        wrfout_god = wrf_files[0]
        wrfout_mor = wrf_files[1]
        wrfout_tho = wrf_files[2]
        wrfout_ws6 = wrf_files[3]

    return wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6

def WC_percen(roid,wrfout):
    """ function that calculates the percentage of water content
    for each hydrometeor """
    
    if 'PAZ1' in roid:
        try:
            interpRay_wrf = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
        except:
            interpRay_wrf = interp_wrf(roid,wrfout)
    else:
        try:
            interpRay_wrf = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
        except:
            interpRay_wrf = interp_wrf(roid,wrfout)
  
    total_wc = np.nansum(interpRay_wrf['snow'] + interpRay_wrf['ice'] + interpRay_wrf['graupel'] + interpRay_wrf['rain'])
    p_snow = 100*np.nansum(interpRay_wrf['snow'])/total_wc
    p_ice = 100*np.nansum(interpRay_wrf['ice'])/total_wc
    p_graupel = 100*np.nansum(interpRay_wrf['graupel'])/total_wc
    p_rain = 100*np.nansum(interpRay_wrf['rain'])/total_wc
  
    percentages = {'snow': p_snow, 'ice': p_ice, 'graupel': p_graupel, 'rain': p_rain}  
  
    return percentages


# Define linear function for fitting
def linear_fit(x, m, b):
    return m * x + b

def scatter_mp(ROID_done_AR):
    """ function that makes a scatter plot with all the cases showing the 
    best options for each mp scheme for each case """
    
    figpath = '/media/antia/Elements/figures/AR'
  
    x_snow = []; x_ice = []; x_graupel = []; mp = []; mp_scheme_best = []
    cost = [];df_ = pd.DataFrame([]); p_snow_all = []; p_rain_all = []
    p_ice_all = []; p_graup_all = []
    x_best_snow = []; x_best_ice = []; x_best_graupel = []
    int_snow_best =[]; int_graup_best = []; int_ice_best = []; int_rain_best = []
    int_dphi_best = []; cost_best = []; p_best_snow = []; chi_best = []

    for i, roid in enumerate(ROID_done_AR):
        try:
            if 'PAZ1' in roid:
                df = pd.read_pickle('/media/antia/Elements/data/interp' + '/' +  roid[0:4] + '_' +
                    roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl')
                wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
          
            else:
                df = pd.read_pickle('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' +  roid[15:17] + '/' + 'df_lstsq_new3.pkl')
                wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
          
        except:
            
            df = lstsq(roid)
            wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
            for j in range(4):
                plt.close()
                
        df_ = pd.concat([df_,df])        
        mp_best = df[df['cost'] == np.nanmin(df['cost'].values)]['mp']
        mp_scheme_best = np.append(mp_scheme_best,[mp_best])
        
        x_snow = np.append(x_snow,df['param_snow'].values)
        x_ice = np.append(x_ice,df['param_ice'].values)
        x_graupel = np.append(x_graupel,df['param_graupel'].values)
        mp = np.append(mp,df['mp'].values)
        # chi = np.append(chi,df['chi'].values)
        c = df['cost'].values
        dphi_PAZ = get_dphaseCal(roid)
        # dphase = np.count_nonzero(~np.isnan(dphi_PAZ[0:120]))
        first_valid_index = np.argmax(~np.isnan(dphi_PAZ[0:120]))
        dphase_ = dphi_PAZ[0:120][first_valid_index:]
        cost = np.append(cost,c)
            
        x_snow =  df[(df['mp'] == mp_best.values[0])]['param_snow']
        x_best_snow = np.append(x_best_snow,x_snow)
        x_ice =  df[(df['mp'] == mp_best.values[0])]['param_ice']
        x_best_ice = np.append(x_best_ice,x_ice)
        x_graupel =  df[(df['mp'] == mp_best.values[0])]['param_graupel']
        x_best_graupel = np.append(x_best_graupel,x_graupel)
    
        # chi_b = chi[(df_['roid'] == cases_done[i]) & (df_['mp_scheme'] == mp_scheme_best[i])]
        # chi_best = np.append(chi_best,chi_b)
        cost_b = df['cost'][(df['mp'] == mp_best.values[0])]
        cost_best = np.append(cost_best,cost_b)
        
       
        int_snow = df['int_snow'][df['mp'] == mp_best.values[0]]
        int_snow_best = np.append(int_snow_best,int_snow)
        int_dphi = df['int_dphi'][df['mp'] == mp_best.values[0]]
        int_dphi_best = np.append(int_dphi_best,int_dphi)
       
        p_snow = df['percen_snow'][df['mp'] == mp_best.values[0]]
        p_best_snow = np.append(p_best_snow,p_snow)
       
    #normalize the chi squared
    chi_norm = (cost_best-np.min(cost_best))/(np.max(cost_best)-np.min(cost_best))
    # s = np.array([(1/(n +1e-3)) for n in (chi_norm)])
    s = 10/(chi_norm + 0.1)
    
    # from scipy.optimize import curve_fit
    import matplotlib.lines as mlines
    square = mlines.Line2D([], [], marker='s', color='black', markersize=10, linestyle='None', label='Goddard')
    triangle = mlines.Line2D([], [], marker='^', color='black', markersize=10, linestyle='None', label='WSM6')
    star = mlines.Line2D([], [], marker='*', color='black', markersize=10, linestyle='None', label='Morrison')
    cercle = mlines.Line2D([], [], marker='o', color='black', markersize=10, linestyle='None', label='Thompson')
    
    # popt, pcov = curve_fit(linear_fit, int_snow_best, int_dphi_best, sigma=s, absolute_sigma=True, p0=[1, 0])

    from scipy.stats import theilslopes
    m_fit, b_fit, _, _ = theilslopes(int_dphi_best, int_snow_best)

    # Extract the fitted slope (m) and intercept (b)
    # m_fit, b_fit = popt
    y_fit = linear_fit(np.array(int_snow_best), m_fit, b_fit)
  
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # Flatten axes for easier iteration
    axes = axes.flatten()
    for i in range(len(ROID_done_AR)):
        if mp_scheme_best[i] == 'Goddard':
            mark = 's'
        elif mp_scheme_best[i] == 'Thompson':
            mark = 'o'
        elif mp_scheme_best[i] == 'WSM6':
            mark = '^'
        elif mp_scheme_best[i] == 'Morrison':
            mark = '*'
        im = axes[0].scatter(int_snow_best[i],int_dphi_best[i],c=x_best_snow[i],s=s[i],vmin=np.nanmin(x_best_snow),vmax=0.2,marker = mark,cmap='rainbow')
    axes[0].plot(int_snow_best, y_fit, color='grey', linestyle='--', linewidth=2, label=f"Weighted Fit: y = {m_fit:.3f}x + {b_fit:.3f}")
    cbar = plt.colorbar(im,extend='max')
    cbar.set_label('x-parameter for snow',rotation=270,fontsize=16,labelpad=20)
    cbar.ax.tick_params(labelsize=16)
    axes[0].set_xlabel('Snow Integrated Water Content ($kg·m^{-2}$)',fontsize=16)
    axes[0].set_ylabel('Integrated Differential Phase Shift (mm)',fontsize=16)
    # plt.gcf().text(0.82,0.5,legend_text,fontsize=14,va='center',bbox = dict(boxstyle='round',facecolor='white',edgecolor='black'))
    axes[0].legend(handles=[square, triangle, star,cercle], fontsize=16)
    axes[0].text(1250,1600,'(a)',fontsize=16)
    axes[0].tick_params(labelsize=16)   
    
    for i in range(len(ROID_done_AR)):
        if mp_scheme_best[i] == 'Goddard':
            mark = 's'
        elif mp_scheme_best[i] == 'Thompson':
            mark = 'o'
        elif mp_scheme_best[i] == 'WSM6':
            mark = '^'
        elif mp_scheme_best[i] == 'Morrison':
            mark = '*'
        im1 = axes[1].scatter(int_snow_best[i],int_dphi_best[i],c=p_best_snow[i],s=s[i],vmin = np.nanmin(p_best_snow),vmax=np.nanmax(p_best_snow),marker = mark,cmap='rainbow')
    axes[1].plot(int_snow_best, y_fit, color='grey', linestyle='--', linewidth=2, label=f"Weighted Fit: y = {m_fit:.3f}x + {b_fit:.3f}")
    axes[1].legend(fontsize=16)
    axes[1].tick_params(labelsize=16) 
    axes[1].text(1250,1600,'(b)',fontsize=16)
    cbar = plt.colorbar(im1)
    cbar.set_label('snow percentage (%)',rotation=270,fontsize=16,labelpad=20)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('Snow Integrated Water Content ($kg·m^{-2}$)',fontsize=16)
    plt.ylabel('Integrated Differential Phase Shift (mm)',fontsize=16)
    # plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/scatter_plot_best_p2.png')
    
    return None



def scatter_maxWC_maxdphi(ROID_done_AR):
    """ function that makes a scatter plot of maxWC vs maxdphi """
    import scipy
    srcpath_col = '/media/antia/Elements/data/collocations/PAZ'
    figpath = '/media/antia/Elements/figures/AR'
    srcpath = '/media/antia/Elements/data/collocations/Spire'

    max_snow = []; max_ice = []; max_graupel = []; max_rain = []
    max_dphi = []; int_snow = []; int_ice = []; int_rain = []; int_graupel = []
    int_dphi = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            fname_col = glob.glob('/media/antia/Elements/data/collocations/PAZ/' + roid[5:9] + '.' + roid[10:13] +
                                  '/iceCol_' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
    
        else:
            try:
                fname_col = glob.glob(srcpath + '/' + 'iceCol*' + roid + '*.nc')
                file = xr.open_dataset(fname_col[0])
            except:
                fname_col = glob.glob(srcpath + '/' + 'icePha*' + roid + '*.nc')
                file = xr.open_dataset(fname_col[0])
                
        # variables of the rays
        try:
            lon = np.array(file.variables['longitude'][:], dtype=float)
            lat = np.array(file.variables['latitude'][:], dtype=float)
            height = np.array(file.variables['height'][:], dtype=float)
            file.close()
        except:
            lon = np.array(file.variables['ray_longitude'][:], dtype=float)
            lat = np.array(file.variables['ray_latitude'][:], dtype=float)
            height = np.array(file.variables['ray_height'][:], dtype=float)
            file.close()


        ray = {}
        ray['lon'] = lon
        ray['lat'] = lat
        ray['h'] = height
        ray['dist'] = af.distlatlonhei(ray['lat'][:, 1:], ray['lon'][:, 1:], ray['h'][:, 1:],
                    ray['lat'][:, :-1], ray['lon'][:, :-1], ray['h'][:, :-1])
        h = np.nanmin(ray['h'], axis=1)
        dphase = get_dphaseCal(roid)
        dphi_PAZ = dphase[:200]
        dphi_PAZ[np.isnan(dphi_PAZ)] = 0
        
      
        wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
   
        wrfout = [wrfout_god,wrfout_mor,wrfout_tho,wrfout_ws6]
        for j in range(4):
            try:
                if 'PAZ1' in roid:
                    # df = pd.read_pickle('/media/antia/Elements/data/interp/' +  roid[0:4] + '_' +
                            # roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_params2.pkl')
                    # sim_dphi_ = np.load('/media/antia/Elements/data/interp/' +  roid[0:4] + '_' +
                            # roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'sim_dphi.npy')
                    i_iwc = np.load('/media/antia/Elements/data/interp/' +  roid[0:4] + '_' +
                                    roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'i_iwc_' + wrfout[j][-32:-29] + '.npy',allow_pickle=True).tolist()
                else:
                    # df = pd.read_pickle('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' +  roid[15:17]    + '/' + 'df_params2.pkl')
                    # sim_dphi_ = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' +  roid[15:17]   + '/' + 'sim_dphi.npy',allow_pickle=True)
                    i_iwc = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' +  roid[15:17]   + '/' + 'i_iwc_' + wrfout[j][-32:-29] + '.npy',allow_pickle=True).tolist()

            except:
                df,sim_dphi_,i_iwc = parameters(roid, wrfout[j])
            
            wc = i_iwc['ice']
            wc = np.append(wc,i_iwc['snow'])
            wc = np.append(wc,i_iwc['graupel'])
            wc = np.reshape(wc,(3,200))
            wc[np.isnan(wc)] = 0
            i_iwc['rain'][np.isnan(i_iwc['rain'])] = 0
        
            max_wc_snow = np.nanmax(i_iwc['snow'])
            max_wc_ice = np.nanmax(i_iwc['ice'])
            max_wc_graupel = np.nanmax(i_iwc['graupel'])
            max_wc_rain = np.nanmax(i_iwc['rain'])
            max_dphiPAZ = np.nanmax(dphi_PAZ)
    
            max_snow = np.append(max_snow,max_wc_snow)
            max_ice = np.append(max_ice,max_wc_ice)
            max_graupel = np.append(max_graupel,max_wc_graupel)
            max_rain = np.append(max_rain,max_wc_rain)
            max_dphi = np.append(max_dphi,max_dphiPAZ)
        
            int_snow = np.append(int_snow,np.nansum(i_iwc['snow']))
            int_rain = np.append(int_rain,np.nansum(i_iwc['rain']))
            int_graupel = np.append(int_graupel,np.nansum(i_iwc['graupel']))
            int_ice = np.append(int_ice,np.nansum(i_iwc['ice']))
            int_dphi = np.append(int_dphi, np.nansum(dphi_PAZ))
        
    # ar = np.arange(0,len(max_dphi),1)
    # from paper_NEXRAD import linear_regresion
   
    # for (hy,color,label) in [(int_snow,'green','snow'),(int_ice,'red','ice'),(int_rain,'blue','rain'),(int_graupel,'violet','graupel')]:
    #     y_new, slope, r = linear_regresion(hy,int_dphi)
    #     plt.figure(figsize=(8,8))
    #     plt.scatter(hy,int_dphi)
    #     plt.plot(hy,y_new,label='slope = ' + str(
    #         np.round(slope[0], 2)) + ' ' + '$r^{2}$ = ' + str(np.round(r, 2)))
    #     plt.xlabel('IWC ' + label + '(kg/kg)')
    #     plt.ylabel('differential phase shift (mm)')
    #     plt.legend()
    #     plt.savefig(figpath + '/' + label + '_int_scatter_dphi_mor.png')
    #     plt.close()
        
    # #FIGURE
    # for (hy,color,label) in [(int_snow,'green','snow'),(int_ice,'red','ice'),(int_rain,'blue','rain'),(int_graupel,'violet','graupel')]:
    #     plt.figure(figsize=(8,8))
    #     plt.plot(ar,hy,color=color,label=label)
    #     plt.ylabel('water content ' + label + '(kg/kg)')
    #     plt.legend()
    #     ax = plt.twinx()
    #     ax.plot(ar,int_dphi,color='black',label='PAZ')
    #     plt.ylabel('differential phase shift (mm)')
    #     plt.savefig(figpath + '/' + label + '_int_dphi_mor.png')
    #     plt.close()
    

    return int_dphi, int_snow, int_rain, int_graupel, int_ice


def scatter_plot(ROID_done_AR):
    """ function that makes an scatter plot showing the values of 
    integrated dphi vs integrated WC """
    
    int_WC = []; int_PRO = []; mp_scheme = []; p_s = []
    best_mp = []; best_intpro = []; best_intwc = []; best_psnow = []
    x_snow = []; x_snow_abs = []; cost = []; cost_best = []
    for roid in ROID_done_AR:
        try:
            if 'PAZ1' in roid:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
            else:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        except:
            df = lstsq(roid)
        
        int_snow = df['int_snow'].values
        int_dphi = df['int_dphi'].values
        int_WC = np.append(int_WC,int_snow)
        int_PRO = np.append(int_PRO,int_dphi)
        
        x_s = df['param_snow'].values
        x_snow = np.append(x_snow,x_s)
        x_s_abs = df['param_snow'][df['cost'] == np.nanmin(df['cost'])].values[0]
        x_snow_abs = np.append(x_snow_abs,x_s_abs)
        
        mp = df['mp'].values
        mp_scheme = np.append(mp_scheme,mp)
        p_snow = df['percen_snow'].values
        p_s = np.append(p_s,p_snow)
        
        c = df['cost'].values
        cost = np.append(cost,c)
        
        #best asbolute values
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values[0]
        best_mp = np.append(best_mp,mp_best)
        intpro_best = df[df['cost'] == np.nanmin(df['cost'])]['int_dphi'].values[0]
        best_intpro = np.append(best_intpro,intpro_best)
        intwc_best = df[df['cost'] == np.nanmin(df['cost'])]['int_snow'].values[0]
        best_intwc = np.append(best_intwc,intwc_best)
        psnow_best = df[df['cost'] == np.nanmin(df['cost'])]['percen_snow'].values[0]
        best_psnow = np.append(best_psnow,psnow_best)
        c_best = df[df['cost'] == np.nanmin(df['cost'])]['cost'].values[0]
        cost_best = np.append(cost_best,c_best)
    
    # Mapear formas ás categorías
    shape_map = {
        'Goddard': 'o',    # Círculo
        'Morrison': 's',    # Cadrado
        'Thompson': '^',    # Triángulo
        'WSM6': 'D'     # Diamante
        }   

    
    cost_norm = cost/np.linalg.norm(cost)
    s = np.array([1/(10*n) for n in (cost_norm)])
    s_god = s[mp_scheme == 'Goddard']
    s_tho = s[mp_scheme == 'Thompson']
    s_mor = s[mp_scheme == 'Morrison']
    s_ws6 = s[mp_scheme == 'WSM6']
    cost_norm_best = cost_best/np.linalg.norm(cost_best)
    s_best = np.array([1/(10*n) for n in (cost_norm_best)])
    
    min_, max_ = np.nanmin(p_s), np.nanmax(p_s)
    # Figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear o scatter plot, filtrando por cada categoría
    plt.scatter(int_WC[mp_scheme=='Goddard'],int_PRO[mp_scheme=='Goddard'],c=p_s[mp_scheme=='Goddard'],s=s_god,vmin=min_,vmax=max_,marker='s',label='Goddard',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='Thompson'],int_PRO[mp_scheme=='Thompson'],c=p_s[mp_scheme=='Thompson'],s=s_tho,vmin=min_,vmax=max_,marker='o',label='Thompson',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='WSM6'],int_PRO[mp_scheme=='WSM6'],c=p_s[mp_scheme=='WSM6'],s=s_ws6,vmin=min_,vmax=max_,marker='^',label='WSM6',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='Morrison'],int_PRO[mp_scheme=='Morrison'],c=p_s[mp_scheme=='Morrison'],s=s_mor,vmin=min_,vmax=max_,marker='*',label='Morrison',cmap='rainbow')
    plt.plot(best_intwc,0.1*best_intwc,'-',color='grey')
    
    # Engadir barra de cores
    plt.colorbar().set_label('snow percentage (%)',rotation=270,labelpad=20,fontsize=14)
    plt.xlim(np.nanmin(int_WC)-1000,np.nanmax(int_WC)+1000)
    plt.ylim(np.nanmin(int_PRO)-100,np.nanmax(int_PRO)+100)
 
    # Engadir etiquetas e lenda
    ax.set_xlabel('Integrated water content for snow (kg/kg)', fontsize=14)
    ax.set_ylabel('Integrated differential phase shift (mm)', fontsize=14)
    ax.legend(fontsize=14)
    
    # Mostrar o gráfico
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/' + 'scatter_plot.png')
    
    ################################################
    min_, max_ = np.nanmin(x_snow), np.nanmax(x_snow)
    # Figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear o scatter plot, filtrando por cada categoría
    plt.scatter(int_WC[mp_scheme=='Goddard'],int_PRO[mp_scheme=='Goddard'],c=x_snow[mp_scheme=='Goddard'],s=s_god,vmin=min_,vmax=max_,marker='s',label='Goddard',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='Thompson'],int_PRO[mp_scheme=='Thompson'],c=x_snow[mp_scheme=='Thompson'],s=s_tho,vmin=min_,vmax=max_,marker='o',label='Thompson',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='WSM6'],int_PRO[mp_scheme=='WSM6'],c=x_snow[mp_scheme=='WSM6'],s=s_ws6,vmin=min_,vmax=max_,marker='^',label='WSM6',cmap='rainbow')
    plt.scatter(int_WC[mp_scheme=='Morrison'],int_PRO[mp_scheme=='Morrison'],c=x_snow[mp_scheme=='Morrison'],s=s_mor,vmin=min_,vmax=max_,marker='*',label='Morrison',cmap='rainbow')
    plt.plot(best_intwc,0.1*best_intwc,'-',color='grey')

    # Engadir barra de cores
    plt.colorbar().set_label('x-parameter snow (%)',rotation=270,labelpad=20,fontsize=14)

    # Engadir etiquetas e lenda
    ax.set_xlabel('Integrated water content for snow (kg/kg)', fontsize=14)
    ax.set_ylabel('Integrated differential phase shift (mm)', fontsize=14)
    ax.legend(fontsize=14)
    plt.xlim(np.nanmin(int_WC)-1000,np.nanmax(int_WC)+1000)
    plt.ylim(np.nanmin(int_PRO)-100,np.nanmax(int_PRO)+100)
    # Mostrar o gráfico
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/' + 'scatter_plot_x.png')

    
    min_,max_ = np.nanmin(p_s), np.nanmax(p_s)
    #figure absolute cases
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(ROID_done_AR)):
        if best_mp[i] == 'Goddard':
            mark = 's'
        elif best_mp[i] == 'Thompson':
            mark = 'o'
        elif best_mp[i] == 'WSM6':
            mark = '^'
        elif best_mp[i] == 'Morrison':
            mark = '*'
        plt.scatter(best_intwc[i],best_intpro[i],c=best_psnow[i],s=s_best[i],vmin=min_,vmax=max_,marker=mark,cmap='rainbow')
    plt.plot(best_intwc,0.1*best_intwc,'-',color='grey')
    # Colorbar
    plt.colorbar().set_label('snow percentage (%)',rotation=270,labelpad=20,fontsize=14)
    plt.xlim(np.nanmin(int_WC)-1000,np.nanmax(int_WC)+1000)
    plt.ylim(np.nanmin(int_PRO)-100,np.nanmax(int_PRO)+100)
 
    # Labels & Legend
    ax.set_xlabel('Integrated water content for snow (kg/kg)', fontsize=14)
    ax.set_ylabel('Integrated differential phase shift (mm)', fontsize=14)
    plt.legend(fontsize=14)

    # Save Figure
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/' + 'scatter_plot_abs.png')
    
        
    min_,max_ = np.nanmin(x_snow), np.nanmax(x_snow)
    #figure absolute cases
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(ROID_done_AR)):
        if best_mp[i] == 'Goddard':
            mark = 's'
        elif best_mp[i] == 'Thompson':
            mark = 'o'
        elif best_mp[i] == 'WSM6':
            mark = '^'
        elif best_mp[i] == 'Morrison':
            mark = '*'
        plt.scatter(best_intwc[i],best_intpro[i],c=x_snow_abs[i],s=s_best[i],vmin=min_,vmax=max_,marker=mark,cmap='rainbow')
    plt.plot(best_intwc,0.1*best_intwc,'-',color='grey')
    # Colorbar
    plt.colorbar().set_label('x-parameter snow',rotation=270,labelpad=20,fontsize=14)
    plt.xlim(np.nanmin(int_WC)-1000,np.nanmax(int_WC)+1000)
    plt.ylim(np.nanmin(int_PRO)-100,np.nanmax(int_PRO)+100)
 
    # Labels & Legend
    ax.set_xlabel('Integrated water content for snow (kg/kg)', fontsize=14)
    ax.set_ylabel('Integrated differential phase shift (mm)', fontsize=14)
    ax.legend(fontsize=14)

    # Save Figure
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/' + 'scatter_plot_abs_x.png')
    

    return None

def hist_particle_ice(ROID_done_AR):
    """ function that makes an histogram of the best particles
    for snow """

    part_ice = []; param_ice = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        
        particle_ice = df['particle_ice'][df['cost'] == np.nanmin(df['cost'])]
        part_ice = np.append(part_ice,particle_ice)
        x_ice = df['param_ice'][df['cost'] == np.nanmin(df['cost'])]
        param_ice = np.append(param_ice,x_ice)
    
    # Ensure all particles are included
    all_particles = list(dict_coefs.keys())  # List of all possible snow particles
    ps_counts = Counter(part_ice)  # Count occurrences of each particle
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])

    # Assign zero frequency to missing particles
    for particle in all_particles:
        if particle not in ps_counts:
            ps_counts[particle] = 0

    # Convert `b_param_coef` into a dictionary mapping particles to their param values
    particle_to_param = {p: b_param_coef[i] for i, p in enumerate(all_particles)}

    # Sort particles based on their param_snow values from `b_param_coef`
    sorted_particles = sorted(all_particles, key=lambda p: particle_to_param[p])

    # Get ordered frequencies
    frequencies = [ps_counts[particle] for particle in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    # Plot histogram
    fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)

    # Main x-axis (Snow Particles)
    ax1.bar(sorted_particles, frequencies, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Best particles for ice', fontsize=14)
    ax1.set_ylabel('Number of AR', fontsize=14)
    ax1.set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)

    # Create secondary x-axis for param_snow
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Ensure both axes align
    ax2.set_xticks(ax1.get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param Ice Value', fontsize=14)

    # Save the figure
    plt.savefig('/media/antia/Elements/figures/AR/hist_best_ice_particles2.png')

    return None

def hist_particle_graupel(ROID_done_AR):
    """ function that makes an histogram of the best particles
    for snow """

    part_graupel = []; param_graupel = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        
        particle_graupel = df['particle_graupel'][df['cost'] == np.nanmin(df['cost'])]
        part_graupel = np.append(part_graupel,particle_graupel)
        x_graupel = df['param_graupel'][df['cost'] == np.nanmin(df['cost'])]
        param_graupel = np.append(param_graupel,x_graupel)
    
    # Ensure all particles are included
    all_particles = list(dict_coefs.keys())  # List of all possible snow particles
    ps_counts = Counter(part_graupel)  # Count occurrences of each particle
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])

    # Assign zero frequency to missing particles
    for particle in all_particles:
        if particle not in ps_counts:
            ps_counts[particle] = 0

    # Convert `b_param_coef` into a dictionary mapping particles to their param values
    particle_to_param = {p: b_param_coef[i] for i, p in enumerate(all_particles)}

    # Sort particles based on their param_snow values from `b_param_coef`
    sorted_particles = sorted(all_particles, key=lambda p: particle_to_param[p])

    # Get ordered frequencies
    frequencies = [ps_counts[particle] for particle in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    # Plot histogram
    fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)

    # Main x-axis (Snow Particles)
    ax1.bar(sorted_particles, frequencies, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Best particles for graupel', fontsize=14)
    ax1.set_ylabel('Number of AR', fontsize=14)
    ax1.set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)

    # Create secondary x-axis for param_snow
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Ensure both axes align
    ax2.set_xticks(ax1.get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param graupel Value', fontsize=14)

    # Save the figure
    plt.savefig('/media/antia/Elements/figures/AR/hist_best_graupel_particles2.png')

    return None

def hist_particle_snow(ROID_done_AR):
    """ function that makes an histogram of the best particles
    for snow """
    import matplotlib.patches as mpatches
    
    p_snow = []; part_snow = []; param_snow = []; cost = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        
        particle_snow = df['particle_snow'][df['cost'] == np.nanmin(df['cost'])]
        part_snow = np.append(part_snow,particle_snow)
        x_snow = df['param_snow'][df['cost'] == np.nanmin(df['cost'])]
        param_snow = np.append(param_snow,x_snow)
        c = df['cost'][df['cost']==np.nanmin(df['cost'])].values
        cost = np.append(cost,c)
        
        percen_snow = df['percen_snow'][df['cost'] == np.nanmin(df['cost'])]
        p_snow = np.append(p_snow,percen_snow)
        
    # Ensure all particles are included
    all_particles = list(dict_coefs.keys())  # List of all possible snow particles
    ps_counts = Counter(part_snow)  # Count occurrences of each particle
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])

    # Assign zero frequency to missing particles
    for particle in all_particles:
        if particle not in ps_counts:
            ps_counts[particle] = 0

    # Convert `b_param_coef` into a dictionary mapping particles to their param values
    particle_to_param = {p: b_param_coef[i] for i, p in enumerate(all_particles)}

    # Sort particles based on their param_snow values from `b_param_coef`
    sorted_particles = sorted(all_particles, key=lambda p: particle_to_param[p])

    # Get ordered frequencies
    frequencies = [ps_counts[particle] for particle in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    # Define cost threshold for highlighting
    cost_threshold = 10

    # Separate high and low cost cases
    # high_cost_indices = cost > cost_threshold
    # low_cost_indices = ~high_cost_indices
    high_percen_snow = p_snow > 75
    low_percen_snow = p_snow < 75

    # Count occurrences of each particle for low and high cost cases
    # low_cost_counts = Counter(part_snow[low_cost_indices])
    # high_cost_counts = Counter(part_snow[high_cost_indices])
    high_percen_counts = Counter(part_snow[high_percen_snow])
    low_percen_counts = Counter(part_snow[low_percen_snow])

    # Ensure all particles are included in both
    # low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
    # high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
    low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
    high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]

    # # Plot histogram with stacked bars
    # fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)

    # # Plot the low-cost bars first (skyblue)
    # bars = ax1.bar(sorted_particles, low_cost_freq, color='skyblue', alpha=0.7, label='Low cost function')

    # # Overlay the high-cost bars on top (red)
    # ax1.bar(sorted_particles, high_cost_freq, bottom=low_cost_freq, color='red', alpha=0.7, label='High cost function')
    
    # for i, bar in enumerate(bars):
    #     if high_percen_freq[i] > 0:  # If the particle appears in high-snow cases
    #         bar.set_edgecolor('black')  # Highlight edge
    #         bar.set_linewidth(2)
    
    
    # # Labels and x-axis formatting
    # ax1.set_xlabel('Best particles for snow', fontsize=14)
    # ax1.set_ylabel('Number of AR', fontsize=14)
    # ax1.set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    # ax1.set_ylim(0,10)
    # # ax1.set_title('north-east Pacific ARs',fontsize=14)
    # ax1.legend()

    # # Create secondary x-axis for param_snow
    # ax2 = ax1.twiny()
    # ax2.set_xlim(ax1.get_xlim())  # Align both axes
    # ax2.set_xticks(ax1.get_xticks())  # Match tick positions
    # ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    # ax2.set_xlabel('x-parameter for snow', fontsize=14)

    # import matplotlib.patches as mpatches  # Import patch for legend proxy

    # # Create proxy bars for legend
    # high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='snow percentage > 75%')
    # low_cost_patch = mpatches.Patch(color='skyblue', label='cost function < 10')
    # high_cost_patch = mpatches.Patch(color='red', label='cost function > 10')

    # # Add legend with all components
    # ax1.legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)


    # # Save the figure
    # plt.savefig('/media/antia/Elements/figures/AR/hist_best_snow_particles_highlighted.png')

    # plt.show()
    # Define cost thresholds
    cost_J1 = cost < 1
    cost_J5 = (cost >= 1) & (cost < 5)
    cost_J10 = (cost >= 5) & (cost <10)
    cost_high = cost >= 10# J >= 5

    # Count occurrences for each cost range
    J1_counts = Counter(part_snow[cost_J1])
    J5_counts = Counter(part_snow[cost_J5])
    J10_counts = Counter(part_snow[cost_J10])
    Jhigh_counts = Counter(part_snow[cost_high])

    # Ensure all particles are included
    J1_freq = [J1_counts[p] if p in J1_counts else 0 for p in sorted_particles]
    J5_freq = [J5_counts[p] if p in J5_counts else 0 for p in sorted_particles]
    J10_freq = [J10_counts[p] if p in J10_counts else 0 for p in sorted_particles]
    Jhigh_freq = [Jhigh_counts[p] if p in Jhigh_counts else 0 for p in sorted_particles]

    # Plot histogram with stacked bars
    fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)

    # Lightest shade for J < 1
    bars_J1 = ax1.bar(sorted_particles, J1_freq, color='lightblue', alpha=0.7, label='J < 1')

    # Medium shade for 1 ≤ J < 5
    bars_J5 = ax1.bar(sorted_particles, J5_freq, bottom=J1_freq, color='dodgerblue', alpha=0.7, label='1 ≤ J < 5')
    
    bars_J10 = ax1.bar(sorted_particles, J10_freq, bottom = np.array(J1_freq) + np.array(J5_freq),color='blue' ,alpha=0.7, label='5 ≤ J < 10')
    # Darkest shade for J ≥ 5
    bars_Jhigh = ax1.bar(sorted_particles, Jhigh_freq, bottom=np.array(J1_freq) + np.array(J5_freq) + np.array(J10_freq),
                     color='darkblue', alpha=0.7, label='J ≥ 10')

    # Highlight bars for high snow percentage
    for bars in [bars_J1, bars_J5, bars_J10,bars_Jhigh]:  # Iterate over all bars
        for i, bar in enumerate(bars):  # Loop through each bar individually
            if high_percen_freq[i] > 0:  
                bar.set_edgecolor('black')  
                bar.set_linewidth(2)

   

    # Labels and formatting
    ax1.set_xlabel('Best particles for snow', fontsize=14)
    ax1.set_ylabel('Number of AR', fontsize=14)
    ax1.set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    ax1.set_ylim(0, max(np.array(J1_freq) + np.array(J5_freq) + np.array(J10_freq) + np.array(Jhigh_freq)) + 2)

    # Create secondary x-axis for param_snow
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  
    ax2.set_xticks(ax1.get_xticks())  
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('x-parameter for snow', fontsize=14)

    # Create legend patches
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Snow % > 75%')
    J1_patch = mpatches.Patch(color='lightblue', label='J < 1')
    J5_patch = mpatches.Patch(color='dodgerblue', label='1 ≤ J < 5')
    J10_patch = mpatches.Patch(color= 'blue', label = '5 ≤ J < 10')
    Jhigh_patch = mpatches.Patch(color='darkblue', label='J ≥ 10')

    # Add legend
    ax1.legend(handles=[J1_patch, J5_patch,J10_patch, Jhigh_patch, high_snow_patch], loc='upper right', fontsize=12)

    # Save the figure
    plt.savefig('/media/antia/Elements/figures/AR/hist_best_snow_particles_J_gradient3.png')
    plt.show()
    

    return None

def hist_cost_function(ROID_done_AR):
    """ function that makes an histogram of the cost
    function for the best abs cases """
    
    cost = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        
        c = df['cost'][df['cost'] == np.nanmin(df['cost'])].values[0]
        cost = np.append(cost,c)
        
    bins = len(ROID_done_AR)
    plt.figure(figsize=(8,8))
    plt.hist(cost,bins=bins,color='orange',alpha=.7)
    plt.xlabel('Cost function')
    plt.ylabel('nº AR')
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_cost.png')

    return None

def hist_particles_mp(ROID_done_AR):
    """ function that makes an histogram of the different
    particles for snow depending on the mp scheme """
    
    p_snow = []; mp =[]; cost = []; part_snow = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            try:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
            except:
                lstsq(roid)
        else:
            try:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
            except:
                lstsq(roid)
        
        particle_snow = df['particle_snow'].values
        part_snow = np.append(part_snow,particle_snow)
        mp_scheme = df['mp'].values
        mp = np.append(mp,mp_scheme)
        c = df['cost'].values
        cost = np.append(cost,c)
        percen_snow = df['percen_snow'].values
        p_snow = np.append(p_snow,percen_snow)
    
    
    # Define unique microphysics schemes
    microphysics_schemes = np.unique(mp)
   
    # Ensure all particles are included
    all_particles = list(dict_coefs.keys())  # List of all possible snow particles
    ps_counts = Counter(part_snow)  # Count occurrences of each particle
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])

    # Assign zero frequency to missing particles
    for particle in all_particles:
        if particle not in ps_counts:
            ps_counts[particle] = 0

    # Convert `b_param_coef` into a dictionary mapping particles to their param values
    particle_to_param = {p: b_param_coef[i] for i, p in enumerate(all_particles)}

    # Sort particles based on their param_snow values from `b_param_coef`
    sorted_particles = sorted(all_particles, key=lambda p: particle_to_param[p])

    # Get ordered frequencies
    frequencies = [ps_counts[particle] for particle in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]
    ####################################################

    fig, axes = plt.subplots(2, 2, figsize=(19, 13))
    axes = axes.flatten()
    #########################################
    # Select particles for the current microphysics scheme
    selected_particles = part_snow[mp == 'Goddard']
    cost_god = cost[mp =='Goddard']
    p_snow_god = p_snow[mp == 'Goddard']

    cost_treshold = 10
    high_cost_indices = cost_god > cost_treshold
    low_cost_indices = ~high_cost_indices
    high_percen_snow = p_snow_god > 75
    low_percen_snow = p_snow_god < 75
    
    low_cost_counts = Counter(selected_particles[low_cost_indices])
    high_cost_counts = Counter(selected_particles[high_cost_indices])
    high_percen_counts = Counter(selected_particles[high_percen_snow])
    low_percen_counts = Counter(selected_particles[low_percen_snow])
   
    low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
    high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
    low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
    high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]
    
    # Count occurrences of each particle
    ps_counts = Counter(selected_particles)

    # Ensure all particles are included (fill missing with zero)
    frequencies = [ps_counts.get(p, 0) for p in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    # selected_particles = p_snow[mp == microphysics_schemes[0]]
    bars1 = axes[0].bar(sorted_particles,low_cost_freq,color='green', alpha=0.7,label='Goddard')
    # Overlay the high-cost bars on top (red)
    bars2 = axes[0].bar(sorted_particles, high_cost_freq, bottom=low_cost_freq, color='red', alpha=0.7, label='High cost function')
 
    for bars in [bars1,bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)

    
    axes[0].set_xticklabels(sorted_particles,rotation=45,ha='right',fontsize=14)
    axes[0].set_xlabel('x-parameter for snow',fontsize=14)
    axes[0].set_ylabel('nº AR',fontsize=14)
    axes[0].tick_params(labelsize=14)
    axes[0].set_xticks(range(len(sorted_particles)))
    axes[0].set_ylim(0,16)
    axes[0].set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    axes[0].legend(fontsize=14)
    axes[0].set_title('Goddard')
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())  # Ensure both axes align
    ax2.set_xticks(axes[0].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param Snow Value', fontsize=14)

    import matplotlib.patches as mpatches  # Import patch for legend proxy

    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='snow percentage > 75%')
    low_cost_patch = mpatches.Patch(color='green', label='cost function < 10')
    high_cost_patch = mpatches.Patch(color='red', label='cost function > 10')
     
    # Add legend with all components
    axes[0].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)


    #########################################
    # Select particles for the current microphysics scheme
    selected_particles = p_snow[mp == 'Morrison']
    # Select particles for the current microphysics scheme
    selected_particles = part_snow[mp == 'Morrison']
    cost_god = cost[mp =='Morrison']
    p_snow_god = p_snow[mp == 'Morrison']

    cost_treshold = 10
    high_cost_indices = cost_god > cost_treshold
    low_cost_indices = ~high_cost_indices
    high_percen_snow = p_snow_god > 75
    low_percen_snow = p_snow_god < 75
    
    low_cost_counts = Counter(selected_particles[low_cost_indices])
    high_cost_counts = Counter(selected_particles[high_cost_indices])
    high_percen_counts = Counter(selected_particles[high_percen_snow])
    low_percen_counts = Counter(selected_particles[low_percen_snow])
   
    low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
    high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
    low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
    high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]
    
    # Count occurrences of each particle
    ps_counts = Counter(selected_particles)

    # Ensure all particles are included (fill missing with zero)
    frequencies = [ps_counts.get(p, 0) for p in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]
    
    # selected_particles = p_snow[mp == microphysics_schemes[0]]
    bars1 = axes[1].bar(sorted_particles,low_cost_freq,color='blue', alpha=0.7,label='Morrison')
    bars2 = axes[1].bar(sorted_particles, high_cost_freq, bottom=low_cost_freq, color='red', alpha=0.7, label='High cost function')
    
    for bars in [bars1,bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)
    
    axes[1].set_xticklabels(sorted_particles,rotation=45,ha='right',fontsize=14)
    axes[1].set_xlabel('x-parameter for snow',fontsize=14)
    axes[1].set_ylabel('nº AR',fontsize=14)
    axes[1].tick_params(labelsize=14)
    axes[1].set_xticks(range(len(sorted_particles)))
    axes[1].set_ylim(0,16)
    axes[1].set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    axes[1].legend(fontsize=14)
    axes[1].set_title('Morrison')
    ax2 = axes[1].twiny()
    ax2.set_xlim(axes[1].get_xlim())  # Ensure both axes align
    ax2.set_xticks(axes[1].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param Snow Value', fontsize=14)
    import matplotlib.patches as mpatches  # Import patch for legend proxy

    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='snow percentage > 75%')
    low_cost_patch = mpatches.Patch(color='blue', label='cost function < 10')
    high_cost_patch = mpatches.Patch(color='red', label='cost function > 10')
     
    # Add legend with all components
    axes[1].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)


    #########################################
    # Select particles for the current microphysics scheme
    selected_particles = p_snow[mp == 'Thompson']
    # Select particles for the current microphysics scheme
    selected_particles = part_snow[mp == 'Thompson']
    cost_god = cost[mp =='Thompson']
    p_snow_god = p_snow[mp == 'Thompson']

    cost_treshold = 10
    high_cost_indices = cost_god > cost_treshold
    low_cost_indices = ~high_cost_indices
    high_percen_snow = p_snow_god > 75
    low_percen_snow = p_snow_god < 75
    
    low_cost_counts = Counter(selected_particles[low_cost_indices])
    high_cost_counts = Counter(selected_particles[high_cost_indices])
    high_percen_counts = Counter(selected_particles[high_percen_snow])
    low_percen_counts = Counter(selected_particles[low_percen_snow])
   
    low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
    high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
    low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
    high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]
    
    # Count occurrences of each particle
    ps_counts = Counter(selected_particles)

    # Ensure all particles are included (fill missing with zero)
    frequencies = [ps_counts.get(p, 0) for p in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]
    
    # selected_particles = p_snow[mp == microphysics_schemes[0]]
    bars1 = axes[2].bar(sorted_particles,low_cost_freq,color='orange', alpha=0.7,label='Thompson')
    bars2 = axes[2].bar(sorted_particles, high_cost_freq, bottom=low_cost_freq, color='red', alpha=0.7, label='High cost function')
    
    for bars in [bars1,bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)
                
    # Count occurrences of each particle
    ps_counts = Counter(selected_particles)

    # Ensure all particles are included (fill missing with zero)
    frequencies = [ps_counts.get(p, 0) for p in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    axes[2].set_xticklabels(sorted_particles,rotation=45,ha='right',fontsize=14)
    axes[2].set_xlabel('x-parameter for snow',fontsize=14)
    axes[2].set_ylabel('nº AR',fontsize=14)
    axes[2].tick_params(labelsize=14)
    axes[2].set_xticks(range(len(sorted_particles)))
    axes[2].set_ylim(0,16)
    axes[2].set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    axes[2].legend(fontsize=14)
    axes[2].set_title('Thompson')
    ax2 = axes[2].twiny()
    ax2.set_xlim(axes[2].get_xlim())  # Ensure both axes align
    ax2.set_xticks(axes[2].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param Snow Value', fontsize=14)

    import matplotlib.patches as mpatches  # Import patch for legend proxy

    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='snow percentage > 75%')
    low_cost_patch = mpatches.Patch(color='orange', label='cost function < 10')
    high_cost_patch = mpatches.Patch(color='red', label='cost function > 10')

    # Add legend with all components
    axes[2].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)


    #########################################
    # Select particles for the current microphysics scheme
    selected_particles = p_snow[mp == 'WSM6']
    # Select particles for the current microphysics scheme
    selected_particles = part_snow[mp == 'WSM6']
    cost_god = cost[mp =='WSM6']
    p_snow_god = p_snow[mp == 'WSM6']

    cost_treshold = 10
    high_cost_indices = cost_god > cost_treshold
    low_cost_indices = ~high_cost_indices
    high_percen_snow = p_snow_god > 75
    low_percen_snow = p_snow_god < 75
    
    low_cost_counts = Counter(selected_particles[low_cost_indices])
    high_cost_counts = Counter(selected_particles[high_cost_indices])
    high_percen_counts = Counter(selected_particles[high_percen_snow])
    low_percen_counts = Counter(selected_particles[low_percen_snow])
   
    low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
    high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
    low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
    high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]
    
    # selected_particles = p_snow[mp == microphysics_schemes[0]]
    bars1 = axes[3].bar(sorted_particles,low_cost_freq,color='violet', alpha=0.7,label='WSM6')
    bars2 = axes[3].bar(sorted_particles, high_cost_freq, bottom=low_cost_freq, color='red', alpha=0.7, label='High cost function')
    
    for bars in [bars1,bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)

    # Count occurrences of each particle
    ps_counts = Counter(selected_particles)

    # Ensure all particles are included (fill missing with zero)
    frequencies = [ps_counts.get(p, 0) for p in sorted_particles]

    # Get corresponding param_snow values from `b_param_coef`
    sorted_param_values = [particle_to_param[p] for p in sorted_particles]

    axes[3].set_xticklabels(sorted_particles,rotation=45,ha='right',fontsize=14)
    axes[3].set_xlabel('x-parameter for snow',fontsize=14)
    axes[3].set_ylabel('nº AR',fontsize=14)
    axes[3].tick_params(labelsize=14)
    axes[3].set_xticks(range(len(sorted_particles)))
    axes[3].set_ylim(0,16)
    axes[3].set_xticklabels(sorted_particles, rotation=45, ha='right', fontsize=14)
    axes[3].legend(fontsize=14)
    axes[3].set_title('WSM6')
    ax2 = axes[3].twiny()
    ax2.set_xlim(axes[3].get_xlim())  # Ensure both axes align
    ax2.set_xticks(axes[3].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('Param Snow Value', fontsize=14)

    import matplotlib.patches as mpatches  # Import patch for legend proxy

    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='snow percentage > 75%')
    low_cost_patch = mpatches.Patch(color='violet', label='cost function < 10')
    high_cost_patch = mpatches.Patch(color='red', label='cost function > 10')

    # Add legend with all components
    axes[3].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_best_snow_particles_' + str(microphysics_schemes[3]) + '2.png')
    
    return None

def hist_mp_best_figure():
    

    import matplotlib.gridspec as gridspec
    
    mp_schemes = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values[0]
        if mp_best == 'd_w':
            mp_best = 'Goddard'
        mp_schemes = np.append(mp_schemes,mp_best)
    
    #counts the occurrences of each scheme
    mp_counts_total = Counter(mp_schemes)
    
    mp_schemes = []
    for roid in ROID_done_Atlantic:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values[0]
        if mp_best == 'd_w':
            mp_best = 'Goddard'
        mp_schemes = np.append(mp_schemes,mp_best)
    
    #counts the occurrences of each scheme
    mp_counts_NA = Counter(mp_schemes)
 
    
    mp_schemes = []
    for roid in ROID_done_EastPacific:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values[0]
        if mp_best == 'd_w':
            mp_best = 'Goddard'
        mp_schemes = np.append(mp_schemes,mp_best)
    
    #counts the occurrences of each scheme
    mp_counts_EP = Counter(mp_schemes)

    
    #separate the keys and the values for plotting
    schemes = list(mp_counts_total.keys())
    frequencies = list(mp_counts_total.values())
    
    # Create figure and gridspec
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Left large subplot (spanning both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.bar(schemes,frequencies,color='skyblue',alpha=0.7)
    plt.xlabel('Microphysics schemes',fontsize=14)
    plt.ylabel('number of AR',fontsize=14)
    ax1.set_yticks(range(0, 19 + 1,3))
    plt.xticks(fontsize=14)
    ax1.text(-0.5,18.3,'(a)',fontsize=14)
    plt.title('Total AR',fontsize=14)
    
    #separate the keys and the values for plotting
    # schemes = list(mp_counts_NA.keys())

    frequencies = [1,3,5,11]
    # f = [9,5,2,4]
    
    # Left large subplot (spanning both rows)
    ax1 = fig.add_subplot(gs[0,1])
    ax1.bar(schemes,frequencies,color='yellowgreen',alpha=0.7)
    plt.xlabel('Microphysics schemes',fontsize=14)
    plt.ylabel('number of AR',fontsize=14)
    plt.xticks(fontsize=14)
    ax1.set_yticks(range(0, 19 + 1,3))
    ax1.text(-0.5,17,'(b)',fontsize=14)
    
    plt.title('North Atlantic AR', fontsize=14)
    
    #separate the keys and the values for plotting
    # schemes = list(mp_counts_EP.keys())
    frequencies = [2,3,6,6]
    # f = [8,6,3,0]
    
    # Left large subplot (spanning both rows)
    ax1 = fig.add_subplot(gs[1,1])
    ax1.bar(schemes,frequencies,color='navajowhite',alpha=0.7)
    plt.xlabel('Microphysics schemes',fontsize=14)
    plt.ylabel('number of AR',fontsize=14)
    plt.xticks(fontsize=14)
    ax1.set_yticks(range(0, 19 + 1,3))
    ax1.text(-0.5,17,'(c)',fontsize=14)
    
    plt.title('East Pacific AR', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR' + '/' + 'hist_mp_best_paper3.png')

    return None

def hist_cost_mp():
    
    cost = []; mp_schemes = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
    
        c = df['cost'][df['cost'] == np.nanmin(df['cost'])].values
        # c = df['cost'].values
        cost = np.append(cost,c)
        mp_best = df[df['cost'] == np.nanmin(df['cost'])]['mp'].values
        # mp_best = df['mp'].values
        mp_schemes = np.append(mp_schemes,mp_best)
    
    # Convert MP schemes to numerical indices for plotting
    mp_unique = np.unique(mp_schemes)
    x_indices = np.array([np.where(mp_unique == scheme)[0][0] for scheme in mp_schemes])

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_indices, cost, alpha=0.7, color='blue')
    
    # Format x-axis
    plt.xticks(ticks=range(len(mp_unique)), labels=mp_unique, rotation=45)
    plt.xlabel("Microphysics Schemes")
    plt.ylabel("Cost Function Value")
    plt.title("Cost Function Distribution by MP Scheme")

    # Improve spacing
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_distrib_cost_mp_total.png')
    
    import seaborn as sns
    
    df = pd.DataFrame({"MP Scheme": mp_schemes, "Cost Function": cost})

    # Plot
    plt.figure(figsize=(8, 6))
    sns.stripplot(x="MP Scheme", y="Cost Function", data=df, jitter=True, size=8, alpha=0.7)

    # Formatting
    plt.xlabel("Microphysics Schemes")
    plt.ylabel("Cost Function Value")
    plt.title("Cost Function Distribution by MP Scheme")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig('/media/antia/Elements/figures/AR/' + 'hist_distrib_cost_mp2_total.png')
    
    
    return None

toDO_Antia =  ['PAZ1.2021.026.04.03.G18', 
 'FM170.2023.223.08.57.G05',  
 'PAZ1.2022.063.17.04.G27',  'PAZ1.2019.233.03.01.G08', 
 'PAZ1.2023.006.15.28.G25',
'PAZ1.2023.001.12.07.G13',  'PAZ1.2019.358.11.45.G07',
'PAZ1.2019.265.11.37.G18', 'PAZ1.2020.304.21.30.G20',
'PAZ1.2021.089.10.09.G19', 'PAZ1.2021.247.10.34.G32',
'PAZ1.2022.030.09.09.G27', 'PAZ1.2018.239.03.26.G08',
 'FM170.2023.279.00.45.E05', 'FM166.2023.141.14.45.C21']

down_ERA5 = ['FM170.2023.223.08.57.G05',
           'PAZ1.2019.265.11.37.G18', 
'PAZ1.2021.089.10.09.G19', 'PAZ1.2021.247.10.34.G32',
'PAZ1.2022.030.09.09.G27', 'PAZ1.2018.239.03.26.G08',
 'FM170.2023.279.00.45.E05']

# cases_done =['FM166.2023.140.21.22.R07', 'FM166.2023.186.00.15.C20',
#        'FM167.2023.139.08.43.E04', 'FM167.2023.203.10.18.C29',
#        'FM170.2023.139.09.10.E04', 'PAZ1.2019.028.11.45.G30'
#        'PAZ1.2019.239.22.14.G20',
       
#        'PAZ1.2020.125.20.46.G30', 'PAZ1.2020.284.19.27.G12',
#        'PAZ1.2020.337.19.58.G10', 'PAZ1.2020.355.15.11.G28',
#        'PAZ1.2020.355.18.18.G25', 'PAZ1.2020.357.20.34.G32',
#        'PAZ1.2021.010.03.58.G13', 'PAZ1.2021.014.15.53.G12',
#        'PAZ1.2021.018.03.16.G13', 'PAZ1.2021.068.16.10.G29',
#        'PAZ1.2021.071.18.28.G27', 'PAZ1.2021.083.05.05.G25',
#        'PAZ1.2021.129.17.13.G16', 'PAZ1.2021.293.06.27.G05',
#        'PAZ1.2021.316.21.52.G08', 'PAZ1.2021.328.11.01.G09',
#        'PAZ1.2021.351.17.01.G25', 'PAZ1.2022.003.03.51.G15',
#        'PAZ1.2022.063.17.05.G27', 'PAZ1.2022.065.18.05.G08',
#        'PAZ1.2022.071.06.24.G32', 'PAZ1.2022.074.17.04.G27',
#        'PAZ1.2022.291.20.37.G16', 
#        'PAZ1.2023.038.10.59.G19', 'PAZ1.2023.089.07.24.G01']


ROID_done_EastPacific =['FM166.2023.140.21.22.R07', 'FM166.2023.239.08.03.R02',
       'PAZ1.2018.239.03.26.G08', 'PAZ1.2019.145.17.38.G11',
       'PAZ1.2019.228.19.01.G02', 'PAZ1.2020.290.17.57.G24',
       'PAZ1.2020.356.05.00.G13', 'PAZ1.2021.014.15.53.G12',
       'PAZ1.2021.164.03.59.G04', 'PAZ1.2021.182.03.34.G27',
       'PAZ1.2022.063.17.05.G27', 'PAZ1.2022.122.01.17.G25',
       'PAZ1.2022.311.05.29.G13', 'PAZ1.2023.165.04.07.G08',
       'PAZ1.2023.289.01.41.G17','FM170.2023.223.08.57.G05',
       'PAZ1.2020.355.18.18.G25']


ROID_done_Atlantic = ['FM167.2023.160.13.53.R15', 'PAZ1.2018.319.09.17.G32',
        'PAZ1.2018.234.06.45.G06',
       'PAZ1.2019.028.11.45.G30', 'PAZ1.2019.142.10.32.G02',
       'PAZ1.2019.286.10.17.G32', 'PAZ1.2019.359.11.29.G07',
       'PAZ1.2020.125.20.46.G30', 'PAZ1.2020.284.19.27.G12',
       'PAZ1.2020.337.19.58.G10', 'PAZ1.2020.357.20.34.G32',
       'PAZ1.2021.129.17.13.G16', 'PAZ1.2021.316.21.52.G08',
       'PAZ1.2021.328.11.01.G09', 'PAZ1.2022.291.20.37.G16',
       'PAZ1.2023.038.10.59.G19','PAZ1.2023.363.10.52.G14',
       'PAZ1.2023.089.07.24.G01','PAZ1.2021.034.10.07.G08',
      'FM167.2023.155.12.21.G23' ]


ROID_done_AR = np.append(ROID_done_EastPacific,ROID_done_Atlantic)


# cases_done_NA_Antia = ['PAZ1.2019.028.11.45.G30', 'PAZ1.2019.286.10.17.G32',
#        				'PAZ1.2019.359.11.29.G07', 'PAZ1.2020.125.20.46.G30',
#        				'PAZ1.2020.284.19.27.G12', 'PAZ1.2020.304.21.30.G20',
#        				'PAZ1.2020.337.19.58.G10', 'PAZ1.2020.357.20.34.G32',
#       					'PAZ1.2021.089.10.09.G19', 'PAZ1.2021.129.17.13.G16',
#        				'PAZ1.2021.247.10.34.G32', 'PAZ1.2021.316.21.52.G08',
#        				'PAZ1.2021.328.11.01.G09', 'PAZ1.2022.291.20.37.G16',
#        				'PAZ1.2023.038.10.59.G19', 'PAZ1.2023.089.07.24.G01',
#        				'PAZ1.2023.363.10.52.G14', 'FM167.2023.100.14.40.C35',
#        				'FM167.2023.160.13.53.R15']

# cases_done_NA_Ramon=['PAZ1.2018.205.12.01.G21', 'PAZ1.2018.234.06.45.G06',
#        				 'PAZ1.2022.030.09.09.G27']


casos_no = ['PAZ1.2020.304.21.30.G20','PAZ1.2020.355.18.18.G25',
            'PAZ1.2018.205.12.01.G21','PAZ1.2019.024.17.38.G32',
           'PAZ1.2023.324.15.42.G24', 'PAZ1.2023.001.12.07.G13',
           'PAZ1.2019.233.03.01.G08','PAZ1.2023.006.15.28.G25']


dict_coefs = {'HongPlate_Id9': [0.758,'snow_plate'] ,
              'LiuThinPlate_Id16' : [0.650,'snow_plate'],
              'HongColumn_Id7': [0.556,'graupel'],
              'IconCloudIce_Id27' : [0.606,'ice'],
              'LiuLongColumn_Id14': [0.425,'graupel'],
              'LiuSectorSnowflake_Id3': [0.666,'snow_plate'],
              'LiuThickPlate_Id15': [0.388,'snow_plate'],
              'LiuShortColumn_Id13': [0.269,'graupel'],
              'HongBulletRosette_Id5': [0.273,'bullet_rosette'],
              'EvansSnowAgg_Id1': [0.051,'snow_agg'],
              'HongBulletRosette_Id11':[0.272,'bullet_rosette'],
              'HongBulletRosette_Id10': [0.160,'bullet_rosette'],
              'HongAggregate_Id8': [0.096,'snow_agg'],
              'HexColAggCrystal_Id22': [0.025,'snow_agg'],
              'HexPlaAggCrystal_Id20': [0.027,'snow_agg'],
              'HexColAggCrystal_Id18': [0.017,'snow_agg'],
              'LiuBlockColumn_Id12': [0.072,'graupel'],
              'HongBulletRosette_Id2': [0.086,'bullet_rosette'],
              'IconSnow_Id28': [0.028,'snow_agg'],
              'HexPlaAggCrystal_Id19': [0.017,'snow_agg'],
              'GemSnow_Id32': [0.008,'snow_agg'],
              'HexColAggCrystal_Id21': [0.017,'ice'],
              'HexColAggCrystal_Id17': [0.010,'snow_agg'],
              'HongBulletRosette_Id4': [0.016,'bullet_rosette'],
              'HongBulletRosette_Id6': [0.012,'bullet_rosette'],
              'Rosette_Id36': [0.017,'rosette'],
              'GemCloudIce_Id31': [0.019,'ice'],
              'GemHail_Id34': [0.003,'hail'],
              'IconHail_Id30':[0.008,'hail']}


plist={'snow_agg':['IconSnow_Id28', 'GemSnow_Id32', 'EvansSnowAgg_Id1', 
               'HongAggregate_Id8', 'TyynelaFernDendAgg_Id26',
               'HexPlaAggCrystal_Id40','HexColAggCrystal_Id22',
               'HexPlaAggCrystal_Id20','HexPlaAggCrystal_Id19',
               'HexColAggCrystal_Id17','HexColAggCrystal_Id18',],
           'snow_plate':[
               'LiuThickPlate_Id15','LiuThinPlate_Id16','HongPlate_Id9',
               'LiuSectorSnowflake_Id3'],
       'hail':['GemHail_Id34', 'IconHail_Id30'],
       'ice':['GemCloudIce_Id31', 'IconCloudIce_Id27',
                    'HexColAggCrystal_Id21',],
       'graupel':['LiuBlockColumn_Id12','LiuShortColumn_Id13',
                 'LiuLongColumn_Id14','HongColumn_Id7']}

def INTWC(roid,wrfout):

    srcpath = '/media/antia/Elements/data/collocations'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
           ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])

    ht = np.nanmin(ray['h'],axis=1)
    dphi_PAZ = get_dphaseCal(roid)

    ray['dist'][np.isnan(ray['dist'])] = 0
    if 'PAZ1' in roid:
        interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()
    else:
        interp = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'interp_' + wrfout[-32:-29] + '.npy',allow_pickle=True).item()

    interp['snow'][np.isnan(interp['snow'])] = 0
    wc_snow = np.trapz(interp['snow'],dx=ray['dist'],axis=1) #integrated WC for an specific WC

    interp['rain'][np.isnan(interp['rain'])] = 0
    wc_rain = np.trapz(interp['rain'],dx=ray['dist'],axis=1)

    interp['graupel'][np.isnan(interp['graupel'])] = 0
    wc_graupel = np.trapz(interp['graupel'],dx=ray['dist'],axis=1)

    interp['ice'][np.isnan(interp['ice'])] = 0
    wc_ice = np.trapz(interp['ice'],dx=ray['dist'],axis=1)
    
    first_valid_index = np.argmax(~np.isnan(dphi_PAZ[0:120]))
    dphase_ = dphi_PAZ[0:120][first_valid_index:]
    wc_rain = np.flip(wc_rain)[0:120][first_valid_index:]
    wc_ice = np.flip(wc_ice)[0:120][first_valid_index:]
    wc_snow = np.flip(wc_snow)[0:120][first_valid_index:]
    wc_graupel = np.flip(wc_graupel)[0:120][first_valid_index:]
    ht = np.flip(ht)[0:120][first_valid_index:]
    
    return wc_rain, ht, wc_ice, wc_snow, wc_graupel, dphase_

def figure_intWC_paper():
    figpath = '/media/antia/Elements/figures/AR'
  
    roid1 = 'PAZ1.2019.142.10.32.G02'
    wrfout_god1, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid1)
    roid2 = 'PAZ1.2023.038.10.59.G19'
    wrfout_god, wrfout_mor2, wrfout_tho, wrfout_ws6 = get_wrfout(roid2)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    wc_rain1, ht1, wc_ice1, wc_snow1, wc_graupel1, dphi_PAZ1 = INTWC(roid1, wrfout_god1)
    wc_rain2, ht2, wc_ice2, wc_snow2, wc_graupel2, dphi_PAZ2 = INTWC(roid2, wrfout_mor2)
    axes = axes.flatten()
    
    im1 = axes[0].plot((wc_rain1),(ht1),color='royalblue',label='rain')
    im2 = axes[0].plot((wc_snow1),(ht1),color='orange',label='snow')
    im3 = axes[0].plot((wc_graupel1),(ht1),color='yellowgreen',label='graupel')
    im4 = axes[0].plot((wc_ice1),(ht1),color='darkblue',label='ice')
    axes[0].set_ylabel('Height (km)',fontsize=16)
    axes[0].set_xlabel('Integrated Water Content ($kg·m^{-2}$)',fontsize=16)
    axes[0].text(95,-1.5,'(a)',fontsize=16)
    axes[0].tick_params(labelsize=16)
    
    twin = axes[0].twiny()
    im5 = twin.plot(dphi_PAZ1,(ht1),color='black',label = 'PRO')
    plt.xticks(fontsize=16)
    # Combine and create a single legend
    lbls = [l.get_label() for l in (im1+im2+im3+im4+im5)]
    plt.ylim(0,12.3)
    plt.legend(im1 + im2 + im3 + im4 + im5, lbls, fontsize=16)
    plt.xlabel('Differential phase shift (mm)',fontsize=16)
    plt.ylabel('Height (km)',fontsize=16)
    
    im1 = axes[1].plot((wc_rain2),(ht2),color='royalblue',label='rain')
    im2 = axes[1].plot((wc_snow2),(ht2),color='orange',label='snow')
    im3 = axes[1].plot((wc_graupel2),(ht2),color='yellowgreen',label='graupel')
    im4 = axes[1].plot((wc_ice2),(ht2),color='darkblue',label='ice')
    axes[1].set_ylabel('Height (km)',fontsize=16)
    axes[1].set_xlabel('Integrated Water Content ($kg·m^{-2}$)',fontsize=16)
    axes[1].text(170,-1.5,'(b)',fontsize=16)
    axes[1].tick_params(labelsize=16)
    
    twin = axes[1].twiny()
    im5 = twin.plot(dphi_PAZ2,(ht2),color='black',label = 'PRO')
    plt.xticks(fontsize=16)
    plt.ylim(0,12.3)
    plt.xlabel('Differential phase shift (mm)',fontsize=16)
    plt.ylabel('Height (km)',fontsize=16)
    plt.tight_layout()
    
    if 'PAZ1' in roid1:
        plt.savefig(figpath + '/' + 'integrated_WC' + '_' + wrfout_god1[-32:-29] + '.png')
    else:
        plt.savefig(figpath + '/' + 'integrated_WC' + '_' + wrfout_mor2[-32:-29] + '.png')

    return None

def LSTSQ(roid):
    
    figpath = '/media/antia/Elements/figures/AR'
    srcpath = '/media/antia/Elements/data/collocations'
    path = '/media/antia/Elements/data/collocations/iceCal'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        try:
            h_flag = get_hflag(roid)[0][0]
            if roid == 'PAZ1.2023.089.07.24.G01':
                h_flag = 1.5
            if roid == 'PAZ1.2023.363.10.52.G14':
                h_flag= 1.

        # h_flag = h_flag.values[0][0]+1
        except:
            h_flag = 0
            if roid == 'PAZ1.2020.356.05.00.G13':
                h_flag = 2.
            elif roid == 'PAZ1.2021.164.03.59.G04':
                h_flag = 2.2
            elif roid == 'PAZ1.2021.182.03.34.G27':
                h_flag = 2.
            elif roid ==  'PAZ1.2021.129.17.13.G16':
                h_flag= 2.
            elif roid == 'PAZ1.2020.355.18.18.G25':
                h_flag = 0.8
        # elif roid == 'PAZ1.2020.125.20.46.G30':
        #     h_flag = 2.5
        dphase = get_dphaseCal(roid)
        # dphase = np.array(file_cal.variables['dph_smooth_grid'][:],dtype=float)
        try:
            fname_col = glob.glob(path + '/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCal_' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_col[0])
            sigma =  np.array(file_cal.variables['dph_smooth_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            
            sigma1 = std_dphi(roid)/np.sqrt(50)
            std1 = 1/sigma1
            
        except:
            sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
        
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
  
        try:
            h_flag = file.height_flag_smooth
            sigma =  np.array(file.variables['dph_smooth_L2_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            dphase = get_dphaseCal(roid)
        except:
            fname_cal = glob.glob(srcpath + '/' + 'Spire/' + 'iceCal*' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_cal[0])
            sigma = np.array(file_cal.variables["dph_smooth_std_lin"][:],dtype=float)/np.sqrt(50)
            h_flag = 1.
            # sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            # dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
            dphase = get_dphaseCal(roid)
        
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                      ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    h = np.nanmin(ray['h'],axis=1)
    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
    wrfouts = [wrfout_god,wrfout_mor,wrfout_tho,wrfout_ws6]

    #coeficients from the dict of ARTS
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])
    particle_habits = list(dict_coefs.keys())
    type_particles = np.array([value[1] for value in dict_coefs.values()])

    mp = ['Goddard','Morrison','Thompson','WSM6']
    color = ['green','blue','red','violet']
    x_s = []; x_i = []; x_g = []; dphi_simu = []; cost_ = []
    cc_ = []; std_stats = []; mean_ = []; particle_snow = []; type_snow = []
    particle_ice = []; type_ice = []; particle_graupel = []; type_graupel = []
    int_dphi_pro = []; int_snow = []; mp_s = []; percen_snow = []
    res = []
    #height flag
    dphase[0:220][np.flip(h) < h_flag] = np.nan
    dphase_fig = dphase[0:220]
    dphase_ = dphase.copy()
    dphase_[0:220][np.flip(h) < h_flag] = np.nan
    first_valid_index = np.argmax(~np.isnan(std[0:120]))
   
    std = std[0:120][first_valid_index:] 
    dphase_ = dphase_[0:120][first_valid_index:]
    sigma = sigma[0:120][first_valid_index:]
    std[np.isnan(std)] = 0
    sigma[np.isnan(sigma)] = 0
    dphase_[np.isnan(dphase_)] = 0
    for i in range(len(wrfouts)):
        wc = []
        df_params, i_iwc, sim_dphi_ = parameters(roid, wrfouts[i])
        mp_ = wrfouts[i][-32:-29] #microphysics scheme
        if mp_ == 'god':
            mp__ = 'Goddard'
        elif mp_ == 'mor':
            mp__ = 'Morrison'
        elif mp_ == 'tho':
            mp__ = 'Thompson'
        elif mp_ == 'ws6':
            mp__ = 'WSM6'
        mp_s = np.append(mp_s,mp__)
        i_iwc['snow'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['snow'] = i_iwc['snow'][0:120][first_valid_index:]
        i_iwc['snow'][np.isnan(i_iwc['snow'])] = 0
        i_iwc['ice'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['ice'] = i_iwc['ice'][0:120][first_valid_index:]
        i_iwc['ice'][np.isnan(i_iwc['ice'])] = 0
        i_iwc['graupel'][np.flip(h)[0:200] < h_flag] = np.nan
        i_iwc['graupel'] = i_iwc['graupel'][0:120][first_valid_index:]
        i_iwc['graupel'][np.isnan(i_iwc['graupel'])] = 0
        wc = np.append(wc,i_iwc['snow'])
        wc = np.append(wc,i_iwc['ice'])
        wc = np.append(wc,i_iwc['graupel'])
        wc = np.reshape(wc,(3,len(i_iwc['snow'])))
        i_iwc['rain'][np.flip(h)[0:200]<h_flag] = np.nan
        i_iwc['rain'] = i_iwc['rain'][0:120][first_valid_index:]
        i_iwc['rain'][np.isnan(i_iwc['rain'])] = 0
   
        # wc_filtered = wc[:, valid_idx]  # Keep only valid columns
    
        x = scipy.optimize.lsq_linear(np.diag(std) @ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([0.008,0.012,0.072],[0.758,0.606,0.556]))['x']
        cost = scipy.optimize.lsq_linear( np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([0.008,0.012,0.072],[0.758,0.606,0.556]))['cost']
        residuals = scipy.optimize.lsq_linear(np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([0.008,0.012,0.072],[0.758,0.606,0.556]))['fun']
 
        res = np.append(res,residuals)
        x_snow = x[0]
        x_ice = x[1]
        x_graupel = x[2]
        x_s = np.append(x_s,x_snow)
        x_i = np.append(x_i,x_ice)
        x_g = np.append(x_g,x_graupel)
        cost_ = np.append(cost_,cost)
    
        sim_dphi = i_iwc['rain'] + i_iwc['snow']*x_snow + i_iwc['ice']*x_ice + i_iwc['graupel']*x_graupel
    
        int_dphi = np.trapz(dphase_[0:120])
        int_dphi_pro = np.append(int_dphi_pro,int_dphi)
        int_IWC_snow = np.trapz(i_iwc['snow'][0:120])
        int_snow = np.append(int_snow,int_IWC_snow)
    
        cc = np.corrcoef(dphase_[0:200],sim_dphi)
        cc_ = np.append(cc_,cc[0,1])
    
        diff_dphi = dphase_[0:200] - sim_dphi
        std_diff = np.nanstd(diff_dphi)
        std_stats = np.append(std_stats,std_diff)
        mean_ = np.append(mean_,np.nanmean(diff_dphi))
    
        index_s = np.argmin(np.abs(x_snow-b_param_coef))
        p_snow = particle_habits[index_s]
        t_snow = type_particles[index_s]
        particle_snow = np.append(particle_snow,p_snow)
        type_snow = np.append(type_snow,t_snow)
    
        index_i = np.argmin(np.abs(x_ice-b_param_coef))
        p_ice = particle_habits[index_i]
        t_ice = type_particles[index_i]
        particle_ice = np.append(particle_ice,p_ice)
        type_ice = np.append(type_ice,t_ice)
    
        index_g = np.argmin(np.abs(x_graupel-b_param_coef))
        p_graupel = particle_habits[index_g]
        t_graupel = type_particles[index_g]
        particle_graupel = np.append(particle_graupel,p_graupel)
        type_graupel = np.append(type_graupel,t_graupel)
    
        dphi_simu = np.append(dphi_simu,sim_dphi)
    
        percentages = WC_percen(roid,wrfouts[i])
        percen_snow = np.append(percen_snow, percentages['snow'])
    
    dphi_simu = np.reshape(dphi_simu,(4,len(dphase_)))
    return dphi_simu

def figure_lstq_paper():
    
    roid = 'PAZ1.2021.034.10.07.G08'
    # roid = 'FM167.2023.155.12.21.G23'
    
    figpath = '/media/antia/Elements/figures/AR'
    srcpath = '/media/antia/Elements/data/collocations'
    path = '/media/antia/Elements/data/collocations/iceCal'

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        try:
            h_flag = get_hflag(roid)[0][0]
            if roid == 'PAZ1.2023.089.07.24.G01':
                h_flag = 1.5
            if roid == 'PAZ1.2023.363.10.52.G14':
                h_flag= 1.

        # h_flag = h_flag.values[0][0]+1
        except:
            h_flag = 0
            if roid == 'PAZ1.2020.356.05.00.G13':
                h_flag = 2.
            elif roid == 'PAZ1.2021.164.03.59.G04':
                h_flag = 2.2
            elif roid == 'PAZ1.2021.182.03.34.G27':
                h_flag = 2.
            elif roid ==  'PAZ1.2021.129.17.13.G16':
                h_flag= 2.
            elif roid == 'PAZ1.2020.355.18.18.G25':
                h_flag = 0.8
        # elif roid == 'PAZ1.2020.125.20.46.G30':
        #     h_flag = 2.5
        dphase = get_dphaseCal(roid)
        # dphase = np.array(file_cal.variables['dph_smooth_grid'][:],dtype=float)
        try:
            fname_col = glob.glob(path + '/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCal_' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_col[0])
            sigma =  np.array(file_cal.variables['dph_smooth_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
        except:
            sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
        
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
  
        try:
            h_flag = file.height_flag_smooth
            sigma =  np.array(file.variables['dph_smooth_L2_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            dphase = get_dphaseCal(roid)
        except:
            fname_cal = glob.glob(srcpath + '/' + 'Spire/' + 'iceCal*' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_cal[0])
            sigma = np.array(file_cal.variables["dph_smooth_std_lin"][:],dtype=float)/np.sqrt(50)
            h_flag = 1.
            # sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            # dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
            dphase = get_dphaseCal(roid)
        
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                      ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    h = np.nanmin(ray['h'],axis=1)
    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
   
    #coeficients from the dict of ARTS
    b_param_coef = np.array([value[0] for value in dict_coefs.values()])
    particle_habits = list(dict_coefs.keys())
    type_particles = np.array([value[1] for value in dict_coefs.values()])

    mp = 'Goddard'
    color = 'green'
    x_s = []; x_i = []; x_g = []; dphi_simu = []; cost_ = []
    cc_ = []; mean_ = []; particle_snow = []; type_snow = []
    particle_ice = []; type_ice = []; particle_graupel = []; type_graupel = []
    int_dphi_pro = []; int_snow = []; mp_s = []; percen_snow = []
    res = []; std_stats = []
    #height flag
    dphase[0:220][np.flip(h) < h_flag] = np.nan
    dphase_fig = dphase[0:220]
    dphase_ = dphase.copy()
    dphase_[0:220][np.flip(h) < h_flag] = np.nan
    first_valid_index = np.argmax(~np.isnan(std[0:120]))
   
    std = std[0:120][first_valid_index:] 
    dphase_ = dphase_[0:120][first_valid_index:]
    sigma = sigma[0:120][first_valid_index:]
    std[np.isnan(std)] = 0
    sigma[np.isnan(sigma)] = 0
    dphase_[np.isnan(dphase_)] = 0
    
    wrfouts= wrfout_god
    
    wc = []
    df_params, i_iwc, sim_dphi_ = parameters(roid, wrfouts)
    mp_ = wrfouts[-32:-29] #microphysics scheme
    if mp_ == 'god':
        mp__ = 'Goddard'
    elif mp_ == 'mor':
        mp__ = 'Morrison'
    elif mp_ == 'tho':
        mp__ = 'Thompson'
    elif mp_ == 'ws6':
        mp__ = 'WSM6'
    mp_s = np.append(mp_s,mp__)
    i_iwc['snow'][np.flip(h)[0:200] < h_flag] = np.nan
    i_iwc['snow'] = i_iwc['snow'][0:120][first_valid_index:]
    i_iwc['snow'][np.isnan(i_iwc['snow'])] = 0
    i_iwc['ice'][np.flip(h)[0:200] < h_flag] = np.nan
    i_iwc['ice'] = i_iwc['ice'][0:120][first_valid_index:]
    i_iwc['ice'][np.isnan(i_iwc['ice'])] = 0
    i_iwc['graupel'][np.flip(h)[0:200] < h_flag] = np.nan
    i_iwc['graupel'] = i_iwc['graupel'][0:120][first_valid_index:]
    i_iwc['graupel'][np.isnan(i_iwc['graupel'])] = 0
    wc = np.append(wc,i_iwc['snow'])
    wc = np.append(wc,i_iwc['ice'])
    wc = np.append(wc,i_iwc['graupel'])
    wc = np.reshape(wc,(3,len(i_iwc['snow'])))
    i_iwc['rain'][np.flip(h)[0:200]<h_flag] = np.nan
    i_iwc['rain'] = i_iwc['rain'][0:120][first_valid_index:]
    i_iwc['rain'][np.isnan(i_iwc['rain'])] = 0
    
    x = scipy.optimize.lsq_linear(np.diag(std) @ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['x']
    cost = scipy.optimize.lsq_linear( np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['cost']
    residuals = scipy.optimize.lsq_linear(np.diag(std)@ wc.T,np.diag(std) @ (dphase_-i_iwc['rain']),bounds = ([1e-3,1e-3,1e-3],[0.758,0.758,0.758]))['fun']
 
    res = np.append(res,residuals)
    x_snow = x[0]
    x_ice = x[1]
    x_graupel = x[2]
    x_s = np.append(x_s,x_snow)
    x_i = np.append(x_i,x_ice)
    x_g = np.append(x_g,x_graupel)
    cost_ = np.append(cost_,cost/len(dphase_))
    
    sim_dphi = i_iwc['rain'] + i_iwc['snow']*x_snow + i_iwc['ice']*x_ice + i_iwc['graupel']*x_graupel
    
    int_dphi = np.trapz(dphase_[0:120])
    int_dphi_pro = np.append(int_dphi_pro,int_dphi)
    int_IWC_snow = np.trapz(i_iwc['snow'][0:120])
    int_snow = np.append(int_snow,int_IWC_snow)
    
    cc = np.corrcoef(dphase_[0:200],sim_dphi)
    cc_ = np.append(cc_,cc[0,1])
    
    diff_dphi = dphase_[0:200] - sim_dphi
    std_diff = np.nanstd(diff_dphi)
    std_stats = np.append(std_stats,std_diff)
    mean_ = np.append(mean_,np.nanmean(diff_dphi))
    
    index_s = np.argmin(np.abs(x_snow-b_param_coef))
    p_snow = particle_habits[index_s]
    t_snow = type_particles[index_s]
    particle_snow = np.append(particle_snow,p_snow)
    type_snow = np.append(type_snow,t_snow)
    
    index_i = np.argmin(np.abs(x_ice-b_param_coef))
    p_ice = particle_habits[index_i]
    t_ice = type_particles[index_i]
    particle_ice = np.append(particle_ice,p_ice)
    type_ice = np.append(type_ice,t_ice)
    
    index_g = np.argmin(np.abs(x_graupel-b_param_coef))
    p_graupel = particle_habits[index_g]
    t_graupel = type_particles[index_g]
    particle_graupel = np.append(particle_graupel,p_graupel)
    type_graupel = np.append(type_graupel,t_graupel)
    
    
    percentages = WC_percen(roid,wrfouts)
    percen_snow = np.append(percen_snow, percentages['snow'])
   
    #sigma dphi
    dphase_sig = dphase_.copy()
    dphase_sig[dphase_sig == 0] = np.nan
    upper_values = dphase_sig + sigma
    lower_values = dphase_sig - sigma
    
    dphi_simu = LSTSQ(roid)
   
    # fig, axes = plt.subplots(1, 2, figsize=(17, 11))
    # axes = axes.flatten()
    # #Figure
    # axes[1].plot(dphase_,np.flip(h)[0:120][first_valid_index:],color='black',label='PRO')
    # axes[1].plot(dphi_simu[0],np.flip(h)[0:120][first_valid_index:],color='green',label=mp)
    # axes[1].fill_betweenx(np.flip(h)[0:120][first_valid_index:],lower_values,upper_values,color='grey',alpha=0.4)
    # axes[1].plot([],[],color = 'white',label = 'x_snow =' + str(np.round(x_snow,5)))
    # axes[1].plot([],[],color = 'white',label = 'x_ice =' + str(np.round(x_ice,5)))
    # axes[1].plot([],[],color = 'white',label = 'x_graupel =' + str(np.round(x_graupel,5)))
    # axes[1].plot([],[],color = 'white', label = 'cost = ' + str(np.round(cost/len(dphase_),2)))
    # axes[1].set_xlabel('Differential phase shift (mm)',fontsize=16)
    # axes[1].set_ylabel('Height (km)',fontsize=16)
    # axes[1].legend(fontsize=16)
    # axes[1].tick_params(labelsize=16)
    # axes[1].text(7.5,-1.3,'(b)',fontsize=16)
    # axes[1].set_ylim(0,12.4)
    # plt.title('Goddard',fontsize=16)
    
    # #Figure
    # axes[0].plot(dphase_,np.flip(h)[0:120][first_valid_index:],color='black',label='PRO')
    # axes[0].plot(dphi_simu[0],np.flip(h)[0:120][first_valid_index:],color='green',label='Goddard')
    # axes[0].fill_betweenx(np.flip(h)[0:120][first_valid_index:],lower_values,upper_values,color='grey',alpha=0.4)
    # axes[0].plot(dphi_simu[1],np.flip(h)[0:120][first_valid_index:],color='blue',label='Morrison')
    # axes[0].plot(dphi_simu[2],np.flip(h)[0:120][first_valid_index:],color='red',label='Thompson')
    # axes[0].plot(dphi_simu[3],np.flip(h)[0:120][first_valid_index:],color='violet',label='WSM6')
    # axes[0].set_xlabel('Differential phase shift (mm)',fontsize=16)
    # axes[0].tick_params(labelsize=16)
    # axes[0].set_ylabel('Height (km)',fontsize=16)
    # axes[0].legend(fontsize=16)
    # axes[0].text(7.5,-1.3,'(a)',fontsize=16)
    # axes[0].set_ylim(0,12.4)
    # plt.savefig(figpath + '/' + 'lstsq_paper2.png')
 
    
    plt.figure(figsize=(10, 11))

    #    Plot profiles
    plt.plot(dphase_, np.flip(h)[0:120][first_valid_index:], color='black', label='PRO')
    plt.plot(dphi_simu[0], np.flip(h)[0:120][first_valid_index:], color='green', label='Goddard')
    plt.plot(dphi_simu[1], np.flip(h)[0:120][first_valid_index:], color='blue', label='Morrison')
    plt.plot(dphi_simu[2], np.flip(h)[0:120][first_valid_index:], color='red', label='Thompson')
    plt.plot(dphi_simu[3], np.flip(h)[0:120][first_valid_index:], color='violet', label='WSM6')

    # Fill uncertainty envelope
    plt.fill_betweenx(np.flip(h)[0:120][first_valid_index:], lower_values, upper_values, color='grey', alpha=0.4)

    # Axes formatting
    plt.xlabel('Differential phase shift (mm)', fontsize=16)
    plt.ylabel('Height (km)', fontsize=16)
    plt.ylim(0, 12.4)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    # plt.title('Differential Phase Profiles', fontsize=16)

    # Annotate x-parameters and cost function (add below the plot)
    textstr = '\n'.join((
        f'$x_{{snow}}$ = {x_snow:.5f}',
        f'$x_{{ice}}$ = {x_ice:.5f}',
        f'$x_{{graupel}}$ = {x_graupel:.5f}',
        f'cost Goddard = {cost/len(dphase_):.2f}'))

    plt.annotate(textstr,
             xy=(0.98, 0.03), xycoords='axes fraction',
             fontsize=14,
             ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    # Optional: add subfigure label
    # plt.text(7.5, -1.3, '(a)', fontsize=16)

    # Save figure
    plt.savefig(figpath + '/' + 'lstsq_paper2_combined.png')

    
    return None

def plot_J_best_mp():
    """ function that makes a plot of the cost function normalized
    and the schemes ordered by best, second, third and fourth """
    
    cost = []; mp_schemes = []
    
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)

        c = df['cost'].values
        dphi_PAZ = get_dphaseCal(roid)
        dphase = np.count_nonzero(~np.isnan(dphi_PAZ[0:120]))
        cost = np.append(cost,c/(dphase))
        mp_best = df['mp'].values
        mp_schemes = np.append(mp_schemes,mp_best)

    #normalize costs
    # c_norm = (cost - np.nanmin(cost))/(np.nanmax(cost) - np.nanmin(cost))
    c_ = np.reshape(cost,(37,4))
    mp_ = np.reshape(mp_schemes,(37,4))
    #order cost and mp (best,second, third, fourth)
    # index_sorted = np.argsort(c_,axis=1)
    
    # c_sorted = np.take_along_axis(c_, index_sorted, axis=0)
    # mp_sorted = np.take_along_axis(mp_, index_sorted, axis=0)

    plt.figure(figsize=(8,8))

    # Define a fixed order for microphysics schemes
    mp_categories = ['Goddard', 'Thompson', 'WSM6', 'Morrison']
    mp_indexes = [0,1,2,3]
    for i in range(len(ROID_done_AR)):
        c_sorted = c_[i][np.argsort(c_[i])]
        mp_sorted = mp_[i][np.argsort(c_[i])]

        # Convert mp_sorted categories into numerical indices for plotting
        mp_cost_dict = dict(zip(mp_sorted, c_sorted))

        # Reorder 'c' based on the fixed 'mp_order'
        c_sorted = np.array([mp_cost_dict[mp_name] for mp_name in mp_categories])
        mp_sorted = np.array(mp_categories)  # This remains the same


        plt.scatter(mp_sorted, c_sorted)
        plt.plot(mp_sorted,c_sorted, '--')

    # Customize x-axis labels
    plt.xticks(ticks=range(len(mp_categories)), labels=mp_categories, rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5) 
    plt.xlabel('Microphysics Scheme')
    plt.ylabel('Cost Function')
    # plt.ylim(-0.02,0.05)
    plt.savefig('/media/antia/Elements/figures/AR/sorted_J.png')

    # plt.figure(figsize=(8,8))
    # for i in range(len(ROID_done_AR)):
    #     c_sorted = c_[i][np.argsort(c_[i])]
    #     mp_sorted = mp_[i][np.argsort(c_[i])]
    #     plt.scatter(mp_sorted,c_sorted)
    #     plt.plot(mp_sorted,c_sorted,'--')
    # plt.xticks(rotation=45, ha='right', fontsize=12) 
    # plt.grid(axis='y', linestyle='--', alpha=0.5) 
    # plt.xlabel('Position of best-performing schemes')
    # plt.ylabel('Cost function')
    # plt.savefig('/media/antia/Elements/figures/AR/sorted_J.png')

    
    return None

def fig_article_particles():
    
    def params_hist(ROID_done_AR):
        part_snow = []; param_snow = []; cost = []
        p_snow = []
        for roid in ROID_done_AR:
            if 'PAZ1' in roid:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
            else:
                df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
    
            particle_snow = df['particle_snow'][df['cost'] == np.nanmin(df['cost'])]
            part_snow = np.append(part_snow,particle_snow)
            x_snow = df['param_snow'][df['cost'] == np.nanmin(df['cost'])]
            param_snow = np.append(param_snow,x_snow)
            c = df['cost'][df['cost']==np.nanmin(df['cost'])].values
            cost = np.append(cost,c)
            percen_snow = df['percen_snow'][df['cost'] == np.nanmin(df['cost'])]
            p_snow = np.append(p_snow,percen_snow)
    
        # Ensure all particles are included
        all_particles = list(dict_coefs.keys())  # List of all possible snow particles
        ps_counts = Counter(part_snow)  # Count occurrences of each particle
        b_param_coef = np.array([value[0] for value in dict_coefs.values()])
        
        # Assign zero frequency to missing particles
        for particle in all_particles:
            if particle not in ps_counts:
                ps_counts[particle] = 0

        # Convert `b_param_coef` into a dictionary mapping particles to their param values
        particle_to_param = {p: b_param_coef[i] for i, p in enumerate(all_particles)}

        # Sort particles based on their param_snow values from `b_param_coef`
        sorted_particles = sorted(all_particles, key=lambda p: particle_to_param[p])

        # Get ordered frequencies
        frequencies = [ps_counts[particle] for particle in sorted_particles]

        # Get corresponding param_snow values from `b_param_coef`
        sorted_param_values = [particle_to_param[p] for p in sorted_particles]
    
        # Define cost threshold for highlighting
        cost_threshold = 10

        # Separate high and low cost cases
        high_cost_indices = cost > cost_threshold
        low_cost_indices = ~high_cost_indices
        high_percen_snow = p_snow > 75
        low_percen_snow = p_snow < 75

        # Count occurrences of each particle for low and high cost cases
        low_cost_counts = Counter(part_snow[low_cost_indices])
        high_cost_counts = Counter(part_snow[high_cost_indices])
        high_percen_counts = Counter(part_snow[high_percen_snow])
        low_percen_counts = Counter(part_snow[low_percen_snow])

        # Ensure all particles are included in both
        low_cost_freq = [low_cost_counts[p] if p in low_cost_counts else 0 for p in sorted_particles]
        high_cost_freq = [high_cost_counts[p] if p in high_cost_counts else 0 for p in sorted_particles]
        low_percen_freq = [low_percen_counts[p] if p in low_percen_counts else 0 for p in sorted_particles]
        high_percen_freq = [high_percen_counts[p] if p in high_percen_counts else 0 for p in sorted_particles]

        return sorted_particles,low_cost_freq, high_cost_freq, sorted_param_values, high_percen_freq
    
    
    sorted_particles_NA,low_cost_freq_NA, high_cost_freq_NA,sorted_param_values_NA,high_percen_freq_NA  = params_hist(ROID_done_Atlantic)
    sorted_particles_EP,low_cost_freq_EP, high_cost_freq_EP, sorted_param_values_EP,high_percen_freq_EP = params_hist(ROID_done_EastPacific)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes = axes.flatten()
    
    # Plot the low-cost bars first (skyblue)
    bars1 = axes[1].bar(sorted_particles_NA, low_cost_freq_NA, color='yellowgreen', alpha=0.7, label='Low cost function')

    # Overlay the high-cost bars on top (red)
    bars2 = axes[1].bar(sorted_particles_NA, high_cost_freq_NA, bottom=low_cost_freq_NA, color='red', alpha=0.7, label='High cost function')

    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq_NA[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)

    # Labels and x-axis formatting
    axes[1].set_xlabel('Best particles for snow', fontsize=14)
    axes[1].set_ylabel('Number of AR', fontsize=14)
    axes[1].set_xticklabels(sorted_particles_NA, rotation=45, ha='right', fontsize=14)
    axes[1].set_ylim(0,7)
    axes[1].set_title('north Atlantic ARs',fontsize=14)
    axes[1].text(-0.5,6.5,'(b)',fontsize=16)
    axes[1].legend()

    # Create secondary x-axis for param_snow
    ax2 = axes[1].twiny()
    ax2.set_xlim(axes[1].get_xlim())  # Align both axes
    ax2.set_xticks(axes[1].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values_NA], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('x-parameter for snow', fontsize=14)
    import matplotlib.patches as mpatches  # Import patch for legend proxy

    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='High snow percentage')
    low_cost_patch = mpatches.Patch(color='yellowgreen', label='Low cost function')
    high_cost_patch = mpatches.Patch(color='red', label='High cost function')
    # Add legend with all components
    axes[1].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)

    # Plot the low-cost bars first (skyblue)
    bars1=axes[0].bar(sorted_particles_EP, low_cost_freq_EP, color='navajowhite', alpha=0.7, label='Low cost function')

    # Overlay the high-cost bars on top (red)
    bars2 = axes[0].bar(sorted_particles_EP, high_cost_freq_EP, bottom=low_cost_freq_EP, color='red', alpha=0.7, label='High cost function')

    for bars in [bars1,bars2]:
        for i, bar in enumerate(bars):
            if high_percen_freq_EP[i] > 0:  # If the particle appears in high-snow cases
                bar.set_edgecolor('black')  # Highlight edge
                bar.set_linewidth(2)

    # Labels and x-axis formatting
    axes[0].set_xlabel('Best particles for snow', fontsize=14)
    axes[0].set_ylabel('Number of AR', fontsize=14)
    axes[0].set_xticklabels(sorted_particles_EP, rotation=45, ha='right', fontsize=14)
    axes[0].set_ylim(0,7)
    axes[0].text(-0.5,6.5,'(a)',fontsize=16)
    axes[0].set_title('north-east Pacific ARs',fontsize=14)
    axes[0].legend()

    # Create secondary x-axis for param_snow
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())  # Align both axes
    ax2.set_xticks(axes[0].get_xticks())  # Match tick positions
    ax2.set_xticklabels([f"{val:.3f}" for val in sorted_param_values_EP], rotation=45, ha='left', fontsize=14)
    ax2.set_xlabel('x-parameter for snow', fontsize=14)
    
    # Create proxy bars for legend
    high_snow_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='High snow percentage')
    low_cost_patch = mpatches.Patch(color='navajowhite', label='Low cost function')
    high_cost_patch = mpatches.Patch(color='red', label='High cost function')
    # Add legend with all components
    axes[0].legend(handles=[low_cost_patch, high_cost_patch, high_snow_patch], loc='upper right', fontsize=12)

    
    plt.tight_layout()
    plt.savefig('/media/antia/Elements/figures/AR/hist_best_snow_particles_highlighted_EP_NA3.png')

    return None

def q_pro_ray(roid):
    """ function that plots four figures for all mp schemes for a case
    where PRO rays are represented with the interpolated mixing ratio of
    different hydrometeors """
    
    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)
    
    srcpath_col = '/media/antia/Elements/data/collocations'
    fname_col = glob.glob(srcpath_col + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
    file = xr.open_dataset(fname_col[0])
    
    domain = wrfout_god[-21:-18]


    # variables of the rays
    lon = np.array(file.variables['longitude'][:], dtype=float)
    lat = np.array(file.variables['latitude'][:], dtype=float)
    hei = np.array(file.variables['height'][:], dtype=float)
    file.close()

    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:, 1:], ray['lon'][:, 1:], ray['h'][:, 1:],
                               ray['lat'][:, :-1], ray['lon'][:, :-1], ray['h'][:, :-1])

    
    qrain = getvar(Dataset(wrfout_god),'QRAIN',timeidx=0)
    units = qrain.units
    
    try:
        interpRay_wrf_god = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout_god[-32:-29] + '.npy',allow_pickle=True).item()
        interpRay_wrf_mor = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout_mor[-32:-29] + '.npy',allow_pickle=True).item()
        interpRay_wrf_tho = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout_tho[-32:-29] + '.npy',allow_pickle=True).item()      
        interpRay_wrf_ws6 = np.load('/media/antia/Elements/data/interp' + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout_ws6[-32:-29] + '.npy',allow_pickle=True).item()
      
    except:
        interp_wrf(roid, wrfout_god)
        interpRay_wrf_god = np.load('/home/aliga/carracedo/Desktop/ICE/data' + '/' + 'WRF' + '/' +
                        roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'interp_' + wrfout_god[48:51] + '.npy', allow_pickle=True).item()

    x, y = np.meshgrid(file.npoint.values, file.nray.values)
    h = file.height.values
    
    # var = 'snow'
    vmax = 0.1
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(18,12))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(bottom = 0.15,top=0.9,left=0.02,right=0.98,wspace=0.01)
 
    ax1 = plt.subplot(gs1[0,0])
    for i in range(44):
        i = i*5
        plt.scatter(x[i, :], h[i, :], c=interpRay_wrf_god['snow']
                    [i, :], marker='.', cmap='RdYlBu_r',vmin=0,vmax=0.1)
     
    plt.xlabel('Distance in points from tangent point',fontsize=9)
    plt.ylabel('Tangencial height (km)',fontsize=9)
    plt.xticks(np.linspace(0,file.npoint.shape[0],7),np.linspace(-150,150,7))
    cbar = plt.colorbar(location='left',pad=0.13,ticks = np.linspace(0,0.1,9))
    cbar.set_label('snow' + ' IWC (' + units + ')',fontsize=9)
    plt.ylim(0,15)
    plt.title('Goddard')
    
    ax2 = plt.subplot(gs1[0,1])
    for i in range(44):
        i = i*5
        plt.scatter(x[i, :], h[i, :], c=interpRay_wrf_god['ice']
                [i, :], marker='.', cmap='RdYlBu_r',vmin=0,vmax=0.2)
    
    plt.xlabel('Distance in points from tangent point',fontsize=9)
    plt.ylabel('Tangencial height (km)',fontsize=9)
    plt.xticks(np.linspace(0,file.npoint.shape[0],7),np.linspace(-150,150,7))
    cbar = plt.colorbar(location='left',pad=0.13,ticks = np.linspace(0,0.2,9))
    cbar.set_label('ice' + ' IWC  (' + units + ')',fontsize=9)
    plt.ylim(0,15)
    plt.title('Morrison')
    
    
    ax3 = plt.subplot(gs1[1,0])
    for i in range(44):
        i = i*5
        plt.scatter(x[i, :], h[i, :], c=interpRay_wrf_god['graupel']
            [i, :], marker='.', cmap='RdYlBu_r',vmin=0,vmax=0.05)
    
    plt.xlabel('Distance in points from tangent point',fontsize=9)
    plt.ylabel('Tangencial height (km)',fontsize=9)
    plt.xticks(np.linspace(0,file.npoint.shape[0],7),np.linspace(-150,150,7))
    cbar = plt.colorbar(location='left',pad=0.13,ticks = np.linspace(0,0.05,9))
    cbar.set_label('graupel' + ' IWC  (' + units + ')',fontsize=9)
    plt.ylim(0,15)
    plt.title('WSM6')
    
    ax4 = plt.subplot(gs1[1,1])
    for i in range(44):
        i = i*5
        plt.scatter(x[i, :], h[i, :], c=interpRay_wrf_god['rain']
                [i, :], marker='.', cmap='RdYlBu_r',vmin=0,vmax=0.2)
     
    plt.xlabel('Distance in points from tangent point',fontsize=9)
    plt.ylabel('Tangencial height (km)',fontsize=9)
    plt.xticks(np.linspace(0,file.npoint.shape[0],7),np.linspace(-150,150,7))
    cbar = plt.colorbar(location='left',pad=0.13,ticks = np.linspace(0,0.2,9))
    cbar.set_label('rain' + ' IWC (' + units + ')',fontsize=9)
    plt.ylim(0,15)
    plt.title('Thompson')
    
    plt.savefig('/media/antia/Elements/figures/AR/' + 'PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'rays_q' + '.png')
    
    
    # Define hydrometeor properties: (colormap, data, vmin, vmax)

    # hydrometeors = {
    #     'snow':    ('cool',    interpRay_wrf_god['snow'],    0,  0.1),
    #     'ice':     ('plasma',  interpRay_wrf_god['ice'],     0,  0.2),
    #     'graupel': ('cividis', interpRay_wrf_god['graupel'], 0,  0.05),
    #     'rain':    ('Greens',  interpRay_wrf_god['rain'],    0,  0.3)
    #     }
  

    # plt.figure(figsize=(18, 12))
    # gs1 = gridspec.GridSpec(2, 2)
    # gs1.update(bottom=0.15, top=0.9, left=0.02, right=0.98, wspace=0.01)

    # # Iterate through subplots (each microphysics scheme)
    # titles = ['Goddard', 'Morrison', 'WSM6', 'Thompson']
    # schemes = [interpRay_wrf_god, interpRay_wrf_mor, interpRay_wrf_ws6, interpRay_wrf_tho]

    # plt.figure(figsize=(12,8))
    
    # # Loop through hydrometeors and plot them with their respective vmin/vmax
    # scatter_plots = []
    # for i in range(44):
    #     i = i * 5
    #     no_hydro = (
    #         (interpRay_wrf_god['snow'][i, :] < 0.001) &
    #         (interpRay_wrf_god['ice'][i, :] < 0.001) &
    #         (interpRay_wrf_god['graupel'][i, :] < 0.001) &
    #         (interpRay_wrf_god['rain'][i, :] < 0.001)
    #         )

    #     # **Plot the "empty" rays first in grey**
    #     plt.scatter(x[i, no_hydro], h[i, no_hydro], color='#D3D3D3', alpha=0.8, s=10, label="No Hydrometeors")

    
    #     # **Now plot each hydrometeor normally**
    # for htype, (cmap, hdata, vmin, vmax) in hydrometeors.items():
    #     sc = plt.scatter(x[i, ~no_hydro], h[i, ~no_hydro], c=hdata[i, ~no_hydro], 
    #                          marker='o', cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, s=10, label=htype)
    # #         scatter_plots.append((sc, htype))
    # # for htype, (cmap, hdata, vmin, vmax) in hydrometeors.items():
    # #     for i in range(44):
    # #         i = i * 5
    # #         sc = plt.scatter(x[i, :], h[i, :], c=hdata[i, :], marker='.', cmap=cmap, vmin=vmin, vmax=vmax)
    # #     scatter_plots.append((sc, htype))  # Store for colorbar

    # plt.xlabel('Distance in points from tangent point', fontsize=9)
    # plt.ylabel('Tangencial height (km)', fontsize=9)
    # plt.xticks(np.linspace(0, file.npoint.shape[0], 7), np.linspace(-150, 150, 7))
    # plt.ylim(0, 15)
    
    # plt.savefig('/media/antia/Elements/figures/AR/PAZ1_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/rays_all_hydrometeors.png')
    # plt.show()

    return None

def best_x_error():
    """ function that makes a plot of the x-params of 
    snow, ice and graupel best for each case with their
    error """
    
    x_param_snow = []; x_param_ice = []; x_param_graupel = []
    error_snow = []; error_ice = []; error_graupel = []
    mp_scheme = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new3.pkl',allow_pickle=True)

        x_snow = df['param_snow'][df['cost'] == np.nanmin(df['cost'])]
        x_param_snow = np.append(x_param_snow,x_snow)
        x_ice = df['param_ice'][df['cost'] == np.nanmin(df['cost'])]
        x_param_ice = np.append(x_param_ice,x_ice)
        x_graupel = df['param_graupel'][df['cost'] == np.nanmin(df['cost'])]
        x_param_graupel = np.append(x_param_graupel,x_graupel)
        
        std_snow = df['std_x_snow'][df['cost'] == np.nanmin(df['cost'])]
        error_snow = np.append(error_snow,std_snow)
        std_ice = df['std_x_ice'][df['cost'] == np.nanmin(df['cost'])]
        error_ice = np.append(error_ice,std_ice)
        std_graupel = df['std_xgraupel'][df['cost'] == np.nanmin(df['cost'])]
        error_graupel = np.append(error_graupel,std_graupel)

        mp = df['mp'][df['cost'] == np.nanmin(df['cost'])]
        mp_scheme = np.append(mp_scheme,mp)

    num_cases = len(ROID_done_AR)
    x_pos = np.arange(num_cases)  # X positions for each case

    # Define colors and markers for clarity
    hydrometeors = {
        'Snow': ('blue', x_param_snow, error_snow),
        'Ice': ('red', x_param_ice, error_ice),
        'Graupel': ('green', x_param_graupel, error_graupel)
        }

    range_snow = [1e-3,0.758]
    range_ice = [0.008,0.606]
    range_graupel = [0.008,0.556]
    
    # plt.figure(figsize=(14, 6))

    # # Plot each hydrometeor type with error bars
    # for label, (color, marker, x_params, errors) in hydrometeors.items():
    #     plt.errorbar(x_pos, x_params, yerr=errors, fmt=marker, color=color, 
    #              capsize=5, label=label, markersize=8, linestyle='None')

    # # Customize x-axis
    # plt.xticks(x_pos, ROID_done_AR, rotation=45, ha='right', fontsize=10)  # ROID labels as xticks
    # plt.xlabel('Atmospheric River Cases (ROID)', fontsize=12)
    # plt.ylabel('X-Parameter Values', fontsize=12)
    # plt.title('Best X-Parameters for Each AR Case', fontsize=14)
    
    # plt.ylim(-0.3,0.8)
    # # Add legend
    # plt.legend(fontsize=10, loc='best')

    # # Show grid for better readability
    # plt.grid(True, linestyle='--', alpha=0.5)

    # # Save the figure
    # plt.savefig('/media/antia/Elements/figures/AR/x_parameters_best_cases.png', dpi=300, bbox_inches='tight')

    # plt.show()
    
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define marker styles for each mp_scheme
    marker_dict = {'Goddard': 's', 'Thompson': 'o', 'WSM6': '*', 'Morrison': '^'}

    # Loop through each case (ROID)
    for i, roid in enumerate(ROID_done_AR):
        mp_type = mp_scheme[i]  # Get the associated mp_scheme
        marker = marker_dict.get(mp_type, 'x')  # Default to 'x' if mp_type is unknown
    
        # Scatter points for Snow, Ice, and Graupel (all using the same marker for the ROID)
        plt.errorbar(x_pos[i],x_param_graupel[i],yerr=error_graupel[i],marker=marker,color='green',capsize=3.)
        plt.errorbar(x_pos[i],x_param_ice[i],yerr=error_ice[i],marker=marker,color='red',capsize=3.)
        plt.errorbar(x_pos[i],x_param_snow[i],yerr=error_snow[i],marker=marker,color='blue',capsize=3.)
        
        
    # Set log scale on y-axis
    ax.set_yscale('log')
    plt.xticks(x_pos, ROID_done_AR, rotation=45, ha='right', fontsize=10)  # ROID labels as xticks
 
    # Labels
    ax.set_xlabel('roid', fontsize=14)
    ax.set_ylabel('x-parameter for Snow/Ice/Graupel (log scale)', fontsize=14)
    import matplotlib.lines as mlines
    colors = {'Snow': 'blue', 'Ice': 'red', 'Graupel': 'green'}
    # --- Custom Legend ---
    # 1. Hydrometeor color legend
    snow_patch = mlines.Line2D([], [], color=colors['Snow'], marker='o', linestyle='None', markersize=10, label='Snow')
    ice_patch = mlines.Line2D([], [], color=colors['Ice'], marker='o', linestyle='None', markersize=10, label='Ice')
    graup_patch = mlines.Line2D([], [], color=colors['Graupel'], marker='o', linestyle='None', markersize=10, label='Graupel')
 
    # 2. Microphysics scheme legend (markers only)
    goddard_patch = mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='Goddard')
    thompson_patch = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Thompson')
    wsm6_patch = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='WSM6')
    morrison_patch = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='Morrison')

    # Combine legends
    legend1 = ax.legend(handles=[snow_patch, ice_patch, graup_patch], title="Hydrometeors", fontsize=12, loc='lower left')
    legend2 = ax.legend(handles=[goddard_patch, thompson_patch, wsm6_patch, morrison_patch], title="Microphysics Schemes", fontsize=12, loc='lower right')

    # Add the first legend back (since the second one replaces it)
    ax.add_artist(legend1)
    # Number of Atlantic cases
    num_atlantic = len(ROID_done_EastPacific)

    # Add a vertical dashed line after the last Atlantic case
    plt.axvline(x=num_atlantic - 0.5, color='black', linestyle='--', linewidth=2, label='Atlantic/Pacific Divider')
    # ax.set_ylim(10e-17,10)
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    # Add shaded regions for each hydrometeor range
    ax.axhspan(range_snow[0], range_snow[1], color='grey', alpha=0.2, label='Snow Range')
    # ax.axhspan(range_ice[0], range_ice[1], color='red', alpha=0.2, label='Ice Range')
    # ax.axhspan(range_graupel[0], range_graupel[1], color='green', alpha=0.2, label='Graupel Range')

    
    plt.savefig('/media/antia/Elements/figures/AR/x_parameters_best_cases.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()
    
    
    return None

def figure_J_x():
    """ function that makes a scatter plot of the x
    parameters being the J the colorbar """
    
    cost = []; x_snow = []; x_ice = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
            
        c = df['cost'][df['cost'] == np.nanmin(df['cost'])]
        cost = np.append(cost,c)
        x_s = df['param_snow'][df['cost'] == np.nanmin(df['cost'])]
        x_snow = np.append(x_snow,x_s)
        x_i = df['param_ice'][df['cost'] == np.nanmin(df['cost'])]
        x_ice = np.append(x_ice,x_i)
        
    plt.figure(figsize=(8,8))
    plt.scatter(x_snow,x_ice,c=cost,vmax=10,cmap='jet')
    cbar = plt.colorbar(extend='max')
    cbar.set_label('Cost function')
    plt.xlabel('x-parameter snow')
    plt.ylabel('x-parameter ice')
    plt.savefig('/media/antia/Elements/figures/AR/J_x_scatter.png')
    
    return None

def map_cases():
    """ function that makes a map of the AR cases """
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    srcpath = '/media/antia/Elements/data/collocations'
    lon_cases = []; lat_cases = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])

        else:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'ice*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])

        lon = file.lon_occ
        lat = file.lat_occ
    
        lon_cases = np.append(lon_cases,lon)
        lat_cases = np.append(lat_cases,lat)

    # Define Atlantic vs Pacific based on longitude (Modify if needed)
    atlantic_cases = [(lon, lat) for lon, lat in zip(lon_cases, lat_cases) if lon > -100]
    pacific_cases = [(lon, lat) for lon, lat in zip(lon_cases, lat_cases) if lon <= -100]

    # Convert into separate lists
    lon_atlantic, lat_atlantic = zip(*atlantic_cases) if atlantic_cases else ([], [])
    lon_pacific, lat_pacific = zip(*pacific_cases) if pacific_cases else ([], [])

    # Create the figure and set projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent (focus on North Atlantic & North-East Pacific)
    ax.set_extent([-180, np.nanmax(lon_cases)+10, np.nanmin(lat_cases)-5, np.nanmax(lat_cases)+5], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False  # Hide top and right labels
    
    # Plot the cases
    ax.scatter(lon_atlantic, lat_atlantic, color='red', label="Atlantic Cases", edgecolor='black', s=100, marker='o')
    ax.scatter(lon_pacific, lat_pacific, color='blue', label="Pacific Cases", edgecolor='black', s=100, marker='s')

    # Add legend
    ax.legend(loc='lower right', fontsize=12)
    
    # Title
    ax.set_title("Atmospheric River Cases in North Atlantic & North-East Pacific", fontsize=14)
    plt.savefig('/media/antia/Elements/figures/AR/map_AR.png')
    # Show plot
    plt.show()
    
    return None

def heatmap_J_x():
    """ function that plots a heatmap of the combinations 
    between x_snow and x_ice and with J as colorbar fro an
    specific case """
    path = '/media/antia/Elements/data/collocations/iceCal'
    srcpath = '/media/antia/Elements/data/collocations'

    roid = 'PAZ1.2018.234.06.45.G06'
   
    from matplotlib.colors import LogNorm
    
    ice_param_list = 10**np.linspace(-3,0,25).reshape(1,1,-1,1,1)
    snow_param_list = 10**np.linspace(-3,0,26).reshape(1,-1,1,1,1)
    graupel_param_list = 10**np.linspace(-3,0,27).reshape(-1,1,1,1,1)

    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        try:
            h_flag = get_hflag(roid)[0][0]
            if roid == 'PAZ1.2023.089.07.24.G01':
                h_flag = 1.5
            if roid == 'PAZ1.2023.363.10.52.G14':
                h_flag= 1.

            # h_flag = h_flag.values[0][0]+1
        except:
            h_flag = 0
            if roid == 'PAZ1.2020.356.05.00.G13':
                h_flag = 2.
            elif roid == 'PAZ1.2021.164.03.59.G04':
                h_flag = 2.2
            elif roid == 'PAZ1.2021.182.03.34.G27':
                h_flag = 2.
            elif roid ==  'PAZ1.2021.129.17.13.G16':
                h_flag= 2.
            elif roid == 'PAZ1.2020.355.18.18.G25':
                h_flag = 0.8
            # elif roid == 'PAZ1.2020.125.20.46.G30':
            #     h_flag = 2.5
        dphase = get_dphaseCal(roid)
        # dphase = np.array(file_cal.variables['dph_smooth_grid'][:],dtype=float)
        try:
            fname_col = glob.glob(path + '/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCal_' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_col[0])
            sigma =  np.array(file_cal.variables['dph_smooth_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
        except:
            sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
      
        try:
            h_flag = file.height_flag_smooth
            sigma =  np.array(file.variables['dph_smooth_L2_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            dphase = get_dphaseCal(roid)
        except:
            fname_cal = glob.glob(srcpath + '/' + 'Spire/' + 'iceCal*' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_cal[0])
            sigma = np.array(file_cal.variables["dph_smooth_std_lin"][:],dtype=float)/np.sqrt(50)
            h_flag = 1.
            # sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            # dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
            dphase = get_dphaseCal(roid)


    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)

    df_params, i_iwc, sim_dphi = parameters(roid, wrfout_ws6)
    sim_dphi = sim_dphi - i_iwc['rain']
            
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
    
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                          ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    h = np.nanmin(ray['h'],axis=1)
   
    dphase[0:220][np.flip(h) < h_flag] = np.nan
    dphase_fig = dphase[0:220]
    dphase_ = dphase.copy()
    dphase_[0:220][np.flip(h) < h_flag] = np.nan
    first_valid_index = np.argmax(~np.isnan(std[0:120]))
   
    std = std[0:120][first_valid_index:] 
    dphase_ = dphase_[0:120][first_valid_index:]
    sigma = sigma[0:120][first_valid_index:]
    std[np.isnan(std)] = 0
    sigma[np.isnan(sigma)] = 0
    dphase_[np.isnan(dphase_)] = 0


    R_inv = 1. / sigma**2  # or just R_inv = 1. / sigma**2 if already trimmed
    R_inv = R_inv[None, None, None, :, None]  # shape to match sim_dphi

    # Compute the weighted cost function (chi-squared style)
    cost_function = np.nanmean(((sim_dphi[:,:,:,:,first_valid_index:120] - dphase_)**2) * R_inv, axis=-1)

    # Reduce to 2D for plotting (fix graupel at a reference index, e.g., index 12)
    cost_function_2d = cost_function[0, :, :, 0]/len(dphase_)  # Shape: (26, 25)
   
    # Extract x-parameters for Snow & Ice, squeezing extra dimensions
    x_param_snow = np.squeeze(snow_param_list)  # Shape: (26,)
    x_param_ice = np.squeeze(ice_param_list)    # Shape: (25,)
    
    x2plot,y2plot=np.meshgrid(x_param_snow,x_param_ice)
    
    #################################################
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes = axes.flatten()
    # Create a heatmap
    #plt.pcolormesh(x_param_snow, x_param_ice, cost_function_2d.T, cmap='viridis',norm=LogNorm())
    im=axes[0].contourf(x2plot,y2plot,np.clip(cost_function_2d,0,5).T,100,cmap='jet')
    contour = axes[0].contour(x2plot,y2plot,np.clip(cost_function_2d,0,5).T,levels=[0,0.0025,0.01],colors=['black'])
    axes[0].clabel(contour, inline=True, fontsize=10, fmt="%.3f")
    # Add colorbar
    cbar = plt.colorbar(im,extend='max')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Cost Function',fontsize=14)
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].tick_params(labelsize=14)   
    # Labels and title
    axes[0].set_xlabel('x-parameter for Snow',fontsize=14)
    axes[0].set_ylabel('x-parameter for Ice',fontsize=14)
    axes[0].set_title(roid,fontsize=14)
    axes[0].text((0.3e-1),0.0004,'(a)',fontsize=14)

    roid = 'PAZ1.2022.122.01.17.G25'
    if 'PAZ1' in roid:
        fname_col = glob.glob(srcpath + '/' + 'PAZ/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCol_' + roid + '*.nc')
        file = xr.open_dataset(fname_col[0])
        try:
            h_flag = get_hflag(roid)[0][0]
            if roid == 'PAZ1.2023.089.07.24.G01':
                h_flag = 1.5
            if roid == 'PAZ1.2023.363.10.52.G14':
                h_flag= 1.

         # h_flag = h_flag.values[0][0]+1
        except:
            h_flag = 0
            if roid == 'PAZ1.2020.356.05.00.G13':
                h_flag = 2.
            elif roid == 'PAZ1.2021.164.03.59.G04':
                h_flag = 2.2
            elif roid == 'PAZ1.2021.182.03.34.G27':
                h_flag = 2.
            elif roid ==  'PAZ1.2021.129.17.13.G16':
                h_flag= 2.
            elif roid == 'PAZ1.2020.355.18.18.G25':
                h_flag = 0.8
         # elif roid == 'PAZ1.2020.125.20.46.G30':
         #     h_flag = 2.5
        dphase = get_dphaseCal(roid)
     # dphase = np.array(file_cal.variables['dph_smooth_grid'][:],dtype=float)
        try:
            fname_col = glob.glob(path + '/' + roid[5:9] + '.' + roid[10:13] + '/' + 'iceCal_' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_col[0])
            sigma =  np.array(file_cal.variables['dph_smooth_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
        except:
            sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
         
    else:
        try:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'iceCol*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
        except:
            fname_col = glob.glob(srcpath + '/' + 'Spire/' + 'icePha*' + roid + '*.nc')
            file = xr.open_dataset(fname_col[0])
   
        try:
            h_flag = file.height_flag_smooth
            sigma =  np.array(file.variables['dph_smooth_L2_grid_std'][:],dtype=float)/np.sqrt(50)
            std = 1/sigma
            dphase = get_dphaseCal(roid)
        except:
            fname_cal = glob.glob(srcpath + '/' + 'Spire/' + 'iceCal*' + roid + '*.nc')
            file_cal = xr.open_dataset(fname_cal[0])
            sigma = np.array(file_cal.variables["dph_smooth_std_lin"][:],dtype=float)/np.sqrt(50)
            h_flag = 1.
            # sigma = std_dphi(roid)/np.sqrt(50)
            std = 1/sigma
            # dphase = np.array(file.variables['dph_smooth_L2_grid'][:],dtype=float)
            dphase = get_dphaseCal(roid)


    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout(roid)

    df_params, i_iwc, sim_dphi = parameters(roid, wrfout_ws6)
    sim_dphi = sim_dphi - i_iwc['rain']
         
    try:
        #variables of the rays
        lon = np.array(file.variables['longitude'][:],dtype=float)
        lat = np.array(file.variables['latitude'][:],dtype=float)
        hei = np.array(file.variables['height'][:],dtype=float)
        file.close()
    except:
        #variables of the rays
        lon = np.array(file.variables['ray_longitude'][:],dtype=float)
        lat = np.array(file.variables['ray_latitude'][:],dtype=float)
        hei = np.array(file.variables['ray_height'][:],dtype=float)
        file.close()
 
    ray = {}
    ray['lon'] = lon
    ray['lat'] = lat
    ray['h'] = hei
    ray['dist'] = af.distlatlonhei(ray['lat'][:,1:],ray['lon'][:,1:],ray['h'][:,1:],
                       ray['lat'][:,:-1],ray['lon'][:,:-1],ray['h'][:,:-1])
    h = np.nanmin(ray['h'],axis=1)

    dphase[0:220][np.flip(h) < h_flag] = np.nan
    dphase_fig = dphase[0:220]
    dphase_ = dphase.copy()
    dphase_[0:220][np.flip(h) < h_flag] = np.nan
    first_valid_index = np.argmax(~np.isnan(std[0:120]))

    std = std[0:120][first_valid_index:] 
    dphase_ = dphase_[0:120][first_valid_index:]
    sigma = sigma[0:120][first_valid_index:]
    std[np.isnan(std)] = 0
    sigma[np.isnan(sigma)] = 0
    dphase_[np.isnan(dphase_)] = 0


    R_inv = 1. / sigma**2  # or just R_inv = 1. / sigma**2 if already trimmed
    R_inv = R_inv[None, None, None, :, None]  # shape to match sim_dphi

    
    roid = 'PAZ1.2022.122.01.17.G25'
    wrfout_god, wrfout_mor, wrfout_tho, wrfout_ws6 = get_wrfout('PAZ1.2022.122.01.17.G25')

    df_params, i_iwc, sim_dphi = parameters(roid, wrfout_mor)
    sim_dphi = sim_dphi - i_iwc['rain']
    # Compute the weighted cost function (chi-squared style)
    cost_function = np.nanmean(((sim_dphi[:,:,:,:,first_valid_index:120] - dphase_)**2) * R_inv, axis=-1)

    # Reduce to 2D for plotting (fix graupel at a reference index, e.g., index 12)
    cost_function_2d = cost_function[0, :, :, 0]/len(dphase_)  # Shape: (26, 25)
 
    
    im=axes[1].contourf(x2plot,y2plot,np.clip(cost_function_2d,0,10).T,100,cmap='jet')
    contour = axes[1].contour(x2plot,y2plot,np.clip(cost_function_2d,0,10).T,levels=[0,0.1,0.5],colors=['black'])
    axes[1].clabel(contour, inline=True, fontsize=10, fmt="%.3f")
    # Add colorbar
    cbar = plt.colorbar(im,extend='max')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Cost Function',fontsize=14)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].tick_params(labelsize=14) 
    # Labels and title
    axes[1].set_xlabel('x-parameter for Snow',fontsize=14)
    axes[1].set_ylabel('x-parameter for Ice',fontsize=14)
    axes[1].set_title(roid,fontsize=14)
    axes[1].text((0.3e-1),0.0004,'(b)',fontsize=14)
    
    plt.tight_layout()

    figpath = '/media/antia/Elements/figures/AR' 
    if 'PAZ1' in roid:
        os.makedirs(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'heatmap_J_x_log.png')
    else:
        os.makedirs(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17],exist_ok=True)
        plt.savefig(figpath + '/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/'  + 'heatmap_J_x_log.png')
     
    return None

def hist_x():
    """ function that makes a histogram of the values of x
    for the three hydrometeors """
   
    param_snow = []; mp_scheme = []
    for roid in ROID_done_AR:
        if 'PAZ1' in roid:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:4] + '_' + roid[5:9] + '_' + roid[10:13] + '_' + roid[14:16] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        else:
            df = np.load('/media/antia/Elements/data/interp/' + roid[0:5] + '_' + roid[6:10] + '_' + roid[11:14] + '_' + roid[15:17] + '/' + 'df_lstsq_new.pkl',allow_pickle=True)
        
        x_snow = df['param_snow'][df['cost'] == np.nanmin(df['cost'])]
        param_snow = np.append(param_snow,x_snow)
        mp = df['mp'][df['cost'] == np.nanmin(df['cost'])]    
        mp_scheme = np.append(mp_scheme,mp)
    
    #select params per each mp scheme
    param_snow_god = param_snow[mp_scheme== 'god']
    
    return None
