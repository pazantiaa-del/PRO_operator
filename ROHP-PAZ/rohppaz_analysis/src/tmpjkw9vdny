#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:48:38 2022

@author: carracedo
"""
import pyart
import math
import numpy as np
import interpolation3D as int3d
import sys
import datetime
import xarray as xr
sys.path.append('/home/aliga/carracedo/ROHP-PAZ/rohppaz_processing/rohppaz/lib/occult')
import angle_functions as af

def vUCARcontrol(dt):
    
    if dt<datetime(2021,4,1):
        versionUcar='2010.2640'
    elif ((dt>=datetime(2021,4,1)) & (dt<datetime(2021,8,1))):
        versionUcar='2021.1590'
    elif dt>=datetime(2021,8,1):
        versionUcar='2021.2080'
    return versionUcar

def interpolate_models(ray,longitude,latitude,height, variables):
    
    known=np.array([longitude.ravel(),latitude.ravel(),
                    height.ravel()]).T
    ask = np.array([ray['lon'].ravel(),
                    ray['lat'].ravel(),
                    ray['h'].ravel()]).T
    
    interpRay = {}
    for v in variables.keys():
        var = variables[v].ravel()
        A=int3d.Interpolation3D_v2(known,var)
        interpRay_=A(ask,nne=8, eps=0.3, p=1, dup=np.inf)
        interpRay[v] = interpRay_.reshape(ray['lon'].shape)
    
    return interpRay

class radar(object):
    
    """ Provides the information inside a radar file from NEXRAD. You are able to obtain the different variables: 
        reflectivity, differential reflectivity, differential phase, cross correlation ratio. It also provides a method 
        for calculating the specific differential phase shift. 
        
        Another method is presented for the correction of the elevation angle, for both the specific differential
        phase and the differential reflectivity.
        
        Usually it is better to treat the variables in cartesian coordinates so a cartesian grid is also made, but the 
        variables are also obtained in radar coordinates (2D, one dimension for the elevation angle and another for 
        the range) 
        
        Attributes:
            Z: reflectivity in dBZ
            Zdr: differential reflectivity in dB
            CC: cross correlation coefficient (adimensional)
            kdp: Specific differential phase in deg/km
            lon: Longitude in grid
            lat: Latitude in grid
            h: height in grid
            gZ: reflectivity in grid
            gZdr: differential reflectivity in grid
            gCC: cross correlation coefficient in grid
            gKdp: specific differential phase in grid
            
        Methods:
            correction_angle: Corrects Kdp and Zdr because at higher elevation angles the signals do not penetrate the 
            hydrometeors perpendicular to their vertical axis.
            
                kdp_corrected: Specific differential phase corrected in function of the elevation angle in radar coordinates
                zdr_corrected: Differential reflectivity corrected in function of the elevation angle in radar coordinates
                cgKdp: Specific differential phase corrected 3D (lon,lat,height)
                cgZdr: Differential reflectivity corrected 3D (lon,lat,height)
            
            interpolation: Makes an interpolation along the ray path of PAZ for obtaining the values of the radar
            variables in PAZ observations.
                
                interp_kdp: Specific differential phase along the ray path
                interp_Z: reflectivity along the ray path
                interp_CC: cross correlation coefficient along ray path
                interp_Zdr_differential reflectivity along ray path
                
            """
    
    def __init__(self,radar_path):
        
        radar = pyart.io.read(radar_path)
        
        #Kdp calculation
        radar.add_field('kdp', pyart.retrieve.kdp_vulpiani(radar,band='S')[0])
        
        #radar variables
        self.Z = radar.fields['reflectivity']['data']
        self.Zdr = radar.fields['differential_reflectivity']['data']
        self.CC = radar.fields['cross_correlation_ratio']['data']
        self.kdp = radar.fields['kdp']['data']
        
        #cartesian grid
        grid = pyart.map.grid_from_radars(radar, 
                                      grid_shape=(40,201,201), 
                                      grid_limits=((0,20000.),
                                                   (-500000.,500000.),
                                                   (-500000.,500000)))
        
        #variables in cartesians
        self.lon = grid.point_longitude['data']
        self.lat = grid.point_latitude['data']
        self.h = grid.point_altitude['data']*1e-3
        self.gZ = grid.fields['reflectivity']['data']
        self.gZdr = grid.fields['differential_reflectivity']['data']
        self.gKdp = grid.fields['kdp']['data']
        self.gCC = grid.fields['cross_correlation_ratio']['data']
        
        def correction_angle(self):
        
            #correction of the elevation angle
            kdp_rad = radar.fields['kdp']['data']
            reflect = radar.fields['reflectivity']['data']
            diff_reflect = radar.fields['differential_reflectivity']['data']
            KDP_rad = []
            Reflec = []
            Diff_reflec = []
            for i in range(kdp_rad.shape[0]):
                coseno = math.cos(math.radians(radar.elevation['data'][i]))
                seno = math.sin(math.radians(radar.elevation['data'][i]))
                zdr = (diff_reflect[i,:]*coseno**4)/(1-np.sqrt(diff_reflect[i,:])*seno**2)**2
                kdp = kdp_rad[i,:]/coseno**2
                KDP_rad = np.append(KDP_rad,kdp)
                Diff_reflec = np.append(Diff_reflec,zdr)
        
            KDP_rad = np.reshape(KDP_rad,(kdp_rad.shape))
            Diff_reflec = np.reshape(Diff_reflec,(diff_reflect.shape))
    
            kdp_dict = {'data': KDP_rad, 'units': 'deg/km', 'long_name': 'Specific Differential Phase',
                            '_FillValue': KDP_rad.fill_value, 'standard_name': 'KDP'}
            
            zdr_dict = {'data': Diff_reflec, 'units': 'dB', 'long_name': 'Differential reflectivity',
                            '_FillValue': Diff_reflec.fill_value, 'standard_name': 'Zdr'}
 
            radar.add_field('kdp_c',kdp_dict)
            radar.add_field('Zdr_c',zdr_dict)
        
            #variables to which the correction of the elevation angle was applied
            self.kdp_corrected = radar.fields['kdp_c']['data']
            self.zdr_corrected = radar.fields['Zdr_c']['data']
            
            self.gKdp = grid.fields['kdp_c']['data']
            self.gZdr = grid.fields['Zdr_c']['data']
            
            return    
        
        
        def interpolation(self,file_col,col_path):
        
            #INTERPOLATION
            file = xr.open_dataset(col_path + '/' + file_col[12:20] + '/' + file_col)  
            
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
            
            
            interp = interpolate_models(ray, self.lon, self.lat, self.h, 
                            {'kdp' :self.gKdp, 'reflectivity': self.gZ,
                             'differential_reflectivity': self.gZdr,
                             'CC': self.gCC})
        
            self.interp_kdp = interp['kdp']
            self.interp_Z = interp['reflectivity']
            self.interp_Zdr = interp['differential_reflectivity']
            self.interp_CC = interp['CC']
