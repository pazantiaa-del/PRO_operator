#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:28:45 2021

@author: padulles
"""

import numpy as np

def distlatlon(lat1,lon1,lat2,lon2,R=6371.):
  """
  #Distance in km between two points along a great circle, calculated using the "haversine" method (Roger Sinnott, 1984)
  """

  lat1=np.radians(lat1)
  lon1=np.radians(lon1)
  lat2=np.radians(lat2)
  lon2=np.radians(lon2)
  
  dLat=lat2-lat1
  dLon=lon2-lon1
  
  a=np.sin(dLat/2.)*np.sin(dLat/2.) + np.sin(dLon/2.)*np.sin(dLon/2.)*np.cos(lat1)*np.cos(lat2)
  c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
  d=R*c
  
  return d

def distlatlonhei(lat,lon,hei,lat2,lon2,hei2,R=6371.):
    d_horiz=distlatlon(lat,lon,lat2,lon2,R=6371.)
    d_verti=(hei-hei2)
    
    dist=np.sqrt(d_horiz**2 + d_verti**2)
    
    return dist

def angle_v(lat,lon,hei):
    
    d_horiz1=distlatlon(lat[:,1:],lon[:,1:],lat[:,:-1],lon[:,:-1],R=6371.)
    d_verti1=(hei[:,1:]-hei[:,:-1])
    
    d_horiz2=distlatlon(lat[:,2:],lon[:,2:],lat[:,:-2],lon[:,:-2],R=6371.)
    d_verti2=(hei[:,2:]-hei[:,:-2])
    
    ang1=np.arctan(d_verti1/d_horiz1)*180/np.pi
    ang2=np.arctan(d_verti2/d_horiz2)*180/np.pi
    
    angle_v=np.zeros(lat.shape)
    angle_v[:,1:-1]=(ang1[:,:-1]+ang2)/2.
    angle_v[:,[0,-1]]=np.array([ang1[:,0],ang1[:,-1]]).T
    
    return angle_v
    

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    try:
        return vector/(np.linalg.norm(vector,axis=1).reshape(-1,1))
    except:
        return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))