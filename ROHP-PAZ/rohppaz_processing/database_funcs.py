#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:54:59 2021

@author: padulles
"""

import numpy as np
import glob
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from configobj import ConfigObj
import os,sys
sys.path.append('/media/antia/easystore/ROHP-PAZ/rohppaz_processing/rohppaz/lib/occult')
#local imports
import paths

pathinsert = paths.projectPath()
config = ConfigObj(str(pathinsert.joinpath('rohppaz','lib','database',
                                           'database_config.ini')))



def engine_db(db=None):
    
    unix_socket="?unix_socket=/var/run/mysqld/mysqld.sock"
    if sys.version_info.major == 3:
        engine_ = "mysql+pymysql://"
    else:
        engine_ = "mysql://"
    engine_ += 'pazopr'
    engine_ += ':'
    engine_ +='pazesfantastic!'
    engine_ += "@localhost/"
    if db==None:
        engine_ += 'rohppaz_db'
    else:
        engine_ +=db
        
    engine_ += unix_socket
    engine = create_engine(engine_)
    
    return engine

def engine_nodb():
    
    unix_socket="?unix_socket=/var/run/mysqld/mysqld.sock"
    if sys.version_info.major == 3:
        engine_ = "mysql+pymysql://"
    else:
        engine_ = "mysql://"
    engine_ += config['db_default']['username']
    engine_ += ':'
    engine_ += config['db_default']['password']
    engine_ += "@localhost/"
        
    engine_ += unix_socket
    engine = create_engine(engine_)
    
    return engine

def get_data_db(sql_query,db=None):
    
    engine = engine_db(db=db)
    
    df = pd.read_sql(sql_query, con = engine)
    engine.dispose()
    return df

##################
def get_radar_station(roid,db=None):
    
    sql_query = "SELECT * FROM collocations_with_nexrad WHERE roid = '"+roid+"';"
    df = get_data_db(sql_query,db=db)
    station_name = df.station_id[0]
    
    return station_name
###################

def domain_vars(roid):
    """ function for obtaining vars of iceCol files when there is 
    no iceCol files """
    
    #date
    year = "SELECT year from pro_events WHERE roid = '"+roid+"';"
    month = "SELECT month from pro_events WHERE roid = '"+roid+"';"
    day = "SELECT day from pro_events WHERE roid = '"+roid+"';"
    hour = "SELECT hour from pro_events WHERE roid = '"+roid+"';"
    lat = "SELECT lat from pro_events WHERE roid = '"+roid+"';"
    lon = "SELECT lon from pro_events WHERE roid = '"+roid+"';"
    
    try:
        y = get_data_db(year,db='rohppaz_db')
        m = get_data_db(month,db='rohppaz_db')
        d = get_data_db(day,db='rohppaz_db')
        h = get_data_db(hour,db='rohppaz_db')
        latitude = get_data_db(lat,db='rohppaz_db')
        longitude = get_data_db(lon,db='rohppaz_db')
    except:
        y = get_data_db(year,db='rohppaz_db_nrt')
        m = get_data_db(month,db='rohppaz_db_nrt')
        d = get_data_db(day,db='rohppaz_db_nrt')
        h = get_data_db(hour,db='rohppaz_db_nrt')
        latitude = get_data_db(lat,db='rohppaz_db_nrt')
        longitude = get_data_db(lon,db='rohppaz_db_nrt')
    
    return y.year[0], m.month[0], (d.day[0]), h.hour[0], latitude.lat[0], longitude.lon[0]

def get_hflag(roid):
    df=get_data_db("SELECT height_flag_smooth from stats_phase WHERE roid= '"+roid+"';"
                 , db='rohppaz_db_nrt')

    hflag = df.values.astype(float)
    
    return hflag

def get_dphaseSpire(roid):
    df=get_data_db("SELECT * from calibrated_phase WHERE roid='"+roid+
                          "' AND variable = 'phase_lin';", db='spirepro_db')
    dph = df.values[0,2:402].astype(float)
    return dph

def get_dphase(roid):

    if 'PAZ' in roid:
        try:
            dphase = get_dphaseCal(roid,db='rohppaz_db' )
        except:
            dphase = get_dphaseCal(roid,db='rohppaz_db_nrt')
          
    else:
        dphase = get_dphaseSpire(roid)

    return dphase

def get_dphaseCal(roid,db=None):
    
    if 'PAZ1' in roid:
        sql_query = "SELECT * FROM calibrated_phase WHERE \
             variable = 'phase' AND roid = '"+roid+"';"
        try:    
            df_phase = get_data_db(sql_query,db='rohppaz_db')
        except:
            df_phase = get_data_db(sql_query,db='rohppaz_db_nrt')
    else:
        sql_query = "SELECT * FROM calibrated_phase WHERE \
                 variable = 'phase_cal' AND roid = '"+roid+"';"
                 
        df_phase = get_data_db(sql_query,db='spirepro_db')
    
    
    try:
        phase = np.array(df_phase.iloc[0,2:].astype(float))
    except:
        df_phase = get_data_db(sql_query,db='rohppaz_db_nrt')
        phase = np.array(df_phase.iloc[0,2:].astype(float))
  
    return phase


def get_dphaseCal_jpl(jplid,db=None):
    sql_query = "SELECT * FROM jpl_calibrated_phase WHERE jplid = '"+jplid+"';"
    df_phase = get_data_db(sql_query,db=db)
    phase = np.array(df_phase.iloc[0,2:].astype(float))
    return phase

def get_wetTemp(roid,db=None):
    sql_query = "SELECT * FROM wet_profiles WHERE roid = '"+roid+"' \
                 AND variable='wet_Temp';"
    wetTemp = get_data_db(sql_query,db=db)
    temperature = np.array(wetTemp.iloc[0,2:].astype(float))
    return temperature

def get_wetVp(roid,db=None):
    sql_query = "SELECT * FROM wet_profiles WHERE roid = '"+roid+"' \
                 AND variable='wet_Vp';"
    wetVp = get_data_db(sql_query,db=db)
    vp = np.array(wetVp.iloc[0,2:].astype(float))
    return vp

def get_wetPres(roid,db=None):
    sql_query = "SELECT * FROM wet_profiles WHERE roid = '"+roid+"' \
                 AND variable='wet_Pres';"
    wetPres = get_data_db(sql_query,db=db)
    p = np.array(wetPres.iloc[0,2:].astype(float))
    return p

def table_names():
    
    table_name = dict(config['db_tables'])
    return table_name

def column_types():
    
    config_dtype = ConfigObj(str(pathinsert.joinpath('rohppaz','lib','database',
                                           'database_columntype.ini')))
    typevar={}
    for c in np.sort(config_dtype.keys()):
        typevar[c]=dict(config_dtype[c])
    
    return typevar

def dataFrame_names():

    config_dtype = ConfigObj(str(pathinsert.joinpath('rohppaz','lib','database',
                                           'database_columntype.ini')))
    typevar={}
    for c in np.sort(config_dtype.keys()):
        cc = int(c[c.rfind('_')+1:])
        typevar[cc]=dict(config_dtype[c])
    columnNames = {}
    for c in np.sort(list(typevar.keys())):
        columnNames[c] = np.sort(list(typevar[c].keys()))
    
    #order dicts:
    for c in [7,9,10]:
        tmpcol = columnNames[c]
        columnNames[c]=['roid','variable']
        columnNames[c].extend(tmpcol[:-2])
    c=12
    tmpcol = columnNames[c]
    columnNames[c]=['jplid','variable']
    columnNames[c].extend(tmpcol[:-2])
    
    return columnNames, typevar

def date_from_roid(roid):
    
    dt = datetime.strptime(roid[5:19], "%Y.%j.%H.%M")
    return dt

def date_from_jplid(jplid):
    
    dt = datetime.strptime(jplid[:-7],"%Y%m%d_%H%M")
    return dt

def get_latestdata_table(table_num):
    
    table_name = table_names()
    t_name = table_name[str(table_num)]
    
    sql_query = "select MAX(pro_events.startimeUTC) as maxT from pro_events "
    
    if t_name != 'pro_events':
        
        if ((table_num==11) | (table_num==12)):
            sql_query += "RIGHT JOIN ucar_jpl ON pro_events.roid = ucar_jpl.roid "
            sql_query += "LEFT JOIN "+t_name+" ON ucar_jpl.jplid = "+t_name+".jplid "
        else:
            sql_query += "RIGHT JOIN "+t_name+" ON pro_events.roid = "+t_name+".roid "
        
    sql_query+=';'
    
    df = get_data_db(sql_query)
    latest_dt = pd.to_datetime(df.maxT).iloc[0].to_pydatetime()

    return latest_dt

def to_db(df,table_num, db=None):
    
    engine=engine_db(db=db)
    
    table_name = table_names()
    
    try: df.empty
    except: return
            
    if df.empty==False:
        df.to_sql(table_name[str(table_num)], 
                  con=engine,
                  if_exists='append', 
                  index=False, 
                  method='multi',
                  chunksize=500)

def check_type_column(df,t_num):
    
    typevar = column_types()
    df = df.astype(typevar['db_columns_'+str(t_num)])
    
    return df

