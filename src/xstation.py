#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:30:03 2022

Station related functions.

For station inventory of simple dictory format, the current convention is:
stadict['network']: np.array of str, network code of each station;
stadict['station']: np.array of str, station code of each station;
stadict['latitude']: np.array of float, latitude in decimal degree of each station;
stadict['longitude']: np.array of float, longitude in decimal degree of each station;
stadict['elevation']: np.array of float, elevation in meters relative to the sea-level (positive for up) of each station;

@author: shipe
"""


import numpy as np
from obspy import read_inventory
from obspy.core.inventory import Inventory, Network, Station
import pandas as pd


def station_select(station, latrg=None, lonrg=None, elerg=None):
    """
    To select stations according to the input criterion.

    Parameters
    ----------
    station : dict
        Input station dict which contains information of each station therein.
        each parameter should be in numpy array format;
        station['station'] : station code (name) of each station;
        station['latitude'] : latitude in degree;
        station['longitude'] : logitude in degree;
        station['elevation'] : elevation in meter;
    latrg : list of float, optional
        latitude range in degree, [lat_min, lat_max].
        The default is None.
    lonrg : list of float, optional
        longitude range in degree, [lon_min, lon_max].
        The default is None.
    elerg : list of float, optional
        elevation range in meter, [ele_min, ele_max].
        The default is None.

    Returns
    -------
    station_s : dict
        The output stations after event selection.

    """
    
    NN = len(station['station'])  # total number of stations
    sindx = np.full((NN,), True)

    # select stations according to latitude range
    if latrg is not None:
        sindx_temp = (station['latitude'] >= latrg[0]) & (station['latitude'] <= latrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select stations according to longitude range
    if lonrg is not None:
        sindx_temp = (station['longitude'] >= lonrg[0]) & (station['longitude'] <= lonrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
        
    # select stations according to elevation range
    if elerg is not None:
        sindx_temp = (station['elevation'] >= elerg[0]) & (station['elevation'] <= elerg[1])
        sindx = np.logical_and(sindx, sindx_temp)

    station_s = {}
    for ikey in list(station.keys()):
        station_s[ikey] = station[ikey][sindx]

    return station_s


def stainv2stadict(stainv):
    """
    Transform obspy station inventory to python dictory.

    Parameters
    ----------
    stainv : obspy station inventory object
        station inventory.

    unique station is identified by network.station.location;
    unique station should have the same depth;

    Returns
    -------
    stadict : dict
        station inventory.
        stadict['network']: network code of each station;
        stadict['station']: station code of each station;
        stadict['latitude']: latitude in decimal degree of each station;
        stadict['longitude']: longitude in decimal degree of each station;
        stadict['elevation']: elevation in meters relative to the sea-level (positive for up) of each station;
        stadict['location']: location code of each station;
        stadict['depth']: depth in meter of each station;
        stadict['channel']: channel code of each station;
    """
    
    stadict = {}
    stadict['network'] = []
    stadict['station'] = []
    stadict['location'] = []
    stadict['latitude'] = [] 
    stadict['longitude'] = []
    stadict['elevation'] = []
    stadict['depth'] = []
    stadict['channel'] = []
    
    for inet in stainv:
        for ista in inet:
            stadict['network'].append(inet.code)
            stadict['station'].append(ista.code)
            stadict['latitude'].append(ista.latitude)
            stadict['longitude'].append(ista.longitude)
            stadict['elevation'].append(ista.elevation)
            if len(ista.channels) > 0:
                # have channel information
                # add location code, depth, instrument code, and component code
                locations = []
                channels = []
                depths = []
                for icha in ista:
                    locations.append(icha.location_code)
                    channels.append(icha.code)
                    depths.append(icha.depth)

            else:
                stadict['location'].append(None)
                stadict['depth'].append(None)
    
    # # convert to numpy array
    # for ikey in list(stadict.keys()):
    #     stadict[ikey] = np.array(stadict[ikey])
    
    return stadict


def load_station(file_station, outformat='obspy'):
    """
    To read in station metadata and returns an obspy invertory object.

    Parameters
    ----------
    file_station : str
        filename (inclusing path) of the station metadata. 
        The data format should be recognizable by ObsPy, such as:
            FDSNWS station text format: *.txt,
            FDSNWS StationXML format: *.xml.
        or a simply CSV file using ',' as the delimiter in which the first row 
        is column name and must contain: 'network', 'station', 'latitude', 
        'longitude', 'elevation'. Latitude and longitude are in decimal degree 
        and elevation in meters relative to the sea-level (positive for up). 
    outformat : str
        specify the format of the loaded station invery in memory;
        'obspy': obspy station inventory object;
        'dict': simple python dictory format;
    
    Returns
    -------
    stainfo : obspy invertory object
        contains station information, such as network code, station code, 
        longitude, latitude, evelvation etc.

    """
    
    # read station metadata
    stafile_suffix = file_station.split('.')[-1]
    if stafile_suffix.lower() == 'xml':
        # input are in StationXML format
        stainfo = read_inventory(file_station, format="STATIONXML")
    elif stafile_suffix.lower() == 'txt':
        # input are in FDSNWS station text file format
        stainfo = read_inventory(file_station, format="STATIONTXT")
    elif stafile_suffix.lower() == 'csv':
        # simple CSV format
        stainfo = read_stainv_csv(file_station)
    else:
        raise ValueError('Wrong input for input inventory file: {}! Format not recognizable!'.format(file_station))
    
    if outformat.lower() == 'dict':
        # obspy station inventory to dict
        stainfo = stainv2stadict(stainfo)
    
    return stainfo


def read_stainv_csv(file_stainv):
    """
    Read the csv format station inventory file, and format it to obspy station inventory object.

    The input CSV file using ',' as the delimiter in which the first row 
    is column name and must contain: 'network', 'station', 'latitude', 
    'longitude', 'elevation'. Latitude and longitude are in decimal degree 
    and elevation in meters relative to the sea-level (positive for up).

    Parameters
    ----------
    file_stainv : str
        filename of the input station inventory of csv format.

    Returns
    -------
    stainv : obspy station inventory object.

    """
    
    stainv = Inventory(networks=[])
    stadf = pd.read_csv(file_stainv, delimiter=',', encoding='utf-8', 
                        header="infer", skipinitialspace=True)
    net_dict = {}
    for rid, row in stadf.iterrows():
        if row['network'] not in net_dict.keys():
            net = Network(code=row['network'], stations=[])
            net_dict[row['network']] = net
        sta = Station(code=row['station'], latitude=row['latitude'], 
                      longitude=row['longitude'], elevation=row['elevation'])
        net_dict[row['network']].stations.append(sta)
    
    for inet in net_dict.keys():
        stainv.networks.append(net_dict[inet])
    
    return stainv


