#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:30:03 2022

Station related functions.

@author: shipe
"""


import numpy as np


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
    catakeys = list(station.keys())
    for ikey in catakeys:
        station_s[ikey] = station[ikey][sindx]

    return station_s

