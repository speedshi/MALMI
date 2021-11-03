#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:52:27 2021

Functions related to event catalogs.

@author: shipe
"""


import numpy as np


def catalog_evselect(catalog, timerg=None, latrg=None, lonrg=None, deprg=None):
    """
    To select events from input catalog.

    Parameters
    ----------
    catalog : dict
        Input catalog which contains information of each event therein.
        each parameter should be in numpy array format;
        catalog['id'] : id of the event;
        catalog['time'] : origin time;
        catalog['latitude'] : latitude in degree;
        catalog['longitude'] : logitude in degree;
        catalog['depth_km'] : depth in km;
    timerg : list of datetime, optional
        Origin time range. The default is None.
    latrg : list of float, optional
        latitude range in degree. The default is None.
    lonrg : list of float, optional
        longitude range in degree. The default is None.
    deprg : list of float, optional
        depth range in km. The default is None.

    Returns
    -------
    catalog_s : dict
        The output catalog after event selection.

    """
    
    n_event = len(catalog['time'])  # total number of event in the input catalog
    sindx = np.full((n_event,), True)
        
    # select events according to origin time range
    if timerg is not None:
        sindx_temp = (catalog['time'] >= timerg[0]) & (catalog['time'] <= timerg[1])
        sindx = np.logical_and(sindx, sindx_temp)

    # select events according to latitude range
    if latrg is not None:
        sindx_temp = (catalog['latitude'] >= latrg[0]) & (catalog['latitude'] <= latrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select events according to longitude range
    if lonrg is not None:
        sindx_temp = (catalog['longitude'] >= lonrg[0]) & (catalog['longitude'] <= lonrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
        
    # select events according to depth range
    if deprg is not None:
        sindx_temp = (catalog['depth_km'] >= deprg[0]) & (catalog['depth_km'] <= deprg[1])
        sindx = np.logical_and(sindx, sindx_temp)

    catalog_s = {}
    catakeys = list(catalog.keys())
    for ikey in catakeys:
        catalog_s[ikey] = catalog[ikey][sindx]

    return catalog_s
