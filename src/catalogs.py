#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:52:27 2021

Functions related to event catalogs.

@author: shipe
"""


import numpy as np
from obspy.geodetics import gps2dist_azimuth


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


def catalog_rmrpev(catalog, thrd_time=0.3, thrd_hdis=None, thrd_depth=None, evkp=None):
    """
    To remove the repeated events in the catalog.

    Parameters
    ----------
    catalog : dict,
        input catalog containing event information.
        catalog['time'] : origin time;
        catalog['latitude'] : latitude in degree;
        catalog['longitude'] : logitude in degree;
        catalog['depth_km'] : depth in km;
        catalog['coherence_max'] : stacking coherence of the events;
        catalog['magnitude'] : event magnitude;
    thrd_time : float, optional
        time threshold in second. Events with origin time differences within 
        this threshold are considered to be repeated events. 
        The default is 0.3.
    thrd_hdis : float, optional
        horizontal distance threshold in km. Events with horizontal distance differences
        within this threshold are considered to be repeated events. 
        The default is None, meaning don't consider the horizontal distance.
    thrd_depth : TYPE, optional
        depth/vertical distance threshold in km. Events with depth distance differences
        within this threshold are considered to be repeated events. 
        The default is None, meaning don't consider the depth distance.
    evkp : str, optional
        specify the key for comparing events, the event with a large value will be kept.
        For example, if evkp='coherence_max', then repeated events with a larger stacking coherence value will be kept;
        if evkp='magnitude', then repeated events with a larger magnitude will be kept.
        The default is None, meaning keep the first event.
    
    Returns
    -------
    catalog_new : dict
        output catalog after removing repeated events.

    """
    
    
    Nev = len(catalog['time'])  # the total number of events in the catalog
    catakeys = list(catalog.keys())
    
    evidlist = []
    catalog_new = {}
    for ikey in catakeys:
        catalog_new[ikey] = []
    
    for iev in range(Nev):
        if iev not in evidlist:
            evtimedfs = np.array([abs(ettemp.total_seconds()) for ettemp in (catalog['time'][iev+1:] - catalog['time'][iev])])  # origin time difference in seconds
            eindx_bool = (evtimedfs <= thrd_time)  # the boolean array indicating whether event origin time match
            eindx = np.flatnonzero(eindx_bool) + iev + 1  # index of events in catalog which matches the origin time of the current event
        
            if (len(eindx) > 0):
                # further check if the location match
                selid = np.full_like(eindx, True, dtype=bool)
                
                if ('latitude' in catalog) and ('longitude' in catalog):
                    # calculate horizontal distance, in km
                    hdist_meter = np.zeros((len(eindx),))
                    for iii, iievref in enumerate(eindx):
                        hdist_meter[iii], _, _ = gps2dist_azimuth(catalog['latitude'][iievref], catalog['longitude'][iievref], 
                                                                 catalog['latitude'][iev], catalog['longitude'][iev])
                    hdist_km = abs(hdist_meter)/1000.0  # meter -> km
                
                    if (thrd_hdis is not None):
                        # ckeck if horizontal distance within limit
                        selid_temp = (hdist_km <= thrd_hdis)
                        selid = np.logical_and(selid, selid_temp)
                
                if ('depth_km' in catalog):
                    # calculate vertival/depth distance with sign, in km
                    vdist_km = catalog['depth_km'][eindx] - catalog['depth_km'][iev]
    
                    if (thrd_depth is not None):
                        # check if vertical/depth distance within limit
                        selid_temp = (np.absolute(vdist_km) <= thrd_depth)
                        selid = np.logical_and(selid, selid_temp)
                
                eindx = eindx[selid]
                
                if (len(eindx) == 0):
                    # no matched events, add the current event into the new catalog
                    for ikey in catakeys:
                        catalog_new[ikey].append(catalog[ikey][iev])
                    
                elif ((evkp is None) or (sum(catalog[evkp][eindx] > catalog[evkp][iev]) == 0)):
                    # have matched events, and the current event is the best one to keep
                    for ikey in catakeys:
                        catalog_new[ikey].append(catalog[ikey][iev])
                    evidlist.append(*eindx)
                    print('Repeated event found at: ', catalog['time'][iev])
                else:
                    # have matched events, the best event to keep is in 'eindx', to be checked in the later loop
                    print('Repeated event found at: ', catalog['time'][iev])
            else:
                # no matched events, add the current events into the new catalog
                for ikey in catakeys:
                    catalog_new[ikey].append(catalog[ikey][iev])
    
    # convert to numpy array
    for ikey in catakeys:
        catalog_new[ikey] = np.array(catalog_new[ikey])
       
    return catalog_new

