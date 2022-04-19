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


def catalog_select(catalog, thrd_cmax=None, thrd_stanum=None, thrd_phsnum=None, thrd_lat=None, thrd_lon=None, thrd_cstd=None, thrd_depth=None):
    """
    This function is used to select events according to input criterions.

    Parameters
    ----------
    catalog : dic
        The input catalog which contains information of each event therein.
        each parameter should be in numpy array format;
        mcatalog['id'] : id of the event;
        mcatalog['time'] : origin time;
        mcatalog['latitude'] : latitude in degree;
        mcatalog['longitude'] : logitude in degree;
        mcatalog['depth_km'] : depth in km;
        mcatalog['coherence_max'] : maximum coherence of migration volume;
        mcatalog['coherence_std'] : standard deviation of migration volume;
        mcatalog['coherence_med'] : median coherence of migration volume;
        mcatalog['starttime'] : detected starttime of the event;
        mcatalog['endtime'] : detected endtime of the event;
        mcatalog['station_num'] : total number of stations triggered of the event;
        mcatalog['phase_num'] : total number of phases triggered of the event;
        mcatalog['dir'] : directory of the migration results of the event.
    thrd_cmax : float, optional
        threshold of minimal coherence. The default is None.
        event coherence must larger than (>=) this threshold.
    thrd_stanum : int, optional
        threshold of minimal number of triggered stations. The default is None.
        event triggered staiton number must large than (>=) this threshold.
    thrd_phsnum : int, optional
        threshold of minimal number of triggered phases. The default is None.
        event triggered phase number must large than (>=) this threshold.
    thrd_lat : list of float, optional
        threshold of latitude range in degree. The default is None.
        event latitude must within this range (include boundary value).
    thrd_lon : list of float, optional
        threshold of longitude range in degree. The default is None.
        event longitude must within this range (include boundary value).
    thrd_cstd: float, optional
        threshold of maximum standard variance of stacking volume. The default is None.
        event stacking volume std must smaller than (<=) this threshold.
    thrd_depth: list of float, optional
        threshold of depth range in km, e.g. [-1, 12]. The default is None.
        event depth must within this range (include boundary value).

    Returns
    -------
    catalog_s : dic
        The catalog containing the selected events.

    """
    
    n_event = len(catalog['time'])  # total number of event in the input catalog
    
    # select events according to the stacking coherence
    if thrd_cmax is not None:
        sindx = (catalog['coherence_max'] >= thrd_cmax)
    else:
        sindx = np.full((n_event,), True)
    
    # select events according to total number of triggered stations
    if thrd_stanum is not None:
        sindx_temp = (catalog['station_num'] >= thrd_stanum)
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select events according to total number of triggered phases
    if thrd_phsnum is not None:
        sindx_temp = (catalog['phase_num'] >= thrd_phsnum)
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select events according to latitude range
    if thrd_lat is not None:
        sindx_temp = (catalog['latitude'] >= thrd_lat[0]) & (catalog['latitude'] <= thrd_lat[1])
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select events according to longitude range
    if thrd_lon is not None:
        sindx_temp = (catalog['longitude'] >= thrd_lon[0]) & (catalog['longitude'] <= thrd_lon[1])
        sindx = np.logical_and(sindx, sindx_temp)
        
    # select events according to standard variance of stacking volume
    if thrd_cstd is not None:
        sindx_temp = (catalog['coherence_std'] <= thrd_cstd)
        sindx = np.logical_and(sindx, sindx_temp)
        
    # select events according to depth range
    if thrd_depth is not None:
        sindx_temp = (catalog['depth_km'] >= thrd_depth[0]) & (catalog['depth_km'] <= thrd_depth[1])
        sindx = np.logical_and(sindx, sindx_temp)
    
    catalog_s = {}
    catakeys = list(catalog.keys())
    for ikey in catakeys:
        catalog_s[ikey] = catalog[ikey][sindx]
    
    return catalog_s


def catalog_matchref(catalog, catalog_ref, thrd_time, thrd_hdis=None, thrd_depth=None, matchmode='time'):
    """
    This function is to compare two input catalogs and match the contained events.

    Input catalog should contain:
        catalog['time']: origin time of each event, in datetime format;
        catalog['longitude'], catalog['latitude']: the latitude and longitude 
        in degree of each event, optional;
        catalog['depth_km']: depth in km of each event, optional;
        catalog['id']: event id of each event, optional.
        catalog['magnitude'] : event magnitude, optional;
    
    None value will be assigned to event with no avaliable information.
        
    NOTE do not modify the input catalogs.

    Parameters
    ----------
    catalog : dict
        the input catalog, usually the newly obtained catalog by users.
    catalog_ref : dict
        the reference catalog for comparison, usually a standard offiical catalog.
    thrd_time : float
        time limit in second, within this limit we can consider two event are identical.
    thrd_hdis : float, optional
        horizontal distance limit in km, within this limit we can consider two event are identical.
        The default is None, means not comparing horizontal distance.
    thrd_depth : float, optional
        depth limit in second, within this limit we can consider two event are identical.
        The default is None, means not comparing depth.
    matchmode : str, optional
        the way to find the best match event when there are multiple events in the reference
        catalog that matches. The default value is 'time'.
        'time' : the closest in origin time;
        'hdist' : the closest in horizontal plane (minimal horizontal distance);
        'dist' : the closest in 3D space;

    Returns
    -------
    catalog_match : dict
        the matched catalog.
        catalog_match['status'] : 'matched' -> find the same event in the reference catalog; 
                                  'new' -> newly detected event not in the reference catalog; 
                                  'undetected' -> missed event that exist in the reference catalog;
        catalog_match['time'], catalog_match['longitude'], catalog_match['latitude'], catalog['id'],
        catalog_match['depth_km'] : information of the 'matched' and the 'new' events in the input catalog;
                                    'undetected' events will have None values for these parameters.
        catalog_match['time_ref'], catalog_match['longitude_ref'], catalog_match['latitude_ref'], catalog['id_ref'],
        catalog_match['depth_km_ref'] : information of the 'matched' and the 'undetected' events in the reference catalog;
                                        'new' events will have None values for these parameters.
        catalog_match['hdist_km'], catalog_match['vdist_km']: the horizontal and vertical/depth distance in km between
                                                              note depth distance can be nagtive: they defined as "catalog_ref - catalog"
                                                              the matched events in the input catalog and the reference catalog.

    """
    
    Nev_cinp = len(catalog['time'])  # number of events in the input catalog
    Nev_cref = len(catalog_ref['time'])  # number of events in the reference catalog
    
    # attached the event ID if the input catalog does not have one
    # default id: linearly increase from 1 to the Number of events
    if 'id' not in catalog:
        catalog['id'] = np.arange(1, Nev_cinp+1)
        
    if 'id' not in catalog_ref:
        catalog_ref['id'] = np.arange(1, Nev_cref+1)
    
    catalog_match = {}  # the output matched catalog
    catalog_match['status'] = []
    catalog_match['time'] = []
    catalog_match['time_ref'] = []
    catalog_match['id'] = []
    catalog_match['id_ref'] = []
    if ('latitude' in catalog) and ('longitude' in catalog):
        catalog_match['latitude'] = []
        catalog_match['latitude_ref'] = []
        catalog_match['longitude'] = []
        catalog_match['longitude_ref'] = []
        catalog_match['hdist_km'] = []
    if ('depth_km' in catalog):
        catalog_match['depth_km'] = []
        catalog_match['depth_km_ref'] = []
        catalog_match['vdist_km'] = []
    if ('magnitude' in catalog):
        catalog_match['magnitude'] = []
        catalog_match['magnitude_ref'] = []
    
    dcevref_id = []
    # loop over each event in the input catalog, compare with events in the reference catalog
    for iev in range(Nev_cinp):
        evtimedfs = np.array([abs(ettemp.total_seconds()) for ettemp in (catalog_ref['time'] - catalog['time'][iev])])  # origin time difference in seconds
        eindx_bool = (evtimedfs <= thrd_time)  # the boolean array indicating whether event origin time matched
        eindx = np.flatnonzero(eindx_bool)  # index of events in the reference catalog which matches the origin time of the current event
        evtimedfs_select = evtimedfs[eindx_bool]  # all the origin time differences in second within the limit

        if (len(eindx) > 0):
            # find events with similar origin times in the reference catalog
            # they could match with the current event
            selid = np.full_like(eindx, True, dtype=bool)
            
            if ('latitude' in catalog) and ('longitude' in catalog):
                # calculate horizontal distance, in km
                hdist_meter = np.zeros((len(eindx),))
                for iii, iievref in enumerate(eindx):
                    hdist_meter[iii], _, _ = gps2dist_azimuth(catalog_ref['latitude'][iievref], catalog_ref['longitude'][iievref], 
                                                             catalog['latitude'][iev], catalog['longitude'][iev])
                hdist_km = (hdist_meter)/1000.0  # meter -> km
            
                if (thrd_hdis is not None):
                    # ckeck if horizontal distance within limit
                    selid_temp = (np.absolute(hdist_km) <= thrd_hdis)
                    selid = np.logical_and(selid, selid_temp)
            
            if ('depth_km' in catalog):
                # calculate vertival/depth distance with sign, in km
                vdist_km = catalog_ref['depth_km'][eindx] - catalog['depth_km'][iev]

                if (thrd_depth is not None):
                    # check if vertical/depth distance within limit
                    selid_temp = (np.absolute(vdist_km) <= thrd_depth)
                    selid = np.logical_and(selid, selid_temp)

            eindx = eindx[selid]
            evtimedfs_select = evtimedfs_select[selid]
            if ('latitude' in catalog) and ('longitude' in catalog):
                hdist_km = hdist_km[selid]
            if ('depth_km' in catalog):
                vdist_km = vdist_km[selid]
            
            if len(eindx) == 0:
                # the current event does not match any event in the reference catalog
                # it should be a newly detected event
                catalog_match['status'].append('new')
                catalog_match['time'].append(catalog['time'][iev])
                catalog_match['time_ref'].append(None)
                catalog_match['id'].append(catalog['id'][iev])
                catalog_match['id_ref'].append(None)
                if ('latitude' in catalog) and ('longitude' in catalog):
                    catalog_match['latitude'].append(catalog['latitude'][iev])
                    catalog_match['latitude_ref'].append(None)
                    catalog_match['longitude'].append(catalog['longitude'][iev])
                    catalog_match['longitude_ref'].append(None)
                    catalog_match['hdist_km'].append(None)
                if ('depth_km' in catalog):
                    catalog_match['depth_km'].append(catalog['depth_km'][iev]) 
                    catalog_match['depth_km_ref'].append(None)
                    catalog_match['vdist_km'].append(None)
                if ('magnitude' in catalog):
                    catalog_match['magnitude'].append(catalog['magnitude'][iev])
                    catalog_match['magnitude_ref'].append(None)
            
            elif len(eindx) == 1:
                # match one event in the reference catalog
                catalog_match['status'].append('matched')
                catalog_match['time'].append(catalog['time'][iev])
                catalog_match['time_ref'].append(catalog_ref['time'][eindx[0]])
                catalog_match['id'].append(catalog['id'][iev])
                catalog_match['id_ref'].append(catalog_ref['id'][eindx[0]])
                if ('latitude' in catalog) and ('longitude' in catalog):
                    catalog_match['latitude'].append(catalog['latitude'][iev])
                    catalog_match['latitude_ref'].append(catalog_ref['latitude'][eindx[0]])
                    catalog_match['longitude'].append(catalog['longitude'][iev])
                    catalog_match['longitude_ref'].append(catalog_ref['longitude'][eindx[0]])
                    catalog_match['hdist_km'].append(hdist_km[0])
                if ('depth_km' in catalog):
                    catalog_match['depth_km'].append(catalog['depth_km'][iev]) 
                    catalog_match['depth_km_ref'].append(catalog_ref['depth_km'][eindx[0]])
                    catalog_match['vdist_km'].append(vdist_km[0])
                if ('magnitude' in catalog):
                    catalog_match['magnitude'].append(catalog['magnitude'][iev])
                    catalog_match['magnitude_ref'].append(catalog_ref['magnitude'][eindx[0]])
                
                dcevref_id.append(eindx[0])  # add the event_ref index in the detection list
                
            elif len(eindx) > 1:
                # more then one event matched
                # need to define which one matches the best
                if (matchmode == 'time') or ('latitude' not in catalog) or ('longitude' not in catalog):
                    # best matched event is the closest in origin time
                    ssid = np.argmin(evtimedfs_select)
                elif matchmode == 'hdist':
                    # best matched event is the closest in horizonal plane
                    ssid = np.argmin(np.absolute(hdist_km))
                elif matchmode == 'dist':
                    # best matched event is the closest in 3D space
                    ssid = np.argmin(np.sqrt(hdist_km*hdist_km + vdist_km*vdist_km))
                else:
                    raise ValueError('Input of matchmode is unrecognized!')
                
                catalog_match['status'].append('matched')
                catalog_match['time'].append(catalog['time'][iev])
                catalog_match['time_ref'].append(catalog_ref['time'][eindx[ssid]])
                catalog_match['id'].append(catalog['id'][iev])
                catalog_match['id_ref'].append(catalog_ref['id'][eindx[ssid]])
                if ('latitude' in catalog) and ('longitude' in catalog):
                    catalog_match['latitude'].append(catalog['latitude'][iev])
                    catalog_match['latitude_ref'].append(catalog_ref['latitude'][eindx[ssid]])
                    catalog_match['longitude'].append(catalog['longitude'][iev])
                    catalog_match['longitude_ref'].append(catalog_ref['longitude'][eindx[ssid]])
                    catalog_match['hdist_km'].append(hdist_km[ssid])
                if ('depth_km' in catalog):
                    catalog_match['depth_km'].append(catalog['depth_km'][iev]) 
                    catalog_match['depth_km_ref'].append(catalog_ref['depth_km'][eindx[ssid]])
                    catalog_match['vdist_km'].append(vdist_km[ssid])
                if ('magnitude' in catalog):
                    catalog_match['magnitude'].append(catalog['magnitude'][iev])
                    catalog_match['magnitude_ref'].append(catalog_ref['magnitude'][eindx[ssid]])
                
                dcevref_id.append(eindx[ssid])  # add the event_ref index in the detection list
            
        else:
            # the current event does not match any event in the reference catalog
            # it should be a newly detected event
            catalog_match['status'].append('new')
            catalog_match['time'].append(catalog['time'][iev])
            catalog_match['time_ref'].append(None)
            catalog_match['id'].append(catalog['id'][iev])
            catalog_match['id_ref'].append(None)
            if ('latitude' in catalog) and ('longitude' in catalog):
                catalog_match['latitude'].append(catalog['latitude'][iev])
                catalog_match['latitude_ref'].append(None)
                catalog_match['longitude'].append(catalog['longitude'][iev])
                catalog_match['longitude_ref'].append(None)
                catalog_match['hdist_km'].append(None)
            if ('depth_km' in catalog):
                catalog_match['depth_km'].append(catalog['depth_km'][iev]) 
                catalog_match['depth_km_ref'].append(None)
                catalog_match['vdist_km'].append(None)
            if ('magnitude' in catalog):
                catalog_match['magnitude'].append(catalog['magnitude'][iev])
                catalog_match['magnitude_ref'].append(None)

    # find and merge undetected events which exist in the reference catalog into the final matched catalog
    for ieref in range(Nev_cref):
        if ieref not in dcevref_id:
            # the event is not detected in the input catalog
            catalog_match['status'].append('undetected')
            catalog_match['time'].append(None)
            catalog_match['time_ref'].append(catalog_ref['time'][ieref])
            catalog_match['id'].append(None)
            catalog_match['id_ref'].append(catalog_ref['id'][ieref])
            if ('latitude' in catalog) and ('longitude' in catalog):
                catalog_match['latitude'].append(None)
                catalog_match['latitude_ref'].append(catalog_ref['latitude'][ieref])
                catalog_match['longitude'].append(None)
                catalog_match['longitude_ref'].append(catalog_ref['longitude'][ieref])
                catalog_match['hdist_km'].append(None)
            if ('depth_km' in catalog):
                catalog_match['depth_km'].append(None) 
                catalog_match['depth_km_ref'].append(catalog_ref['depth_km'][ieref])
                catalog_match['vdist_km'].append(None)
            if ('magnitude' in catalog):
                catalog_match['magnitude'].append(None)
                catalog_match['magnitude_ref'].append(catalog_ref['magnitude'][ieref])

    # convert to numpy array
    catalog_match['status'] = np.array(catalog_match['status'])
    catalog_match['time'] = np.array(catalog_match['time'])
    catalog_match['time_ref'] = np.array(catalog_match['time_ref'])
    catalog_match['id'] = np.array(catalog_match['id'])
    catalog_match['id_ref'] = np.array(catalog_match['id_ref'])
    if ('latitude' in catalog_match) and ('longitude' in catalog_match):
        catalog_match['latitude'] = np.array(catalog_match['latitude'])
        catalog_match['latitude_ref'] = np.array(catalog_match['latitude_ref'])
        catalog_match['longitude'] = np.array(catalog_match['longitude'])
        catalog_match['longitude_ref'] = np.array(catalog_match['longitude_ref'])
        catalog_match['hdist_km'] = np.array(catalog_match['hdist_km'])
    if ('depth_km' in catalog_match):
        catalog_match['depth_km'] = np.array(catalog_match['depth_km'])
        catalog_match['depth_km_ref'] = np.array(catalog_match['depth_km_ref'])
        catalog_match['vdist_km'] = np.array(catalog_match['vdist_km'])
    if ('magnitude' in catalog_match):
        catalog_match['magnitude'] = np.array(catalog_match['magnitude'])
        catalog_match['magnitude_ref'] = np.array(catalog_match['magnitude_ref'])

    return catalog_match




