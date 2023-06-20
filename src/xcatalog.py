#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:52:27 2021

Functions related to event catalogs.

For event catalog of simple dictory format, the current convention is:
catalog['id']: np.array of str, id of each event;
catalog['time']: np.array of UTCDateTime, origin time of each event;
catalog['latitude']: np.array of float, latitude in degree of each event;
catalog['longitude']: np.array of float, logitude in degree of each event;
catalog['depth_km']: np.array of float, depth in km below the sea-level (down->positive) of each event;
catalog['magnitude']: np.array of float, magnitude of each event;
catalog['magnitude_type']: np.array of str or str, magnitude type, e.g. 'M', 'ML', 'Mw';
catalog['pick']: np.array or list of dict, len(list or np.array) = number of events, each dict is the picking results, e.g. dict['network.station.location.channel_code']['P'] for P-phase pick time, dict['network.station.location.channel_code']['S'] for S-phase pick time;
catalog['arrivaltime']: np.array or list of dict, similar to 'pick', but is the theoretical calculated arrivaltimes of P- and S-phases;

@author: shipe
"""


import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
import os
import glob
from ioformatting import read_lokicatalog, read_malmipsdetect, dict2csv, csv2dict, read_arrivaltimes
from utils_dataprocess import get_picknumber, pickarrvt_rmsd, pickarrvt_mae
import pickle
import copy
import datetime
import warnings
from obspy import UTCDateTime
from obspy.core.event import read_events
from obspy.core.event import Catalog as obspy_Catalog
from obspy.core.event import Event as obspy_Event
from obspy.core.event import Origin as obspy_Origin
from obspy.core.event import Magnitude as obspy_Magnitude
from obspy.core.event import Pick as obspy_Pick
from obspy.core.event import WaveformStreamID
from obspy.core.event import Arrival as obspy_Arrival
from obspy.core.event import OriginQuality as obspy_OriginQuality
from obspy.core.event.base import Comment
from xpick import picks_select


def retrive_catalog(dir_dateset, cata_ftag='catalogue', dete_ftag='event_station_phase_info.txt', cata_fold='*', dete_fold='*', search_fold=None, evidtag='malmi', picktag='.MLpicks', arrvttag='.phs'):
    """
    This function is used to concatenate the catalogs together from the data base.

    Parameters
    ----------
    dir_dateset : str
        Path to the parent folder of catalog file and phase file.
        (corresponding to the 'dir_MIG' folder in the MALMI main.py script).
    cata_ftag : str, optional
        Catalog filename. The default is 'catalogue'.
    dete_ftag : str, optional
        Detection filename. The default is 'event_station_phase_info.txt'.
    cata_fold : str, optional
        Catalog-file parent folder name. The default is '*'.
        (corresponding to the 'fld_migresult' folder in the MALMI main.py script).
    dete_fold : str, optional
        Detection-file parent folder name. The default is '*'.
        (corresponding to the 'fld_prob' folder in the MALMI main.py script).
    search_fold : list of str, optional
        The MALMI result folders which contains catalog files. 
        (corresponding to the 'fd_seismic' folder in the MALMI main.py script).
        The default is None, which means all avaliable folders in 'dir_dateset'.
    evidtag : str, default is 'malmi'
        event id tage;
    picktag : str, default is '.MLpicks'
        picking file filename tage for extracting picks;
        if None, not extracting picks;
    arrvttag : str, default is '.phs'
        arrivaltime file filename tage for extracting theoretical arrivaltimes;
        if None, not extracting arrivaltimes;

    Returns
    -------
    mcatalog : dic
        The final obtained catalog which concatenates all the catalog files.
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
        mcatalog['dir'] : directory of the migration results of the event;
        mcatalog['pick'] : picking time;
        mcatalog['arrivaltime'] : theoretical arrivaltimes;
        mcatalog['asso_station_all'] : total number of stations associated with picks;
        mcatalog['asso_station_PS'] : total number of stations associated with both P and S picks;
        mcatalog['asso_station_P'] : total number of stations associated with only P picks;
        mcatalog['asso_station_S'] : total number of stations associated with only S picks;
        mcatalog['asso_P_all'] : total number of P picks;
        mcatalog['asso_S_all'] : total number of S picks;
        mcatalog['asso_phase_all'] : total number of phase picks;
        mcatalog['rms_pickarvt'] : the root-mean-square deviation between picking times and theoretical arrivaltimes;   
        mcatalog['mae_pickarvt'] : the mean absolute error between picking times and theoretical arrivaltimes;
    """
    
    assert(os.path.exists(dir_dateset))
    
    if search_fold is None:
        file_cata = sorted(glob.glob(os.path.join(dir_dateset, '**/{}/{}'.format(cata_fold,cata_ftag)), recursive=True))  # file list of catalogue files
        file_dete = sorted(glob.glob(os.path.join(dir_dateset, '**/{}/{}'.format(dete_fold,dete_ftag)), recursive=True))  # file list of detection files
    elif isinstance(search_fold, list) and (isinstance(search_fold[0], str)):
        file_cata = []
        file_dete = []
        for ifld in search_fold:
            file_cata += sorted(glob.glob(os.path.join(dir_dateset, '{}/{}/{}'.format(ifld,cata_fold,cata_ftag)), recursive=True))  # file list of catalogue files
            file_dete += sorted(glob.glob(os.path.join(dir_dateset, '{}/{}/{}'.format(ifld,dete_fold,dete_ftag)), recursive=True))  # file list of detection files
    else:
        raise ValueError('Wrong input format for: {}! Can only be None or list of str!'.format(search_fold))
    
    assert(len(file_cata) == len(file_dete))  # should correspond
        
    eid = 0
    # loop over each catalog/detection file and concatenate them together to make the final version
    for ii in range(len(file_cata)):
        assert(file_cata[ii].split('/')[-3] == file_dete[ii].split('/')[-3])  # they should share the common parent path
            
        # load catalog file
        ctemp = read_lokicatalog(file_cata[ii])
        
        # load detection file
        dtemp = read_malmipsdetect(file_dete[ii])
        
        if ii == 0:
            # initialize the final catalog
            mcatalog = {}
            mcatalog['id'] = []  # id of the event
            for ickey in ctemp:
                # inherit keys from loki catalog dictionary
                mcatalog[ickey] = []
            for idkey in dtemp:
                # inherit keys from detection dictionary
                mcatalog[idkey] = []
            mcatalog['dir'] = []  # directory of the migration results of the event
            if picktag is not None:
                mcatalog['pick'] = []
                mcatalog['asso_station_all'] = []
                mcatalog['asso_station_PS'] = []
                mcatalog['asso_station_P'] = []
                mcatalog['asso_station_S'] = []
                mcatalog['asso_P_all'] = []
                mcatalog['asso_S_all'] = []
                mcatalog['asso_phase_all'] = []
            if arrvttag is not None:
                mcatalog['arrivaltime'] = []
            if (picktag is not None) and (arrvttag is not None):
                mcatalog['rms_pickarvt'] = []  # the root-mean-square deviation between picking times and theoretical arrivaltimes    
                mcatalog['mae_pickarvt'] = []  # the mean absolute error between picking times and theoretical arrivaltimes
                
        if ctemp:  # not empty catalog
            assert(len(ctemp['time']) == len(dtemp['phase_num']))  # event number should be the same
            for iev in range(len(ctemp['time'])):
                assert(ctemp['time'][iev] <= dtemp['endtime'][iev])
                
                for ickey in ctemp:
                    # inherit keys from loki catalog dictionary
                    mcatalog[ickey].append(ctemp[ickey][iev])
                for idkey in dtemp:
                    # inherit keys from detection dictionary
                    mcatalog[idkey].append(dtemp[idkey][iev])
                eid = eid + 1  # start from 1
                evtimeid = ctemp['time'][iev].strftime('%Y%m%d%H%M%S%f')  # UTCDateTime to string
                mcatalog['id'].append('{}_{}_{:06d}'.format(evidtag, evtimeid, eid))  # generate event identification
                sss = file_cata[ii].split(os.path.sep)
                if sss[0] == '':
                    # the input path: 'dir_dateset' is a absolute address
                    dir_ers = '{}'.format(os.path.sep)
                else:
                    # the input path: 'dir_dateset' is a relative address
                    dir_ers = ''
                    
                for pstr in sss[:-1]:
                    dir_ers = os.path.join(dir_ers, pstr)
                dir_ers =  os.path.join(dir_ers, dtemp['starttime'][iev].isoformat())
                assert(os.path.exists(dir_ers))
                mcatalog['dir'].append(dir_ers)  # migration result direcotry of the event
                
                if picktag is not None:
                    # load picking time
                    file_pkev = glob.glob(os.path.join(dir_ers, '*{}*'.format(picktag)))
                    assert(len(file_pkev)==1)
                    pick_iev = read_arrivaltimes(file_pkev[0])
                    mcatalog['pick'].append(pick_iev)
                    num_station_all, num_station_PS, num_station_P, num_station_S, num_P_all, num_S_all = get_picknumber(picks=pick_iev)
                    mcatalog['asso_station_all'].append(num_station_all)  # total number of stations associated with picks
                    mcatalog['asso_station_PS'].append(num_station_PS)  # total number of stations having both P and S picks
                    mcatalog['asso_station_P'].append(num_station_P)  # total number of stations having only P picks
                    mcatalog['asso_station_S'].append(num_station_S)  # total number of stations having only S picks
                    mcatalog['asso_P_all'].append(num_P_all)  # total number of P picks
                    mcatalog['asso_S_all'].append(num_S_all)  # total number of S picks
                    mcatalog['asso_phase_all'].append(num_P_all+num_S_all)  # total number of phase picks
                
                if arrvttag is not None:
                    # load theoretical arrival time
                    file_arvtev = glob.glob(os.path.join(dir_ers, '*{}*'.format(arrvttag)))
                    assert(len(file_arvtev)==1)
                    arvt_iev = read_arrivaltimes(file_arvtev[0])
                    mcatalog['arrivaltime'].append(arvt_iev)
                
                if (picktag is not None) and (arrvttag is not None):
                    # calculate the root-mean-square deviation and mean-absolute-error between picked arrivaltimes and theoretical arrivaltimes
                    mcatalog['rms_pickarvt'].append(pickarrvt_rmsd(pick_iev, arvt_iev))  # rms
                    mcatalog['mae_pickarvt'].append(pickarrvt_mae(pick_iev, arvt_iev))  # mae
            del ctemp, dtemp
    
    # convert to numpy array
    for ikey in list(mcatalog.keys()):
        mcatalog[ikey] = np.array(mcatalog[ikey])
    
    return mcatalog


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
    for ikey in list(catalog.keys()):
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
            evtimedfs = np.array([abs(ettemp) for ettemp in (catalog['time'][iev+1:] - catalog['time'][iev])])  # origin time difference in seconds
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
                    evidlist.extend(eindx)
                    print('Duplicate events found at time around: ', catalog['time'][iev])
                else:
                    # have matched events, the best event to keep is in 'eindx', to be checked in the later loop
                    print('Duplicate events found at time around: ', catalog['time'][iev])
            else:
                # no matched events, add the current events into the new catalog
                for ikey in catakeys:
                    catalog_new[ikey].append(catalog[ikey][iev])
    
    # convert to numpy array
    for ikey in catakeys:
        catalog_new[ikey] = np.array(catalog_new[ikey])
       
    return catalog_new


def catalog_evchoose(catalog, select):
    """
    Choose events in the catalog that fullfill the select requirement.

    Parameters
    ----------
    catalog : dict
        input catalog.
    select : dict
        selection criterion.

    Returns
    -------
    catalog_s : dict
        catalog after selection.

    """
    
    cat_keys = list(catalog.keys())  # keys of the input catalog
    n_event = len(catalog[cat_keys[0]])  # total number of events in the input catalog
    sindx = np.full((n_event,), True)  # true index
    
    sel_keys = list(select.keys())  # selecting keys
    for ikey in sel_keys:
        if (select[ikey] is not None) and (isinstance(select[ikey],list)) and (len(select[ikey])==2):
            catalog[ikey] = np.where(catalog[ikey] == None, np.nan, catalog[ikey])  # replace None with np.nan to enable comparison operation
            sindx_temp = (catalog[ikey] >= select[ikey][0]) & (catalog[ikey] <= select[ikey][1])
            sindx = np.logical_and(sindx, sindx_temp)
    
    catalog_s = {}
    for ikey2 in cat_keys:
        catalog_s[ikey2] = catalog[ikey2][sindx]
    
    return catalog_s


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
    for ikey in list(catalog.keys()):
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
        the input catalog, i.e. the newly obtained catalog by users.
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
        if isinstance(catalog['time'][0], datetime.datetime):
            # datetime expressed in datetime.datetime format
            evtimedfs = np.array([abs(ettemp.total_seconds()) for ettemp in (catalog_ref['time'] - catalog['time'][iev])])  # origin time difference in seconds
        else:
            # datetime expressed in UTCDateTime format
            evtimedfs = np.array([abs(ettemp) for ettemp in (catalog_ref['time'] - catalog['time'][iev])])  # origin time difference in seconds
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
    for jjkey in catalog_match.keys():
        catalog_match[jjkey] = np.array(catalog_match[jjkey])

    return catalog_match


def retrive_catalog_from_MALMI_database(CAT):
    """
        Retrive earthquake catalog from the MALMI result database.

        Parameters
        ----------
        CAT : dict
            parameters controlling the process of retriving catalog.
            CAT['dir_dateset'] : str
                database directory, i.e. the path to the parent folder of MALMI catalog and phase file.
                (corresponding to the 'dir_MIG' folder in the MALMI main.py script).
            CAT['cata_fold'] : str,
                Catalog-file parent folder name.
                (corresponding to the 'fld_migresult' folder in the MALMI main.py script).
            CAT['dete_fold'] : str
                Detection-file parent folder name.
                (corresponding to the 'fld_prob' folder in the MALMI main.py script).
            CAT['evidtag'] : str
                event id tage for generating event id;
            CAT['extract'] : str, default is None.
                The filename including path of the catalog to load.
                If None, will extrace catalog from MALMI processing result database 'CAT['dir_dateset']'.
            CAT['search_fold'] : list of str, optional
                The MALMI result folders which contains catalog files.
                Will search catalog files from this fold list.
                (This corresponds to the 'fd_seismic' folder).
                The default is None, which means all avaliable folders in 'dir_dateset'.
            CAT['dir_output'] : str
                directory for outputting catalogs.
                default value is related to the current project directory.
            CAT['fname'] : str
                the output catalog filename.
                The default is 'MALMI_catalog_original'.
            CAT['fformat'] : str or list of str
                the format of the output catalog file, can be 'pickle' and 'csv'.
                Or other obspy compatible format, such as "QUAKEML", "NLLOC_OBS", "SC3ML". 
                The default is 'pickle'.
            CAT['rmrpev'] : boolen, default is 'True'
                whether to remove the deplicated events in the catalog.
            CAT['evselect'] : dict, default is {}
                parameters controlling the selection of events from original catalog,
                i.e. quality control of the orgiginal catalog.
                type 1:
                    CAT['evselect']['thrd_cmax'] : float, default is 0.036
                        threshold of minimal coherence.
                    CAT['evselect']['thrd_cstd'] : float, default is 0.119
                        threshold of maximum standard variance of stacking volume.
                    CAT['evselect']['thrd_stanum'] : int, default is None
                        threshold of minimal number of triggered stations.
                    CAT['evselect']['thrd_phsnum'] : int, default is None
                        threshold of minimal number of triggered phases.
                    CAT['evselect']['thrd_llbd'] : float, default is 0.002    
                        lat/lon in degree for excluding migration boundary.
                        e.g. latitude boundary is [lat_min, lat_max], event coordinates 
                        must then within [lat_min+thrd_llbd, lat_max-thrd_llbd].
                    CAT['evselect']['latitude'] : list of float
                        threshold of latitude range in degree, e.g. [63.88, 64.14].
                        default values are determined by 'thrd_llbd' and 'mgregion'.
                    CAT['evselect']['longitude'] : list of float
                        threshold of longitude range in degree, e.g. [-21.67, -21.06].
                        default values are determined by 'thrd_llbd' and 'mgregion'.
                    CAT['evselect']['thrd_depth'] : list of float, default is None
                        threshold of depth range in km, e.g. [-1, 12].

        Returns
        -------
        catalog : dict
            retrived earthquake catalog.

        """
    
    # set catalog output path
    if not os.path.exists(CAT['dir_output']):
        os.makedirs(CAT['dir_output'], exist_ok=True)

    if CAT['extract'] is None:
        # extract catlog from MALMI processing result database
        catalog = retrive_catalog(dir_dateset=CAT['dir_dateset'], cata_ftag='catalogue', dete_ftag='event_station_phase_info.txt', 
                                  cata_fold=CAT['cata_fold'], dete_fold=CAT['dete_fold'], search_fold=CAT['search_fold'], evidtag=CAT['evidtag'])
    else:
        # directly load saved existing catalog
        if CAT['extract'].split('.')[-1].lower() == 'pickle':
            with open(CAT['extract'], 'rb') as handle:
                catalog = pickle.load(handle)
        elif CAT['extract'].split('.')[-1].lower() == 'csv':
            catalog = csv2dict(CAT['extract'], delimiter=',')
        else:
            try:
                # try using obspy to read event file
                catalog = read_events(CAT['extract'])
            except:
                raise ValueError("Wrong input for CAT[\'fformat\']: {}!".format(CAT['fformat']))
    
    if CAT['evselect'] is not None:
        # select events from the original catalog using quality control parameters
        if ('pick_snr' in CAT['evselect']) and (CAT['evselect']['pick_snr'] is not None):
            # need to select picks according to snr threshold
            snr_para = {}
            snr_para['P'] = CAT['evselect']['pick_snr']
            snr_para['S'] = CAT['evselect']['pick_snr']
            assert(len(catalog['id']) == len(catalog['time']))
            Nevt = len(catalog['id'])  # total number of events in the catalog
            for iiev in range(Nevt):
                picks_s = picks_select(picks=catalog['pick'][iiev], arriv_para=None, snr_para=snr_para)
                catalog['pick'][iiev] = picks_s
                num_station_all, num_station_PS, num_station_P, num_station_S, num_P_all, num_S_all = get_picknumber(picks=picks_s)
                catalog['asso_station_all'][iiev]= num_station_all  # total number of stations associated with picks
                catalog['asso_station_PS'][iiev] = num_station_PS  # total number of stations having both P and S picks
                catalog['asso_station_P'][iiev] = num_station_P  # total number of stations having only P picks
                catalog['asso_station_S'][iiev] = num_station_S  # total number of stations having only S picks
                catalog['asso_P_all'][iiev] = num_P_all  # total number of P picks
                catalog['asso_S_all'][iiev] = num_S_all  # total number of S picks
                catalog['asso_phase_all'][iiev] = num_P_all+num_S_all  # total number of phase picks
                catalog['rms_pickarvt'][iiev] = pickarrvt_rmsd(picks_s, catalog['arrivaltime'][iiev])  # rms
                catalog['mae_pickarvt'][iiev] = pickarrvt_mae(picks_s, catalog['arrivaltime'][iiev])  # mae

        if 'thrd_cmax' in CAT['evselect']:
            # select type 1
            catalog = catalog_select(catalog, thrd_cmax=CAT['evselect']['thrd_cmax'], 
                                              thrd_stanum=CAT['evselect']['thrd_stanum'], 
                                              thrd_phsnum=CAT['evselect']['thrd_phsnum'], 
                                              thrd_lat=CAT['evselect']['latitude'], 
                                              thrd_lon=CAT['evselect']['longitude'], 
                                              thrd_cstd=CAT['evselect']['thrd_cstd'], 
                                              thrd_depth=CAT['evselect']['thrd_depth'])
        else:
            # select type 2
            catalog = catalog_evchoose(catalog, select=CAT['evselect'])
        
    # remove repeated events
    if CAT['rmrpev']:
        catalog = catalog_rmrpev(catalog=catalog, thrd_time=CAT['rmrpev']['thrd_time'], thrd_hdis=CAT['rmrpev']['thrd_hdis'], thrd_depth=CAT['rmrpev']['thrd_depth'], evkp='coherence_max')
    
    # save catalog
    if (CAT['fname'] is not None) and (CAT['fformat'] is not None):
        if isinstance(CAT['fformat'], str):
            CAT['fformat'] = [CAT['fformat']]

        for ifformat in CAT['fformat']:
            cfname = os.path.join(CAT['dir_output'], CAT['fname']+'.'+ifformat)
            if ifformat.lower() == 'pickle':
                # save the extracted original catalog in pickle format
                with open(cfname, 'wb') as handle:
                    pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif ifformat.lower() == 'csv':
                # save the extracted original catalog in csv format
                dict2csv(catalog, cfname, mode='w')
            else:
                try:
                    # obspy output catalog
                    catalog_obspy = dict2catalog(catalog)
                    catalog_obspy.write(cfname, format=ifformat)
                except:
                    raise ValueError("Wrong input for CAT[\'fformat\']: {}!".format(ifformat))

    return catalog


def catalog2dict(catalog):
    """
    Transform obspy catalog object to simple python dictory object.
    The pick information if there are any in the input catalog will be discard. Need to incorperate this in the dict...(to do)

    Parameters
    ----------
    catalog : obspy catalog object.
        earthquake event catalog.

    Returns
    -------
    catalog_dict : dict
        earthquake event catalog.
        catalog_dict['id']: np.array of str, event id of each catalog event;
        catalog_dict['time']: np.array of UTCDateTime, event origin time of each catalog event;
        catalog_dict['latitude']: np.array of float, event latitude in degree of each catalog event;
        catalog_dict['longitude']: np.array of float, event longitude in degree of each catalog event;
        catalog_dict['depth_km']: np.array of float, event depth in km (relative to the sea-level, down->positive) of each catalog event;
        catalog_dict['magnitude']: np.array of float, event magnitude of each catalog event;
        catalog['magnitude_type']: np.array of str, event magnitude type.
    """
    
    catalog_dict= {}
    catalog_dict['id'] = []
    catalog_dict['time'] = []
    catalog_dict['latitude'] = []
    catalog_dict['longitude'] = []
    catalog_dict['depth_km'] = []
    catalog_dict['magnitude'] = []
    catalog_dict['magnitude_type'] = []
    
    for ievent in catalog:
        catalog_dict['id'].append(ievent.resource_id.id)
        
        try:
            ievent_origin = ievent.preferred_origin()
        except:
            ievent_origin = ievent.origins[0]
    
        catalog_dict['time'].append(ievent_origin.time)    
        catalog_dict['latitude'].append(ievent_origin.latitude)
        catalog_dict['longitude'].append(ievent_origin.longitude)
        catalog_dict['depth_km'].append(ievent_origin.depth/1000.0)
        
        try:
            ievent_magnitude = ievent.preferred_magnitude()
        except:
            ievent_magnitude = ievent.magnitudes[0]
    
        catalog_dict['magnitude'].append(ievent_magnitude.mag)
        catalog_dict['magnitude_type'].append(ievent_magnitude.magnitude_type)
    
    # convert to numpy array
    for ikey in list(catalog_dict.keys()):
        catalog_dict[ikey] = np.array(catalog_dict[ikey])
        
    return catalog_dict


def dict2catalog(cat_dict):
    """
    Transform simple python dictory object to obspy catalog object.

    Parameters
    ----------
    cat_dict : dict
        input catalog of dict format.

    Returns
    -------
    catalog : obspy catalog object
        output catalog.

    """
    
    keys_catdict = list(cat_dict.keys())  # keys of the input catalog dict
    nev = len(cat_dict[keys_catdict[0]])  # total number of events
    for ikeycat in keys_catdict:
        if isinstance(cat_dict[ikeycat], (list, np.ndarray, pd.core.series.Series)):
            assert(len(cat_dict[ikeycat]) == nev)
    
    catalog = obspy_Catalog()
    for iev in range(nev):
        iorigin = obspy_Origin()
        try:
            iorigin.resource_id = '{}_origin'.format(cat_dict['id'][iev])
        except:
            pass
        try:
            iorigin.time = cat_dict['time'][iev] 
        except:
            pass
        try:
            iorigin.latitude = cat_dict['latitude'][iev]
        except:
            pass
        try:
            iorigin.longitude = cat_dict['longitude'][iev]
        except:
            pass
        try:
            iorigin.depth = cat_dict['depth_km'][iev] * 1000.0  # unit in meters
        except:
            pass
        # iorigin.depth_type = 'from location'
        
        ioriginqlt = obspy_OriginQuality()
        try:
            ioriginqlt.associated_phase_count = cat_dict['asso_phase_all'][iev]
        except:
            pass
        try:
            ioriginqlt.associated_station_count = cat_dict['asso_station_all'][iev]
        except:
            pass
        try:
            ioriginqlt.standard_error = cat_dict['rms_pickarvt'][iev]
        except:
            pass
        # ioriginqlt.azimuthal_gap = 
        iorigin.quality = ioriginqlt

        if 'magnitude' in cat_dict:
            imag = obspy_Magnitude()
            imag.resource_id = '{}_magnitude'.format(cat_dict['id'][iev])
            imag.mag = cat_dict['magnitude'][iev]
            imag.origin_id = iorigin.resource_id
            if 'magnitude_type' in cat_dict:
                if isinstance(cat_dict['magnitude_type'], str):
                    imag.magnitude_type = cat_dict['magnitude_type']
                else:
                    imag.magnitude_type = cat_dict['magnitude_type'][iev]
            else:     
                imag.magnitude_type = 'M'
        else:
            imag = None
        
        arrivals_list = []  # obspy arrival list
        arrivals_ids = []  # resource identifier of Arrivals
        if 'pick' in cat_dict:
            picks_list = []  # obspy pick list
            stationids_pick = list(cat_dict['pick'][iev].keys())  # station_id list that have picks for the current event
            # station_id should be composed of 'network.station.location.channel', e.g. 'BW.FUR..EHZ'
            for ipsta in stationids_pick:
                # loop over each picked station of the current event
                for iphase in list(cat_dict['pick'][iev][ipsta].keys()):  
                    # loop over each picked phase of the current station
                    ipick = obspy_Pick()
                    ipick.resource_id = "{}_pick_{}_{}".format(cat_dict['id'][iev],ipsta,iphase)
                    ipick.time = cat_dict['pick'][iev][ipsta][iphase]
                    ipick.waveform_id = WaveformStreamID(seed_string=ipsta)  # use 'seed_string' or station_code=ipsta
                    ipick.phase_hint = iphase
                    # ipick.evaluation_mode = "automatic"
                    picks_list.append(ipick)

                    # associate this pick to an arrival in origin
                    iarrival = obspy_Arrival()
                    iarrival_id = "{}_arrival_{}_{}".format(cat_dict['id'][iev],ipsta,iphase)
                    iarrival.resource_id = iarrival_id
                    iarrival.pick_id = ipick.resource_id  # Refers to the resource_id of a Pick
                    iarrival.phase = iphase
                    try:
                        iarrival.comments = [Comment(text="{}".format(cat_dict['arrivaltime'][iev][ipsta][iphase]))]
                        iarrival.time_residual = cat_dict['pick'][iev][ipsta][iphase] - cat_dict['arrivaltime'][iev][ipsta][iphase]  # Residual between observed and expected arrival time. Unit: second
                    except:
                        pass
                    arrivals_list.append(iarrival)
                    arrivals_ids.append(iarrival_id)
        else:
            picks_list = None
        
        if 'arrivaltime' in cat_dict:
            stationids_arrival = list(cat_dict['arrivaltime'][iev].keys())  # station_id list having theoretical arrivaltimes for the current event
            # station_id should be composed of 'network.station.location.channel', e.g. 'BW.FUR..EHZ'
            for iasta in stationids_arrival:  # loop over each station of the current event
                for iarr in list(cat_dict['arrivaltime'][iev][iasta].keys()):  # loop over each phase of the current station
                    iarr_id = "{}_arrival_{}_{}".format(cat_dict['id'][iev],iasta,iarr)
                    if iarr_id not in arrivals_ids:
                        # no arrival yet
                        iarrival = obspy_Arrival()
                        iarrival.resource_id = iarr_id
                        iarrival.phase = iarr
                        iarrival.comments = [Comment(text="{}".format(cat_dict['arrivaltime'][iev][iasta][iarr]))]  # absolute theoretical arrival times
                        arrivals_list.append(iarrival)
        
        iorigin.arrivals = arrivals_list
        ievent = obspy_Event(origins=[iorigin], magnitudes=[imag], picks=picks_list)
        ievent.resource_id = cat_dict['id'][iev]
        ievent.preferred_origin_id = iorigin.resource_id
        ievent.preferred_magnitude_id = imag.resource_id
        catalog.append(ievent)
    
    return catalog


def load_catalog(catafile, outformat='original'):
    """
    Load catalog into memory.

    Parameters
    ----------
    catafile : str
        filename including path of the catlaog
        can be 'csv', 'xml', or 'pickle' format.
    outformat : str
        format of the catalog in memory;
        'original': keep the loaded catalog directly as it is loaded;
        'dict': simple python dictory format;
        'obspy': obspy catalog object;
    
    Returns
    -------
    catalog : format depends on 'outformat'
        loaded catalog.

    """
    
    catalog_suffix = catafile.split('.')[-1]
    
    if catalog_suffix.lower() == 'csv':
        catalog = csv2dict(catafile)
    elif catalog_suffix.lower() == 'xml':
        catalog = read_events(catafile)
    elif catalog_suffix.lower() == 'pickle':
        with open(catafile, 'rb') as handle:
            catalog = pickle.load(handle)
    else:
        raise ValueError('Wrong input for input catalog file: {}! Format not recognizable!'.format(catafile))
    
    # for dict format, the origin time of each event should be in UTCDateTime format
    if isinstance(catalog, dict):
        if not isinstance(catalog['time'][0], UTCDateTime):
            catalog['time'] = np.array([UTCDateTime(ttt) for ttt in catalog['time']])
    
    # check output format
    if (outformat.lower() == 'dict') and (isinstance(catalog, obspy_Catalog)):
        # obspy catalog object to dict
        catalog = catalog2dict(catalog)
    elif (outformat.lower() == 'obspy') and (isinstance(catalog, dict)):
        # dict to obspy catalog object
        catalog = dict2catalog(catalog)
    
    return catalog










