#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:04:00 2021

Functions related to earthquake magnitude estimation, calculation, conversion etc.

@author: shipe
"""


import numpy as np
from obspy.geodetics import gps2dist_azimuth
import copy
import os
import obspy
import glob
import datetime
from obspy import UTCDateTime
import warnings
from xcatalog import catalog_matchref


def relative_amp(catalog, catalog_ref, catalog_match, stations, mgcalpara=None, mode='closest', distmode='3D', sorder='amplitude', staavenum='all'):
    """
    To calculate magnitude according to relative amplitude ratio.
    At least one event must match between the input catalog and the reference catalog.

    Parameters
    ----------
    catalog : dict
        the input catalog in which the event magnitude need to be determined.
        requried keys:
            catalog['id']
            catalog['latitude']
            catalog['longitude']
            catalog['depth_km']
            catalog['dir']
    catalog_ref : dict
        the reference catalog where event magnitude already exist.
        required keys:
            catalog_ref['id']
            catalog_ref['magnitude']
            catalog_ref['latitude']
            catalog_ref['longitude']
            catalog_ref['depth_km']
    catalog_match : dict
        a matched catalog between input catalog and the reference catalog, see 
        function "catalog_matchref" for more detail.
        required keys:
            catalog_match['status']
            catalog_match['id']
            catalog_match['id_ref']
    stations : dict
        contains station information.
        required keys:
            stations['station'] : the station name
            stations['latitude']
            stations['longitude']
            stations['elevation']
    mgcalpara : dict, optional
        parameters realted to magnitude estimation.
        The default is None, means use default parameters.
        required keys:
            mgcalpara['freq'] : filtering frequency band for pre-processing seismic data;
            mgcalpara['phase'] : use which phase for determine event magnitude:
                'P' for using P-phase only;
                'S' for using S-phase only;
                'PS' for using P-phase and S-phase to get event magnitude independently, then averge the two to get the final value;
            mgcalpara['P_start'] : start time relative to P-picking time for getting P-phase amplitude;
            mgcalpara['P_end'] : end time relative to P-picking time for getting P-phase amplitude.
            mgcalpara['S_start'] : start time relative to S-picking time for getting S-phase amplitude;
            mgcalpara['S_end'] : end time relative to S-picking time for getting S-phase amplitude.
            Note nagtive means time before the picking time, positive means time after the picking time.
            Maximum absolute amplitude (3D partical motion) between starttime and endtime
            is chosen as the phase amplitude.
    mode : char, optional
        Determine the order of refernce events for magnitude estimation. 
        "closest" : determine event magnitude from the closest avaliable matched-event in the reference catalog;
        "largest" : determine event magnitude from the largest avaliable matched-event in the reference catalog;
        The default is 'closest'.
    distmode : char, optional
        clarify how to calculate inter-event distance.
        "3D" : calculating distance in 3D.
        "horizontal" : only calculate horizontal distance, ignoring depth information.
        The default is '3D'.
    sorder : char, optional
        Determine the order of comment stations for magnitude estimation. 
        "amplitude" : determine event magnitude from the station with largest amplitude;
        "station_dist" : determine event magnitude from the closest avaliable station;
        The default is 'amplitude'.
    staavenum : int or str, optional
        specify the total number of stations to average for calculating magnitude.
        if staavenum is "all", then use all avaliable stations to averge;
        if staavenum > the total number of avaliable stations, then use all avaliable stations;
        For example staavenum = 1, i.e. use the first amplitude ratio in the station list (sorted according to 'sorder')
        to calculate magnitude.
        Default is 'all', i.e. use all avaliable stations to obtain the avarege value to calculate magnitude.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    catalog_new : dict
        input catalog with magnitude of all included events determined.

    """
    
    from ioformatting import read_arrivaltimes
    
    seismic_foldername = 'seis_evstream'  # the foldername of seismic data segment
    phase_ftage = 'MLpicks'  # the filename tage for picking or arrivaltime file
    
    if mgcalpara is None:
        mgcalpara = {}
        mgcalpara['freq'] = [3, 40]  # in Hz
        mgcalpara['phase'] = 'S'  # which phase to use for extracting amplitude ratio, can be 'P', 'S' or 'PS'
        mgcalpara['P_start'] = -0.5  # negtive value means time duration before a datetime
        mgcalpara['P_end'] = 0.6
        mgcalpara['S_start'] = -0.5  # negtive value means time duration before a datetime
        mgcalpara['S_end'] = 0.8
    
    Nev_in = len(catalog['time'])  # total number of events in the input catalog   
      
    # initialize the output catalog
    catalog_new = copy.deepcopy(catalog)
    catalog_new['magnitude'] = []  # this is the magnitude infor we want to resolve
    
    # get information of all matched events
    event_match_latitude = []
    event_match_longitude = []
    event_match_depth_km = []
    event_match_magnitude = []
    event_match_dir = []
    for iemt in range(len(catalog_match['status'])):
        if catalog_match['status'][iemt] == 'matched':
            # collect information for the matched events
            idxtemp = (catalog_ref['id'] == catalog_match['id_ref'][iemt])
            assert(sum(idxtemp)==1)  # only one event should match
            event_match_latitude.append(catalog_ref['latitude'][idxtemp][0])
            event_match_longitude.append(catalog_ref['longitude'][idxtemp][0])
            event_match_depth_km.append(catalog_ref['depth_km'][idxtemp][0])
            event_match_magnitude.append(catalog_ref['magnitude'][idxtemp][0])
            idxtemp = (catalog['id'] == catalog_match['id'][iemt])
            assert(sum(idxtemp)==1)  # only one event should match
            event_match_dir.append(catalog['dir'][idxtemp][0])
    Nev_match = len(event_match_magnitude)
    event_match_latitude = np.array(event_match_latitude)
    event_match_longitude = np.array(event_match_longitude)
    event_match_depth_km = np.array(event_match_depth_km)
    event_match_magnitude = np.array(event_match_magnitude)
    event_match_dir = np.array(event_match_dir)
    if mode == 'largest':
        # sorted index according to event magnitude in descending order
        eidxref_sorted = np.argsort(-1*event_match_magnitude)

    # assign the magnitude of matched events to the input catalog
    for iev in range(Nev_in):
        eidx = (catalog_match['id'] == catalog['id'][iev])  # note we should find the event by its id, it's unique and constant in catalog_match
        assert(sum(eidx)==1)  # only one event should match this id
        if catalog_match['status'][eidx][0] == 'matched':
            # this event has a match in the reference catalog
            # directly assign the magnitude of matched event to it
            eidx_ref = (catalog_ref['id'] == catalog_match['id_ref'][eidx][0])
            assert(sum(eidx_ref)==1)  # only one event should mathch this id_ref
            catalog_new['magnitude'].append(catalog_ref['magnitude'][eidx_ref][0])
            
        elif catalog_match['status'][eidx][0] == 'new':
            # this event is a new event with no event in the reference catalog can match
            # need to determine the magnitude of this event.
            
            magnitude_done = False
            
            if mode == 'closest':
                # determine magnitude according to the closest matched event in the reference catalog
                # calculate inter-event distances between the current event and 
                # all matched events, distance in km
                distmx = np.zeros(shape=(Nev_match,))
                if distmode == 'horizontal':
                    # horizontal distance not include event depth
                    for jj in range(Nev_match):
                        hdist_meter, _, _ = gps2dist_azimuth(event_match_latitude[jj], event_match_longitude[jj], 
                                                             catalog['latitude'][iev], catalog['longitude'][iev])
                        distmx[jj] = abs(hdist_meter)/1000.0
                elif distmode == '3D':
                    # real distance include event depth
                    for jj in range(Nev_match):
                        hdist_meter, _, _ = gps2dist_azimuth(event_match_latitude[jj], event_match_longitude[jj], 
                                                             catalog['latitude'][iev], catalog['longitude'][iev])
                        hdist_km = hdist_meter / 1000.0  # from meter to km
                        vdist_km = event_match_depth_km[jj] - catalog['depth_km'][iev]
                        distmx[jj] = np.sqrt(hdist_km*hdist_km + vdist_km*vdist_km)
                else:
                    raise ValueError('Input distmode not recognized!')    
                eidxref_sorted = np.argsort(distmx)  # sorted index according to inter-event distances in ascending order    
            elif mode == 'largest':
                # determine magnitude according to the largest matched event in the reference catalog
                pass  # 'eidxref_sorted' already calculated before
            else:
                raise ValueError('Input mode not recognized!')
            
            # get data path information
            ev_resdir = catalog['dir'][iev]  # result directory of the current event whose magnitude need to be determined
            pathtemp = ev_resdir.split('/')
            pathtemp[-2] = seismic_foldername
            ev_seisdir = ''
            for pptt in pathtemp:
                # seismic data segment directory of current event
                ev_seisdir = os.path.join(ev_seisdir, pptt)
            assert(os.path.exists(ev_seisdir))
            
            # load phase file
            ev_phasefile = glob.glob(os.path.join(ev_resdir, '*'+phase_ftage+'*'))
            assert(len(ev_phasefile)==1)  # should be only one file
            arrvt = read_arrivaltimes(ev_phasefile[0])  # arrivaltimes at different stations
            
            # load seismic data segment
            stream = obspy.read(os.path.join(ev_seisdir,'*'))
            if ('freq' in mgcalpara) and (mgcalpara['freq'] is not None):
                stream.filter('bandpass', freqmin=mgcalpara['freq'][0], freqmax=mgcalpara['freq'][1], zerophase=True)
            stream.interpolate(sampling_rate=1000)  # up-sampling to avoide getting data of different size when slicing data
            
            # get the amplitude of certain phase at available stations
            ev_stalist = []
            ev_ssdist = []
            ev_amplitude = []
            ev_Pamplitude = []
            ev_Samplitude = []
            for sta in arrvt:
                if mgcalpara['phase'] == 'P':
                    if 'P' in arrvt[sta]:
                        tt1 = arrvt[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_start'])
                        tt2 = arrvt[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_end'])
                        stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                        if (stream_sta.count()==3) and (len(stream_sta[0].data)==len(stream_sta[1].data)==len(stream_sta[2].data)):
                            # should have 3 component data
                            ev_stalist.append(sta)
                            ev_amplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                            stream_sta[1].data*stream_sta[1].data + 
                                                            stream_sta[2].data*stream_sta[2].data)))
                            
                            staindx = (stations['station'] == sta)
                            assert(sum(staindx)==1)
                            hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                 catalog['latitude'][iev], catalog['longitude'][iev])
                            vdist_meter = catalog['depth_km'][iev]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                            ev_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                        
                elif mgcalpara['phase'] == 'S':
                    if 'S' in arrvt[sta]:
                        tt1 = arrvt[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_start'])
                        tt2 = arrvt[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_end'])
                        stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                        if (stream_sta.count()==3) and (len(stream_sta[0].data)==len(stream_sta[1].data)==len(stream_sta[2].data)):
                            # should have 3 component data
                            ev_stalist.append(sta)
                            ev_amplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                            stream_sta[1].data*stream_sta[1].data + 
                                                            stream_sta[2].data*stream_sta[2].data)))
                            
                            staindx = (stations['station'] == sta)
                            assert(sum(staindx)==1)
                            hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                 catalog['latitude'][iev], catalog['longitude'][iev])
                            vdist_meter = catalog['depth_km'][iev]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                            ev_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                
                elif mgcalpara['phase'] == 'PS':
                    if ('P' in arrvt[sta]) and ('S' in arrvt[sta]):  # note require having both P and S phases
                        if stream.select(station=sta).count()==3:
                            # should have 3 component data
                            ev_stalist.append(sta)
                            tt1 = arrvt[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_start'])
                            tt2 = arrvt[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_end'])
                            stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                            ev_Pamplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                             stream_sta[1].data*stream_sta[1].data + 
                                                             stream_sta[2].data*stream_sta[2].data)))
                            
                            tt1 = arrvt[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_start'])
                            tt2 = arrvt[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_end'])
                            stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                            ev_Samplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                             stream_sta[1].data*stream_sta[1].data + 
                                                             stream_sta[2].data*stream_sta[2].data)))
                            
                            staindx = (stations['station'] == sta)
                            assert(sum(staindx)==1)
                            hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                 catalog['latitude'][iev], catalog['longitude'][iev])
                            vdist_meter = catalog['depth_km'][iev]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                            ev_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                
                else:
                    raise ValueError('Incorrent input for mgcalpara[\'phase\']!')
            
            ev_stalist = np.array(ev_stalist)
            ev_ssdist = np.array(ev_ssdist)
            if (mgcalpara['phase'] == 'P') or (mgcalpara['phase'] == 'S'):
                ev_amplitude = np.array(ev_amplitude)
                if sorder == 'amplitude':
                    llindex = np.argsort(-1.0*ev_amplitude)  # get the station ordered according to amplitude descending order
                elif sorder == 'station_dist':
                    llindex = np.argsort(ev_ssdist)  # get the station ordered according to event-station distance ascending order
                ev_amplitude = ev_amplitude[llindex]
                ev_stalist = ev_stalist[llindex]
                ev_ssdist = ev_ssdist[llindex]
            elif mgcalpara['phase'] == 'PS':
                ev_Pamplitude = np.array(ev_Pamplitude)
                ev_Samplitude = np.array(ev_Samplitude)
                if sorder == 'amplitude':
                    llindex = np.argsort(-1.0*(ev_Pamplitude+ev_Samplitude))  # get the station ordered according to amplitude descending order. Note here use the sum amplitude of P and S
                elif sorder == 'station_dist':
                    llindex = np.argsort(ev_ssdist)  # get the station ordered according to event-station distance ascending order
                ev_Pamplitude = ev_Pamplitude[llindex]
                ev_Samplitude = ev_Samplitude[llindex]
                ev_stalist = ev_stalist[llindex]
                ev_ssdist = ev_ssdist[llindex]
            else:
                raise ValueError('Incorrent input for mgcalpara[\'phase\']!')
            
            del stream, stream_sta, arrvt
            
            for ekk in eidxref_sorted:
                # loop over each matched event in certain order untill we can determine the magnitude successfully
                evref_resdir = event_match_dir[ekk]  # result directory of the matched event
                pathtemp_ref = evref_resdir.split('/')
                pathtemp_ref[-2] = seismic_foldername
                evref_seisdir = ''
                for pptt in pathtemp_ref:
                    # seismic data segment directory of current reference event
                    evref_seisdir = os.path.join(evref_seisdir, pptt)
                assert(os.path.exists(evref_seisdir))
                
                # load phase file
                evref_phasefile = glob.glob(os.path.join(evref_resdir, '*'+phase_ftage+'*'))
                assert(len(evref_phasefile)==1)  # should be only one file
                arrvt_ref = read_arrivaltimes(evref_phasefile[0])  # arrivaltimes at different stations
                
                # load seismic data segment
                stream = obspy.read(os.path.join(evref_seisdir,'*'))
                if ('freq' in mgcalpara) and (mgcalpara['freq'] is not None):
                    stream.filter('bandpass', freqmin=mgcalpara['freq'][0], freqmax=mgcalpara['freq'][1], zerophase=True)
                stream.interpolate(sampling_rate=1000)  # up-sampling to avoide getting data of different size when slicing data
                
                # get the amplitude of certain phase at available stations
                evref_stalist = []
                evref_ssdist = []
                evref_amplitude = []
                evref_Pamplitude = []
                evref_Samplitude = []
                for sta in arrvt_ref:
                    if mgcalpara['phase'] == 'P':
                        if 'P' in arrvt_ref[sta]:
                            tt1 = arrvt_ref[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_start'])
                            tt2 = arrvt_ref[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_end'])
                            stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                            if (stream_sta.count()==3) and (len(stream_sta[0].data)==len(stream_sta[1].data)==len(stream_sta[2].data)):
                                # should have 3 component data
                                evref_stalist.append(sta)
                                evref_amplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                                   stream_sta[1].data*stream_sta[1].data + 
                                                                   stream_sta[2].data*stream_sta[2].data)))
                                
                                staindx = (stations['station'] == sta)
                                assert(sum(staindx)==1)
                                hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                     event_match_latitude[ekk], event_match_longitude[ekk])
                                vdist_meter = event_match_depth_km[ekk]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                                evref_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                            
                    elif mgcalpara['phase'] == 'S':
                        if 'S' in arrvt_ref[sta]:
                            tt1 = arrvt_ref[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_start'])
                            tt2 = arrvt_ref[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_end'])
                            stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                            if (stream_sta.count()==3) and (len(stream_sta[0].data)==len(stream_sta[1].data)==len(stream_sta[2].data)):
                                # should have 3 component data
                                evref_stalist.append(sta)
                                evref_amplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                                   stream_sta[1].data*stream_sta[1].data + 
                                                                   stream_sta[2].data*stream_sta[2].data)))
                                
                                staindx = (stations['station'] == sta)
                                assert(sum(staindx)==1)
                                hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                     event_match_latitude[ekk], event_match_longitude[ekk])
                                vdist_meter = event_match_depth_km[ekk]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                                evref_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                    
                    elif mgcalpara['phase'] == 'PS':
                        if ('P' in arrvt_ref[sta]) and ('S' in arrvt_ref[sta]):
                            if stream.select(station=sta).count()==3:
                                # should have 3 component data
                                evref_stalist.append(sta)
                                tt1 = arrvt_ref[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_start'])
                                tt2 = arrvt_ref[sta]['P'] + datetime.timedelta(seconds=mgcalpara['P_end'])
                                stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                                evref_Pamplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                                    stream_sta[1].data*stream_sta[1].data + 
                                                                    stream_sta[2].data*stream_sta[2].data)))
                                
                                tt1 = arrvt_ref[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_start'])
                                tt2 = arrvt_ref[sta]['S'] + datetime.timedelta(seconds=mgcalpara['S_end'])
                                stream_sta = stream.select(station=sta).slice(starttime=UTCDateTime(tt1), endtime=UTCDateTime(tt2))
                                evref_Samplitude.append(max(np.sqrt(stream_sta[0].data*stream_sta[0].data + 
                                                                    stream_sta[1].data*stream_sta[1].data + 
                                                                    stream_sta[2].data*stream_sta[2].data)))
                                
                                staindx = (stations['station'] == sta)
                                assert(sum(staindx)==1)
                                hdist_meter, _, _ = gps2dist_azimuth(stations['latitude'][staindx][0], stations['longitude'][staindx][0], 
                                                                     event_match_latitude[ekk], event_match_longitude[ekk])
                                vdist_meter = event_match_depth_km[ekk]*1000.0 - (-1.0*stations['elevation'][staindx][0])  # note the difference between elevation and depth
                                evref_ssdist.append(np.sqrt(hdist_meter*hdist_meter + vdist_meter*vdist_meter))
                    
                    else:
                        raise ValueError('Incorrent input for mgcalpara[\'phase\']!')
                
                del stream, stream_sta, arrvt_ref
                
                evref_stalist = np.array(evref_stalist)
                evref_ssdist = np.array(evref_ssdist)
                evref_amplitude = np.array(evref_amplitude)
                evref_Pamplitude = np.array(evref_Pamplitude)
                evref_Samplitude = np.array(evref_Samplitude)
                ampratio = []
                ampratio_P = []
                ampratio_S = []
                for iqq in range(len(ev_stalist)):
                    # loop over each station in the station list of the new event
                    if ev_stalist[iqq] in evref_stalist:
                        # the current station both exist in the station list of the new event and the reference event
                        starefindx = (evref_stalist == ev_stalist[iqq])
                        assert(sum(starefindx)==1)
                        if (mgcalpara['phase'] == 'P') or (mgcalpara['phase'] == 'S'):
                            ampratio.append((ev_amplitude[iqq] * ev_ssdist[iqq]) / (evref_amplitude[starefindx][0] * evref_ssdist[starefindx][0]))  # note here we correct the amplitude ratio using event-station distance to account for geometric spreading
                        elif mgcalpara['phase'] == 'PS':
                            ampratio_P.append((ev_Pamplitude[iqq] * ev_ssdist[iqq]) / (evref_Pamplitude[starefindx][0] * evref_ssdist[starefindx][0]))
                            ampratio_S.append((ev_Samplitude[iqq] * ev_ssdist[iqq]) / (evref_Samplitude[starefindx][0] * evref_ssdist[starefindx][0]))
                        else:
                            raise ValueError('Incorrent input for mgcalpara[\'phase\']!')
                    else:
                        # the current station does not exist in the station list of the reference event
                        pass
                
                if (mgcalpara['phase'] == 'P') or (mgcalpara['phase'] == 'S'):
                    NN_stacom = len(ampratio)
                    if NN_stacom > 0:
                        if (staavenum == 'all') or (staavenum > NN_stacom):
                            # average over all avaliable stations
                            ampratio_ave = np.array(ampratio).mean()
                        else:
                            # average over 'staavenum' stations
                            ampratio_ave = np.array(ampratio[:staavenum]).mean()
                        catalog_new['magnitude'].append(event_match_magnitude[ekk] + np.log10(ampratio_ave))
                        magnitude_done = True
                        break  # magnitude already determined, no need to look at the rest of the matched events
                elif mgcalpara['phase'] == 'PS':
                    NN_stacom = min(len(ampratio_P), len(ampratio_S))
                    if NN_stacom > 0:
                        if (staavenum == 'all') or (staavenum > NN_stacom):
                            # average over all avaliable stations
                            ampratio_ave = np.array(ampratio_P+ampratio_S).mean()
                        else:
                            # average over 'staavenum' stations
                            ampratio_ave = np.array(ampratio_P[:staavenum]+ampratio_S[:staavenum]).mean()
                        catalog_new['magnitude'].append(event_match_magnitude[ekk] + np.log10(ampratio_ave))
                        magnitude_done = True
                        break  # magnitude already determined, no need to look at the rest of the matched events
            
            if not magnitude_done:
                # no common station matched, cannot determine magnitude
                catalog_new['magnitude'].append(None)
                warnings.warn("Magnitude of event_id: {} cannot be determined! No common station found beteen this event and the reference events.".format(catalog['id'][iev]))
        else:
            raise ValueError('match status wrong for the input catalog! Should either be matched or new!')
        
    catalog_new['magnitude'] = np.array(catalog_new['magnitude'])
    assert(len(catalog_new['magnitude']) == len(catalog_new['id']))
    return catalog_new


def estimate_magnitude(MAGNI):
    """
    Determine event magnitude.

    Parameters
    ----------
    MAGNI : dict
        MAGNI['catalog']: dict, input catalog where event magnitude need to be determined;
                          format see in the 'relative_amp' function;
        MAGNI['engine']: str, method used for magnitude determination;
                         options: 'relative';
        if MAGNI['engine'] = 'relative', we also need:
            MAGNI['catalog_ref']: dict, reference catalog with event magnitude;
                                  format see in the 'relative_amp' function;
            MAGNI['match_thrd_time']: float, time in second, detail see 'catalog_matchref';
            MAGNI['stations']: dict, station information, detail see 'relative_amp';
            MAGNI['mgcalpara']: dict, magnitude determination parameters, detail see 'relative_amp';
        

    Returns
    -------
    catalog : dict
        catalog with event magnitude determined.

    """

    if MAGNI['engine'] == 'relative':
        # use relative amplitude ratio to determine magnitude 
        
        # match the input catalog and the reference catalog
        catalog_match = catalog_matchref(catalog=MAGNI['catalog'], catalog_ref=MAGNI['catalog_ref'], thrd_time=MAGNI['match_thrd_time'], thrd_hdis=None, thrd_depth=None, matchmode='time')
        
        catalog = relative_amp(catalog=MAGNI['catalog'], catalog_ref=MAGNI['catalog_ref'], 
                               catalog_match=catalog_match, stations=MAGNI['stations'], 
                               mgcalpara=MAGNI['mgcalpara'], mode='closest', distmode='3D', sorder='amplitude', staavenum='all')
    else:
        # other methods to be continue
        raise ValueError('Wroing input for MAGNI[\'engine\']: {}!'.format(MAGNI['engine']))

    return catalog


