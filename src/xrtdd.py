#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:48:50 2022

@author: shipe
"""


import pandas as pd
import numpy as np
import os
import csv
import glob
from ioformatting import read_arrivaltimes


def write_rtddstation(file_station, dir_output='./', filename='station.csv'):
    """
    This function is used to format the station file for scrtdd.

    Parameters
    ----------
    file_station : str
        filename of the input station file.
    dir_output : str, optional
        Directory for output file. The default is './'.
    filename : str, optional
        filename of output station file. The default is 'station.csv'.

    Returns
    -------
    a csv format station file for rtdd.
    
    # example of output data fromat
    # latitude,longitude,elevation,networkCode,stationCode,locationCode
    # 45.980278,7.670195,3463.0,4D,MH36,A
    # 45.978720,7.663000,4003.0,4D,MH48,A
    # 46.585719,8.383171,2320.4,4D,RA43,
    # 45.903349,6.885881,2250.0,8D,AMIDI,00
    # 46.371345,6.873937,379.0,8D,NVL3,

    """
    
    # load station infomation: SED COSEISMIQ CSV format, temporary format
    stadf = pd.read_csv(file_station, delimiter=',', encoding='utf-8', 
                        names=['net','location','sta code','description','lat',
                               'lon','altitude','type','up since','details'], 
                        dtype={'location':np.str}, keep_default_na=False)
    
    # make sure the output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    sf = open(os.path.join(dir_output, filename), 'w', newline='')
    sfcsv = csv.writer(sf, delimiter=',', lineterminator="\n")
    
    sfheader = ['latitude', 'longitude', 'elevation', 'networkCode', 'stationCode', 'locationCode']
    sfcsv.writerow(sfheader)
    sf.flush()
    
    for ista in range(len(stadf)):
        # staid = '{}.{}.{}'.format(stadf.loc[ista,'net'], stadf.loc[ista, 'sta code'], stadf.loc[ista, 'location'])
        latitude = stadf.loc[ista, 'lat']
        longitude = stadf.loc[ista, 'lon']
        elevation = stadf.loc[ista, 'altitude']
        networkCode = stadf.loc[ista, 'net']
        stationCode = stadf.loc[ista, 'sta code']
        locationCode = stadf.loc[ista, 'location']
        
        sfcsv.writerow([latitude, longitude, elevation, networkCode, stationCode, locationCode])
        sf.flush()
    
    sf.close()
    return


def output_rtddstation(stainv, dir_output='./', filename='station.csv'):
    """
    This function is used to format the station file for scrtdd.
    Note the input 'stainv' is a obspy station inventory.

    Parameters
    ----------
    stainv : obspy station invertory
        contains station information, such as network code, station code, 
        longitude, latitude, evelvation etc.
    dir_output : str, optional
        Directory for output file. The default is './'.
    filename : str, optional
        filename of output station file. The default is 'station.csv'.

    Returns
    -------
    a csv format station file for rtdd.
    
    # example of output data fromat
    # latitude,longitude,elevation,networkCode,stationCode,locationCode
    # 45.980278,7.670195,3463.0,4D,MH36,A
    # 45.978720,7.663000,4003.0,4D,MH48,A
    # 46.585719,8.383171,2320.4,4D,RA43,
    # 45.903349,6.885881,2250.0,8D,AMIDI,00
    # 46.371345,6.873937,379.0,8D,NVL3,

    """
    
    # make sure the output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    sf = open(os.path.join(dir_output, filename), 'w', newline='')
    sfcsv = csv.writer(sf, delimiter=',', lineterminator="\n")
    
    sfheader = ['latitude', 'longitude', 'elevation', 'networkCode', 'stationCode', 'locationCode']
    sfcsv.writerow(sfheader)
    sf.flush()
    
    for inet in stainv:
        for ista in inet:
            latitude = ista.latitude
            longitude = ista.longitude
            elevation = ista.elevation
            networkCode = inet.code
            stationCode = ista.code
            if ista.channels:
                locationCode = ista.channels[0].location_code
            else:
                locationCode = ''
        
            sfcsv.writerow([latitude, longitude, elevation, networkCode, stationCode, locationCode])
            sf.flush()
    
    sf.close()
    return


def write_rtddeventphase(catalog, file_station, dir_output='./', filename_event='event.csv', filename_phase='phase.csv', phaseart_ftage='.phs'):
    """
    This function is used to format the event and phase files for scrtdd.

    Parameters
    ----------
    catalog : dic
        Event catalog information;
        mcatalog['id'] : id of the event;
        mcatalog['time'] : origin time;
        mcatalog['latitude'] : latitude in degree;
        mcatalog['longitude'] : logitude in degree;
        mcatalog['depth_km'] : depth in km;
        mcatalog['magnitude'] : magnitude;
        mcatalog['dir'] : directory of the migration results of the event.
    file_station : str
        filename of the input station file.
    dir_output : str, optional
        Directory for output file. The default is './'.
    filename_event : str, optional
        filename of output event file. The default is 'event.csv'.
    filename_phase : str, optional
        filename of output phase file. The default is 'phase.csv'.
    phaseart_ftage : str, optional
        filename of the phase arrivaltime or picking time file.
        The default is '.phs'.

    Returns
    -------
    a csv format phase file for rtdd;
    a csv format event file for rtdd;
    
    # example of event file:
    # id,isotime,latitude,longitude,depth,magnitude,rms
    # 1,2019-11-05T00:54:21.256705Z,46.318264,7.365509,4.7881,3.32,0.174
    # 2,2019-11-05T01:03:06.484287Z,46.320718,7.365435,4.2041,0.64,0.138
    # 3,2019-11-05T01:06:27.140654Z,46.325626,7.356148,3.9756,0.84,0.083
    # 4,2019-11-05T01:12:25.753816Z,46.325012,7.353627,3.7090,0.39,0.144

    # example of phase file:
    # eventId,isotime,lowerUncertainty,upperUncertainty,type,networkCode,stationCode,locationCode,channelCode,evalMode
    # 1,2019-11-05T00:54:22.64478Z,0.025,0.025,Pg,8D,RAW2,,HHZ,automatic
    # 1,2019-11-05T00:54:23.58254Z,0.100,0.100,Sg,8D,RAW2,,HHT,manual
    # 1,2019-11-05T00:54:22.7681Z,0.025,0.025,Pg,CH,SAYF2,,HGZ,manual
    # 1,2019-11-05T00:54:24.007619Z,0.050,0.050,Sg,CH,STSW2,,HGT,manual
    # 2,2019-11-05T01:03:08.867835Z,0.050,0.050,S,8D,RAW2,,HHT,manual
    # 2,2019-11-05T01:03:07.977432Z,0.025,0.025,P,CH,SAYF2,,HGZ,manual
    # 2,2019-11-05T01:03:08.9947Z,0.050,0.050,Sg,CH,SAYF2,,HGT,automatic
    # 2,2019-11-05T01:03:09.12808Z,0.050,0.050,P,CH,STSW2,,HGR,manual
    # 2,2019-11-05T01:03:09.409276Z,0.025,0.025,Sg,CH,SENIN,,HHT,automatic

    """
    
    datetime_format = '%Y-%m-%dT%H:%M:%S.%fZ'  # datetime format in the output file
    
    # load station infomation: SED COSEISMIQ CSV format, temporary format
    stadf = pd.read_csv(file_station, delimiter=',', encoding='utf-8', 
                        names=['net','location','sta code','description','lat',
                               'lon','altitude','type','up since','details'], 
                        dtype={'location':np.str}, keep_default_na=False)
    
    # make sure the output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    # initialize event file
    eventf = open(os.path.join(dir_output, filename_event), 'w', newline='')
    eventf_writer = csv.writer(eventf, delimiter=',', lineterminator="\n")
    eventf_header = ['id', 'isotime', 'latitude', 'longitude', 'depth', 'magnitude', 'rms']
    eventf_writer.writerow(eventf_header)
    eventf.flush()
    
    # initialize phase file
    phasef = open(os.path.join(dir_output, filename_phase), 'w', newline='')
    phasef_writer = csv.writer(phasef, delimiter=',', lineterminator="\n")
    phasef_header = ['eventId','isotime','lowerUncertainty','upperUncertainty','type','networkCode','stationCode','locationCode','channelCode','evalMode']
    phasef_writer.writerow(phasef_header)
    phasef.flush()
    
    # loop over each event for output
    for iev in range(len(catalog['id'])):
        eventId = catalog['id'][iev]
        isotime = catalog['time'][iev].strftime(datetime_format)
        latitude = catalog['latitude'][iev]
        longitude = catalog['longitude'][iev]
        depth = catalog['depth_km'][iev]
        if 'magnitude' in catalog:
            # have magnitude info
            magnitude = catalog['magnitude'][iev]
        else:
            # no magnitude info
            magnitude = 1.0
        rms = 0.2
        eventf_writer.writerow([eventId, isotime, latitude, longitude, depth, magnitude, rms])
        eventf.flush()
        
        # load phase information and output
        file_phase = glob.glob(os.path.join(catalog['dir'][iev], '*'+phaseart_ftage+'*'))
        assert(len(file_phase) == 1)
        arrvtt = read_arrivaltimes(file_phase[0])
        stations_art = list(arrvtt.keys())  # station names which have arrivaltimes
        for sta in stations_art:
            # loop over the arrivaltimes at each station and output
            df_csta = stadf.loc[stadf['sta code']==sta, :]  # get the current station information 
            assert(len(df_csta) == 1)
            assert(len(df_csta['type'].item())==2)
            lowerUncertainty = 0.2
            upperUncertainty = 0.2
            networkCode = df_csta['net'].item()
            stationCode = df_csta['sta code'].item()
            locationCode = df_csta['location'].item()
            evalMode = 'automatic'
            
            if 'P' in arrvtt[sta]:
                isotime = arrvtt[sta]['P'].strftime(datetime_format)
                PHStype = 'P'
                channelCode = '{}Z'.format(df_csta['type'].item())
                phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                        networkCode, stationCode, locationCode, channelCode, evalMode])
                phasef.flush()
                
            if 'S' in arrvtt[sta]:
                isotime = arrvtt[sta]['S'].strftime(datetime_format)
                PHStype = 'S'
                channelCode = '{}N'.format(df_csta['type'].item())
                phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                        networkCode, stationCode, locationCode, channelCode, evalMode])
                phasef.flush()
                
                channelCode = '{}E'.format(df_csta['type'].item())
                phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                        networkCode, stationCode, locationCode, channelCode, evalMode])
                phasef.flush()    

    eventf.close()
    phasef.close()
    return


def output_rtddeventphase(catalog, stainv, dir_output='./', filename_event='event.csv', filename_phase='phase.csv', phaseart_ftage='.phs', station_channel_codes=['HHZ', 'HHN', 'HHE']):
    """
    This function is used to format the event and phase files for scrtdd.
    Note the input 'stainv' is a obspy station inventory.

    Parameters
    ----------
    catalog : dic
        Event catalog information;
        mcatalog['id'] : id of the event;
        mcatalog['time'] : origin time;
        mcatalog['latitude'] : latitude in degree;
        mcatalog['longitude'] : logitude in degree;
        mcatalog['depth_km'] : depth in km;
        mcatalog['magnitude'] : magnitude;
        mcatalog['dir'] : directory of the migration results of the event.
    stainv : obspy station invertory
        contains station information, such as network code, station code, 
        longitude, latitude, evelvation etc.
    dir_output : str, optional
        Directory for output file. The default is './'.
    filename_event : str, optional
        filename of output event file. The default is 'event.csv'.
    filename_phase : str, optional
        filename of output phase file. The default is 'phase.csv'.
    phaseart_ftage : str, optional
        filename of the phase arrivaltime or picking time file.
        The default is '.phs'.
    station_channel_codes : list of str,
        if input inventory does not provide channel codes, use this to generate
        phase file. Default is ['HHZ', 'HHN', 'HHE'].

    Returns
    -------
    a csv format phase file for rtdd;
    a csv format event file for rtdd;
    
    Note the event id in the event file and phase file must be a integer!
    
    # example of event file:
    # id,isotime,latitude,longitude,depth,magnitude,rms
    # 1,2019-11-05T00:54:21.256705Z,46.318264,7.365509,4.7881,3.32,0.174
    # 2,2019-11-05T01:03:06.484287Z,46.320718,7.365435,4.2041,0.64,0.138
    # 3,2019-11-05T01:06:27.140654Z,46.325626,7.356148,3.9756,0.84,0.083
    # 4,2019-11-05T01:12:25.753816Z,46.325012,7.353627,3.7090,0.39,0.144

    # example of phase file:
    # eventId,isotime,lowerUncertainty,upperUncertainty,type,networkCode,stationCode,locationCode,channelCode,evalMode
    # 1,2019-11-05T00:54:22.64478Z,0.025,0.025,Pg,8D,RAW2,,HHZ,automatic
    # 1,2019-11-05T00:54:23.58254Z,0.100,0.100,Sg,8D,RAW2,,HHT,manual
    # 1,2019-11-05T00:54:22.7681Z,0.025,0.025,Pg,CH,SAYF2,,HGZ,manual
    # 1,2019-11-05T00:54:24.007619Z,0.050,0.050,Sg,CH,STSW2,,HGT,manual
    # 2,2019-11-05T01:03:08.867835Z,0.050,0.050,S,8D,RAW2,,HHT,manual
    # 2,2019-11-05T01:03:07.977432Z,0.025,0.025,P,CH,SAYF2,,HGZ,manual
    # 2,2019-11-05T01:03:08.9947Z,0.050,0.050,Sg,CH,SAYF2,,HGT,automatic
    # 2,2019-11-05T01:03:09.12808Z,0.050,0.050,P,CH,STSW2,,HGR,manual
    # 2,2019-11-05T01:03:09.409276Z,0.025,0.025,Sg,CH,SENIN,,HHT,automatic

    """
    
    datetime_format = '%Y-%m-%dT%H:%M:%S.%fZ'  # datetime format in the output file
    
    # make sure the output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    # initialize event file
    eventf = open(os.path.join(dir_output, filename_event), 'w', newline='')
    eventf_writer = csv.writer(eventf, delimiter=',', lineterminator="\n")
    eventf_header = ['id', 'isotime', 'latitude', 'longitude', 'depth', 'magnitude', 'rms']
    eventf_writer.writerow(eventf_header)
    eventf.flush()
    
    # initialize phase file
    phasef = open(os.path.join(dir_output, filename_phase), 'w', newline='')
    phasef_writer = csv.writer(phasef, delimiter=',', lineterminator="\n")
    phasef_header = ['eventId','isotime','lowerUncertainty','upperUncertainty','type','networkCode','stationCode','locationCode','channelCode','evalMode']
    phasef_writer.writerow(phasef_header)
    phasef.flush()
    
    # loop over each event for output
    for iev in range(len(catalog['id'])):
        try:
            eventId = int(catalog['id'][iev])  # note the event id should be an integer for rtdd
        except:
            eventId = iev + 1
        isotime = catalog['time'][iev].strftime(datetime_format)
        latitude = catalog['latitude'][iev]
        longitude = catalog['longitude'][iev]
        depth = catalog['depth_km'][iev]
        if 'magnitude' in catalog:
            # have magnitude info
            magnitude = catalog['magnitude'][iev]
        else:
            # no magnitude info
            magnitude = 1.0
        rms = 0.2
        eventf_writer.writerow([eventId, isotime, latitude, longitude, depth, magnitude, rms])
        eventf.flush()
        
        # load phase information and output
        file_phase = glob.glob(os.path.join(catalog['dir'][iev], '*'+phaseart_ftage+'*'))
        assert(len(file_phase) == 1)
        arrvtt = read_arrivaltimes(file_phase[0])
        stations_art = list(arrvtt.keys())  # station names which have arrivaltimes
        for sta in stations_art:
            # loop over the arrivaltimes at each station and output
            istainv = stainv.select(station=sta)  # get the current station inventory 
            assert(len(istainv) == 1)
            assert(istainv[0][0].code == sta)
            lowerUncertainty = 0.2
            upperUncertainty = 0.2
            networkCode = istainv.networks[0].code
            stationCode = istainv.networks[0].stations[0].code
            if istainv.networks[0].stations[0].channels:
                locationCode = istainv.networks[0].stations[0].channels[0].location_code
                channel_codes = []
                for icha in istainv.networks[0].stations[0].channels:
                    channel_codes.append(icha.code)
            else:
                locationCode = ''
                channel_codes = station_channel_codes
            evalMode = 'automatic'
            
            if 'P' in arrvtt[sta]:
                # P arrivaltime
                isotime = arrvtt[sta]['P'].strftime(datetime_format)
                PHStype = 'P'
                
                schannel = [iich for iich in channel_codes if 'Z' in iich]
                if len(schannel)==1:
                    channelCode = schannel[0]
                    phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                            networkCode, stationCode, locationCode, channelCode, evalMode])
                    phasef.flush()
                else:
                    for iich in channel_codes:
                        if '3' in iich:
                            channelCode = iich
                            phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                    networkCode, stationCode, locationCode, channelCode, evalMode])
                            phasef.flush()
                            break
                        elif '2' in iich:
                            channelCode = iich
                            phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                    networkCode, stationCode, locationCode, channelCode, evalMode])
                            phasef.flush()
                            break
                        elif '1' in iich:
                            channelCode = iich
                            phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                    networkCode, stationCode, locationCode, channelCode, evalMode])
                            phasef.flush()
                            break
                
            if 'S' in arrvtt[sta]:
                # S arrivaltime
                isotime = arrvtt[sta]['S'].strftime(datetime_format)
                PHStype = 'S'
                
                for iich in channel_codes:
                    if 'N' in iich:
                        channelCode = iich
                        phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                networkCode, stationCode, locationCode, channelCode, evalMode])
                        phasef.flush()
                        break
                    elif 'E' in iich:
                        channelCode = iich
                        phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                networkCode, stationCode, locationCode, channelCode, evalMode])
                        phasef.flush()
                        break
                    elif '1' in iich:
                        channelCode = iich
                        phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                networkCode, stationCode, locationCode, channelCode, evalMode])
                        phasef.flush()
                        break
                    elif '2' in iich:
                        channelCode = iich
                        phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                networkCode, stationCode, locationCode, channelCode, evalMode])
                        phasef.flush()
                        break
                    elif '3' in iich:
                        channelCode = iich
                        phasef_writer.writerow([eventId, isotime, lowerUncertainty, upperUncertainty, PHStype,
                                                networkCode, stationCode, locationCode, channelCode, evalMode])
                        phasef.flush()
                        break

    eventf.close()
    phasef.close()
    return




