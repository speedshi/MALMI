#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:23:38 2021

@author: Peidong SHI
@email: speedshi@hotmail.com
"""


from pandas import to_datetime
import os
from obspy import read_inventory
import json


def stream2EQTinput(stream, dir_output):
    """
    This function is used to format the input obspy stream into the EQ-Transformer 
    acceptable seismic data inputs.
    
    The three component seismic data of a station should be downloaded at the same time range.
    The output filename contains the time range of the data. For a perticular station,
    the starttime and endtime with a wider range of the three component is used as the 
    unified time range in the output filename.
    So don't split different component data of the same station to differnt stream,
    they must be kept in the same stream and be checked for outputting. You can simply
    merge different streams to a final complete stream which contrain all stations or at
    least all components of the same station, and then pass the steam to this function.

    In general the seismic data (stream) span a day (data are usually downloaded daily). 
    However, this function also accept data time range longer or smaller than a day.
    But using daily data segment is highly recommended, because by default the EQ-Transformer
    are set to process this kind of data (daily date segment). It now also works for longer or 
    short time range. But there is no guarantee that the future updated version will also 
    support this feature.

    Parameters
    ----------
    stream : obspy stream
        input seismic data.
    dir_output : str
        directory for outputting.

    Returns
    -------
    None.

    Example
    -------
    dir_output = '/Users/human/eqt/examples/mseeds'
    stream2EQTinput(stream, dir_output)
    """
    
    
    timeformat = "%Y%m%dT%H%M%SZ"  # NOTE here output until second
    components = ["E", "N", "Z", "1", "2", "3"]
    
    # scan all traces to get the station names
    stations = []
    for tr in stream:
        sname = tr.stats.station
        if sname not in stations:
            stations.append(sname)
    del tr
    
    # for a particular station, first check starttime and endtime, then output data
    for ista in stations:
        
        # scan different components for getting a unified time range (choose the wider one) for a perticular station
        dcount = 0
        for icomp in components:
            stdata = stream.select(station=ista, component=icomp)
            if stdata.count() > 0:
                for tr in stdata:
                    if dcount == 0:
                        starttime = tr.stats.starttime
                        endtime = tr.stats.endtime
                    else:
                        starttime = min(starttime, tr.stats.starttime)
                        endtime = max(endtime, tr.stats.endtime)
                    dcount += 1
            del stdata
    
        # round datetime to the nearest second, and convert to the setted string format
        starttime_str = to_datetime(starttime.datetime).round('1s').strftime(timeformat)
        endtime_str = to_datetime(endtime.datetime).round('1s').strftime(timeformat)
    
        # output data for each station, the data from the same station are output 
        # to the same folder
        # creat a folder for each station and output data in the folder
        dir_output_sta = os.path.join(dir_output, ista)
        if not os.path.exists(dir_output_sta):
            os.makedirs(dir_output_sta)
        
        # Output data for each station and each component
        # For a particular station, the three component (if there are) share
        # the same time range in the final filename.
        for icomp in components:
            stdata = stream.select(station=ista, component=icomp)
            if stdata.count() > 0:
                OfileName = stdata[0].id + '__' + starttime_str + '__' + endtime_str + '.mseed'
                stdata.write(os.path.join(dir_output_sta, OfileName), format="MSEED")
                    
                    
    return


def stainv2json(file_station, mseed_directory, dir_json):
    """
    Parameters
    ----------
    file_station : str
        filename (inclusing path) of the station metadata. 
        Must be in FDSNWS station text file format: *.txt;
        or StationXML format: *.xml.
    mseed_directory : str
        String specifying the path to the directory containing miniseed files. 
        Directory must contain subdirectories of station names, which contain miniseed files 
        in the EQTransformer format. 
        Each component must be a seperate miniseed file, and the naming
        convention is GS.CA06.00.HH1__20190901T000000Z__20190902T000000Z.mseed, 
        or more generally NETWORK.STATION.LOCATION.CHANNEL__STARTTIMESTAMP__ENDTIMESTAMP.mseed
    dir_json : str
        String specifying the path to the output json file.

    Returns
    -------
    stations_list.json: A dictionary (json file) containing information for the available stations.
    
    Example
    -------
    file_station = "../data/station/all_stations_inv.xml"
    mseed_directory = "../data/seismic_data/EQT/mseeds/"
    dir_json = "../data/seismic_data/EQT/json"
    stainv2json(file_station, mseed_directory, dir_json)
    """
    
    # read station metadata
    stafile_suffix = file_station.split('.')[-1]
    if stafile_suffix == 'xml' or stafile_suffix == 'XML':
        # input are in StationXML format
        stainfo = read_inventory(file_station, format="STATIONXML")
    elif stafile_suffix == 'txt' or stafile_suffix == 'TXT':
        # input are in FDSNWS station text file format
        stainfo = read_inventory(file_station, format="STATIONTXT")
    
    # get the name of all used stations
    sta_names = sorted([dname for dname in os.listdir(mseed_directory) if os.path.isdir(os.path.join(mseed_directory, dname))])
    station_list = {}
    
    # loop over each station for config the station jaso file    
    for network in stainfo:
        for station in network:
            if station.code in sta_names:
                # get the channel list for the current station
                
                # try to get the channel information from station inventory file
                sta_channels = []
                for channel in station:
                    sta_channels.append(channel.code)
                
                # correct station inventory file should contain channel information
                # otherwise check the MSEED filename in 'mseed_directory' for "network" and "channels"
                if not sta_channels:
                    # the input station inventory file does not contain channel information
                    # we need to find that information find the filename of the MSEED seismic data
                    data_dir = os.path.join(mseed_directory, station.code)
                    seedf_names = sorted([sfname for sfname in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, sfname))])
                    sta_channels = list(set([seedfl.split('.')[3].split('__')[0] for seedfl in seedf_names]))
                    
                # add entry to station list for the current station
                station_list[station.code] = {"network": network.code, 
                                              "channels": sta_channels, 
                                              "coords": [station.latitude, station.longitude, station.elevation]}
       
    # output station json file    
    if not os.path.exists(dir_json):
        os.makedirs(dir_json)
    jfilename = os.path.join(dir_json, 'station_list.json')
    with open(jfilename, 'w') as fp:
        json.dump(station_list, fp)     
    
    
    return

