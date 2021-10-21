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
from obspy.core.inventory import Inventory, Network, Station
import json
import obspy
from obspy import UTCDateTime
import warnings
import pandas as pd
import datetime
import numpy as np
import gc
import glob
import csv
import copy


def read_seismic_fromfd(dir_seismic, channels=None):
    """
    read in continuous seismic data as obspy stream from a specified folder.

    Parameters
    ----------
    dir_seismic : str
        path to the folder where all seismic data are saved.

    Returns
    -------
    stream : obspy stream
        containing all seismic data.
    channels : str
        channel name for loading data, default: None;
        if None then loading all available channels in the stream.

    """
    
    import fnmatch
    
    # obtain the filename of each seismic data file 
    file_seismicin = sorted([fname for fname in os.listdir(dir_seismic) if os.path.isfile(os.path.join(dir_seismic, fname))])
    
    # read in seismic data
    stream = obspy.Stream()
    for dfile in file_seismicin:
        stream += obspy.read(os.path.join(dir_seismic, dfile))
    
    if channels is not None:
        # select channels
        for tr in stream:
            if not any([fnmatch.fnmatch(tr.stats.channel, cha) for cha in channels]):
                # the channel of current trace not in the specified channel list
                # remove this trace
                stream.remove(tr)
    
    return stream


def output_seissegment(stream, dir_output, starttime, endtime):
    """
    This function is used to output seismic data segment accroding to input time
    range.

    Parameters
    ----------
    stream : obspy stream
        obspy stream data containing all seismic traces.
    dir_output : str
        path to the output directory.
    starttime : datetime
        starttime of data segment.
    endtime : datetime
        endtime of data segment.

    Returns
    -------
    None.

    """
    
    # get station names
    # scan all traces to get the station names
    stations = []
    for tr in stream:
        sname = tr.stats.station
        if sname not in stations:
            stations.append(sname)
    del tr
    
    # get channel names
    # search for all available channels in the input stream data
    channels = []
    for tr in stream:
        if tr.stats.channel not in channels:
            channels.append(tr.stats.channel)
    del tr
    
    # make sure the output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    # output data for each station and channel
    timeformat = "%Y%m%dT%H%M%SZ"
    for ista in stations:
        for ichan in channels:
            stdata = (stream.select(station=ista, channel=ichan)).copy()
            if stdata.count() > 0:
                stdata.trim(UTCDateTime(starttime), UTCDateTime(endtime), pad=False, fill_value=0)
                stdata.merge(fill_value=0)
                if stdata.count() > 0:
                    # make sure after trim there are data existing
                    assert(stdata.count()==1)  # should contain only one trace
                    starttime_str = starttime.strftime(timeformat)
                    endtime_str = endtime.strftime(timeformat)
                    ofname = os.path.join(dir_output, stdata[0].id + '__' + starttime_str + '__' + endtime_str + '.sac')
                    stdata.write(ofname, format="SAC")
                
                del stdata
    
    # del stream
    gc.collect()
    return


def stream2EQTinput(stream, dir_output, channels=["*HE", "*HN", "*HZ", "*H1", "*H2", "*H3"]):
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
    channels : str
        channel name for outputting, default: ["*HE", "*HN", "*HZ", "*H1", "*H2", "*H3"];
        if None or [], then searching for all channels in the input stream.

    Returns
    -------
    None.

    Example
    -------
    dir_output = '/Users/human/eqt/examples/mseeds'
    stream2EQTinput(stream, dir_output)
    """
    
    
    timeformat = "%Y%m%dT%H%M%SZ"  # NOTE here output until second
    
    if not channels:
        # no input channel names
        # search for all available channels in the input stream data
        channels = []
        for tr in stream:
            if tr.stats.channel not in channels:
                channels.append(tr.stats.channel)
        del tr
        gc.collect()
    
    # scan all traces to get the station names
    stations = []
    for tr in stream:
        sname = tr.stats.station
        if sname not in stations:
            stations.append(sname)
    del tr
    gc.collect()
    
    # for a particular station, first check starttime and endtime, then output data
    for ista in stations:
        # scan different channels for getting a unified time range (choose the wider one) for a perticular station
        dcount = 0
        for ichan in channels:
            stdata = stream.select(station=ista, channel=ichan)
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
            gc.collect()
    
        # round datetime to the nearest second, and convert to the setted string format
        starttime_str = to_datetime(starttime.datetime).round('1s').strftime(timeformat)
        endtime_str = to_datetime(endtime.datetime).round('1s').strftime(timeformat)
    
        # output data for each station, the data from the same station are output 
        # to the same folder
        # creat a folder for each station and output data in the folder
        dir_output_sta = os.path.join(dir_output, ista)
        if not os.path.exists(dir_output_sta):
            os.makedirs(dir_output_sta)
        
        # Output data for each station and each channel
        # For a particular station, the three channel (if there are) share
        # the same time range in the final filename.
        for ichan in channels:
            stdata = stream.select(station=ista, channel=ichan)
            if stdata.count() > 0:
                OfileName = stdata[0].id + '__' + starttime_str + '__' + endtime_str + '.mseed'
                stdata.write(os.path.join(dir_output_sta, OfileName), format="MSEED")
            del stdata
            gc.collect()
    
    gc.collect()                
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
    elif stafile_suffix == 'csv' or stafile_suffix == 'CSV':
        # SED COSEISMIQ CSV format, temporary format
        stainfo = Inventory(networks=[])
        stadf = pd.read_csv(file_station, delimiter=',', encoding='utf-8', 
                            names=['net','agency','sta code','description','lat','lon','altitude','type','up since','details'])
        for rid, row in stadf.iterrows():
            net = Network(code=row['net'], stations=[])
            sta = Station(code=row['sta code'],latitude=row['lat'],longitude=row['lon'],elevation=row['altitude'])
            net.stations.append(sta)
            stainfo.networks.append(net)
            
            
    # get the name of all used stations
    sta_names = sorted([dname for dname in os.listdir(mseed_directory) if os.path.isdir(os.path.join(mseed_directory, dname))])
    station_list = {}
    
    # loop over each station for config the station jason file    
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
                    sta_channels = sorted(list(set([seedfl.split('.')[3].split('__')[0] for seedfl in seedf_names])))
                    
                # add entry to station list for the current station
                station_list[station.code] = {"network": network.code, 
                                              "channels": sta_channels, 
                                              "coords": [station.latitude, station.longitude, station.elevation]}
    
    # check if all data have been assigned the station info  
    for ista in sta_names:
        if ista not in station_list:
            warnings.warn("Station info not found for station: {}.".format(ista))
        
    # output station json file    
    if not os.path.exists(dir_json):
        os.makedirs(dir_json)
    jfilename = os.path.join(dir_json, 'station_list.json')
    with open(jfilename, 'w') as fp:
        json.dump(station_list, fp)     
    
    gc.collect()
    return


def vector2trace(datainfo, data, dir_output='./'):
    """
    Write a data vector to an obspy trace.
    
    Parameters
    ----------
    datainfo : dictionary
        contains information about the station and data, includes:
            datainfo['station_name']: str, the name of the station, required;
            datainfo['channel_name']: str, the channel name of the trace, required;
                                      NOTE len(channel_name) MUST <= 3;
            datainfo['dt']: time sampling interval of the data in second, required;
            datainfo['starttime']: datetime, the starting time of the trace, required;
            datainfo['network']: str, network name of the trace, optional;
            datainfo['location']: str,  location name of the trace, optional;
            
    data : numpy vector
        the data vector to be written, shape: npts*1.
    dir_output : str
        the directory for output file.

    Returns
    -------
    None.

    """
    
    
    trace = obspy.Trace()  # initilize an empty obspy trace
    
    # set the trace header information
    trace.stats.station = datainfo['station_name']
    trace.stats.channel = datainfo['channel_name']
    trace.stats.delta = datainfo['dt']
    trace.stats.starttime = UTCDateTime(datainfo['starttime'])
    if 'network' in datainfo:
        trace.stats.network = datainfo['network']
    if 'location' in datainfo:
        trace.stats.location = datainfo['location']
    
    # assign data to the trace
    trace.data = data
    
    # set the displayed datetime format in the output filename
    # NOTE here output until second
    timeformat = "%Y%m%dT%H%M%SZ"  
    
    # make sure output directory exist
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    # output to file
    if trace.id[0] == '.':
        nametag = trace.id[1:] + '.' + trace.stats.starttime.datetime.strftime(timeformat) + '.mseed'
    else:
        nametag = trace.id + '.' + trace.stats.starttime.datetime.strftime(timeformat) + '.mseed'
    fname = os.path.join(dir_output, nametag)
    trace.write(fname, format="MSEED")
    
    del trace
    gc.collect()
    
    return


def EQTprob2trace(dir_probinput, dir_output, ev_otimes):
    """
    Transform probability output of EQ-Transformer to obspy trace.
    Parameters
    ----------
    dir_probinput : str
        path to the EQT probability data set of different stations.
    dir_output : str
        path for data outputs.
    ev_otimes : datetime list
        the datetimes we want to extract probability data segment, can be 
        viewed as the rough origin time of an event.

    Returns
    -------
    None.

    """
    
    import h5py
    
    # set internal parameters
    datainfo = {}
    datainfo['dt'] = 0.01  # time sampling rate of data in second, for EQT probability output, by default is 0.01 s
    data_size = 6000  # the data size in date points, by default the EQT output are 1 minutes long, thus 6000 points
    pbfname = 'prediction_probabilities.hdf5'  # the common filename of probability file for each station
    dtformat_EQT = '%Y-%m-%dT%H:%M:%S.%fZ'  # the datetime format using in EQT probability hdf5 outputs
    trtmax = 15  # time in second
    
    # obtain the folder name for the results of each station, each folder contain the probability data of one station
    dirnames = sorted([fdname for fdname in os.listdir(dir_probinput) if os.path.isdir(os.path.join(dir_probinput, fdname))])
    for sfdname in dirnames:
        # loop over each station folder, read and write results
        station_name = sfdname.split('_')[0]
        datainfo['station_name'] = station_name
        pbfile = os.path.join(dir_probinput, sfdname, pbfname)  # the filename of picking probability
        
        # load probability data set
        pbdf = h5py.File(pbfile, 'r')
        dsg_name = list(pbdf['probabilities'].keys())  # get the name of each probability data segment 
        dsg_starttime = [datetime.datetime.strptime(idsgnm.split('_')[-1], dtformat_EQT) for idsgnm in dsg_name]  # get the starttime of each probability data segment 
        
        for evotime in ev_otimes:
            # loop over each input event time for outputting data segments corresponding to the input time
            dir_output_ev = dir_output + '/' + evotime.strftime(dtformat_EQT)    
            
            timedfis = np.array([(ttt - evotime).total_seconds() for ttt in dsg_starttime])  # time different in second between the starttime of each data segment with the input event time
            slsegidx = np.where(timedfis <= 0, timedfis, -np.inf).argmax()  # obtain the selected data segment index: find the one with starttime before the input event time and also most closest in time
            
            # need to check if the input event time is within the starttime and endtime of the selected data segment    
            datainfo['starttime'] = dsg_starttime[slsegidx]
            if (evotime >= (datainfo['starttime'] - datetime.timedelta(seconds=trtmax))) and (evotime < (datainfo['starttime'] + datetime.timedelta(seconds=(data_size-1)*datainfo['dt']))):
                # input event time within the selceted time range -> output data
                pbdata = np.zeros((data_size, 3), dtype=np.float32)
                pbdf['probabilities'][dsg_name[slsegidx]].read_direct(pbdata)
            
                prob_D = pbdata[:,0]  # detection probability
                prob_P = pbdata[:,1]  # P-phase picking probability
                prob_S = pbdata[:,2]  # S-phase picking probability
                
                # output P-phase picking probability
                datainfo['channel_name'] = 'PBP'  # note maximum three characters, the last one must be 'P'
                vector2trace(datainfo, prob_P, dir_output_ev)
                
                # output S-phase picking probability
                datainfo['channel_name'] = 'PBS'  # note maximum three characters, the last one must be 'S'
                vector2trace(datainfo, prob_S, dir_output_ev)
                
                del pbdata, prob_D, prob_P, prob_S
                gc.collect()
                
            else:
                # input event time outside the selceted time range -> no output, generate warning
                warnings.warn('No data segment found around {} for station: {}.'.format(evotime, station_name))
    
    gc.collect()
    return


def read_lokicatalog(file_catalog):
    """
    This function is used to read the loki generated catalogue file and returns
    the event origin time, latitude, longitude and depth information.
    Parameters
    ----------
    file_catalog : str
        filename including path of the catalog file.

    Returns
    -------
    catalog : dic
         contains the catalog information.
        catalog['time'] : list of datetime
            origin time of catalog events.
        catalog['latitude'] : list of float
            longitude in degree of catalog events.
        catalog['longitude'] : list of float
            latitude in degree of catalog events.
        catalog['depth_km'] : list of float
            depth in km of catalog events.
        catalog['coherence_max'] : list of float
            coherence of catalog events.
        catalog['coherence_std'] : list of float
            standard deviation of migration volume.
        catalog['coherence_med'] : list of float
            median coherence of migration volume.

    """
    
    # set catalog format
    format_catalog = ['time', 'latitude', 'longitude', 'depth_km', 'cstd', 'cmed', 'cmax']  # indicate the meaning of each colume
    datetime_format_26 = '%Y-%m-%dT%H:%M:%S.%f'  # datetime format in the input file
    datetime_format_19 = '%Y-%m-%dT%H:%M:%S'  # datetime format in the input file
    
    # read catalog
    cadf = pd.read_csv(file_catalog, delimiter=' ', header=None, names=format_catalog,
                       skipinitialspace=True, encoding='utf-8')
    
    # format catalog information
    catalog = {}
    etimes = list(cadf['time'])
    catalog['time'] = []
    for itime in etimes:
        if len(itime) == 19:
            catalog['time'].append(datetime.datetime.strptime(itime, datetime_format_19))  # origin time
        elif len(itime) == 26:
            catalog['time'].append(datetime.datetime.strptime(itime, datetime_format_26))  # origin time
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
    catalog['latitude'] = list(cadf['latitude'])  # latitude in degree
    catalog['longitude'] = list(cadf['longitude'])  # logitude in degree
    catalog['depth_km'] = list(cadf['depth_km'])  # depth in km
    catalog['coherence_max'] = list(cadf['cmax'])  # maximum coherence of migration volume
    catalog['coherence_std'] = list(cadf['cstd'])  # standard deviation of migration volume
    catalog['coherence_med'] = list(cadf['cmed'])  # median coherence of migration volume
    
    del cadf, etimes
    gc.collect()
    return catalog


def read_malmipsdetect(file_detect):
    """
    This function is used to read the MALMI detection file which contains detection
    information, that is for each detected event how many stations are triggered,
    how many phases are triggered. Those information can be used for quality control.

    Parameters
    ----------
    file_detect : str
        The filename including path of the input file.

    Raises
    ------
    ValueError
        datetime format is not consistent with defined one.

    Returns
    -------
    detect_info : dic
        detect_info['starttime'] : list of datetime
            starttime and folder name of the detected event;
        detect_info['endtime'] : list of datetime
            endtime of the detected event;
        detect_info['station'] : list of float
            number of stations triggered of the detected event;
        detect_info['phase'] : list of float
            number of phase triggered of the detected event;

    """
    
    format_f = ['starttime', 'endtime', 'station', 'phase']
    datetime_format_26 = '%Y-%m-%dT%H:%M:%S.%f'  # datetime format in the input file
    datetime_format_19 = '%Y-%m-%dT%H:%M:%S'  # datetime format in the input file
    
    # read file
    df = pd.read_csv(file_detect, delimiter=' ', header=None, names=format_f,
                       skipinitialspace=True, encoding='utf-8', comment='#')
    
    # format output data
    detect_info = {}
    detect_info['starttime'] = []
    detect_info['endtime'] = []
    for ii in range(len(df)):
        if len(df.loc[ii,'starttime']) == 19:
            detect_info['starttime'].append(datetime.datetime.strptime(df.loc[ii,'starttime'], datetime_format_19))  # origin time
        elif len(df.loc[ii,'starttime']) == 26:
            detect_info['starttime'].append(datetime.datetime.strptime(df.loc[ii,'starttime'], datetime_format_26))  # origin time
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
        
        if len(df.loc[ii,'endtime']) == 19:
            detect_info['endtime'].append(datetime.datetime.strptime(df.loc[ii,'endtime'], datetime_format_19))  # origin time
        elif len(df.loc[ii,'endtime']) == 26:
            detect_info['endtime'].append(datetime.datetime.strptime(df.loc[ii,'endtime'], datetime_format_26))  # origin time
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
            
    detect_info['station'] = list(df['station'])
    detect_info['phase'] = list(df['phase'])
    
    return detect_info


def get_MLpicks_ftheart(dir_prob, dir_io, maxtd_p=3.0, maxtd_s=3.0, P_thrd=0.1, S_thrd=0.1, thephase_ftage='.phs', ofname=None):
    """
    This function is used to extract ML picks according to the calculated 
    theoratical arrivaltimes.

    Parameters
    ----------
    dir_prob : str
        The directory to where ML probabilities are saved.
    dir_io : str
        The directory to where the theoratical arrivaltimes file are saved, and
        is also the directory for outputting the ML pick file.
    maxtd_p : float, optional
        time duration in second, [P_theoratical_arrt-maxtd_p, P_theoratical_arrt+maxtd_p] 
        is the time range to consider possible ML picks for P-phase.
        The default is 3.0.
    maxtd_s : TYPE, optional
        time duration in second, [S_theoratical_arrt-maxtd_s, S_theoratical_arrt+maxtd_s] 
        is the time range to consider possible ML picks for S-phase.
        The default is 3.0.
    P_thrd : float, optional
        probability threshold above which is considered as acceptable P-picks. 
        The default is 0.1.
    S_thrd : float, optional
        probability threshold above which is considered as acceptable S-picks. 
        The default is 0.1.
    thephase_ftage : str, optional
        The filename tage of theoratical arrivaltime file, better use the suffix ('.phs') 
        of the theoratical arrivaltime file. The default is '.phs'.
    ofname : TYPE, optional
        The output ML picking filename. The default is None, then it share the 
        same filename as the theoratical arrivaltime file.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # check if the input data directory exist
    if not os.path.exists(dir_prob):
        raise ValueError('Input direcotry {} does not exist!'.format(dir_prob))
    if not os.path.exists(dir_io):
        raise ValueError('Input direcotry {} does not exist!'.format(dir_io))
    
    # read the theoretical phase arrivaltimes
    file_thephase = glob.glob(os.path.join(dir_io, '*'+thephase_ftage+'*'))
    assert(len(file_thephase) == 1)  # should contains only one theoretical phase arrivaltime file
    thearrvtt = read_arrivaltimes(file_thephase[0])
    stations = list(thearrvtt.keys())  # station list which have theoretical arrivaltimes
    
    if ofname is None:
        # set default output filename if it is not setted by inputs
        ofname = file_thephase[0].split('/')[-1].split(thephase_ftage)[0] + '.MLpicks'
    
    # initialize the output file
    ofile = open(os.path.join(dir_io, ofname), 'a')
    ofile.write('# station P_arrivaltime S_arrivaltime \n')
    ofile.flush()
    
    # loop over each station to find the ML picks and output to file
    for sta in stations:
        if 'P' in thearrvtt[sta]:
            # P-phase theoretical arrivaltime exist
            fprob_P = glob.glob(os.path.join(dir_prob, sta+'*PBP*'))
            assert(len(fprob_P) < 2)
            if len(fprob_P) == 1:
                stream = obspy.read(fprob_P[0])  # load P-phase probability
                art_start = thearrvtt[sta]['P'] - datetime.timedelta(seconds=maxtd_p)  # earliest possible P-phase arrivaltime
                art_end = thearrvtt[sta]['P'] + datetime.timedelta(seconds=maxtd_p)  # latest possible P-phase arrivaltime
                stream_sl = stream.slice(starttime=UTCDateTime(art_start), endtime=UTCDateTime(art_end))  # the probability segment between the earliest and latest possible phase arrivaltimes
                if (stream_sl.count() > 0) and (stream_sl[0].data.max() >= P_thrd):
                    # larger than threshold, is a good pick
                    P_picks = stream_sl[0].times(type='utcdatetime')[np.argmax(stream_sl[0].data)].datetime  # P-phase pick time
                else:
                    # P-phase probability not larger than threshold, no acceptable picks
                    P_picks = None
            else:
                # no P-phase probabilities
                P_picks = None
        else:
            # no P-phase theoretical arrivaltime
            P_picks = None
        
        if 'S' in thearrvtt[sta]:
            # S-phase theoretical arrivaltime exist
            fprob_S = glob.glob(os.path.join(dir_prob, sta+'*PBS*'))
            assert(len(fprob_S) < 2)
            if len(fprob_S) == 1:
                stream = obspy.read(fprob_S[0])  # load P-phase probability
                art_start = thearrvtt[sta]['S'] - datetime.timedelta(seconds=maxtd_s)  # earliest possible S-phase arrivaltime
                art_end = thearrvtt[sta]['S'] + datetime.timedelta(seconds=maxtd_s)  # latest possible S-phase arrivaltime
                stream_sl = stream.slice(starttime=UTCDateTime(art_start), endtime=UTCDateTime(art_end))  # the probability segment between the earliest and latest possible phase arrivaltimes
                if (stream_sl.count() > 0) and (stream_sl[0].data.max() >= S_thrd):
                    # larger than threshold, is a good pick
                    S_picks = stream_sl[0].times(type='utcdatetime')[np.argmax(stream_sl[0].data)].datetime  # S-phase pick time
                else:
                    # S-phase probability not larger than threshold, no acceptable picks
                    S_picks = None
            else:
                # no S-phase probabilities
                S_picks = None
        else:
            # no S-phase theoretical arrivaltime
            S_picks = None
        
        # output to file
        if (P_picks is not None) and (S_picks is not None):
            ofile.write(sta+' '+P_picks.isoformat()+' '+S_picks.isoformat()+'\n')
            ofile.flush()
        elif (P_picks is not None) and (S_picks is None):
            ofile.write(sta+' '+P_picks.isoformat()+' None\n')
            ofile.flush()
        elif (P_picks is None) and (S_picks is not None):
            ofile.write(sta+' None '+S_picks.isoformat()+'\n')
            ofile.flush()

    ofile.close()
    return


def read_arrivaltimes(file_arrvt):
    """
    This function is used to load the arrivaltimes of different stations.

    Parameters
    ----------
    file_arrvt : str
        the filename including path of the arrivaltime file in text format.

    Returns
    -------
    arrvtt : dic
        dictionary contains P- and S-wave arrivaltime information of different
        stations.
        arrvtt['station']['P'] : P-wave arrivaltime;
        arrvtt['station']['S'] : S-wave arrivaltime.

    """
    
    # set arrivaltime file input format
    format_arrvt = ['station', 'Pt', 'St']  # indicate the meaning of each colume
    datetime_format_26 = '%Y-%m-%dT%H:%M:%S.%f'  # datetime format in the input file
    datetime_format_19 = '%Y-%m-%dT%H:%M:%S'  # datetime format in the input file
    
    # read arrivaltime file
    arvtdf = pd.read_csv(file_arrvt, delimiter=' ', header=None, names=format_arrvt,
                         skipinitialspace=True, encoding='utf-8', comment='#')
    
    arrvtt = {}
    for ii in range(len(arvtdf)):
        arrvtt[arvtdf.loc[ii, 'station']] = {}
        if len(arvtdf.loc[ii, 'Pt']) == 26:
            arrvtt[arvtdf.loc[ii, 'station']]['P'] = datetime.datetime.strptime(arvtdf.loc[ii, 'Pt'], datetime_format_26)
        elif len(arvtdf.loc[ii, 'Pt']) == 19:
            arrvtt[arvtdf.loc[ii, 'station']]['P'] = datetime.datetime.strptime(arvtdf.loc[ii, 'Pt'], datetime_format_19)
        elif arvtdf.loc[ii, 'Pt'] == 'None':
            # no P-phase arrivaltimes
            pass
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
            
        if len(arvtdf.loc[ii, 'St']) == 26:
            arrvtt[arvtdf.loc[ii, 'station']]['S'] = datetime.datetime.strptime(arvtdf.loc[ii, 'St'], datetime_format_26)
        elif len(arvtdf.loc[ii, 'St']) == 19:
            arrvtt[arvtdf.loc[ii, 'station']]['S'] = datetime.datetime.strptime(arvtdf.loc[ii, 'St'], datetime_format_19)
        elif arvtdf.loc[ii, 'St'] == 'None':
            # no S-phase arrivaltimes
            pass
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
    
    return arrvtt


def retrive_catalog(dir_dateset, cata_ftag='catalogue', dete_ftag='event_station_phase_info.txt', cata_fold='*'):
    """
    This function is used to concatenate the catalogs together from the data base.

    Parameters
    ----------
    dir_dateset : str
        Path to the parent folder of catalog file and phase file.
    cata_ftag : str, optional
        Catalog filename. The default is 'catalogue'.
    dete_ftag : str, optional
        Detection filename. The default is 'event_station_phase_info.txt'.
    cata_fold : str, optional
        Catalog parent folder name. The default is '*'.

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
        mcatalog['dir'] : directory of the migration results of the event.
    """
    
    assert(os.path.exists(dir_dateset))

    file_cata = sorted(glob.glob(os.path.join(dir_dateset, '**/{}/{}'.format(cata_fold,cata_ftag)), recursive = True))  # file list of catalogue files
    file_dete = sorted(glob.glob(os.path.join(dir_dateset, '**/{}'.format(dete_ftag)), recursive = True))  # file list of detection files
    
    assert(len(file_cata) == len(file_dete))  # should correspond
    
    # initialize the final catalog
    mcatalog = {}
    mcatalog['id'] = []  # id of the event
    mcatalog['time'] = []  # origin time
    mcatalog['latitude'] = []  # latitude in degree
    mcatalog['longitude'] = []  # logitude in degree
    mcatalog['depth_km'] = []  # depth in km
    mcatalog['coherence_max'] = []  # maximum coherence of migration volume
    mcatalog['coherence_std'] = []  # standard deviation of migration volume
    mcatalog['coherence_med'] = []  # median coherence of migration volume
    mcatalog['starttime'] = []  # detected starttime of the event
    mcatalog['endtime'] = []  # detected endtime of the event
    mcatalog['station_num'] = []  # total number of stations triggered of the event
    mcatalog['phase_num'] = []  # total number of phases triggered of the event
    mcatalog['dir'] = []  # directory of the migration results of the event
    
    eid = 0
    # loop over each catalog/detection file and concatenate them together to make the final version
    for ii in range(len(file_cata)):
        assert(file_cata[ii].split('/')[-3] == file_dete[ii].split('/')[-3])  # they should share the common parent path
    
        # load catalog file
        ctemp = read_lokicatalog(file_cata[ii])
        
        # load detection file
        dtemp = read_malmipsdetect(file_dete[ii])
        
        assert(len(ctemp['time']) == len(dtemp['phase']))  # event number should be the same
        for iev in range(len(ctemp['time'])):
            assert(ctemp['time'][iev] <= dtemp['endtime'][iev])
            
            eid = eid + 1
            mcatalog['id'].append(eid)  # event id
            mcatalog['time'].append(ctemp['time'][iev])  # origin time
            mcatalog['latitude'].append(ctemp['latitude'][iev])  # latitude in degree
            mcatalog['longitude'].append(ctemp['longitude'][iev])  # logitude in degree
            mcatalog['depth_km'].append(ctemp['depth_km'][iev])  # depth in km
            mcatalog['coherence_max'].append(ctemp['coherence_max'][iev])  # maximum coherence of migration volume
            mcatalog['coherence_std'].append(ctemp['coherence_std'][iev])  # standard deviation of migration volume
            mcatalog['coherence_med'].append(ctemp['coherence_med'][iev])  # median coherence of migration volume
            mcatalog['starttime'].append(dtemp['starttime'][iev])  # starttime of the event
            mcatalog['endtime'].append(dtemp['endtime'][iev])  # endtime of the event
            mcatalog['station_num'].append(dtemp['station'][iev])  # total number of stations triggered
            mcatalog['phase_num'].append(dtemp['phase'][iev])  # total number of phases triggered
            dir_ers = ''
            sss = file_cata[ii].split('/')
            for pstr in sss[:-1]:
                dir_ers = os.path.join(dir_ers, pstr)
            dir_ers =  os.path.join(dir_ers, dtemp['starttime'][iev].isoformat())
            assert(os.path.exists(dir_ers))
            mcatalog['dir'].append(dir_ers)  # migration result direcotry of the event
            
        del ctemp, dtemp
    
    mcatalog['id'] = np.array(mcatalog['id'])
    mcatalog['time'] = np.array(mcatalog['time'])  
    mcatalog['latitude'] = np.array(mcatalog['latitude'])
    mcatalog['longitude'] = np.array(mcatalog['longitude'])
    mcatalog['depth_km'] = np.array(mcatalog['depth_km'])
    mcatalog['coherence_max'] = np.array(mcatalog['coherence_max'])
    mcatalog['coherence_std'] = np.array(mcatalog['coherence_std'])
    mcatalog['coherence_med'] = np.array(mcatalog['coherence_med'])
    mcatalog['starttime'] = np.array(mcatalog['starttime'])
    mcatalog['endtime'] = np.array(mcatalog['endtime'])
    mcatalog['station_num'] = np.array(mcatalog['station_num'])
    mcatalog['phase_num'] = np.array(mcatalog['phase_num'])
    mcatalog['dir'] = np.array(mcatalog['dir'])
    
    return mcatalog


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
            # stationId = '{}.{}.{}'.format(df_csta['net'].item(), df_csta['sta code'].item(), df_csta['location'].item())
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


def dict2csv(indic, filename=None):
    """
    Write a input dictory to a CSC file.

    Parameters
    ----------
    indic : dict
        The input dictionary.
    filename : str, optional
        The output filename including path. The default is None.

    Returns
    -------
    None.

    """
    
    if filename is None:
        filename = 'output.csv'
    
    outfile = open(filename, 'w', newline='')
    ofcsv = csv.writer(outfile, delimiter=',', lineterminator="\n")
    
    # write keys, i.e. CSV header row
    dickeys = list(indic.keys())
    ofcsv.writerow(dickeys)
    outfile.flush()
    
    # write each row
    NN = len(indic[dickeys[0]])  # total number of rows
    for ii in range(NN):
        crow = []
        for ikey in dickeys:
            crow.append(indic[ikey][ii])
    
        ofcsv.writerow(crow)
        outfile.flush()
    
    outfile.close()
    return


def csv2dict(file_csv):
    """
    This function is used to load the csv file and return a dict which contains
    the information of the csv file. The first row of the csv file contains the
    column names.

    Parameters
    ----------
    file_csv : str
        The input filename including path of the csv file.

    Returns
    -------
    outdic : dict
        The return dict which contains all information in the csv file.

    """
    
    # load station infomation: SED COSEISMIQ CSV format, temporary format
    df = pd.read_csv(file_csv, delimiter=',', header="infer", skipinitialspace=True, encoding='utf-8')
    
    outdic = {}
    for column in df:
       outdic[column] = copy.deepcopy(df[column].values)
    
    return outdic



