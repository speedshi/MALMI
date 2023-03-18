#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:23:38 2021

@author: Peidong SHI
@email: speedshi@hotmail.com
"""


import os
import json
import obspy
from obspy import UTCDateTime
import warnings
import pandas as pd
import datetime
import numpy as np
import glob
import csv
import copy
from utils_dataprocess import stfilter
from xsnr import estimate_snr


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
    channels : list of str
        channel name for loading data, e.g. ['HHE', 'HHN', 'HHZ'].
        default: None; if None then loading all available channels in the stream.

    """
    
    import fnmatch
    
    # # obtain the filename of each seismic data file 
    # file_seismicin = sorted([fname for fname in os.listdir(dir_seismic) if os.path.isfile(os.path.join(dir_seismic, fname))])
    # # read in seismic data
    # stream = obspy.Stream()
    # for dfile in file_seismicin:
    #     stream += obspy.read(os.path.join(dir_seismic, dfile))
    
    stream = obspy.read(os.path.join(dir_seismic, '*'))
    
    if channels is not None:
        # select channels
        for tr in stream:
            if not any([fnmatch.fnmatch(tr.stats.channel, cha.upper()) for cha in channels]):
                # the channel of current trace not in the specified channel list
                # remove this trace
                stream.remove(tr)
    
    return stream


def output_seissegment(stream, dir_output, starttime, endtime, freqband=None):
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
    freqband : list of float
        frequency range in Hz for filtering seismic data, 
        e.g. [3, 45] meaning filter seismic data to 3-45 Hz.
        default is None, means no filtering.

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
                if freqband is not None:
                    # filter data in specified frequency range
                    # note need to process in this way to avoide glitch after filtering
                    stdata.detrend('demean')
                    stdata.detrend('simple')
                    stdata.filter('bandpass', freqmin=freqband[0], freqmax=freqband[1], corners=2, zerophase=True)
                    stdata.taper(max_percentage=0.001, type='cosine', max_length=1)  # to avoid anormaly at bounday
                stdata.merge(method=1, fill_value=0)
                stdata.trim(UTCDateTime(starttime), UTCDateTime(endtime), pad=False, fill_value=0)
                if stdata.count() > 0:
                    # make sure after trim there are data existing
                    assert(stdata.count()==1)  # should contain only one trace
                    starttime_str = starttime.strftime(timeformat)
                    endtime_str = endtime.strftime(timeformat)
                    ofname = os.path.join(dir_output, stdata[0].id + '__' + starttime_str + '__' + endtime_str + '.sac')
                    stdata.write(ofname, format="SAC")
                
                del stdata
    
    return


def stainv2json(stainfo, mseed_directory=None, dir_json='./'):
    """
    Parameters
    ----------
    stainfo : obspy invertory object
        contains station information, such as network code, station code, 
        longitude, latitude, evelvation etc. can be obtained using function: 'xstation.load_station'.
    mseed_directory : str, default in None
        String specifying the path to the directory containing miniseed files. 
        Directory must contain subdirectories of station names, which contain miniseed files 
        in the EQTransformer format. 
        Each component must be a seperate miniseed file, and the naming
        convention is GS.CA06.00.HH1__20190901T000000Z__20190902T000000Z.mseed, 
        or more generally NETWORK.STATION.LOCATION.CHANNEL__STARTTIMESTAMP__ENDTIMESTAMP.mseed
    dir_json : str, default is './'
        String specifying the path to the output json file.

    Returns
    -------
    stations_list.json: A dictionary (json file) containing information for the available stations.
    
    Example
    -------
    mseed_directory = "../data/seismic_data/EQT/mseeds/"
    dir_json = "../data/seismic_data/EQT/json"
    stainv2json(stainfo, mseed_directory, dir_json)
    """
            
    # get the name of all used stations
    if mseed_directory is not None:
        sta_names = sorted([dname for dname in os.listdir(mseed_directory) if os.path.isdir(os.path.join(mseed_directory, dname))])
    else:
        # no input mseed_directory
        sta_names = None
    
    station_list = {}
    
    # loop over each station for config the station jason file    
    for network in stainfo:
        for station in network:
            if (sta_names is None) or (station.code in sta_names):
                # get the channel list for the current station
                
                # try to get the channel information from station inventory file first
                sta_channels = []
                for channel in station:
                    sta_channels.append(channel.code)
                
                # correct station inventory file should contain channel information
                # otherwise check the MSEED filename in 'mseed_directory' for "network" and "channels"
                # in this situation we must input 'mseed_directory'
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
        nametag = trace.id[1:] + '.' + trace.stats.starttime.strftime(timeformat) + '.mseed'
    else:
        nametag = trace.id + '.' + trace.stats.starttime.strftime(timeformat) + '.mseed'
    fname = os.path.join(dir_output, nametag)
    trace.write(fname, format="MSEED")
    
    del trace
    
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
                
            else:
                # input event time outside the selceted time range -> no output, generate warning
                warnings.warn('No data segment found around {} for station: {}.'.format(evotime, station_name))

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
        catalog['time'] : list of UTCDateTime
            origin time of catalog events.
        catalog['latitude'] : list of float
            latitude in degree of catalog events.
        catalog['longitude'] : list of float
            longitude in degree of catalog events.
        catalog['depth_km'] : list of float
            depth in km of catalog events.
        catalog['coherence_max'] : list of float
            coherence of catalog events.
        catalog['coherence_std'] : list of float
            standard deviation of migration volume.
        catalog['coherence_med'] : list of float
            median coherence of migration volume.

    """
    
    ff = open(file_catalog)
    line1 = ff.readline()
    ff.close()
    
    catalog = {}
    
    if len(line1) > 0:  # not empty
        if (len(line1.split()[0])==19) or (len(line1.split()[0])==26):  # the first colume is time
            # time format
            datetime_format_26 = '%Y-%m-%dT%H:%M:%S.%f'  # datetime format in the input file
            datetime_format_19 = '%Y-%m-%dT%H:%M:%S'  # datetime format in the input file
            
            # set catalog format
            Ncol = len(line1.split())  # total number of columes
            if Ncol == 7:  # indicate the meaning of each colume
                format_catalog = ['time', 'latitude', 'longitude', 'depth_km', 'coherence_std', 'coherence_med', 'coherence_max']  
            elif Ncol == 12:
                format_catalog = ['time', 'latitude', 'longitude', 'depth_km', 'coherence_std', 'coherence_med', 'coherence_max', 
                                  'coherence_mean', 'coherence_min', 'coherence_MAD', 'coherence_kurtosis', 'coherence_skewness']
            elif Ncol == 16:
                format_catalog = ['time', 'latitude', 'longitude', 'depth_km', 'coherence_std', 'coherence_med', 'coherence_max', 
                                  'coherence_mean', 'coherence_min', 'coherence_MAD', 'coherence_kurtosis', 'coherence_skewness', 
                                  'coherence_normstd', 'coherence_normMAD', 'coherence_normkurtosis', 'coherence_normskewness']
            else:
                raise ValueError('Unrecognized catalog format for {}!'.format(file_catalog))
            
            # read catalog
            cadf = pd.read_csv(file_catalog, delimiter=' ', header=None, names=format_catalog,
                               skipinitialspace=True, encoding='utf-8')
            
            # format catalog information
            # need to parse time information
            etimes = list(cadf['time'])
            catalog['time'] = []
            for itime in etimes:
                if len(itime) == 19:
                    catalog['time'].append(UTCDateTime(datetime.datetime.strptime(itime, datetime_format_19)))  # origin time
                elif len(itime) == 26:
                    catalog['time'].append(UTCDateTime(datetime.datetime.strptime(itime, datetime_format_26)))  # origin time
                else:
                    raise ValueError('Error! Input datetime format not recoginzed!')
            
            for ikey in format_catalog:
                if ikey.lower() != 'time':
                    catalog[ikey] = list(cadf[ikey])
        else:
            raise ValueError('Unrecognized catalog format for {}!'.format(file_catalog))
        
        del cadf, etimes
    
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
    
    format_f = ['starttime', 'endtime', 'station_num', 'phase_num']
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
    
    for ikey in format_f:
        if (ikey != 'starttime') and (ikey != 'endtime'):
            detect_info[ikey] = list(df[ikey])
            detect_info[ikey] = list(df[ikey])
    
    return detect_info


def get_MLpicks_ftheart(dir_prob, dir_io, maxtd_p=3.0, maxtd_s=3.0, P_thrd=0.1, S_thrd=0.1, thephase_ftage='.phs', ofname=None, dir_seis=None, snr_para=None):
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
        The default is 3.0 second.
    maxtd_s : float, optional
        time duration in second, [S_theoratical_arrt-maxtd_s, S_theoratical_arrt+maxtd_s] 
        is the time range to consider possible ML picks for S-phase.
        The default is 3.0 second.
    P_thrd : float, optional
        probability threshold above which is considered as acceptable P-picks. 
        The default is 0.1.
    S_thrd : float, optional
        probability threshold above which is considered as acceptable S-picks. 
        The default is 0.1.
    thephase_ftage : str, optional
        The filename tage of theoratical arrivaltime file, such as use the suffix ('.phs') 
        of the theoratical arrivaltime file. The default is '.phs'.
    ofname : str, optional
        The output ML picking filename. The default is None, then it share the 
        same filename as the theoratical arrivaltime file.
    dir_seis : str
        The directory to where seismic data are stored.
    snr_para : dict
        Parameters related to SNR estimation of picks.

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
    stations = list(thearrvtt.keys())  # station list which have theoretical arrivaltimes, should be in the format of 'network.station.location.instrument'
    
    # load probability data
    stream_all = read_seismic_fromfd(dir_prob, channels=None)
    
    # load seismic data which are needed for calculating SNR
    if dir_seis is not None:
        # set default snr calculation parameters
        if snr_para is None:
            snr_para = {}
        if 'fband' not in snr_para:
            snr_para['fband'] = None
        if 'method' not in snr_para:
            snr_para['method'] = 'maxamp'
        if 'noise_window_P' not in snr_para:
            snr_para['noise_window_P'] = [-2.5, -1.5]
        if 'signal_window_P' not in snr_para:
            snr_para['signal_window_P'] = [-0.5, 0.7]
        if 'noise_window_S' not in snr_para:
            snr_para['noise_window_S'] = [-3.5, -2.5]
        if 'signal_window_S' not in snr_para:
            snr_para['signal_window_S'] = [-0.4, 1.0]

        seismic_all = read_seismic_fromfd(dir_seis, channels=None)
        if snr_para['fband'] is not None:
            stfilter(seismic_all, snr_para['fband'])

    if ofname is None:
        # set default output filename if it is not setted by inputs
        ofname = file_thephase[0].split('/')[-1].split(thephase_ftage)[0] + '.MLpicks'
    
    # initialize the output file
    ofile = open(os.path.join(dir_io, ofname), 'w', newline='')
    ofcsv = csv.writer(ofile, delimiter=',', lineterminator="\n")
    ofcsv.writerow(['station', 'P', 'P_snr', 'S', 'S_snr'])
    ofile.flush()
    
    # loop over each station to find the ML picks and output to file
    for sta in stations:
        if 'P' in thearrvtt[sta]:
            # P-phase theoretical arrivaltime exist
            if len(sta.split('.'))==4:  # 'network.station.location.instrument'
                stream = stream_all.select(id=sta+"P")  # get P-phase probability for the current station
            elif len(sta.split('.'))==2:  # 'network.station'
                stream = stream_all.select(network=sta.split('.')[0], station=sta.split('.')[1], component="P")
            else:
                raise ValueError("Unrecoginze station identificator: {}!".format(sta))
            # fprob_P = glob.glob(os.path.join(dir_prob, sta+'*PBP*'))
            if stream.count() == 1:
                art_start = thearrvtt[sta]['P'] - datetime.timedelta(seconds=maxtd_p)  # earliest possible P-phase arrivaltime
                art_end = thearrvtt[sta]['P'] + datetime.timedelta(seconds=maxtd_p)  # latest possible P-phase arrivaltime
                stream_sl = stream.slice(starttime=UTCDateTime(art_start), endtime=UTCDateTime(art_end))  # the probability segment between the earliest and latest possible phase arrivaltimes
                if (stream_sl.count() > 0) and (stream_sl[0].data.max() >= P_thrd):
                    # larger than threshold, claim a pick
                    P_picks = stream_sl[0].times(type='utcdatetime')[np.argmax(stream_sl[0].data)].datetime  # P-phase pick time

                    # estimate the SNR of this pick
                    if dir_seis is not None:
                        P_snr = estimate_snr(trace=seismic_all.select(id=sta+"*").merge(), stime=P_picks,
                                             noise_window=snr_para['noise_window_P'], signal_window=snr_para['signal_window_P'], method=snr_para['method'])
                    else:
                        P_snr = None
                else:
                    # P-phase probability not larger than threshold, no acceptable picks
                    P_picks = None
                    P_snr = None
            elif stream.count() == 0:
                # no P-phase probabilities
                warnings.warn("No P-phase probabilities are found for station: {}!".format(sta))
                P_picks = None
                P_snr = None
            else:
                print(stream)
                raise ValueError('More than one P-prob trace are found for station: {}!'.format(sta))
        else:
            # no P-phase theoretical arrivaltime
            P_picks = None
            P_snr = None
        
        if 'S' in thearrvtt[sta]:
            # S-phase theoretical arrivaltime exist
            if len(sta.split('.'))==4:  
                stream = stream_all.select(id=sta+"S")
            elif len(sta.split('.'))==2:  # 'network.station'
                stream = stream_all.select(network=sta.split('.')[0], station=sta.split('.')[1], component="S")  # get S-phase probability for the current station
            else:
                raise ValueError("Unrecoginze station identificator: {}!".format(sta))
            if stream.count() == 1:
                art_start = thearrvtt[sta]['S'] - datetime.timedelta(seconds=maxtd_s)  # earliest possible S-phase arrivaltime
                art_end = thearrvtt[sta]['S'] + datetime.timedelta(seconds=maxtd_s)  # latest possible S-phase arrivaltime
                stream_sl = stream.slice(starttime=UTCDateTime(art_start), endtime=UTCDateTime(art_end))  # the probability segment between the earliest and latest possible phase arrivaltimes
                if (stream_sl.count() > 0) and (stream_sl[0].data.max() >= S_thrd):
                    # larger than threshold, claim a pick
                    S_picks = stream_sl[0].times(type='utcdatetime')[np.argmax(stream_sl[0].data)].datetime  # S-phase pick time

                    # estimate the SNR of this pick
                    if dir_seis is not None:
                        S_snr = estimate_snr(trace=seismic_all.select(id=sta+"*").merge(), stime=S_picks,
                                             noise_window=snr_para['noise_window_S'], signal_window=snr_para['signal_window_S'], method=snr_para['method'])
                    else:
                        S_snr = None
                else:
                    # S-phase probability not larger than threshold, no acceptable picks
                    S_picks = None
                    S_snr = None
            elif stream.count() == 0:
                # no S-phase probabilities
                warnings.warn("No S-phase probabilities are found for station: {}!".format(sta))
                S_picks = None
                S_snr = None
            else:
                print(stream)
                raise ValueError('More than one S-prob trace are found for station: {}!'.format(sta))
        else:
            # no S-phase theoretical arrivaltime
            S_picks = None
            S_snr = None
        
        # output picks to file for the current station
        t2sfromat = "%Y-%m-%dT%H:%M:%S.%f"  # output datetime format, presion down to millisecond, e.g. '2009-08-24T00:20:03.000000'
        if (P_picks is not None) and (S_picks is not None):
            ofcsv.writerow([sta, P_picks.strftime(t2sfromat), str(P_snr), S_picks.strftime(t2sfromat), str(S_snr)])
            ofile.flush()
        elif (P_picks is not None) and (S_picks is None):
            ofcsv.writerow([sta, P_picks.strftime(t2sfromat), str(P_snr), 'None', str(S_snr)])
            ofile.flush()
        elif (P_picks is None) and (S_picks is not None):
            ofcsv.writerow([sta, 'None', str(P_snr), S_picks.strftime(t2sfromat), str(S_snr)])
            ofile.flush()
        elif (P_picks is None) and (S_picks is None):
            # no picks for P and S-phases, no output
            pass

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
        arrvtt['station_name']['P'] : P-wave arrivaltime;
        arrvtt['station_name']['P_snr'] : P-wave pick signal_noise_ratio;
        arrvtt['station_name']['S'] : S-wave arrivaltime;
        arrvtt['station_name']['S_snr'] : S-wave pick signal_noise_ratio;

    """
    
    # set arrivaltime file input format
    datetime_format_26 = '%Y-%m-%dT%H:%M:%S.%f'  # datetime format in the input file
    datetime_format_19 = '%Y-%m-%dT%H:%M:%S'  # datetime format in the input file
    
    # read the first line to determine the input file format
    file1 = open(file_arrvt, 'r')
    line1 = file1.readline()
    file1.close()
    
    # read arrivaltime file
    if line1[0] == '#':
        format_arrvt = ['station', 'P', 'S']  # indicate the meaning of each colume
        arvtdf = pd.read_csv(file_arrvt, delimiter=' ', header=None, names=format_arrvt,
                             skipinitialspace=True, encoding='utf-8', comment='#')
    else:
        # csv format with headline
        arvtdf = csv2dict(file_arrvt, delimiter=',')
    
    arrvtt = {}
    for ii in range(len(arvtdf['station'])):
        ista = arvtdf['station'][ii]  # station name
        arrvtt[ista] = {}
        
        if len(arvtdf['P'][ii]) == 26:
            arrvtt[ista]['P'] = UTCDateTime.strptime(arvtdf['P'][ii], datetime_format_26)
        elif len(arvtdf['P'][ii]) == 19:
            arrvtt[ista]['P'] = UTCDateTime.strptime(arvtdf['P'][ii], datetime_format_19)
        elif arvtdf['P'][ii] == 'None':
            # no P-phase arrivaltimes
            pass
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')

        if 'P_snr' in arvtdf:
            if arvtdf['P_snr'][ii] == 'None':
                # no P-pick snr
                pass
            else:
                arrvtt[ista]['P_snr'] = float(arvtdf['P_snr'][ii])

        if len(arvtdf['S'][ii]) == 26:
            arrvtt[ista]['S'] = UTCDateTime.strptime(arvtdf['S'][ii], datetime_format_26)
        elif len(arvtdf['S'][ii]) == 19:
            arrvtt[ista]['S'] = UTCDateTime.strptime(arvtdf['S'][ii], datetime_format_19)
        elif arvtdf['S'][ii] == 'None':
            # no S-phase arrivaltimes
            pass
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
    
        if 'S_snr' in arvtdf:
            if arvtdf['S_snr'][ii] == 'None':
                # no S-pick snr
                pass
            else:
                arrvtt[ista]['S_snr'] = float(arvtdf['S_snr'][ii]) 
            
    return arrvtt


def dict2csv(indic, filename=None, mode='auto'):
    """
    Write an input dictionary to a CSV file.

    Parameters
    ----------
    indic : dict
        The input dictionary.
        inside this dictionary, each entry much be a list;
    filename : str, optional
        The output filename including path. The default is None.
    mode : str, optional, default is 'auto'
        writing mode; 'a+' for appending; 'w' for rewrite;
        'auto: 'If a file already exist, use append mode; otherwise use rewrite mode.   
    
    Returns
    -------
    None.

    """
    
    if filename is None:
        filename = 'output.csv'
    
    if mode.lower() == 'auto':
        if os.path.exists(filename):
            mode = 'a+'
        else:
            mode = 'w'
        
    outfile = open(filename, mode, newline='')
    ofcsv = csv.writer(outfile, delimiter=',', lineterminator="\n")
    
    if indic:
        # write keys, i.e. CSV header row
        dickeys = list(indic.keys())
        if (mode.lower()[0] != 'a'):  # no need to wirte hearder row in append mode
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


def csv2dict(file_csv, delimiter=','):
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
    
    # load csv file
    df = pd.read_csv(file_csv, delimiter=delimiter, header="infer", skipinitialspace=True, encoding='utf-8')
    
    outdic = {}
    for column in df:
       outdic[column] = copy.deepcopy(df[column].values)
    
    return outdic










