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


def read_seismic_fromfd(dir_seismic):
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

    """
    
    # obtain the filename of each seismic data file 
    file_seismicin = sorted([fname for fname in os.listdir(dir_seismic) if os.path.isfile(os.path.join(dir_seismic, fname))])
    
    # read in seismic data
    stream = obspy.Stream()
    for dfile in file_seismicin:
        stream += obspy.read(os.path.join(dir_seismic, dfile))
    
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
                stdata.trim(UTCDateTime(starttime), UTCDateTime(endtime), pad=True, fill_value=0)
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
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
            
        if len(arvtdf.loc[ii, 'St']) == 26:
            arrvtt[arvtdf.loc[ii, 'station']]['S'] = datetime.datetime.strptime(arvtdf.loc[ii, 'St'], datetime_format_26)
        elif len(arvtdf.loc[ii, 'St']) == 19:
            arrvtt[arvtdf.loc[ii, 'station']]['S'] = datetime.datetime.strptime(arvtdf.loc[ii, 'St'], datetime_format_19)
        else:
            raise ValueError('Error! Input datetime format not recoginzed!')
    
    return arrvtt


def retrive_catalog(dir_dateset, cata_ftag='catalogue', dete_ftag='event_station_phase_info.txt'):
    """
    This function is used to concatenate the catalogs together from the data base.

    Parameters
    ----------
    dir_dateset : str
        Path the catalog data set.
    cata_ftag : str, optional
        Catalog filename. The default is 'catalogue'.
    dete_ftag : str, optional
        Detection filename. The default is 'event_station_phase_info.txt'.

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
    
    file_cata = sorted(glob.glob(dir_dateset + '/**/{}'.format(cata_ftag), recursive = True))  # file list of catalogue file
    file_dete = sorted(glob.glob(dir_dateset + '/**/{}'.format(dete_ftag), recursive = True))  # file list of detection file
    
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
            assert(ctemp['time'][iev] >= dtemp['starttime'][iev])
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
            mcatalog['station_num'].append[dtemp['station'][iev]]  # total number of stations triggered
            mcatalog['phase_num'].append(dtemp['phase_num'][iev])  # total number of phases triggered
            dir_ers = ''
            sss = file_cata[ii].split('/')
            for pstr in sss[:-1]:
                dir_ers = os.path.join(dir_ers, pstr)
            dir_ers =  os.path.join(dir_ers, dtemp['starttime'][iev].isoformat())
            assert(os.path.exists(dir_ers))
            mcatalog['dir'].append(dir_ers)  # migration result direcotry of the event
            
        del ctemp, dtemp
    
    return mcatalog


def write_rtddstation(dir_tt, hdr_filename='header.hdr', dir_output='./', filename=None):
    
    
    from loki import traveltimes
    
    # load station metadata from traveltime table data set
    tobj = traveltimes.Traveltimes(dir_tt, hdr_filename)
    station_name = []  # station name
    station_lon = []  # station longitude in degree
    station_lat = []  # station latitude in degree
    station_ele = []  # station elevation in km
    for staname in tobj.db_stations:
        station_name.append(staname)
        station_lat.append(tobj.stations_coordinates[staname][0])
        station_lon.append(tobj.stations_coordinates[staname][1])
        station_ele.append(tobj.stations_coordinates[staname][2])
    
    ####### To be continue....
    
    
    return




