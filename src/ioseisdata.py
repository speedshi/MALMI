#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:32:40 2022

Functions related to input and output of seismic data.

@author: shipe
"""


import os
import obspy
import glob
import warnings
from ioformatting import read_seismic_fromfd
from pandas import to_datetime
import fnmatch


def seisdata_format_4ML(DFMT):
    
    if DFMT['seisdatastru_input'] == "AIO":
        # input seismic data files are stored simply in one folder
        # suitable for formatting small data set
        seisdate = format_AIO(dir_seismic=DFMT['dir_seismic_input'], instrument_code=DFMT['instrument_code'], 
                              dir_output=DFMT['dir_seismic_output'], freqband=DFMT['freqband'])
    elif DFMT['seisdatastru_input'] == 'SDS':
        # input seismic data files are organized in SDS
        # suitable for formatting large or long-duration data set
        format_SDS(seisdate=DFMT['seismic_date'], stainv=DFMT['stainv'], 
                   dir_seismic=DFMT['dir_seismic_input'], dir_output=DFMT['dir_seismic_output'], 
                   instrument_code=DFMT['instrument_code'], freqband=DFMT['freqband'])
    else:
        raise ValueError('Unrecognized input for: the input seismic data structure! Can\'t determine the structure of the input seismic data files!')
    
    return


def format_AIO(dir_seismic, instrument_code, dir_output, freqband=None):
    """
    Format seismic data stored simply in one folder so that the ouput data
    can be feed to various ML models.
    Seismic data sets are loaded and formated together for all station.
    Suitable for formatting small data set.
    
    Load all avaliable data in the input directory having the instrument codes listed.

    Parameters
    ----------
    dir_seismic : str
        path to the directory where seismic data are stored all in this folder.
    instrument_code : list of str
        the used instrument codes of the input seismic data,
        such as ["HH", "BH", "EH", "SH", "HG", "HN"];
        try to format and output data for all the listed instrument codes;
    dir_output : str
        directory for outputting seismic data, 
        NOTE do not add '/' at the last.
    freqband : list of float
        frequency range in Hz for filtering seismic data, 
        e.g. [3, 45] meaning filter seismic data to 3-45 Hz.
        default is None, means no filtering.

    Returns
    -------
    None.

    """

    # read in all continuous seismic data in the input folder as an obspy stream
    stream = read_seismic_fromfd(dir_seismic)
    seisdate = (stream[0].stats.starttime + (stream[0].stats.endtime - stream[0].stats.starttime)*0.5).date  # date when data exist
    
    # output to the seismic data format that QET can handle 
    stream2EQTinput(stream=stream, dir_output=dir_output, instrument_code=instrument_code, freqband=freqband)
    del stream
    
    return seisdate


def format_SDS(seisdate, stainv, dir_seismic, dir_output, instrument_code, location_code=['','00','R1', 'BT', 'SF', '*'], freqband=None):
    """
    Format seismic data organized in SDS data structure so that the ouput data
    can be feed to various ML models.
    Seismic data sets are formated per station.
    Suitable for formatting large or long-duration data set.
    
    SDS fromat of data archiving structure:
        year/network_code/station_code/channel_code.D/network_code.station_code.location_code.channel_code.D.year.day_of_year
        for example: 2020/CH/VDR/HHZ.D/CH.VDR..HHZ.D.2020.234

    Instrument code has a higher priority than location code.
    Both instrument code list and location code list are priority code list, the 
    program will try load only one instrument code and one location code, the code
    listed in front has higher priority.

    Parameters
    ----------
    seisdate : datetime.date
        the date of seismic data to be formated.
    stainv : obspy station inventory object.
        obspy station inventory containing the station information.
    dir_seismic : str
        path to the SDS archive directory.
    instrument_code : list of str
        the perfered list of instrument code of the input seismic data.
        We need the instrument code to look for data in SDS dirctory, this program will
        loop over this list until it can find data. Code listed first has higher priority.
        such as ["HH", "BH", "EH", "SH", "HG", "HN"] (in this example, 'HH' has the highest priority).    
    dir_output : str
        directory for outputting seismic data, 
        NOTE do not add '/' at the last.
    location_code : list of str, optional, default is ['','00','R1', 'BT', 'SF', '*'] (Note the last code '*' will match any location code it can find).
        the prefered list of location cods; specifying the perference order to load the data;
        For example: ['','00','R1'], in this situation '' will have the highest priority.
        If you only want load a specific location, just specify the perferred one, such as ['00'].
        If you don't want to spcify location code, use None which will use the first location_code where it can load data.
    freqband : list of float
        frequency range in Hz for filtering seismic data, 
        e.g. [3, 45] meaning filter seismic data to 3-45 Hz.
        default is None, means no filtering.
        
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    tdate = seisdate  # the date to be processed for SDS data files
    tyear = tdate.year  # year
    tday = tdate.timetuple().tm_yday  # day of the year

    for network in stainv:
        for station in network:
            # loop over each station for formatting input date set
            dir_stalevel = os.path.join(dir_seismic, str(tyear), network.code, station.code)  # station level
            
            if os.path.exists(dir_stalevel):
                # station folder exist
                
                for iinstru in instrument_code:
                    # loop over instrument code list to check and load data
                    dir_chalevel_want = os.path.join(dir_stalevel, iinstru+'*')
                    dir_chalevel = glob.glob(dir_chalevel_want)  # channel level                    
                    if len(dir_chalevel) == 0:
                        # folder of current instrument code does not exist
                        print("No data found for path: {}! Pass!".format(dir_chalevel_want))
                    elif len(dir_chalevel) <= 3:
                        # folder of current instrument code exists                    
                        
                        # determine the location code
                        ilocation = None
                        if isinstance(location_code, list) and (len(location_code)==1) and (location_code[0] != '*'):
                            # have a specific location code; only load data of that location
                            ilocation = location_code[0]
                        elif (location_code is None) or ((len(location_code)==1) and (location_code[0] == '*')):
                            # no specifying location code list, use the first location code it can find
                            for dir_icha in dir_chalevel:
                                dir_datelevel = os.path.join(dir_icha, '*.{:03d}'.format(tday))
                                sdatafile = glob.glob(dir_datelevel)
                                if len(sdatafile) > 0:
                                    ilocation = sdatafile[0].split(os.sep)[-1].split('.')[2]
                                    break
                        else:
                            # search avaliable location codes from the input location code preferece list
                            data_location_codes = []
                            for dir_icha in dir_chalevel:
                                dir_datelevel = os.path.join(dir_icha, '*.{:03d}'.format(tday))
                                sdatafile = glob.glob(dir_datelevel)
                                for ifile in sdatafile:
                                    data_location_codes.append(ifile.split(os.sep)[-1].split('.')[2])
                            data_location_codes = list(set(data_location_codes))
                            for iicd in location_code:
                                location_code_filtered = fnmatch.filter(data_location_codes, iicd.upper())
                                if len(location_code_filtered) == 1:
                                    ilocation = location_code_filtered[0]
                                    print('Find data at the prefered station location code: {}.'.format(ilocation))
                                    break
                                elif len(location_code_filtered) > 1:
                                    ilocation = location_code_filtered[0]
                                    warnings.warn('Find multiple location codes ({}) matching the current tested code {}. Choose the first one as the prefered station location code: {}.'
                                                  .format(location_code_filtered, iicd, ilocation))
                                    break

                        stream = obspy.Stream()  # initilize an empty obspy stream
                        if ilocation is not None:
                            for dir_icha in dir_chalevel:
                                # loop over each channel folder to load data of the current station
                                dir_datelevel = os.path.join(dir_icha, '*.{}.*.{:03d}'.format(ilocation, tday))  # date and location level, the final filename, use day of the year to identify data
                                sdatafile = glob.glob(dir_datelevel)  # final seismic data filename for the specified station, component and date
                                
                                if len(sdatafile)==0:
                                    print("No data found for {}! Pass!".format(dir_datelevel))
                                elif len(sdatafile)==1:
                                    print('Load data: {}.'.format(sdatafile[0]))
                                    stream += obspy.read(sdatafile[0])
                                else:
                                    raise ValueError("More than one file exist: {}! This should not happen.".format(sdatafile))
                        else:
                            warnings.warn('Cannot find data from the input preferred location code list: {}.'.format(location_code))
                            
                        # output data for the current station
                        if stream.count() > 0:
                            # have at least one component data
                            stream2EQTinput(stream=stream, dir_output=dir_output, instrument_code=None, freqband=freqband)
                            break  # already find and output data for this instrument code, no need to look at the rest instrument codes
                            del stream
                        else:
                            warnings.warn('No data found at station {} for the specified instrument codes {}, date {} and location code {}!'.format(station.code, instrument_code, seisdate, location_code))
                            del stream

                    else:
                        warnings.warn('More than 3 folders ({}) found for the instrument code {}! Pass!'.format(dir_chalevel, iinstru))

            else:
                # station folder does not exist, no data
                warnings.warn('No data found for: {}! Pass!'.format(dir_stalevel))
    
    return


def stream2EQTinput(stream, dir_output, instrument_code=["HH", "BH", "EH", "SH", "HG", "HN"], component_code=['Z','N','E','1','2','3'], freqband=None):
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
    instrument_code : list of str
        instrument_code for outputting, default: ["HH", "BH", "EH", "SH", "HG", "HN"];
        only output data which have the listed instrument codes;
        if None or [], then searching for all avaliable instrument code in the input stream.
    component_code : list of str
        instrument_code for outputting, default: ['Z','N','E','1','2','3'];
        only output data of the listed components; 
        Note complete data should have at least three component data, such as ['Z','N','E'];
        if None or [], then searching for all avaliable component code in the input stream.
    freqband : list of float
        frequency range in Hz for filtering seismic data, 
        e.g. [3, 45] meaning filter seismic data to 3-45 Hz.
        default is None, means no filtering.

    Returns
    -------
    None.

    Example
    -------
    dir_output = '/Users/human/eqt/examples/mseeds'
    stream2EQTinput(stream, dir_output)
    """
    
    timeformat = "%Y%m%dT%H%M%SZ"  # NOTE here output until second
    
    if not instrument_code:
        # no input instrument codes
        # search for all available instrument codes in the input stream data
        instrument_code = []
        for tr in stream:
            if tr.stats.channel[:-1] not in instrument_code:
                instrument_code.append(tr.stats.channel[:-1])
        del tr
    
    if not component_code:
        # no input component codes
        # search for all available component codes in the input stream data
        component_code = []
        for tr in stream:
            if tr.stats.channel[-1] not in component_code:
                component_code.append(tr.stats.channel[-1])
        del tr
    
    # scan all traces to get the station names
    stations = []
    for tr in stream:
        sname = tr.stats.station
        if sname not in stations:
            stations.append(sname)
    del tr
    
    # for a particular station, first check starttime and endtime, then output data
    for ista in stations:
        # select and output data for a perticular station
        
        # creat a folder for each station and output data in the folder
        # the data from the same station are output to the same folder
        dir_output_sta = os.path.join(dir_output, ista)
        if not os.path.exists(dir_output_sta):
            os.makedirs(dir_output_sta)
        
        for iinstru in instrument_code:
            # select and output data for a perticular instrument code
        
            # scan different channels for getting a unified time range (choose the wider one) at a perticular station
            stdata = stream.select(station=ista, channel=iinstru+'*')  # stream data of an instrument code
            if stdata.count() > 0:
                dcount = 0
                for tr in stdata:
                    if dcount == 0:
                        starttime = tr.stats.starttime
                        endtime = tr.stats.endtime
                    else:
                        starttime = min(starttime, tr.stats.starttime)
                        endtime = max(endtime, tr.stats.endtime)
                    dcount += 1
        
                # round datetime to the nearest second, and convert to the setted string format
                starttime_str = to_datetime(starttime.datetime).round('1s').strftime(timeformat)
                endtime_str = to_datetime(endtime.datetime).round('1s').strftime(timeformat)
            
                # Output data for each station and each channel
                # For a particular station, the three channel (if there are) share
                # the same time range in the final output filename.
                for icomp in component_code:  # not all component exist
                    trdata = stdata.select(station=ista, component=icomp)  # stream data of a component
                    if trdata.count() > 0:
                        if freqband is not None:
                            # filter data in specified frequency range
                            # note need to process in this way to avoide glitch after filtering
                            trdata.detrend('demean')
                            trdata.detrend('simple')
                            trdata.filter('bandpass', freqmin=freqband[0], freqmax=freqband[1], corners=2, zerophase=True)
                            trdata.taper(max_percentage=0.001, type='cosine', max_length=2)  # to avoid anormaly at bounday
                        
                        OfileName = trdata[0].id + '__' + starttime_str + '__' + endtime_str + '.mseed'
                        trdata.write(os.path.join(dir_output_sta, OfileName), format="MSEED")

    return


