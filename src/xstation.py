#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:30:03 2022

Station related functions.

For station inventory of simple dictory format, the convention is:
stadict['network']: np.array of str, network code of each station;
stadict['station']: np.array of str, station code of each station;
stadict['latitude']: np.array of float, latitude in decimal degree of each station;
stadict['longitude']: np.array of float, longitude in decimal degree of each station;
stadict['elevation']: np.array of float, elevation in meters relative to the sea-level (positive for up) of each station;
stadict['location']: location code of each station, e.g. "00", "01";
stadict['depth']: depth in meter of each station; 
                  The local depth or overburden of the instruments location. 
                  For downhole instruments, the depth of the instrument under the surface ground level. 
                  For underground vaults, the distance from the instrument to the local ground level above.
                  so depth = elevaltion_of_surface - elevation_of_instrument.
stadict['instrument']: instrument code of each station, e.g. "SH", "HH", "FP";
stadict['component']: component code for each station, e.g. "ZNE", "Z12";

Each unique station is identified by network.station.location.instrument (such as: TA.N59A..BH)
Elevation is the true elevation of the actual station, it does not need to be corrected by depth.

@author: shipe
"""


import numpy as np
from obspy import read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel
import pandas as pd
from xcoordinate import coordsystem
from ioformatting import dict2csv


def get_station_ids(stainv):
    """
    Get the unique station id and location information.
    If channel information exist in station inventory, 
    each unique station is identified by "network.station.location.instrument" (such as: TA.N59A..BH).
    If no channel information exist in station inventory,
    each unique station is identified by "network.station" (such as: TA.N59A).

    INPUT:
        stainv: obspy station inventory object.

    RETURNS:
        staids: list containing station id.
        stainfo: dict containing station location information.

    """

    staids = []
    stainfo = {}
    for network in stainv:
        for station in network:
            # loop over each station
            if len(station.channels) > 0:
                # have channel information
                for ichannel in station.channels:
                    istaid = "{}.{}.{}.{}".format(network.code, station.code, ichannel.location_code, ichannel.code[:-1])  # station identification: network.station.location.instrument
                    if istaid not in staids:
                        staids.append(istaid)
                        stainfo[istaid] = {}
                        stainfo[istaid]['network'] = network.code
                        stainfo[istaid]['station'] = station.code
                        stainfo[istaid]['latitude'] = station.latitude
                        stainfo[istaid]['longitude'] = station.longitude
                        stainfo[istaid]['elevation'] = station.elevation
                        stainfo[istaid]['location'] = ichannel.location_code
                        stainfo[istaid]['instrument'] = ichannel.code[:-1]
                        stainfo[istaid]['depth'] = ichannel.depth
                        stainfo[istaid]['component'] = "{}".format(ichannel.code[-1])
                    else:
                        assert(stainfo[istaid]['depth'] == ichannel.depth)  # should have the same depth
                        assert(stainfo[istaid]['location'] == ichannel.location_code)  # should have the same location code
                        assert(stainfo[istaid]['instrument'] == ichannel.code[:-1])  # should have the same instrument code
                        if ichannel.code[-1] not in stainfo[istaid]['component']:
                            stainfo[istaid]['component'] += ichannel.code[-1]  # append the component code
            else:
                # no channel information
                istaid = "{}.{}".format(network.code, station.code)  # station identification: network.station
                if istaid not in staids:
                    staids.append(istaid)
                    stainfo[istaid] = {}
                    stainfo[istaid]['network'] = network.code
                    stainfo[istaid]['station'] = station.code
                    stainfo[istaid]['latitude'] = station.latitude
                    stainfo[istaid]['longitude'] = station.longitude
                    stainfo[istaid]['elevation'] = station.elevation
    assert(staids == list(stainfo.keys()))
    return staids, stainfo


def station_select(station, latrg=None, lonrg=None, elerg=None):
    """
    To select stations according to the input criterion.

    Parameters
    ----------
    station : dict
        Input station dict which contains information of each station therein.
        each parameter should be in numpy array format;
        station['station'] : station code (name) of each station;
        station['latitude'] : latitude in degree;
        station['longitude'] : logitude in degree;
        station['elevation'] : elevation in meter;
    latrg : list of float, optional
        latitude range in degree, [lat_min, lat_max].
        The default is None.
    lonrg : list of float, optional
        longitude range in degree, [lon_min, lon_max].
        The default is None.
    elerg : list of float, optional
        elevation range in meter, [ele_min, ele_max].
        The default is None.

    Returns
    -------
    station_s : dict
        The output stations after event selection.

    """
    
    NN = len(station['station'])  # total number of stations
    sindx = np.full((NN,), True)

    # select stations according to latitude range
    if latrg is not None:
        sindx_temp = (station['latitude'] >= latrg[0]) & (station['latitude'] <= latrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
    
    # select stations according to longitude range
    if lonrg is not None:
        sindx_temp = (station['longitude'] >= lonrg[0]) & (station['longitude'] <= lonrg[1])
        sindx = np.logical_and(sindx, sindx_temp)
        
    # select stations according to elevation range
    if elerg is not None:
        sindx_temp = (station['elevation'] >= elerg[0]) & (station['elevation'] <= elerg[1])
        sindx = np.logical_and(sindx, sindx_temp)

    station_s = {}
    for ikey in list(station.keys()):
        station_s[ikey] = station[ikey][sindx]

    return station_s


def stainv2stadict(stainv):
    """
    Transform obspy station inventory to python dictory.

    Parameters
    ----------
    stainv : obspy station inventory object
        station inventory.

    unique station is identified by network.station.location.instrument;
    unique station should have the same latitude, longitude, elevation, depth, and component;
    Note latitude, longitude, elevation are taken from the channel level.

    Returns
    -------
    stadict : dict
        station inventory.
        stadict['network']: network code of each station;
        stadict['station']: station code of each station;
        stadict['latitude']: latitude in decimal degree of each station;
        stadict['longitude']: longitude in decimal degree of each station;
        stadict['elevation']: elevation in meters relative to the sea-level (positive for up) of each station;
        stadict['location']: location code of each station;
        stadict['depth']: depth in meter of each station;
        stadict['instrument']: instrument code of each station, e.g. "SH", "HH";
        stadict['component']: component code for each station, e.g. "ZNE", "Z12";
    """
    
    stadict = {}
    stadict['network'] = []
    stadict['station'] = []
    stadict['latitude'] = [] 
    stadict['longitude'] = []
    stadict['elevation'] = []
    stadict['location'] = []
    stadict['depth'] = []
    stadict['instrument'] = []
    stadict['component'] = []
    
    for inet in stainv:
        for ista in inet:
            stadict['network'].append(inet.code)
            stadict['station'].append(ista.code)
                
            if len(ista.channels) > 0:
                # have channel information
                # add latitude, longitude, elevation from channel level
                # add location code, depth, instrument code, and component code
                chas = {}
                for icha in ista:
                    cha_key = "{}.{}.{}.{}".format(inet.code, ista.code, icha.location_code, icha.code[:-1])  # network.station.location.instrument
                    if cha_key not in chas:
                        # current station identification not exist
                        chas[cha_key] = {}
                        chas[cha_key]['latitude'] = icha.latitude
                        chas[cha_key]['longitude'] = icha.longitude
                        chas[cha_key]['elevation'] = icha.elevation
                        chas[cha_key]['location'] = icha.location_code
                        chas[cha_key]['depth'] = icha.depth
                        chas[cha_key]['instrument'] = icha.code[:-1]
                        chas[cha_key]['component'] = "{}".format(icha.code[-1])
                    else:
                        # current station identification exist 
                        assert(chas[cha_key]['latitude'] == icha.latitude)
                        assert(chas[cha_key]['longitude'] == icha.longitude)
                        assert(chas[cha_key]['elevation'] == icha.elevation)
                        assert(chas[cha_key]['location'] == icha.location_code)
                        assert(chas[cha_key]['depth'] == icha.depth)
                        assert(chas[cha_key]['instrument'] == icha.code[:-1])
                        if icha.code[-1] not in chas[cha_key]['component']:
                            chas[cha_key]['component'] += icha.code[-1]
                for jstac in list(chas.keys()):
                    stadict['latitude'].append(chas[jstac]['latitude'])
                    stadict['longitude'].append(chas[jstac]['longitude'])
                    stadict['elevation'].append(chas[jstac]['elevation'])
                    stadict['location'].append(chas[jstac]['location'])
                    stadict['depth'].append(chas[jstac]['depth'])
                    stadict['instrument'].append(chas[jstac]['instrument'])
                    stadict['component'].append(chas[jstac]['component'])
            else:
                # no channel information
                # get latitude, longitude, elevation from station level
                stadict['latitude'].append(ista.latitude)
                stadict['longitude'].append(ista.longitude)
                stadict['elevation'].append(ista.elevation)
                stadict['location'].append(None)
                stadict['depth'].append(None)
                stadict['instrument'].append(None)
                stadict['component'].append(None)
    
    NN = len(stadict['station'])
    for ikey in list(stadict.keys()):
        assert(len(stadict[ikey]) == NN)
        # stadict[ikey] = np.array(stadict[ikey])
    
    return stadict


def load_station(file_station, outformat='obspy'):
    """
    To read in station metadata and returns an obspy invertory object.

    Parameters
    ----------
    file_station : str
        filename (inclusing path) of the station metadata. 
        The data format should be recognizable by ObsPy, such as:
            FDSNWS station text format: *.txt,
            FDSNWS StationXML format: *.xml.
        or a simply CSV file using ',' as the delimiter in which the first row 
        is column name and must contain: 'network', 'station', 'latitude', 
        'longitude', 'elevation'. Latitude and longitude are in decimal degree 
        and elevation in meters relative to the sea-level (positive for up). 
    outformat : str
        specify the format of the loaded station invery in memory;
        'obspy': obspy station inventory object;
        'dict': simple python dictory format;
    
    Returns
    -------
    stainfo : obspy invertory object
        contains station information, such as network code, station code, 
        longitude, latitude, evelvation etc.

    """
    
    # read station metadata
    stafile_suffix = file_station.split('.')[-1]
    if stafile_suffix.lower() == 'xml':
        # input are in StationXML format
        try:
            stainfo = read_inventory(file_station, format="STATIONXML")
        except:
            stainfo = read_inventory(file_station, format="SC3ML")
    elif stafile_suffix.lower() == 'txt':
        # input are in FDSNWS station text file format
        stainfo = read_inventory(file_station, format="STATIONTXT")
    elif stafile_suffix.lower() == 'csv':
        # simple CSV format
        stainfo = read_stainv_csv(file_station, outformat=outformat)
    else:
        raise ValueError('Wrong input for input inventory file: {}! Format not recognizable!'.format(file_station))
    
    if (outformat.lower() == 'dict') and (stafile_suffix.lower() != 'csv'):
        # obspy station inventory to dict
        stainfo = stainv2stadict(stainfo)
    
    return stainfo


def read_stainv_csv(file_stainv, outformat='obspy'):
    """
    Read the csv format station inventory file, and format it to obspy station inventory object.

    Each station is identified by network.station.location.instrument,
    for example: TA.N59A..BH

    The input CSV file using ',' as the delimiter in which the first row 
    is column name and must contain: 'network', 'station', 'latitude', 
    'longitude', 'elevation'. Latitude and longitude are in decimal degree 
    and elevation in meters relative to the sea-level (positive for above the sea-level).
    Elevation is the true elevation of the actual station, it does not need to be corrected by depth.

    Other optional colume names are:
    location: location code of station, such as "00", "", "01". Default: "";
    instrument: instrument code, such as "SH", "HH", "BH", "FP";
    component: component code, shch as "ZNE", "Z12";
    depth: the local depth or overburden of the instruments location in meter.
           For downhole instruments, the depth of the instrument under the surface ground level.
           so depth = elevaltion_of_surface - elevation_of_instrument.
           Default: 0.

    Parameters
    ----------
    file_stainv : str
        filename of the input station inventory of csv format.
    outformat : str, optional
        specify the format of the loaded station invery in memory;
        'obspy': obspy station inventory object;
        'dict': simple python dictory format;
        The default is 'obspy'.

    Returns
    -------
    stainv : obspy station inventory object.

    """
    
    stadf = pd.read_csv(file_stainv, delimiter=',', encoding='utf-8', 
                        header="infer", skipinitialspace=True)
    
    # check if the input csv file contains the required columns
    if ('network' not in stadf.columns):
        raise ValueError("The input csv file must contain 'network' information!")
    if ('station' not in stadf.columns):
        raise ValueError("The input csv file must contain 'station' information!")
    if ('latitude' not in stadf.columns):
        raise ValueError("The input csv file must contain 'latitude' information!")
    if ('longitude' not in stadf.columns):
        raise ValueError("The input csv file must contain 'longitude' information!")
    if ('elevation' not in stadf.columns):
        raise ValueError("The input csv file must contain 'elevation' information!")
    if ('instrument' in stadf.columns) and ('component' in stadf.columns):
        have_channels = True
    elif ('instrument' not in stadf.columns) and ('component' not in stadf.columns):
        have_channels = False
    else:
        raise ValueError("Instrument code and component code must exist at the same time! You cannot present only one of them!")

    if outformat.lower() == 'obspy':
        stainv = Inventory(networks=[])
        
        net_dict = {}
        for rid, row in stadf.iterrows():
            if row['network'] not in list(net_dict.keys()):
                # network not include in net_dict
                net = Network(code=row['network'], stations=[])
                net_dict[row['network']] = net
            
            sta = Station(code=row['station'], latitude=row['latitude'], 
                        longitude=row['longitude'], elevation=row['elevation'])
            if have_channels:
                # add channel information
                if len(row['component']) != 3:
                    raise ValueError("Must input three components! Current is {}!".format(row['component']))

                if ('location' in row) and (row['location'] is not None) and (not np.isnan(row['location'])):
                    jlocation = row['location']
                else:
                    jlocation = ""  # default location code

                if 'depth' in row:
                    jdepth = row['depth']
                else:
                    jdepth = 0  # default depth 

                for icomp in row['component']:
                    jcha = Channel(code=row['instrument']+icomp, location_code=jlocation,
                                latitude=row['latitude'], longitude=row['longitude'], 
                                elevation=row['elevation'], depth=jdepth)
                    sta.channels.append(jcha)
                
            net_dict[row['network']].stations.append(sta)
        
        for inet in net_dict.keys():
            stainv.networks.append(net_dict[inet])

    elif outformat.lower() == 'dict':
        stainv = {}
        for ikey in list(stadf.columns):
            stainv[ikey] = stadf[ikey].values

        if 'location' in list(stadf.columns):
            stainv['location'] = np.where(np.isnan(stainv['location']), "", stainv['location'])

    else:
        raise ValueError("Wrong input for outformat: {}!".format(outformat))
    
    return stainv



class station():
    """
    Seismic station class.
    Station id is defined as "network_code.station_code.location_code.instrument_code".
    """

    def __init__(self, file_station):
        sta_dict = load_station(file_station, outformat='dict')
        self.file = file_station
        if 'network' in sta_dict:
            self.network = sta_dict['network']
        else:
            self.network = None
        if 'station' in sta_dict:
            self.station = sta_dict['station']
            self.number = len(self.station)
        else:
            raise ValueError("The station code must be provided!")
        if 'location' in sta_dict:
            self.location = sta_dict['location']
        else:    
            self.location = None
        if 'instrument' in sta_dict:
            self.instrument = sta_dict['instrument']
        else:    
            self.instrument = None
        if 'component' in sta_dict:
            self.component = sta_dict['component']
        else:        
            self.component = None
        if 'latitude' in sta_dict:
            self.latitude = sta_dict['latitude']
            self.latitude_min = np.nanmin(self.latitude)
            self.latitude_max = np.nanmax(self.latitude)
        else:    
            self.latitude = None
        if 'longitude' in sta_dict:
            self.longitude = sta_dict['longitude']
            self.longitude_min = np.nanmin(self.longitude)
            self.longitude_max = np.nanmax(self.longitude)
        else:
            self.longitude = None
        if 'elevation' in sta_dict:
            self.elevation = sta_dict['elevation']
            self.elevation_min = np.nanmin(self.elevation)
            self.elevation_max = np.nanmax(self.elevation)
        else:
            self.elevation = None

        self.id = np.array(["{}.{}.{}.{}".format(self.network[i], self.station[i], self.location[i], self.instrument[i]) for i in range(self.number)])

    def compute_xyz(self, utm_crs=None, z_scale=-1.0):
        """
        Compute the x, y, z UTM coordinates in meters of the stations.
        x: easting in meters;
        y: northing in meters;
        z: up or down in meters.

        """

        if utm_crs is None:
            # determine the CRS (Coordinate Reference System) for the UTM zone
            stacs = coordsystem()
            stacs.compute_crs(longitude=self.longitude, latitude=self.latitude)
            self.utm_crs = stacs.utm_crs
        else:
            self.utm_crs = utm_crs
            stacs = coordsystem(utm_crs=self.utm_crs)
            
        self.x, self.y = stacs.lonlat2xy(longitude=self.longitude, latitude=self.latitude)
        self.z = self.elevation * z_scale

        return

    def append_xyz(self, x=None, y=None, z=None):
        """
        Add the x, y, z (UTM coordinates in meters) of the stations 
        to the station class.
        """

        if x is not None:
            assert(len(x) == self.number), "The length of x is not equal to the number of stations!"
            self.x = np.array(x)
        
        if y is not None:
            assert(len(y) == self.number), "The length of y is not equal to the number of stations!"
            self.y = np.array(y)
        
        if z is not None:
            assert(len(z) == self.number), "The length of z is not equal to the number of stations!"
            self.z = np.array(z)

        return

    def write_csv(self, file_station='./station.csv'):
        """
        Save the station information to a CSV file.

        """

        sta_dict = {}
        for iattr_name, iattr_value in vars(self).items():
            if (isinstance(iattr_value, np.ndarray)) and (iattr_value.size == self.number):
                sta_dict[iattr_name] = iattr_value
            elif (isinstance(iattr_value, list)) and (len(iattr_value) == self.number):
                sta_dict[iattr_name] = np.array(iattr_value)
            else:
                pass

        dict2csv(sta_dict, filename=file_station)
        return

    def __str__(self):
        cstr = ""
        for key, value in self.__dict__.items():
            cstr += f"{key}: {value}, "
        return cstr.rstrip(", ")
