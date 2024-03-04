#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:52:23 2022

Functions related to coordinates, such as coordinate extraction and conversion.

@author: shipe
"""


import numpy as np
import pyproj
from pyproj import Proj, Transformer
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def grid2mgregion(grid):
    """
    Retrive the boundary latitude and lontidue of the migration region from grid parameters.
    Only applies to rectrangular region without rotation and shift, 
    i.e.  X-axis -> East, Y-axis -> North.

    Parameters
    ----------
    grid : dict
        grid['LatOrig']: latitude in decimal degrees of the origin point of the rectangular migration region (float, min:-90.0, max:90.0);
        grid['LongOrig']: longitude in decimal degrees of the origin point of the rectangular migration region (float, min:-180.0, max:180.0);
        grid['xOrig']: X location of the grid origin in km relative to the geographic origin (positive: east).
        grid['yOrig']: Y location of the grid origin in km relative to the geographic origin (positive: north).
        grid['xNum']: number of grid nodes in the X direction;
        grid['yNum']: number of grid nodes in the Y direction;
        grid['dgrid']: grid spacing in kilometers.

    Returns
    -------
    mgregion : dict
        the lat/lon boundary of migration region in degree.
        mgregion['latitude_min']
        mgregion['latitude_max']
        mgregion['longitude_min']
        mgregion['longitude_max']

    """
    
    from loki import LatLongUTMconversion

    # determine the lon/lat of the migration area
    refell=23
    (UTMZone, eorig, norig) = LatLongUTMconversion.LLtoUTM(refell, grid['LatOrig'], grid['LongOrig'])  # the Cartesian coordinate of the origin point in meter
    east_m_last = (grid['xOrig'] + (grid['xNum']-1) * grid['dgrid']) * 1000.0  # the East coordinate (X) of the last point for the migration area in meter
    north_m_last = (grid['yOrig'] + (grid['yNum']-1) * grid['dgrid']) * 1000.0  # the North coordinate (Y) of the last point for the migration area in meter
    latitude_last, longitude_last = LatLongUTMconversion.UTMtoLL(refell, north_m_last+norig, east_m_last+eorig, UTMZone)  # from Cartisian to latitude, longitude
    east_m_first = grid['xOrig'] * 1000.0  # the East coordinate (X) of the first point for the migration area in meter
    north_m_first = grid['yOrig'] * 1000.0  # the North coordinate (Y) of the first point for the migration area in meter
    latitude_first, longitude_first = LatLongUTMconversion.UTMtoLL(refell, north_m_first+norig, east_m_first+eorig, UTMZone)  # from Cartisian to latitude, longitude
    mgregion = {}
    mgregion['longitude_min'] = min(longitude_last, grid['LongOrig'], longitude_first)
    mgregion['longitude_max'] = max(longitude_last, grid['LongOrig'], longitude_first)
    mgregion['latitude_min'] = min(latitude_last, grid['LatOrig'], latitude_first)
    mgregion['latitude_max'] = max(latitude_last, grid['LatOrig'], latitude_first)
    
    return mgregion


def get_lokicoord(dir_tt, hdr_filename='header.hdr', extr=0.05, consider_mgregion=True):
    """
    This function is used to get the coordinates of station, plotting region
    and migration region from traveltime table data set of LOKI.

    Parameters
    ----------
    dir_tt : str
        path to the travetime data directory of LOKI.
    hdr_filename : str, optional
        travetime data set header filename. The default is 'header.hdr'.
    extr : float, optional
        extend ratio for automatically get the plotting region. 
        The default is 0.05.
    consider_mgregion : boolen
        indicate whether consider 'mgregion' when calculating 'region'.

    Returns
    -------
    inv : obspy invertory format
        station inventory containing station metadata.
    region : list of float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    mgregion : list of float
        the lat/lon boundary of migration region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    
    """
    
    from loki import traveltimes
    from obspy.core.inventory import Inventory, Network, Station
    from loki import LatLongUTMconversion
    
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
    
    # create an obspy invertory according to station information
    inv = Inventory(networks=[], source="MALMI_tt")
    net = Network(code="XX", stations=[], description="Tempory network generated by MALMI.")
    for ista in range(len(station_name)):
        sta = Station(code=station_name[ista], latitude=station_lat[ista],
                      longitude=station_lon[ista], elevation=station_ele[ista]*1000)  # note elevation need to transfer to meter
        net.stations.append(sta)
    inv.networks.append(net)
    
    # determine the lon/lat of migration area
    refell=23
    (UTMZone, eorig, norig) = LatLongUTMconversion.LLtoUTM(refell, tobj.lat0, tobj.lon0)  # the Cartesian coordinate of the origin point in meter
    elast_m = tobj.x[-1]*1000.0  # the East coordinate (X) of the last point for the migration area in meter
    nlast_m = tobj.y[-1]*1000.0  # the North coordinate (Y) of the last point for the migration area in meter
    late_last, lone_last = LatLongUTMconversion.UTMtoLL(refell, nlast_m+norig, elast_m+eorig, UTMZone)  # latitude, longitude
    efirst_m = tobj.x[0]*1000.0  # the East coordinate (X) of the first point for the migration area in meter
    nfirst_m = tobj.y[0]*1000.0  # the North coordinate (Y) of the first point for the migration area in meter
    late_first, lone_first = LatLongUTMconversion.UTMtoLL(refell, nfirst_m+norig, efirst_m+eorig, UTMZone)  # latitude, longitude
    mgregion = []
    mgregion.append(min(lone_last, tobj.lon0, lone_first))
    mgregion.append(max(lone_last, tobj.lon0, lone_first))
    mgregion.append(min(late_last, tobj.lat0, late_first))
    mgregion.append(max(late_last, tobj.lat0, late_first))
    
    # determine the lon/lat for plotting the basemap
    if consider_mgregion:
        lon_min = min(np.min(station_lon), mgregion[0])
        lon_max = max(np.max(station_lon), mgregion[1])
        lat_min = min(np.min(station_lat), mgregion[2])
        lat_max = max(np.max(station_lat), mgregion[3])
    else:
        lon_min = np.min(station_lon)
        lon_max = np.max(station_lon)
        lat_min = np.min(station_lat)
        lat_max = np.max(station_lat)
    region = [lon_min-extr*(lon_max-lon_min), lon_max+extr*(lon_max-lon_min),
              lat_min-extr*(lat_max-lat_min), lat_max+extr*(lat_max-lat_min)]
    
    return inv, region, mgregion



def get_regioncoord(grid, stainv, extr=0.05, consider_mgregion=True):
    """
    Get the coordinates of plotting region and migration region from grid.

    Parameters
    ----------
    grid : dict
        MALMI grid dictionary.
    stainv : obspy invertory object
        station inventory.
    extr : float, optional
        extend ratio for automatically get the plotting region. 
        The default is 0.05.
    consider_mgregion : boolen
        indicate whether consider 'mgregion' when calculating 'region'.

    Returns
    -------
    region : list of float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    mgregion : list of float
        the lat/lon boundary of migration region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.

    """
    
    mgregion_dict = grid2mgregion(grid)
    mgregion = [mgregion_dict['longitude_min'], mgregion_dict['longitude_max'], mgregion_dict['latitude_min'], mgregion_dict['latitude_max']]
    
    station_lon = []
    station_lat = []
    for inet in stainv:
        for ista in inet:
            station_lon.append(ista.longitude)
            station_lat.append(ista.latitude)
            
    # determine the lon/lat range for plotting basemap
    if consider_mgregion:
        lon_min = min(np.min(station_lon), mgregion[0])
        lon_max = max(np.max(station_lon), mgregion[1])
        lat_min = min(np.min(station_lat), mgregion[2])
        lat_max = max(np.max(station_lat), mgregion[3])
    else:
        lon_min = np.min(station_lon)
        lon_max = np.max(station_lon)
        lat_min = np.min(station_lat)
        lat_max = np.max(station_lat)
    region = [lon_min-extr*(lon_max-lon_min), lon_max+extr*(lon_max-lon_min),
              lat_min-extr*(lat_max-lat_min), lat_max+extr*(lat_max-lat_min)]
    
    return region, mgregion



def get_utm_zone(longitude, latitude):
    '''
    Automatically calculate the UTM zone from the longitude and latitude.
    '''

    # use the mean value if the input is a list or numpy array
    if isinstance(longitude, (list, np.ndarray)):
        longitude_c = np.nanmean(longitude)
    else:
        longitude_c = longitude
    if isinstance(latitude, (list, np.ndarray)):
        latitude_c = np.nanmean(latitude)
    else:
        latitude_c = latitude

    zone_number = (int((longitude_c + 180) / 6) % 60) + 1
    if latitude_c >= 0:
        zone_letter = 'N'
    else:
        zone_letter = 'S'
    return zone_number, zone_letter


def lonlat2xy(longitude, latitude, utm_zone=None):
    if utm_zone is None:
        zone_number, zone_letter = get_utm_zone(longitude, latitude)
    else:
        zone_number = utm_zone
    LonLat_To_XY = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84', datum='WGS84', preserve_units=True)
    x, y = LonLat_To_XY(longitude, latitude)  # convert lon/lat to x/y in meter
    return x, y


def xy2lonlat(x, y, utm_zone):
    XY_To_LonLat = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84', preserve_units=True, inverse=True)
    longitude, latitude = XY_To_LonLat(x, y, inverse=True)  # convert x/y to lon/lat in degree
    return longitude, latitude


class coordsystem:
    """
    Coordinate system class.

    Transformer.from_crs("epsg:4326", self.utm_crs, always_xy=True) 
    creates a Transformer object that can convert from the WGS84 
    geographic coordinate system (EPSG:4326) to the UTM coordinate system. 
    The transform method of this object is then used to convert the 
    latitude and longitude to UTM coordinates.

    Please note that the always_xy=True argument is used to specify that 
    the input and output coordinates should be in (longitude, latitude) or (easting, northing) order, 
    which is the standard in GIS applications. 
    If you don't include this argument, 
    pyproj will expect the coordinates in (latitude, longitude) order, 
    which can be confusing.

    From lat/lon to UTM: utm_easting, utm_northing = transformer.transform(longitude, latitude)
    From UTM to lat/lon: longitude, latitude = transformer.transform(utm_easting, utm_northing, direction='INVERSE')

    The default "ele_to_depth_scale" is -1.0, which means the elevation is converted to depth by multiplying -1.0.
    i.e. depth is negative of elevation, downward is positive.

    """

    def __init__(self, utm_crs=None, elevation_to_depth_scale=-1.0):
        self.utm_crs = utm_crs
        self.elevation_to_depth_scale = elevation_to_depth_scale

    def compute_crs(self, longitude, latitude):
        # determine the CRS (Coordinate Reference System) for the UTM zone
        longitude_min = np.nanmin(longitude)
        longitude_max = np.nanmax(longitude)
        latitude_min = np.nanmin(latitude)
        latitude_max = np.nanmax(latitude)
        utm_crs_list = query_utm_crs_info(datum_name="WGS84",
                                          area_of_interest=AreaOfInterest(
                                          west_lon_degree=longitude_min,
                                          south_lat_degree=latitude_min,
                                          east_lon_degree=longitude_max,
                                          north_lat_degree=latitude_max))
        self.utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        return 

    def lonlat2xy(self, longitude, latitude):
        # convert latitude and longitude in degree to UTM in meter
        assert(self.utm_crs is not None), "Please compute the UTM CRS first."
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs, always_xy=True)  
        x, y = transformer.transform(longitude, latitude)
        return x, y
    
    def lonlatele2xyz(self, longitude, latitude, elevation):
        # convert latitude and longitude in degree to UTM in meter
        # convert elevation in meter to depth in meter
        assert(self.utm_crs is not None), "Please compute the UTM CRS first."
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs, always_xy=True)  
        x, y = transformer.transform(longitude, latitude)
        z = elevation * self.elevation_to_depth_scale
        return x, y, z

    def xy2lonlat(self, utm_easting, utm_northing):
        # convert UTM in meter to latitude and longitude in degree
        assert(self.utm_crs is not None), "Please compute the UTM CRS first."
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs, always_xy=True)  
        longitude, latitude = transformer.transform(utm_easting, utm_northing, direction='INVERSE')
        return longitude, latitude
    
    def xyz2lonlatele(self, utm_easting, utm_northing, utm_depth):
        # convert UTM in meter to latitude and longitude in degree
        # convert depth in meter to elevation in meter
        assert(self.utm_crs is not None), "Please compute the UTM CRS first."
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs, always_xy=True)  
        longitude, latitude = transformer.transform(utm_easting, utm_northing, direction='INVERSE')
        elevation = utm_depth / self.elevation_to_depth_scale
        return longitude, latitude, elevation
