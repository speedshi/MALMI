#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:48:10 2021

@author: shipe
"""


import numpy as np
import matplotlib.dates as mdates
import pygmt
from xcatalog import catalog_select


def catalog_plot_depth(region, catalog, depthrg=None, cmap="hot", sta_inv=None, mkregion=None, fname="./basemap.png", plot_stationname=False, eq_size=0.17, markers=None):
    """
    To plot the basemap with seismic events color-coded using event depth.
    
    Parameters
    ----------
    region : list of float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    catalog: dict,
        containing catalog event information;
        catalog['longitude'] : list or numpy.array of float
            longitude of seismic events in degree.
        catalog['latitude'] : list or numpy.array of float
            latitude of seismic events in degree.
        catalog['depth_km'] : list or numpy.array of float
            depth of seismic events in km.
    depthrg : float or list of float
        when input is a float or list with only 1 entry, it specify the maximum depth in km for showing. 
        when input is a list of two entries, it specify the depth range in km for showing.
        Default is None, i.e. show all depths.    
    cmap : string, name of the colormap,
        such as "polar", "plasma", "hot", "viridis";
    sta_inv : obspy invertory format, optional
        station inventory containing station metadata. The default is None.
    mkregion : list of float or list of lists, optional
        if list of float:
            the lat/lon boundary of a marked region for highlighting, in format of 
            [lon_min, lon_max, lat_min, lat_max] in degree. 
        if list of lists:
            plot several marked regions, [[lon_min, lon_max, lat_min, lat_max], ...]
        The default is None, i.e. not plotting the highlighting area.
    fname : str, optional
        filename of the output figure. The default is "./basemap.png".
    plot_stationname : boolen, optional
        specify whether to plot the station names on the map. Default is yes.
    eq_size : list of float or float
        the size of the plotted seismic events. If input is a float, then plot 
        the events using the same size; if input is a list of float (must be in
        the same size as events), then plot events in different sizes.
        The default is to plot with the same size of 0.17.
    markers : dict, for plotting additional markers;
        markers['latitude'] : list of float, latitude in degree of markers;
        markers['longitude'] : list of float, longitude in degree of markers;
        markers['shape'] : list of str or str, specify the shape of markers, 
                           length of 1 or same as markers, e.g. 'c' or ['c', 't', 's', ...]
        markers['size'] : list of float or float, specify the size of markers, 
                          same length as markers['shape'], e.g. 0.3 or [0.3, 0.4, 0.2, ...]
        markers['color'] : list of str or str, specify the colors of markers, 
                           same length as markers['shape'], e.g. 'black' or ['black', 'white', 'red', ...]
                           if None, not filling with colors;
        markers['pen'] : list of str or str, specify the pen for plotting the markers,
                         same length as markers['shape'], e.g. "0.7p,black" or ["0.7p,black", "0.5p,red", "0.3p,blue", ...]
        default is None, not plotting.
        
    Returns
    -------
    None.

    """  
    
    fig = pygmt.Figure()
    
    with pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="fancy", FONT_ANNOT_PRIMARY='16p,Helvetica-Bold,black', FONT_LABEL='16p,Helvetica-Bold,black'):
        fig.coast(region = region,  # Set the x-range and the y-range of the map  -23/-18/63.4/65
                  projection="M15c",  # Set projection to Mercator, and the figure size to 15 cm
                  water="skyblue",  # Set the color of the land t
                  borders="1/0.5p",  # Display the national borders and set the pen thickness
                  shorelines="1/0.5p",  # Display the shorelines and set the pen thickness
                  frame=["xa0.2", "ya0.1"],  # Set the frame to display annotations and gridlines
                  land="gray",  # Set the color of the land
                  #map_scale='g-22.6/63.14+c-22/64+w100k+f+u',  # map scale for local one
                  )
        
        # plot stations
        for net in sta_inv:
            for sta in net:
                fig.plot(x=sta.longitude, y=sta.latitude, style="t0.35c", color="black", pen="0.35p,black") 
                if plot_stationname:
                    fig.text(text=sta.code, x=sta.longitude, y=sta.latitude, font='6p,Helvetica-Bold,black', justify='CT', D='0/-0.15c')
        
        # highlight a rectangular area on the map
        if mkregion is not None:
            if isinstance(mkregion[0], float):
                # plot one marked rectangular region
                fig.plot(data=np.array([[mkregion[0], mkregion[2], mkregion[1], mkregion[3]]]), style='r+s', pen="1p,yellow")
            elif isinstance(mkregion[0], list):
                # plot several marked rectangular region
                for imkrg in mkregion:
                    fig.plot(data=np.array([[imkrg[0], imkrg[2], imkrg[1], imkrg[3]]]), style='r+s', pen="1p,black")
                    
        # plot events
        if depthrg is not None:
            if isinstance(depthrg, float):
                pygmt.makecpt(cmap=cmap, series=[catalog['depth_km'].min(), depthrg])
            elif isinstance(depthrg, list) and (len(depthrg)==1):
                pygmt.makecpt(cmap=cmap, series=[catalog['depth_km'].min(), depthrg[0]])
            elif isinstance(depthrg, list) and (len(depthrg)==2):
                pygmt.makecpt(cmap=cmap, series=[depthrg[0], depthrg[1]])
            else:
                raise ValueError('Input depthrg not recognized!')
        else:
            pygmt.makecpt(cmap=cmap, series=[catalog['depth_km'].min(), catalog['depth_km'].max()])
        if isinstance(eq_size, float):
            fig.plot(x=catalog['longitude'], y=catalog['latitude'], color=catalog['depth_km'], cmap=True, style="c{}c".format(eq_size), pen="0.02p,black", transparency=20)  # , no_clip="r"
        else:
            fig.plot(x=catalog['longitude'], y=catalog['latitude'], size=eq_size, color=catalog['depth_km'], cmap=True, style="cc", pen="0.02p,black", transparency=20)  # , no_clip="r"
        fig.colorbar(frame='af+l"Depth (km)"')  # frame='a2f+l"Depth (km)"', position="JMR"
        
        # plot the markers
        if markers is not None:
            if isinstance(markers['shape'], str):
                # plot markers with the same size, shape and color
                if markers['color'] is not None:
                    fig.plot(x=markers['longitude'], y=markers['latitude'], 
                             style="{}{}c".format(markers['shape'], markers['size']), 
                             color="{}".format(markers['color']), pen=markers['pen'], transparency=10) 
                else:
                    fig.plot(x=markers['longitude'], y=markers['latitude'], 
                             style="{}{}c".format(markers['shape'], markers['size']), 
                             pen=markers['pen'], transparency=10) 
            else:
                # plot marker with different size, shape and color
                for iim in range(len(markers['shape'])):
                    if markers['color'][iim] is not None:
                        fig.plot(x=markers['longitude'][iim], y=markers['latitude'][iim], 
                             style="{}{}c".format(markers['shape'][iim], markers['size'][iim]), 
                             color="{}".format(markers['color'][iim]), pen=markers['pen'][iim], transparency=10)
                    else:
                        fig.plot(x=markers['longitude'][iim], y=markers['latitude'][iim], 
                             style="{}{}c".format(markers['shape'][iim], markers['size'][iim]), 
                             pen=markers['pen'][iim], transparency=10)
        
        # show how many events in total
        fig.text(text='{} events'.format(len(catalog['longitude'])), position='BR', font='16p,Helvetica-Bold,black', justify='BR', offset='-0.4/0.4')
        
        # save figure
        fig.savefig(fname, dpi=600)

    return


def catalog_plot_otime(region, catalog, time_ref=None, cmap="hot", sta_inv=None, mkregion=None, fname="./basemap.png", plot_stationname=False, eq_size=0.17, markers=None):
    """
    To plot the basemap with seismic events color-coded using event origin time.
    
    Parameters
    ----------
    region : list float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    catalog: dict,
        containing catalog event information;
        catalog['longitude'] : list or numpy.array of float
            longitude of seismic events in degree.
        catalog['latitude'] : list or numpy.array of float
            latitude of seismic events in degree.
        catalog['time'] : numpy.array of datetime
            origin times of seismic events in datetime format.
    time_ref : datetime
        Reference time for calculate time difference. Default is None, 
        i.e. maximum origin time of the input event.   
    cmap : string, name of the colormap,
        such as "polar", "plasma", "hot", "viridis";
    sta_inv : obspy invertory format, optional
        station inventory containing station metadata. The default is None.
    mkregion : list of float or list of lists, optional
        if list of float:
            the lat/lon boundary of a marked region for highlighting, in format of 
            [lon_min, lon_max, lat_min, lat_max] in degree. 
        if list of lists:
            plot several marked regions, [[lon_min, lon_max, lat_min, lat_max], ...]
        The default is None, i.e. not plotting the highlighting area.
    fname : str, optional
        filename of the output figure. The default is "./basemap.png".
    plot_stationname : boolen, optional
        specify whether to plot the station names on the map. Default is yes.
    eq_size : list of float or float
        the size of the plotted seismic events. If input is a float, then plot 
        the events using the same size; if input is a list of float (must be in
        the same size as eq_longi), then plot events in different sizes.
        The default is to plot with the same size of 0.17.
    markers : dict, for plotting additional markers;
        markers['latitude'] : list of float, latitude in degree of markers;
        markers['longitude'] : list of float, longitude in degree of markers;
        markers['shape'] : list of str or str, specify the shape of markers, 
                           length of 1 or same as markers, e.g. 'c' or ['c', 't', 's', ...]
        markers['size'] : list of float or float, specify the size of markers, 
                          same length as markers['shape'], e.g. 0.3 or [0.3, 0.4, 0.2, ...]
        markers['color'] : list of str or str, specify the colors of markers, 
                           same length as markers['shape'], e.g. 'black' or ['black', 'white', 'red', ...]
                           if None, not filling with colors;
        markers['pen'] : list of str or str, specify the pen for plotting the markers,
                         same length as markers['shape'], e.g. "0.7p,black" or ["0.7p,black", "0.5p,red", "0.3p,blue", ...]
        default is None, not plotting.
        
    Returns
    -------
    None.

    """  

    fig = pygmt.Figure()
    
    with pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="fancy", FONT_ANNOT_PRIMARY='16p,Helvetica-Bold,black', FONT_LABEL='16p,Helvetica-Bold,black'):
        fig.coast(region = region,  # Set the x-range and the y-range of the map  -23/-18/63.4/65
                  projection="M15c",  # Set projection to Mercator, and the figure size to 15 cm
                  water="skyblue",  # Set the color of the land t
                  borders="1/0.5p",  # Display the national borders and set the pen thickness
                  shorelines="1/0.5p",  # Display the shorelines and set the pen thickness
                  frame=["xa0.2", "ya0.1"],  # Set the frame to display annotations and gridlines
                  land="gray",  # Set the color of the land
                  #map_scale='g-22.6/63.14+c-22/64+w100k+f+u',  # map scale for local one
                  )
        
        # plot stations
        for net in sta_inv:
            for sta in net:
                fig.plot(x=sta.longitude, y=sta.latitude, style="t0.35c", color="black", pen="0.35p,black") 
                if plot_stationname:
                    fig.text(text=sta.code, x=sta.longitude, y=sta.latitude, font='6p,Helvetica-Bold,black', justify='CT', D='0/-0.15c')
        
        # highlight a rectangular area on the map
        if mkregion is not None:
            if isinstance(mkregion[0], float):
                # plot one marked rectangular region
                fig.plot(data=np.array([[mkregion[0], mkregion[2], mkregion[1], mkregion[3]]]), style='r+s', pen="1p,yellow")
            elif isinstance(mkregion[0], list):
                # plot several marked rectangular region
                for imkrg in mkregion:
                    fig.plot(data=np.array([[imkrg[0], imkrg[2], imkrg[1], imkrg[3]]]), style='r+s', pen="1p,black")
        
        # plot events
        if not time_ref:
            # set the default reference time to be the 
            time_ref = max(catalog['time'])
        eq_tref = mdates.date2num(catalog['time']) - mdates.date2num(time_ref)
        pygmt.makecpt(cmap=cmap, series=[eq_tref.min(), eq_tref.max()])
        if isinstance(eq_size, float):
            fig.plot(x=catalog['longitude'], y=catalog['latitude'], color=eq_tref, cmap=True, style="c{}c".format(eq_size), pen="0.01p,black")  # , no_clip="r", transparency=30, 
        else:
            fig.plot(x=catalog['longitude'], y=catalog['latitude'], size=eq_size, color=eq_tref, cmap=True, style="cc", transparency=10, pen="0.01p,black")  # , no_clip="r"
        fig.colorbar(frame='af+l"Days relative to {}"'.format(time_ref))
        
        # plot the markers
        if markers is not None:
            if isinstance(markers['shape'], str):
                # plot markers with the same size, shape and color
                if markers['color'] is not None:
                    fig.plot(x=markers['longitude'], y=markers['latitude'], 
                             style="{}{}c".format(markers['shape'], markers['size']), 
                             color="{}".format(markers['color']), pen=markers['pen'], transparency=10) 
                else:
                    fig.plot(x=markers['longitude'], y=markers['latitude'], 
                             style="{}{}c".format(markers['shape'], markers['size']), 
                             pen=markers['pen'], transparency=10) 
            else:
                # plot marker with different size, shape and color
                for iim in range(len(markers['shape'])):
                    if markers['color'][iim] is not None:
                        fig.plot(x=markers['longitude'][iim], y=markers['latitude'][iim], 
                             style="{}{}c".format(markers['shape'][iim], markers['size'][iim]), 
                             color="{}".format(markers['color'][iim]), pen=markers['pen'][iim], transparency=10)
                    else:
                        fig.plot(x=markers['longitude'][iim], y=markers['latitude'][iim], 
                             style="{}{}c".format(markers['shape'][iim], markers['size'][iim]), 
                             pen=markers['pen'][iim], transparency=10)
        
        # show how many events in total
        fig.text(text='{} events'.format(len(catalog['longitude'])), position='BR', font='16p,Helvetica-Bold,black', justify='BR', offset='-0.4/0.4')
        
        # save figure
        fig.savefig(fname, dpi=600)

    return


def catalog_plot_profile(catalog, pfregion, pfazimuth, depthrg, figsize=(18,6), fname='./event_profiles.pdf', pfspace=0):
    """
    To extract and plot event profiles.

    Parameters
    ----------
    catalog : dict
        input sesimic catalog, containing catalog event information;
        catalog['longitude'] : list or numpy.array of float
            longitude of seismic events in degree.
        catalog['latitude'] : list or numpy.array of float
            latitude of seismic events in degree.
        catalog['depth_km'] : list or numpy.array of float
            depth of seismic events in km.
    pfregion : list of list of 4 floats
        the regions to extract events and plot profiles, each list represents 
        a profile: [[lon_min, lon_max, lat_min, lat_max], ...], profiles are
        extracted relative to the center point of each region along an azimuth angle.
    pfazimuth : list of float
        azimuth angle in degree for extracing profiles. same length as 'pfregion'.
        in degrees clockwise from North.
    depthrg : list of float
        depth range in km for plotting.
    figsize : tuple of float, optional
        figure size in cm (width, hight). The default is (18,6).
    fname : str, optional
        output figure name. The default is './event_profiles.pdf'.
    pfspace : float, optional
        the space between different profiles in cm. 
        default is 0, meaning no spacing, profiles are plotted just next to
        each other like concatenate together.

    Returns
    -------
    None.

    """
    
    NN = len(pfregion)  # total number of profiles to plot
    
    # loop over each region to extract corresponding seismic events and project to a profile
    xx = []
    xxmax = []
    xxmin = []
    yy = []
    for ii in range(NN):
        catalog_rg = catalog_select(catalog, thrd_lat=[pfregion[ii][2], pfregion[ii][3]], thrd_lon=[pfregion[ii][0], pfregion[ii][1]], thrd_depth=None)
        evdata =  np.concatenate((catalog_rg['longitude'][:,None], catalog_rg['latitude'][:,None], catalog_rg['depth_km'][:,None]), axis=1)
        evpjs = pygmt.project(data=evdata, 
                              center=[0.5*(pfregion[ii][0]+pfregion[ii][1]), 0.5*(pfregion[ii][2]+pfregion[ii][3])], 
                              azimuth=pfazimuth[ii], convention='pz', unit=True)
        xx.append(evpjs.iloc[:,0].to_numpy())
        xxmax.append(max(evpjs.iloc[:,0].to_numpy())+0.2)
        xxmin.append(min(evpjs.iloc[:,0].to_numpy())-0.2)
        yy.append(evpjs.iloc[:,1].to_numpy()) 
    
    xxlens = np.array(xxmax) - np.array(xxmin)
    
    fig = pygmt.Figure()
    with pygmt.config(MAP_TICK_LENGTH='-0.1c'):  # let axis tick inside the map
        for ii in range(NN):
            if ii==0:
                fig.plot(x=xx[ii], y=yy[ii], 
                         style="c0.08c", color="black", transparency=10, pen="0.01p,black",
                         projection="X{}c/-{}c".format(xxlens[ii]/xxlens.sum()*figsize[0], figsize[1]), 
                         frame=["WSrt", "xa1f0.2", "yaf"],
                         region=[xxmin[ii], xxmax[ii], depthrg[0], depthrg[1]]) #
            else:
                fig.shift_origin(xshift="w+{}c".format(pfspace))
                fig.plot(x=xx[ii], y=yy[ii], 
                         style="c0.08c", color="black", transparency=10, pen="0.01p,black",
                         projection="X{}c/-{}c".format(xxlens[ii]/xxlens.sum()*figsize[0], figsize[1]), 
                         frame=["wSrt", "xa1f0.2", "yaf"],
                         region=[xxmin[ii], xxmax[ii], depthrg[0], depthrg[1]])

    # save figure
    fig.savefig(fname, dpi=600)
    
    return
    

def catalog_plot_profile_2cat(catalog1, catalog2, pfregion, pfazimuth, depthrg, figsize=(18,6), fname='./event_profiles.pdf', pfspace=0):
    """
    To extract and plot event profiles from two catalogs.

    Parameters
    ----------
    catalog1 : dict
        the first input sesimic catalog, containing catalog event information;
        catalog1['longitude'] : list or numpy.array of float
            longitude of seismic events in degree.
        catalog1['latitude'] : list or numpy.array of float
            latitude of seismic events in degree.
        catalog1['depth_km'] : list or numpy.array of float
            depth of seismic events in km.
    catalog2 : dict
        the first input sesimic catalog, containing catalog event information;
        catalog2['longitude'] : list or numpy.array of float
            longitude of seismic events in degree.
        catalog2['latitude'] : list or numpy.array of float
            latitude of seismic events in degree.
        catalog2['depth_km'] : list or numpy.array of float
            depth of seismic events in km.
    pfregion : list of list of 4 floats
        the regions to extract events and plot profiles, each list represents 
        a profile: [[lon_min, lon_max, lat_min, lat_max], ...], profiles are
        extracted relative to the center point of each region along an azimuth angle.
    pfazimuth : list of float
        azimuth angle in degree for extracing profiles. same length as 'pfregion'.
        in degrees clockwise from North.
    depthrg : list of float
        depth range in km for plotting.
    figsize : tuple of float, optional
        figure size in cm (width, hight). The default is (18,6).
    fname : str, optional
        output figure name. The default is './event_profiles.pdf'.
    pfspace : float, optional
        the space between different profiles in cm. 
        default is 0, meaning no spacing, profiles are plotted just next to
        each other like concatenate together.

    Returns
    -------
    None.

    """
    
    NN = len(pfregion)  # total number of profiles to plot
    
    # loop over each region to extract corresponding seismic events and project to a profile
    xx1 = []
    xxmax = []
    xxmin = []
    yy1 = []
    xx2 = []
    yy2 = []
    for ii in range(NN):
        catalog_rg = catalog_select(catalog1, thrd_lat=[pfregion[ii][2], pfregion[ii][3]], thrd_lon=[pfregion[ii][0], pfregion[ii][1]], thrd_depth=None)
        evdata =  np.concatenate((catalog_rg['longitude'][:,None], catalog_rg['latitude'][:,None], catalog_rg['depth_km'][:,None]), axis=1)
        evpjs = pygmt.project(data=evdata, 
                              center=[0.5*(pfregion[ii][0]+pfregion[ii][1]), 0.5*(pfregion[ii][2]+pfregion[ii][3])], 
                              azimuth=pfazimuth[ii], convention='pz', unit=True)
        xx1.append(evpjs.iloc[:,0].to_numpy())
        xxmax.append(max(evpjs.iloc[:,0].to_numpy())+0.2)
        xxmin.append(min(evpjs.iloc[:,0].to_numpy())-0.2)
        yy1.append(evpjs.iloc[:,1].to_numpy()) 
        
        catalog_rg = catalog_select(catalog2, thrd_lat=[pfregion[ii][2], pfregion[ii][3]], thrd_lon=[pfregion[ii][0], pfregion[ii][1]], thrd_depth=None)
        evdata =  np.concatenate((catalog_rg['longitude'][:,None], catalog_rg['latitude'][:,None], catalog_rg['depth_km'][:,None]), axis=1)
        evpjs = pygmt.project(data=evdata, 
                              center=[0.5*(pfregion[ii][0]+pfregion[ii][1]), 0.5*(pfregion[ii][2]+pfregion[ii][3])], 
                              azimuth=pfazimuth[ii], convention='pz', unit=True)
        xx2.append(evpjs.iloc[:,0].to_numpy())
        yy2.append(evpjs.iloc[:,1].to_numpy()) 
    
    xxlens = np.array(xxmax) - np.array(xxmin)
    
    fig = pygmt.Figure()
    with pygmt.config(MAP_TICK_LENGTH='-0.1c', FONT_ANNOT_PRIMARY='12p,Helvetica-Bold,black', FONT_LABEL='12p,Helvetica-Bold,black'):  # let axis tick inside the map
        for ii in range(NN):
            if ii==0:
                fig.plot(x=xx1[ii], y=yy1[ii], 
                         style="c0.08c", color="black", transparency=10, pen="0.01p,black",
                         projection="X{}c/-{}c".format(xxlens[ii]/xxlens.sum()*figsize[0], figsize[1]), 
                         frame=["WSrt", "xa1f0.2", "yaf"],
                         region=[xxmin[ii], xxmax[ii], depthrg[0], depthrg[1]]) # plot the first catalog events
                
                fig.plot(x=xx2[ii], y=yy2[ii], style="c0.08c", color="red", transparency=30, pen="0.01p,black") # plot the second catalog events
                
            else:
                fig.shift_origin(xshift="w+{}c".format(pfspace))
                fig.plot(x=xx1[ii], y=yy1[ii], 
                         style="c0.08c", color="black", transparency=10, pen="0.01p,black",
                         projection="X{}c/-{}c".format(xxlens[ii]/xxlens.sum()*figsize[0], figsize[1]), 
                         frame=["wSrt", "xa1f0.2", "yaf"],
                         region=[xxmin[ii], xxmax[ii], depthrg[0], depthrg[1]]) # plot the first catalog events
                
                fig.plot(x=xx2[ii], y=yy2[ii], style="c0.08c", color="red", transparency=30, pen="0.01p,black") # plot the second catalog events

    # save figure
    fig.savefig(fname, dpi=600)
    
    return