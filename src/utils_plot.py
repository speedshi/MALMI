#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:26:55 2021

Plot related functions.

@author: shipe
"""


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import obspy
import pandas as pd
from utils_dataprocess import dnormlz


def events_magcum(time, ydata, bins_dt=1, yname='Magnitude', fname='./event_magnitude_cumulative_number.png', ydata_thrd=4.5):
    """
    
    Parameters
    ----------
    time : nparray, shape (n_events,1)
        the origin times of events, datetime format.
    ydata : nparray, shape (n_events,1)
        an attribute of events for plotting on the Y-axis, e.g. magnitude.
    bins_dt : float
        bin segments (time segment) for cumulative plot, in days.
    yname : str
        the title for left Y-axis.
    fname : str
        output filename.
    ydata_thrd : float
        a threshold of ydata related to plotting;    

    Returns
    -------
    None.

    """
    
    
    fig =  plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(111) 
    
    time = mdates.date2num(time)
    size = dnormlz(ydata, 6, 120)
    
    # plot detection property vs detection time
    ax1.scatter(time, ydata, size, c='r', marker='o', alpha=0.5, linewidths=0)  # c=depth, cmap='Reds_r', vmin=0, vmax=10
    ax1.xaxis_date()
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_ylabel(yname,color='k',fontsize=14)
    ax1.autoscale(enable=True, axis='x', tight=True)
    
    # calculate accumulated numbers
    time_min = np.floor(np.amin(time))  # the earliest time
    time_max = np.ceil(np.amax(time))  # the lastest time
    
    bins = np.arange(time_min,time_max+bins_dt,bins_dt)  # the edges of bins for accumulated plot
    events_hist, bin_edges = np.histogram(time,bins=bins)  # number of events in each bin
    events_cum = events_hist.cumsum()  # accumulated number
    
    # plot accumulated number of detections
    ax2 = ax1.twinx()
    ax2.plot(bin_edges[1:],events_cum,c='b',lw=2.2,alpha=0.8,zorder=1)  # accumulated event before a date (not include)
    # ax2.plot(bin_edges[1:],events_hist,c='g',lw=1.5,zorder=1)  # plot the number of events per bin/time-period
    ax2.set_ylabel('Cumulative num.',color='b',fontsize=14)
    ax2.tick_params(axis='y', colors='b')
    
    # plot events larger than a certain threshold on the cumulative curve
    eidx = (np.array(ydata) >= ydata_thrd)
    chse_t = time[eidx]
    for ee in chse_t:
        res = next(x for x, val in enumerate(bin_edges) if val > ee)
        # ax2.plot(ee, events_cum[res], marker='*', mew=0, mfc='lime', ms=9)
        ax2.plot(bin_edges[0:][res-1], events_cum[res-2], marker='*', mew=0, mfc='lime', ms=9)
    
    # output figure
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    return


def probin_plot(dir_input, dir_output, figsize, normv=None, ppower=None, tag=None, staname=None, arrvtt=None):
    """
    To plot the input probability data of different stations.

    Parameters
    ----------
    dir_input : str
        path to the input data set, input data should be stored in a format that
        obspy can read.
    dir_output : str
        prth to the output figure.
    figsize : tuple
        specify the output figure size, e.g (12, 12).
    normv : float, default: None
        a threshold value, if trace data large than this threshold, then normalize
        this trace let maximum to be 1. None for no mormalize.
    ppower : float, default: None
        compute array element wise power over the input phase probabilities.
    tag : char, default: None
        a filename tage for output figures;
    staname : list of str, default: None
        specify the stations to show;
    arrvtt : dic, default: None
        the arrivaltimes of P- and S-waves at different stations.
        arrvtt['station']['P'] : P-wave arrivaltime;
        arrvtt['station']['S'] : S-wave arrivaltime.

    Returns
    -------
    None.

    """
    
    # set internal parameters
    dyy = 1.2  # the y-axis interval between different station data when plotting
    
    # load data set
    file_seismicin = sorted([fname for fname in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, fname))])
    stream = obspy.Stream()
    for indx, dfile in enumerate(file_seismicin):
        stream += obspy.read(os.path.join(dir_input, dfile))
    
    # get the date info of data set
    this_date = stream[0].stats.starttime.date
    
    # get station names
    if staname is None:
        staname = []
        for tr in stream:
            if tr.stats.station not in staname:
                staname.append(tr.stats.station)
    
    # plot data for each station
    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(111)
    ydev = [ii*dyy for ii in range(len(staname))]
    for ii in range(len(staname)):
        # plot P-phase probability
        tr = stream.select(station=staname[ii], component="P")
        if tr.count() > 0:
            tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
            if (normv is not None) and (max(abs(tr[0].data)) >= normv):
                vdata = tr[0].data / max(abs(tr[0].data))
            else:
                vdata = tr[0].data
            if ppower:
                vdata = vdata**ppower
            ax.plot(tt, vdata+ydev[ii], 'r', linewidth=1.2)
            del vdata, tt
        del tr
        
        # plot S-phase probability
        tr = stream.select(station=staname[ii], component="S")
        if tr.count() > 0:
            tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
            if (normv is not None) and (max(abs(tr[0].data)) >= normv):
                vdata = tr[0].data / max(abs(tr[0].data))
            else:
                vdata = tr[0].data
            if ppower:
                vdata = vdata**ppower
            ax.plot(tt, vdata+ydev[ii], 'b', linewidth=1.2)
            del vdata, tt
        del tr
        
        # plot event probability
        tr = stream.select(station=staname[ii], component="D")
        if tr.count() > 0:
            tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
            if (normv is not None) and (max(abs(tr[0].data)) >= normv):
                vdata = tr[0].data / max(abs(tr[0].data))
            else:
                vdata = tr[0].data
            ax.plot(tt, vdata+ydev[ii], 'k--', linewidth=1.2)
            del vdata, tt
        del tr
        
        # plot phase arrivaltimes
        if arrvtt is not None:
            if staname[ii] in arrvtt:
                if 'P' in arrvtt[staname[ii]]:
                    # plot P arrivaltimes
                    ax.vlines(arrvtt[staname[ii]]['P'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
                if 'S' in arrvtt[staname[ii]]:
                    # plot S arrivaltimes
                    ax.vlines(arrvtt[staname[ii]]['S'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
        
    ax.set_yticks(ydev)
    ax.set_yticklabels(staname, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('Input data [{}]'.format(this_date), fontsize=16, fontweight ="bold")
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if tag:
        fname = os.path.join(dir_output, 'input_data_{}.png'.format(tag))
    else:
        fname = os.path.join(dir_output, 'input_data.png')
    fig.savefig(fname, bbox_inches='tight', dpi=600)
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    del stream
    
    return


def seisin_plot(dir_input, dir_output, figsize, comp=['Z','N','E'], dyy=1.8, fband=None, tag=None, staname=None, arrvtt=None):
    """
    To plot the input seismic data of different stations.

    Parameters
    ----------
    dir_input : str
        path to the input data set, input data should be stored in a format that
        obspy can read.
    dir_output : str
        prth to the output figure.
    figsize : tuple
        specify the output figure size, e.g (12, 12).
    comp : list of str, default: ['Z','N','E']
        specify to plot which component, each component is plot in one figure.
    dyy : float, default: 1.8
        the y-axis interval between different station data when plotting.
    fband : list of float, default: None
        the frequency band for filtering input seismic data before plotting,
        defult value None means no filtering is applied.
    tag : char, default: None
        a filename tage for output figures;
    staname : list of str, default: None
        specify the stations to show;
    arrvtt : dic, default: None
        the arrivaltimes of P- and S-waves at different stations.
        arrvtt['station']['P'] : P-wave arrivaltime;
        arrvtt['station']['S'] : S-wave arrivaltime.

    Returns
    -------
    Figures in output directory.

    """
    
    # load data set
    file_seismicin = sorted([fname for fname in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, fname))])
    stream = obspy.Stream()
    for indx, dfile in enumerate(file_seismicin):
        stream += obspy.read(os.path.join(dir_input, dfile))
    
    # get the date info of data set
    this_date = stream[0].stats.starttime.date
    
    # get station names
    if staname is None:
        staname = []
        for tr in stream:
            if tr.stats.station not in staname:
                staname.append(tr.stats.station)
    
    ydev = [ii*dyy for ii in range(len(staname))]  # set y-axis ticks
    
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    if fband:
        stream.detrend('simple')
        stream.filter('bandpass', freqmin=fband[0], freqmax=fband[1])
        stream.taper(max_percentage=0.05, max_length=2)  # to avoid anormaly at bounday
    
    for icomp in comp:
        # plot data of all stations for each component
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111)
        
        for ii in range(len(staname)):
            # plot input seismic data of one component
            tr = stream.select(station=staname[ii], component=icomp)
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (max(abs(tr[0].data)) > 0):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                ax.plot(tt, vdata+ydev[ii], 'k', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot phase arrivaltimes
            if arrvtt is not None:
                if staname[ii] in arrvtt:
                    if 'P' in arrvtt[staname[ii]]:
                        # plot P arrivaltimes
                        ax.vlines(arrvtt[staname[ii]]['P'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
                    if 'S' in arrvtt[staname[ii]]:
                        # plot S arrivaltimes
                        ax.vlines(arrvtt[staname[ii]]['S'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
        
        ax.set_yticks(ydev)
        ax.set_yticklabels(staname, fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_title('Input data [{}]'.format(this_date), fontsize=16, fontweight ="bold")
        if tag:
            fname = os.path.join(dir_output, 'input_data_{}_{}.png'.format(icomp, tag))
        else:
            fname = os.path.join(dir_output, 'input_data_{}.png'.format(icomp))
        fig.savefig(fname, bbox_inches='tight')
        plt.cla()
        fig.clear()
        plt.close(fig)
    
    del stream
    
    return


def seischar_plot(dir_seis, dir_char, dir_output, figsize, comp=['Z','N','E'], dyy=1.8, fband=None, normv=None, ppower=None, tag=None, staname=None, arrvtt=None):
    """
    To plot the input seismic data of different stations with the characteristic 
    functions overlayed on the seismogram.

    Parameters
    ----------
    dir_seis : str
        path to the input seismic data set, input data should be stored in a 
        format that obspy can read.
    dir_char : str
        path to the input characteristic functions, input data should be stored 
        in a format that obspy can read.
    dir_output : str
        prth to the output figure.
    figsize : tuple
        specify the output figure size, e.g (12, 12).
    comp : list of str, default: ['Z','N','E']
        specify to plot which component, each component is plot in one figure.
    dyy : float, default: 1.8
        the y-axis interval between different station data when plotting.
    fband : list of float, default: None
        the frequency band for filtering input seismic data before plotting,
        defult value None means no filtering is applied.
    normv : float, default: None
        a threshold value, if trace data large than this threshold, then normalize
        the characteristi function let maximum to be 1. None for no mormalize.
    ppower : float, default: None
        compute array element wise power over the input characteristi function.
    tag : char, default: None
        a filename tage for output figures;
    staname : list of str, default: None
        specify the stations to show;
    arrvtt : dic, default: None
        the arrivaltimes of P- and S-waves at different stations.
        arrvtt['station']['P'] : P-wave arrivaltime;
        arrvtt['station']['S'] : S-wave arrivaltime.

    Returns
    -------
    Figures in output directory.

    """
    
    # load seismic data set
    file_seismicin = sorted([fname for fname in os.listdir(dir_seis) if os.path.isfile(os.path.join(dir_seis, fname))])
    stream = obspy.Stream()
    for indx, dfile in enumerate(file_seismicin):
        stream += obspy.read(os.path.join(dir_seis, dfile))
    
    # load characteristic function data set
    file_charfin = sorted([fname for fname in os.listdir(dir_char) if os.path.isfile(os.path.join(dir_char, fname))])
    charfs = obspy.Stream()
    for indx, dfile in enumerate(file_charfin):
        charfs += obspy.read(os.path.join(dir_char, dfile))
    
    # get the date info of data set
    this_date = stream[0].stats.starttime.date
    
    # get station names
    if staname is None:
        staname = []
        for tr in stream:
            if tr.stats.station not in staname:
                staname.append(tr.stats.station)
    
    ydev = [ii*dyy for ii in range(len(staname))]  # set y-axis ticks
    
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    if fband:
        stream.detrend('simple')
        stream.filter('bandpass', freqmin=fband[0], freqmax=fband[1])
        stream.taper(max_percentage=0.05, max_length=2)  # to avoid anormaly at bounday
    
    for icomp in comp:
        # plot data of all stations for each component
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111)
        
        for ii in range(len(staname)):
            
            # plot input seismic data of one component
            tr = stream.select(station=staname[ii], component=icomp)
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (max(abs(tr[0].data)) > 0):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                ax.plot(tt, vdata+ydev[ii], 'k', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot P-phase characteristic function
            tr = charfs.select(station=staname[ii], component="P")
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (normv is not None) and (max(abs(tr[0].data)) >= normv):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                if ppower:
                    vdata = vdata**ppower
                ax.plot(tt, vdata+ydev[ii], 'r', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot S-phase characteristic function
            tr = charfs.select(station=staname[ii], component="S")
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (normv is not None) and (max(abs(tr[0].data)) >= normv):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                if ppower:
                    vdata = vdata**ppower
                ax.plot(tt, vdata+ydev[ii], 'b', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot phase arrivaltimes
            if arrvtt is not None:
                if staname[ii] in arrvtt:
                    if 'P' in arrvtt[staname[ii]]:
                        # plot P arrivaltimes
                        ax.vlines(arrvtt[staname[ii]]['P'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
                    if 'S' in arrvtt[staname[ii]]:
                        # plot S arrivaltimes
                        ax.vlines(arrvtt[staname[ii]]['S'], ydev[ii], ydev[ii]+0.95, colors='lime', linewidth=0.9, alpha=0.95, zorder=3)
        
        ax.set_yticks(ydev)
        ax.set_yticklabels(staname, fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_title('Input data [{}]'.format(this_date), fontsize=16, fontweight ="bold")
        if tag:
            fname = os.path.join(dir_output, 'input_data_with_cf_{}_{}.png'.format(icomp, tag))
        else:
            fname = os.path.join(dir_output, 'input_data_with_cf_{}.png'.format(icomp))
        fig.savefig(fname, bbox_inches='tight')
        plt.cla()
        fig.clear()
        plt.close(fig)
    
    del stream
    
    return


def migmatrix_plot(file_corrmatrix, dir_tt, hdr_filename='header.hdr', colormap='RdBu_r', dir_output=None):
    """
    This function is to visualize the migration volume, plot profiles along 
    X-, Y- and Z-directions, and display the isosurface along contour-value of 
    migration volume.

    Parameters
    ----------
    file_corrmatrix : str
        filename including path of the migration vloume.
    dir_tt : str
        directory to the header file of traveltime date set.
    hdr_filename : str, optional
        filename of the header file of traveltime date set. Header file is used
        to get the coordinate of the migration area. The default is 'header.hdr'.
    colormap : str, optional
        colormap name used for plotting. The default is 'RdBu_r'.
    dir_output : str, optional
        output directory for the generated figures. The default is None.

    Returns
    -------
    None.

    """
    
    from loki import traveltimes
    from mayavi import mlab
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d import Axes3D
    
    # set default output directory
    if dir_output is None:
        dir_output = ''
        for ifd in file_corrmatrix.split('/')[:-1]:
            dir_output = os.path.join(dir_output, ifd)
    
    cmap = plt.cm.get_cmap(colormap)  # set colormap
    
    # load migration volume
    corrmatrix = np.load(file_corrmatrix)
    
    # load migration area coordinate info
    tobj = traveltimes.Traveltimes(dir_tt, hdr_filename)
    
    # obtain the maximum projection along one dimension: XY
    nx, ny, nz = np.shape(corrmatrix)
    CXY = np.zeros([ny, nx])
    for i in range(ny):
        for j in range(nx):
            CXY[i,j]=np.max(corrmatrix[j,i,:])
    
    fig, ax = plt.subplots(dpi=600, subplot_kw={"projection": "3d"})  
    XX, YY = np.meshgrid(tobj.x, tobj.y)      
    surf = ax.plot_surface(XX, YY, CXY, rcount=200, ccount=200, cmap=cmap,
                            linewidth=0, edgecolor='none')  # , antialiased=False, alpha=0.9
    #cset = ax.contourf(XX, YY, CXY, zdir='z', offset=-1, cmap=cm.coolwarm) 
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.08, label='Coherence')
    ax.set_xlim(np.min(tobj.x), np.max(tobj.x))
    ax.set_ylim(np.min(tobj.y), np.max(tobj.y))
    ax.set_zlim(0, np.max(CXY))
    ax.set_xlabel('X (Km)')
    ax.set_ylabel('Y (Km)')
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title('Coherence matrix X-Y', fontsize=14, fontweight='bold') 
    fname = os.path.join(dir_output, 'coherence_matrix_xy_surf.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    fig = plt.figure(dpi=600)
    fig.suptitle('Coherence matrix X-Y', fontsize=14, fontweight='bold')
    ax = fig.gca()
    cs = plt.contourf(tobj.x, tobj.y, CXY, 20, cmap=cmap, interpolation='bilinear')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    cbar = plt.colorbar(cs)
    ax.set_aspect('equal')
    fname = os.path.join(dir_output, 'coherence_matrix_xy.png')
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    # obtain the maximum projection along one dimension: XZ
    CXZ = np.zeros([nz, nx])
    for i in range(nz):
        for j in range(nx):
            CXZ[i, j] = np.max(corrmatrix[j,:,i])

    fig, ax = plt.subplots(dpi=600, subplot_kw={"projection": "3d"})  
    XX, ZZ = np.meshgrid(tobj.x, tobj.z)      
    surf = ax.plot_surface(XX, ZZ, CXZ, rcount=200, ccount=200, cmap=cmap,
                            linewidth=0, edgecolor='none')  # , antialiased=False, alpha=0.9
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.08, label='Coherence')
    ax.set_xlim(np.min(tobj.x), np.max(tobj.x))
    ax.set_ylim(np.min(tobj.z), np.max(tobj.z))
    ax.set_zlim(0, np.max(CXZ))
    ax.set_xlabel('X (Km)')
    ax.set_ylabel('Z (Km)')
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title('Coherence matrix X-Z', fontsize=14, fontweight='bold') 
    # zasp = ((np.max(tobj.z)-np.min(tobj.z)))/ (np.max(tobj.x)-np.min(tobj.x)) 
    # ax.set_box_aspect((1, zasp, 1))
    fname = os.path.join(dir_output, 'coherence_matrix_xz_surf.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    fig = plt.figure(dpi=600)
    fig.suptitle('Coherence matrix X-Z', fontsize=14, fontweight='bold')
    ax = fig.gca()
    cs = plt.contourf(tobj.x, tobj.z, CXZ, 20, cmap=cmap, interpolation='bilinear')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    cbar = plt.colorbar(cs)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    fname = os.path.join(dir_output, 'coherence_matrix_xz.png')
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    # obtain the maximum projection along one dimension: YZ
    CYZ = np.zeros([nz, ny])
    for i in range(nz):
        for j in range(ny):
            CYZ[i, j] = np.max(corrmatrix[:,j,i])
    
    fig, ax = plt.subplots(dpi=600, subplot_kw={"projection": "3d"})  
    YY, ZZ = np.meshgrid(tobj.y, tobj.z)      
    surf = ax.plot_surface(YY, ZZ, CYZ, rcount=200, ccount=200, cmap=cmap,
                            linewidth=0, edgecolor='none')  # , antialiased=False, alpha=0.9
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.08, label='Coherence')
    ax.set_xlim(np.min(tobj.y), np.max(tobj.y))
    ax.set_ylim(np.min(tobj.z), np.max(tobj.z))
    ax.set_zlim(0, np.max(CYZ))
    ax.set_xlabel('Y (Km)')
    ax.set_ylabel('Z (Km)')
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title('Coherence matrix Y-Z', fontsize=14, fontweight='bold') 
    fname = os.path.join(dir_output, 'coherence_matrix_yz_surf.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    fig = plt.figure(dpi=600)
    fig.suptitle('Coherence matrix Y-Z', fontsize=14, fontweight='bold')
    ax = fig.gca()
    cs = plt.contourf(tobj.y, tobj.z, CYZ, 20, cmap=cmap, interpolation='bilinear')
    ax.set_xlabel('Y (km)')
    ax.set_ylabel('Z (km)')
    cbar = plt.colorbar(cs)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    fname = os.path.join(dir_output, 'coherence_matrix_yz.png')
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    # plot 3D isosurface of migration volume
    mlab.options.offscreen = True
    cmax = np.max(corrmatrix)
    cmin = np.min(corrmatrix)
    fig = mlab.figure(size=(2000, 2000))
    src = mlab.pipeline.scalar_field(corrmatrix)
    mlab.outline()
    mlab.pipeline.iso_surface(src, contours=[cmin + 0.5*(cmax-cmin), ], opacity=0.3, figure=fig)
    mlab.pipeline.iso_surface(src, contours=[cmin + 0.8*(cmax-cmin), ], figure=fig)
    #mlab.pipeline.volume(src, vmin=0.3*cmax, vmax=0.8*cmax)
    # plane_orientation = 'z_axes'
    # cut = mlab.pipeline.scalar_cut_plane(src.children[0], plane_orientation=plane_orientation)
    # cut.enable_contours = True
    # cut.contour.number_of_contours = 1
    fname = os.path.join(dir_output, '3D_isosurf.png')
    mlab.orientation_axes()
    #mlab.show()
    mlab.savefig(fname, size=(2000,2000))
    
    # plot 3D isosurface of migration volume
    iso_val = cmin + 0.8*(cmax-cmin)
    verts, faces, _, _ = marching_cubes(corrmatrix, iso_val, spacing=(0.01, 0.01, 0.01))  # Use marching cubes to obtain the surface mesh
    fig = plt.figure(dpi=600, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], color='y', lw=1)
    fname = os.path.join(dir_output, '3D_isosurf2.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    return


def plot_basemap(region, sta_inv=None, mkregion=None, fname="./basemap.png", plot_stationname=True):
    """
    To plot the basemap with station and/or a regtangular area.
    
    Parameters
    ----------
    region : list of float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    sta_inv : obspy invertory format, optional
        station inventory containing station metadata. The default is None.
    mkregion : list of float, optional
        the lat/lon boundary of a marked region for highlighting, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree. The default is None, i.e.
        not plotting the highlighting area.
    fname : str, optional
        filename of the output figure. The default is "./basemap.png".
    plot_stationname : boolen, optional
        specify whether to plot the station names on the map. Default is yes.
        
    Returns
    -------
    None.

    """  
    
    import pygmt
           
    # plot and save map---------------------------------------------------------
    # load topography dataset
    grid = pygmt.datasets.load_earth_relief('03s', region=region, registration="gridline")
    fig = pygmt.Figure()
    fig.grdimage(region=region, projection="M15c", grid=grid, cmap="grayC", shading="l+d", dpi=600)  # plot topography
    fig.coast(region = region,  # Set the x-range and the y-range of the map  -23/-18/63.4/65
              projection="M15c",  # Set projection to Mercator, and the figure size to 15 cm
              water="skyblue",  # Set the color of the land t
              borders="1/0.5p",  # Display the national borders and set the pen thickness
              shorelines="1/0.5p",  # Display the shorelines and set the pen thickness
              frame="a",  # Set the frame to display annotations and gridlines
              #land="#666666",  # Set the color of the land
              #map_scale='g-22.6/63.14+c-22/64+w100k+f+u',  # map scale for local one
              )
    
    # plot stations
    if sta_inv:
        for net in sta_inv:
            for sta in net:
                fig.plot(x=sta.longitude, y=sta.latitude, style="t0.35c", color="blue", pen="0.35p,black")  
                if plot_stationname:
                    fig.text(text=sta.code, x=sta.longitude, y=sta.latitude, font='6p,Helvetica-Bold,black', justify='CT', D='0/-0.15c')
    
    # highlight a rectangular area
    if mkregion is not None:
        fig.plot(data=np.array([[mkregion[0], mkregion[2], mkregion[1], mkregion[3]]]), style='r+s', pen="1p,yellow")
    
    # save figure
    fig.savefig(fname, dpi=600)
    
    return


def plot_evmap_depth(region, eq_longi, eq_latit, eq_depth, depth_max=None, cmap="polar", sta_inv=None, mkregion=None, fname="./basemap.png", plot_stationname=False):
    """
    To plot the basemap with seismic events color-coded using event depth.
    
    Parameters
    ----------
    region : list of float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    eq_longi : list or numpy.array of float
        longitude of seismic events in degree.
    eq_latit : list or numpy.array of float
        latitude of seismic events in degree.
    eq_depth : list or numpy.array of float
        depth of seismic events in km.
    depth_max : float
        maximum depth in km for showing. Default is None, i.e. show all depths.    
    sta_inv : obspy invertory format, optional
        station inventory containing station metadata. The default is None.
    mkregion : list of float, optional
        the lat/lon boundary of a marked region for highlighting, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree. The default is None, i.e.
        not plotting the highlighting area.
    fname : str, optional
        filename of the output figure. The default is "./basemap.png".
    plot_stationname : boolen, optional
        specify whether to plot the station names on the map. Default is yes.
        
    Returns
    -------
    None.

    """  
    
    import pygmt

    fig = pygmt.Figure()
    fig.coast(region = region,  # Set the x-range and the y-range of the map  -23/-18/63.4/65
              projection="M15c",  # Set projection to Mercator, and the figure size to 15 cm
              water="skyblue",  # Set the color of the land t
              borders="1/0.5p",  # Display the national borders and set the pen thickness
              shorelines="1/0.5p",  # Display the shorelines and set the pen thickness
              frame="a",  # Set the frame to display annotations and gridlines
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
        fig.plot(data=np.array([[mkregion[0], mkregion[2], mkregion[1], mkregion[3]]]), style='r+s', pen="1p,yellow")
    
    # plot events
    if depth_max:
        pygmt.makecpt(cmap=cmap, series=[eq_depth.min(), depth_max])
    else:
        pygmt.makecpt(cmap=cmap, series=[eq_depth.min(), eq_depth.max()])
    fig.plot(eq_longi, eq_latit, color=eq_depth, cmap=True, style='c0.2c', pen='0.4p,black')  # , transparency=30
    fig.colorbar(frame='af+l"Depth (km)"')
    
    # show how many events in total
    fig.text(text='{} events'.format(len(eq_longi)), position='BR', font='14p,Helvetica-Bold,black', justify='BR', offset='-0.4/0.4')
    
    # save figure
    fig.savefig(fname, dpi=600)

    return



def plot_evmap_otime(region, eq_longi, eq_latit, eq_times, time_ref=None, cmap="polar", sta_inv=None, mkregion=None, fname="./basemap.png", plot_stationname=False):
    """
    To plot the basemap with seismic events color-coded using event origin time.
    
    Parameters
    ----------
    region : list float
        the lat/lon boundary of plotting region, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree.
    eq_longi : list or numpy.array of float
        longitude of seismic events in degree.
    eq_latit : list or numpy.array of float
        latitude of seismic events in degree.
    eq_times : numpy.array of datetime
        origin times of seismic events in datetime format.
    time_ref : datetime
        Reference time for calculate time difference. Default is None, 
        i.e. maximum origin time of the input event.    
    sta_inv : obspy invertory format, optional
        station inventory containing station metadata. The default is None.
    mkregion : list of float, optional
        the lat/lon boundary of a marked region for highlighting, in format of 
        [lon_min, lon_max, lat_min, lat_max] in degree. The default is None, i.e.
        not plotting the highlighting area.
    fname : str, optional
        filename of the output figure. The default is "./basemap.png".
    plot_stationname : boolen, optional
        specify whether to plot the station names on the map. Default is yes.
        
    Returns
    -------
    None.

    """  
    
    import pygmt

    fig = pygmt.Figure()
    fig.coast(region = region,  # Set the x-range and the y-range of the map  -23/-18/63.4/65
              projection="M15c",  # Set projection to Mercator, and the figure size to 15 cm
              water="skyblue",  # Set the color of the land t
              borders="1/0.5p",  # Display the national borders and set the pen thickness
              shorelines="1/0.5p",  # Display the shorelines and set the pen thickness
              frame="a",  # Set the frame to display annotations and gridlines
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
        fig.plot(data=np.array([[mkregion[0], mkregion[2], mkregion[1], mkregion[3]]]), style='r+s', pen="1p,yellow")
    
    # plot events
    if not time_ref:
        # set the default reference time to be the 
        time_ref = max(eq_times)
    eq_tref = mdates.date2num(eq_times) - mdates.date2num(time_ref)
    pygmt.makecpt(cmap=cmap, series=[eq_tref.min(), eq_tref.max()])
    fig.plot(eq_longi, eq_latit, color=eq_tref, cmap=True, style='c0.2c', pen='0.4p,black')  # , transparency=30
    fig.colorbar(frame='af+l"Days relative to {}"'.format(time_ref))
    
    # show how many events in total
    fig.text(text='{} events'.format(len(eq_longi)), position='BR', font='14p,Helvetica-Bold,black', justify='BR', offset='-0.4/0.4')
    
    # save figure
    fig.savefig(fname, dpi=600)

    return


def catlogmatch_pieplot(catalog_mt, dir_fig='.'):
    """
    To plot the pie figure after comparing two catalogs.

    Parameters
    ----------
    catalog_mt : dic
        a comparison catalog, for detail see 'utils_dataprocess.catalog_match'.
        catalog_mt['status'] contains the comparison results.
    dir_fig : str, optional
        dirctory for saving fiugre. The default is '.'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # plot pie chart
    N_matched = sum(catalog_mt['status'] == 'matched')  # total number of matched events
    N_new = sum(catalog_mt['status'] == 'new')  # total number of new events
    N_undetected = sum(catalog_mt['status'] == 'undetected')  # total number of undetected/missed events
    
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n{:d}".format(pct, absolute)
    
    fig = plt.figure(dpi=600)
    ax = fig.add_axes([0,0,1,1])
    labels = ['Matched', 'New', 'Undetected']
    sizes = [N_matched, N_new, N_undetected]
    explode = (0.05, 0.05, 0.05)  # whether "explode" any slice 
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=lambda pct: func(pct, sizes), 
           pctdistance=0.85, shadow=False, startangle=90)
    
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # output figure
    ax.axis('equal')
    fname = os.path.join(dir_fig, 'catalog_statistical_pie.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    return


def catalogcomp_barplot(catalog, catalog_ref, bins_dt=1, figsize=(6,6), dir_fig='.'):
    """
    Plot bar char of events per time slot for comparing two catalogs.

    Parameters
    ----------
    catalog : dict
        Input catalog.
        catalog['time'] : origin times of each event;
    catalog_ref : dic
        Reference atalog for comparing.
        catalog_ref['time'] : origin times of each event;
    bins_dt : float, optional
        time interval in days for generating time slots. The default is 1, 
        e.g. count event number per day.
    figsize : tuple, optional
        the figure size. The default is (6,6).
    dir_fig : str, optional
        dirctory for saving figure. The default is '.'.

    Returns
    -------
    None.

    """
    
    import matplotlib.ticker as ticker
    
    evtimes = mdates.date2num(catalog['time'])
    evtimes_ref = mdates.date2num(catalog_ref['time'])
    
    # calculate accumulated numbers
    time_min = np.floor(min(evtimes.min(), evtimes_ref.min()))  # the earliest time
    time_max = np.ceil(max(evtimes.max(),evtimes_ref.max()))  # the lastest time
    
    bins = np.arange(time_min, time_max+bins_dt, bins_dt)  # the edges of bins for accumulated plot
    events_hist, bin_edges = np.histogram(evtimes, bins=bins)  # number of events in each bin
    events_hist_ref, bin_edges = np.histogram(evtimes_ref, bins=bins)  # number of events in each bin
    
    fig = plt.figure(figsize=figsize, dpi=600)
    ax1 = fig.add_subplot(111) 
    ax1.hist(evtimes, bin_edges, rwidth=1.0, color='blue', label='New catalog', edgecolor='black',  linewidth=0.2, alpha=0.8)
    ax1.hist(evtimes_ref, bin_edges, rwidth=0.55, color='gray', label='Ref. catalog', edgecolor='black',  linewidth=0.1, alpha=0.7)
    ax1.legend(loc='upper left')
    ax1.xaxis_date()
    ax1.set_xlabel('Time', color='k', fontsize=14)
    ax1.set_ylabel('# Events', color='k', fontsize=14)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10)) # forced the horizontal major ticks to appear by steps of x units
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # forced the horizontal minor ticks to appear by steps of 1 units

    fname = os.path.join(dir_fig, 'catalog_compare_bar.png')
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    return

