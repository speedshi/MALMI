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


def probin_plot(dir_input, dir_output, figsize, normv=None, ppower=None, tag=None, staname=None):
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
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ydev = [ii*dyy for ii in range(len(staname))]
    for ii in range(len(staname)):
        # plot P-phase probability
        tr = stream.select(station=staname[ii], component="P")
        if tr.count() > 0:
            tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
            if (normv is not None) and (max(abs(tr[0].data)) > normv):
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
            if (normv is not None) and (max(abs(tr[0].data)) > normv):
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
            if (normv is not None) and (max(abs(tr[0].data)) > normv):
                vdata = tr[0].data / max(abs(tr[0].data))
            else:
                vdata = tr[0].data
            ax.plot(tt, vdata+ydev[ii], 'k--', linewidth=1.2)
            del vdata, tt
        del tr
        
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
    fig.savefig(fname, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    del stream
    
    return


def seisin_plot(dir_input, dir_output, figsize, comp=['Z','N','E'], dyy=1.8, fband=None, tag=None, staname=None):
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
                vdata = tr[0].data / max(abs(tr[0].data))
                ax.plot(tt, vdata+ydev[ii], 'k', linewidth=1.2)
                del vdata, tt
            del tr
        
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


def seischar_plot(dir_seis, dir_char, dir_output, figsize, comp=['Z','N','E'], dyy=1.8, fband=None, normv=None, ppower=None, tag=None, staname=None):
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
                vdata = tr[0].data / max(abs(tr[0].data))
                ax.plot(tt, vdata+ydev[ii], 'k', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot P-phase probability
            tr = charfs.select(station=staname[ii], component="P")
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (normv is not None) and (max(abs(tr[0].data)) > normv):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                if ppower:
                    vdata = vdata**ppower
                ax.plot(tt, vdata+ydev[ii], 'r', linewidth=1.2)
                del vdata, tt
            del tr
            
            # plot S-phase probability
            tr = charfs.select(station=staname[ii], component="S")
            if tr.count() > 0:
                tt = pd.date_range(tr[0].stats.starttime.datetime, tr[0].stats.endtime.datetime, tr[0].stats.npts)
                if (normv is not None) and (max(abs(tr[0].data)) > normv):
                    vdata = tr[0].data / max(abs(tr[0].data))
                else:
                    vdata = tr[0].data
                if ppower:
                    vdata = vdata**ppower
                ax.plot(tt, vdata+ydev[ii], 'b', linewidth=1.2)
                del vdata, tt
            del tr
        
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



