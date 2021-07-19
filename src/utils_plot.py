#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:26:55 2021

Plot related functions.

@author: shipe
"""


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import obspy
import pandas as pd


def events_magcum(time, magnitude, bins_dt=1, fname='./event_magnitude_cumulative_number.png', mgthd = 4.5):
    """
    
    Parameters
    ----------
    time : nparray, shape (n_events,1);
        the origin times of events, datetime format.
    data : nparray, shape (n_events,1);
        the magnitude of events.
    bins_dt: scalar;
        bin segments (time segment) for cumulative plot, in days.

    Returns
    -------
    None.

    """
    
    import mymod.maths as mymath
    
    fig =  plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(111) 
    
    time = mdates.date2num(time)
    size = mymath.dnormlz(magnitude, 6, 120)
    
    # plot detection property vs detection time
    ax1.scatter(time, magnitude, size, c='r', marker='o', alpha=0.5, linewidths=0)  # c=depth, cmap='Reds_r', vmin=0, vmax=10
    ax1.xaxis_date()
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_ylabel('Magnitude',color='k',fontsize=14)
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
    
    # plot events larger than a certain magnitude on the cumulative curve
    eidx = (np.array(magnitude) >= mgthd)
    chse_t = time[eidx]
    for ee in chse_t:
        res = next(x for x, val in enumerate(bin_edges) if val > ee)
        ax2.plot(ee, events_cum[res], marker='*', mew=0, mfc='lime', ms=9)
    
    # output figure
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    
    return


def probin_plot(dir_input, dir_output, figsize):
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

    Returns
    -------
    None.

    """
    
    # set internal parameters
    dyy = 1.0  # the y-axis interval between different station data when plotting
    
    # load data set
    file_seismicin = sorted([fname for fname in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, fname))])
    stream = obspy.Stream()
    for indx, dfile in enumerate(file_seismicin):
        stream += obspy.read(os.path.join(dir_input, dfile))
    
    # get station names
    staname = []
    for tr in stream:
        if tr.stats.station not in staname:
            staname.append(tr.stats.station)
    
    # plot data for each station
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    ydev = [ii*dyy for ii in range(len(staname))]
    for ii in range(len(staname)):
        tr = stream.select(station=staname[ii], component="P")[0]
        tt = pd.date_range(tr.stats.starttime.datetime, tr.stats.endtime.datetime, tr.stats.npts)
        ax.plot(tt, tr.data+ydev[ii], 'r', linewidth=1)
        tr = stream.select(station=staname[ii], component="S")[0]
        tt = pd.date_range(tr.stats.starttime.datetime, tr.stats.endtime.datetime, tr.stats.npts)
        ax.plot(tt, tr.data+ydev[ii], 'k', linewidth=1)
    ax.set_yticks(ydev)
    ax.set_yticklabels(staname, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('Input data [{}]'.format(tr.stats.starttime.date), fontsize=16, fontweight ="bold")
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    fname = os.path.join(dir_output, 'input_data.png')
    fig.savefig(fname, bbox_inches='tight')
    
    return
