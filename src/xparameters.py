#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:43:10 2022

@author: shipe
"""


from ioformatting import read_malmipsdetect
import numpy as np
import matplotlib.pyplot as plt
import os


def staphs_trigger_ana(file_detection, dir_out='./', xmax=None):
    """
    Statistical analysis of the number of station and phase triggered.
    Help to determine the optimal threshold of triggered station number (N_sta) and 
    triggered phase number (N_pha).

    Parameters
    ----------
    file_detection : str
        the filename including path of the detection results.
    dir_out : str, optional
        Directory for outputs. The default is './'.
    xmax : int, optional
        The maximum number of triggered phases/stations for displaying.
    Returns
    -------
    None.

    """
    
    # load detection files
    detections = read_malmipsdetect(file_detection)  # the detection results
    station_min = min(detections['station'])  # minimal number of stations triggered
    station_max = max(detections['station'])  # maximal number of stations triggered
    phase_min = min(detections['phase'])  # minimal number of phase triggered
    phase_max = max(detections['phase'])  # maximal number of phase triggered
    
    # triggered station number statistics
    sta_num = np.arange(station_min, station_max+1)  # station number triggered
    sta_acu = np.zeros_like(sta_num)  # accumulated number
    sta_nac = np.zeros_like(sta_num)  # non-accumulated number
    for ii, ii_sta in enumerate(sta_num):
        sta_acu[ii] = sum(np.array(detections['station']) >= ii_sta)
        sta_nac[ii] = sum(np.array(detections['station']) == ii_sta)
    
    # calculate indicator
    dy = np.gradient(sta_acu)  # first derivatives
    d2y = np.gradient(dy)  # second derivatives
    
    # plot
    # set the maximum number of triggered stations for displaying
    if xmax is None:
        x_max = station_max  
    else:
        x_max = xmax
    ixx_thrd = np.argmax(d2y)
    fig = plt.figure(figsize=(12,8), dpi=300)
    ax1 = fig.add_subplot(211)
    ax1.plot(sta_num, sta_acu, marker='o', markersize=10, c='k', linewidth=2.5, label='Cumulative')
    ax1.bar(sta_num, sta_nac, width=0.3, color='blue', edgecolor='k', label='Non-cumulative')
    ax1.legend(loc ="upper right", fontsize='large', markerscale=1.0)
    ax1.plot(sta_num[ixx_thrd], sta_acu[ixx_thrd], marker='o', markersize=10, c='r')
    ax1.set_xlim([station_min-0.5, x_max+0.5])
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.set_ylabel('Event number', fontsize=16, fontweight ="bold")
    plt.tick_params('x', labelbottom=False)
    ax2 = fig.add_subplot(212, sharex = ax1)
    ax2.plot(sta_num, d2y, marker='o', markersize=10, c='k', linewidth=2.5)
    ax2.plot(sta_num[ixx_thrd], d2y[ixx_thrd], marker='o', markersize=10, c='r')
    ax2.axvline(x=sta_num[ixx_thrd], c='r', ls='--', lw=1.5)
    ax2.set_xlim([station_min-0.5, x_max+0.5])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_xlabel('Triggered station number', fontsize=16, fontweight ="bold")
    ax2.set_ylabel('Indicator', fontsize=16, fontweight ="bold")
    fname = os.path.join(dir_out, "triggered_station_number_variation.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    # triggered phase number statistics
    phs_num = np.arange(phase_min, phase_max+1)  # phase number triggered
    phs_acu = np.zeros_like(phs_num)  # accumulated number
    phs_nac = np.zeros_like(phs_num)  # non-accumulated number
    for ii, ii_phs in enumerate(phs_num):
        phs_acu[ii] = sum(np.array(detections['phase']) >= ii_phs)
        phs_nac[ii] = sum(np.array(detections['phase']) == ii_phs)
        
    # calculate indicator
    dy = np.gradient(phs_acu)  # first derivatives
    d2y = np.gradient(dy)  # second derivatives  
    
    # plot
    # set the maximum number of triggered phases for displaying
    if xmax is None:
        x_max = phase_max  
    else:
        x_max = xmax
    ixx_thrd = np.argmax(d2y)
    fig = plt.figure(figsize=(12,8), dpi=300)
    ax1 = fig.add_subplot(211)
    ax1.plot(phs_num, phs_acu, marker='o', markersize=10, c='k', linewidth=2.5, label='Cumulative')
    ax1.bar(phs_num, phs_nac, width=0.3, color='blue', edgecolor='k', label='Non-cumulative')
    ax1.legend(loc ="upper right", fontsize='large', markerscale=1.0)
    ax1.plot(phs_num[ixx_thrd], phs_acu[ixx_thrd], marker='o', markersize=10, c='r')
    ax1.set_xlim([station_min-0.5, x_max+0.5])
    ax1.tick_params(axis='y', which='major', labelsize=16)
    ax1.set_ylabel('Event number', fontsize=16, fontweight ="bold")
    plt.tick_params('x', labelbottom=False)
    ax2 = fig.add_subplot(212, sharex = ax1)
    ax2.plot(phs_num, d2y, marker='o', markersize=10, c='k', linewidth=2.5)
    ax2.plot(phs_num[ixx_thrd], d2y[ixx_thrd], marker='o', markersize=10, c='r')
    ax2.axvline(x=phs_num[ixx_thrd], c='r', ls='--', lw=1.5)
    ax2.set_xlim([station_min-0.5, x_max+0.5])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_xlabel('Triggered phase number', fontsize=16, fontweight ="bold")
    ax2.set_ylabel('Indicator', fontsize=16, fontweight ="bold")
    fname = os.path.join(dir_out, "triggered_phase_number_variation.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.cla()
    fig.clear()
    plt.close(fig)
    
    return
    
    
    
    
