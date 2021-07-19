#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:26:55 2021

@author: shipe
"""




def stream_resampling(stream, sampling_rate):
    """
    To resample the input seismic data.
    Parameters
    ----------
    stream : obspy stream
        input seismic data.
    sampling_rate : float
        required sampling rate in Hz.

    Returns
    -------
    stream : obspy stream
        output seismic data after resampling.

    """
    
    for tr in stream:
        if tr.stats.sampling_rate != sampling_rate:
            # need to do resampling
            if tr.stats.sampling_rate > sampling_rate:
                # need lowpass filter before resampling
                tr.filter('lowpass',freq=0.5*sampling_rate,zerophase=True)
            
            # perform resampling
            tr.resample(sampling_rate=sampling_rate)
    
    return stream

