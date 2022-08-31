#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:21:02 2022

Functions related to seisbench.

@author: shipe
"""


import os
import obspy
import seisbench.models as sbm
from utils_dataprocess import merge_dict
from ioformatting import dict2csv
import numpy as np


def seisbench_geneprob(spara):
    """
    Use seisbench to generate continuous phase probabilites;

    Parameters
    ----------
    spara : dict
        spara['evsegments']: boolen, specify whether data in the input directory
                            are stored in the form of event segments;
                            True means input data are event segments;
                            False means input data are continous data;
        spara['dir_in']: directory to where seismic data of different stations are stored.
        spara['dir_out']: directory to save the generated phase probabilities;
        spara['model']: str, pretrained model info in seisbench in form of 'modelname.datasetname',
                        such as 'PhaseNet.stead', 'EQTransformer.original', 'GPD.scedc';
        spara['P_thrd']: float, probability threshold for detecting P-phases;
        spara['S_thrd']: float, probability threshold for detecting S-phases;

    Returns
    -------
    None.

    """
    
    sbm_model, sbm_pretrained = spara['model'].split('.')
    
    if (sbm_model.lower() == 'eqtransformer') or (sbm_model.lower() == 'eqt'):
        model = sbm.EQTransformer.from_pretrained(sbm_pretrained)
    elif (sbm_model.lower() == 'phasenet') or (sbm_model.lower() == 'pnt'):
        model = sbm.PhaseNet.from_pretrained(sbm_pretrained)
    elif (sbm_model.lower() == 'gpd'):
        model = sbm.GPD.from_pretrained(sbm_pretrained)
    else:
        raise ValueError('Unrecognized input for spara[\'model\']: {}!'.format(spara['model']))
    
    sbs2ppara = {}
    sbs2ppara['P_threshold'] = spara['P_thrd']
    sbs2ppara['S_threshold'] = spara['S_thrd']
    
    if spara['evsegments']:
        # input data are event segments
        loop_folders = sorted([fdname for fdname in os.listdir(spara['dir_in']) if os.path.isdir(os.path.join(spara['dir_in'], fdname))])
    else:
        # input data are continuous data
        loop_folders = ['']  # current folder
        
    for jjfd in loop_folders:  
        dir_seismicsta = os.path.join(spara['dir_in'], jjfd)
        dir_probsave = os.path.join(spara['dir_out'], jjfd)
        staflds = sorted([fdname for fdname in os.listdir(dir_seismicsta) if os.path.isdir(os.path.join(dir_seismicsta, fdname))])  # station folders
        
        for ifd in staflds:
            iddir = os.path.join(dir_seismicsta, ifd)
            stream = obspy.read(os.path.join(iddir, '*'))  # 3-component data of a station
            annotations, picks = seisbench_stream2prob(stream=stream, model=model, paras=sbs2ppara)
        
            if annotations.count()>0:  # in case no enough data for a successful prediction
                idir_save = os.path.join(dir_probsave, ifd)
                if not os.path.exists(idir_save):
                    os.makedirs(idir_save)
            
                # save prediction results
                for tr in annotations:
                    tr.stats.channel = 'PB'+tr.stats.channel[-1].upper()  # save mseed file, channel name maximum 3 char
                fname = os.path.join(idir_save, 'prediction_probabilities.mseed')
                annotations.write(fname, format='MSEED')
                file_pk = os.path.join(idir_save, 'X_prediction_results.csv')
                picks_dict = {}
                for ipick in picks:
                    ipk_dict = ipick.__dict__
                    for ikk in ipk_dict:
                        ipk_dict[ikk] = [ipk_dict[ikk]]  # to merge dict, item of each entry should be a list or numpy array
                    picks_dict = merge_dict(dict1=picks_dict, dict2=ipk_dict)
                dict2csv(indic=picks_dict, filename=file_pk, mode='w')
        
    return


def seisbench_stream2prob(stream, model, paras):
    
    stream.merge()
    if stream.count()<3:
        # having only 1 or 2 component
        comps = [itr.stats.channel[-1] for itr in stream]
        if ('1' in comps) or ('2' in comps) or ('3' in comps):
            for iii in ['1','2','3']:
                if iii not in comps: 
                    comps_add = [iii]
                    break
        else:
            comps_add = list(set(['Z','N','E']) - set(comps))
        for icomp in comps_add:
            itrace = stream[0].copy()
            itrace.stats.channel = stream[0].stats.channel[:-1]+icomp
            itrace.data = np.zeros_like(stream[0].data)
            stream.append(itrace.copy())

    # trim data to the same time range
    starttime = [itr.stats.starttime for itr in stream]
    endtime = [itr.stats.endtime for itr in stream]
    starttime_min = min(starttime)
    endtime_max = max(endtime)
    stream.trim(starttime=starttime_min, endtime=endtime_max, pad=True, fill_value=0)

    annotations = model.annotate(stream)
    
    if model.name == 'EQTransformer':
        picks, _ = model.classify(stream, P_threshold=paras['P_threshold'], S_threshold=paras['S_threshold'])
    else:
        picks = model.classify(stream, P_threshold=paras['P_threshold'], S_threshold=paras['S_threshold'])
    
    return annotations, picks



