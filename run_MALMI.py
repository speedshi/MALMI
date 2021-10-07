#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:13:03 2021

@author: shipe
"""


import sys
sys.path.append('/home/peidong/xresearch/code/MALMI/src')
from main import MALMI
import gc


dir_seismic = "../data/seismic_data_raw/seismic_raw_20181206"  # path to raw continuous seismic data 
dir_output = "../data"  # path for outputs
dir_tt = '../data/traveltime/tt_150m'  # path to travetime data set
tt_ftage = 'layer'  # traveltime data set filename tage
n_processor = 6  # number of CPU processors for parallel processing
seismic_channels = ["*HE", "*HN", "*HZ"]  # channels of the input seismic data
coseismiq = MALMI(dir_seismic, dir_output, dir_tt, tt_ftage, n_processor, seismic_channels)
gc.collect()


file_station = '../data/station/station_location.csv'  #  station metadata file, in FDSNWS station text format: *.txt or StationXML format: *.xml
coseismiq.format_ML_inputs(file_station)
gc.collect()


input_MLmodel = '/home/peidong/xresearch/code/EQTransformer/ModelsAndSampleData/EqT_model.h5'  # path to a trained EQT model
overlap = 0.8  # overlap rate of time window for generating probabilities. e.g. 0.6 means 60% of time window are overlapped
coseismiq.generate_prob(input_MLmodel, overlap)
gc.collect()


twind_srch = None  # time window length in second where events will be searched in this range
twlex = 1.0  # time in second for extend the time window, roughly equal to the width of P- or S-probability envelope
P_thrd = 0.05  # probability threshold for detecting P-phases/events from the ML-predicted phase probabilities
S_thrd = 0.05  # probability threshold for detecting S-phases/events from the ML-predicted phase probabilities
nsta_thrd = 3  # minimal number of stations triggered during the specified event time period
npha_thrd = 6  # minimal number of phases triggered during the specified event time period
coseismiq.event_detect_ouput(twind_srch, twlex, P_thrd, S_thrd, nsta_thrd, npha_thrd)
gc.collect()


probthrd = 0.01  # if maximum value of the input phase probabilites is larger than this threshold, the input trace will be normalized (to 1)
coseismiq.migration(probthrd)
gc.collect()


coseismiq.rsprocess_view()
gc.collect()


coseismiq.clear_interm()
gc.collect()


