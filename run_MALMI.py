#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:13:03 2021

@author: shipe
"""


import sys
sys.path.append('/home/peidong/xresearch/code/MALMI/src')
from main import MALMI


dir_seismic = "../data/seismic_data_raw/seismic_raw_20181201"  # path to raw continuous seismic data 
dir_output = "../data"  # path for outputs
n_processor = 14  # number of CPU processors for parallel processing
coseismiq = MALMI(dir_seismic, dir_output, n_processor=n_processor)


file_station = '../data/station/station_location.csv'  #  station metadata file, in FDSNWS station text format: *.txt or StationXML format: *.xml
channels = ["*HE", "*HN", "*HZ"]  # channels of the input seismic data
coseismiq.format_ML_inputs(file_station, channels)


input_MLmodel = '/home/peidong/xresearch/code/EQTransformer/ModelsAndSampleData/EqT_model.h5'  # path to a trained EQT model
overlap = 0.8  # overlap rate of time window for generating probabilities. e.g. 0.6 means 60% of time window are overlapped
coseismiq.generate_prob(input_MLmodel, overlap)


sttd_max = 8  # maximum P-P traveltime difference between different stations for the imaging area, in second
spttdf_ssmax = 4  # the maximal P to S arrivaltime difference for a perticular station in second for the imaging area, no need to be very accurate
twlex = 2  # time in second for extend the time window, roughly equal to the width of P- or S-probability envelope
d_thrd = 0.1  # detection threshold for detect events from ML predicted event probabilities
nsta_thrd = 3  # minimal number of stations triggered during a specified time period
coseismiq.event_detect_ouput(sttd_max, spttdf_ssmax, twlex, d_thrd, nsta_thrd)


dir_tt = '../data/traveltime/tt_loki'  # path to travetime data set
tt_ftage = 'layer'  # traveltime data set filename tage
probthrd = 0.001  # if maximum value of the input phase probabilites is larger than this threshold, the input trace will be normalized (to 1)
coseismiq.migration(dir_tt, tt_ftage, probthrd)


coseismiq.clear_interm()


