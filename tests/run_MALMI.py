#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:13:03 2021

@author: shipe
"""


import sys
sys.path.append('/home/shipe/codes/MALMI/src')  # user need to change the path here!
from main import MALMI
import datetime
import os


# %% seismic parameters========================================================
seismic = {}
seismic['dir'] = "./inputs/seismic_data/seismic_raw_20181230"  # path to the SDS archive directory or parent directory where seismic data are stored all in one folder
seismic['stainvf'] = './inputs/station_inventory/stations.csv'  # path to the station inventory file
seismic['datastru'] =  'AIO'  # the input seismic data file structure, can be 'AIO' or 'SDS' or 'EVS'
seismic['freqband'] = [2, 50]  # frequency bandpass range in Hz, sometimes lower frequencies are not necessary and filtering may improve the ML performance


# %% control parameters========================================================
control = {}
control['dir_output'] = "./outputs/malmi_result"  # path for outputs
control['n_processor'] = 6  # number of CPU processors for parallel processing


# %% traveltime parameters=====================================================
tt = {}
tt['vmodel'] = './inputs/velocity_model/local.txt'  # filename including path of the velocity model
tt['dir'] = './outputs/traveltime/tt_400m'  # path to travetime data directory
tt['build'] = False  # if you want to build the traveltime table or not, traveltime trable will be built when initialize MALMI class and only need to be built once 


# %% grid parameters used for migration========================================
# if load existing traveltime tables of NonLinLoc format, set grid as None
# note all stations must lie within the grid
grid = {}
grid['LatOrig'] = 63.5  # latitude in decimal degrees of the origin point of the rectangular migration region (float, min:-90.0, max:90.0)
grid['LongOrig'] = -22.0  # longitude in decimal degrees of the origin point of the rectangular migration region (float, min:-180.0, max:180.0)
grid['zOrig'] = -2.0  # Z location of the grid origin in km relative to the sea-level. Nagative value means above the sea-level; Positive values for below the sea-level;
grid['xNum'] = 250  # number of grid nodes in the X direction (East)
grid['yNum'] = 250  # number of grid nodes in the Y direction (North)
grid['zNum'] = 60  # number of grid nodes in the Z direction (Vertical-down)
grid['dgrid'] = 0.4  # grid spacing in kilometers


# %% detection parameters======================================================
detect = {}
detect['twind_srch'] = 14  # time window length in second where events will be searched in this range
detect['twlex'] = 1.0  # time in second for extend the time window, roughly equal to the width of P- or S-probability envelope
detect['P_thrd'] = 0.05  # probability threshold for detecting P-phases/events from the ML-predicted phase probabilities
detect['S_thrd'] = 0.05  # probability threshold for detecting S-phases/events from the ML-predicted phase probabilities
detect['nsta_thrd'] = 3  # minimal number of stations triggered during the specified event time period
detect['npha_thrd'] = 6  # minimal number of phases triggered during the specified event time period


# %% migration parameters======================================================
MIG = {}
MIG['output_migv'] = False  # do we want to save a 3D migration data for each detected event, note turn on this may need large disk space


# %% Initialize MALMI
myworkflow = MALMI(seismic=seismic, tt=tt, grid=grid, control=control, detect=detect, MIG=MIG)


# %% Format input data set
myworkflow.format_ML_inputs()


# %% Run ML models to get continuous phase probabilities
ML = {}
ML['model'] = '/home/shipe/codes/EQTransformer/ModelsAndSampleData/EqT_model.h5'  # path to a trained EQT model
ML['overlap'] = 0.8  # overlap rate of time window for generating probabilities. e.g. 0.6 means 60% of time window are overlapped
myworkflow.generate_prob(ML)


# %% Detect locatable events from continuous phase probabilities
myworkflow.event_detect_ouput()


# %% Migration location for each event
myworkflow.migration()


# %% Generate waveform plots for each event
myworkflow.rsprocess_view()


# %% Delete the input continuous seismic data for ML models for saving disk space
myworkflow.clear_interm()


# %% retrive earthquake catalog from processing results
catalog = myworkflow.get_catalog()


