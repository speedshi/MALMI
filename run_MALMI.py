#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 21:13:03 2021

@author: shipe
"""


import sys
sys.path.append('/home/peidong/xresearch/code/MALMI/src')  # user need to change the path here!
from main import MALMI
import gc
import datetime
import os


# %% seismic parameters========================================================
seismic = {}
seismic['dir'] = "../data/seismic_data/SDS"  # path to the SDS archive directory or parent directory where seismic data are stored all in one folder
seismic['channels'] = ["*HE", "*HN", "*HZ"]  # used channels of the input seismic data
seismic['stainvf'] = '../data/station/station_location.csv'  # path to the station inventory file
seismic['datastru'] = sys.argv[1]  # the input seismic data file structure, can be 'AIO' or 'SDS'
if seismic['datastru'] == 'SDS':
    seismic['date'] = datetime.datetime.strptime(sys.argv[2], "%Y%m%d").date()  # the date of seismic data to be precessed, for example "20181205", only needed when seisdatastru is 'SDS'
elif seismic['datastru'] == 'AIO':
    seismic['date'] = None  # the date of seismic data to be precessed, only needed when seisdatastru is 'SDS', otherwize use 'None'
    seismic['dir'] = os.path.join(seismic['dir'], sys.argv[2])
else:
    raise ValueError('Unrecognized input for: seisdatastru! Can\'t determine the structure of the input seismic data files!')


# %% control parameters========================================================
control = {}
control['dir_output'] = "../data/region1"  # path for outputs
control['n_processor'] = 6  # number of CPU processors for parallel processing


# %% traveltime parameters=====================================================
tt = {}
tt['vmodel'] = '../data/traveltime/velocity.txt'  # filename including path of the velocity model
tt['dir'] = '../data/traveltime/tt_150m'  # path to travetime data directory
tt['ftage'] = 'layer'  # traveltime data set filename tage as used for NonLinLoc


# %% grid parameters used for migration========================================
# if load existing traveltime tables of NonLinLoc format, set grid as None
# note all stations must lie within the grid
grid = {}
grid['LatOrig'] = 63.5  # latitude in decimal degrees of the origin point of the rectangular migration region (float, min:-90.0, max:90.0)
grid['LongOrig'] = -22.0  # longitude in decimal degrees of the origin point of the rectangular migration region (float, min:-180.0, max:180.0)
grid['rotAngle'] = 0.0  # rotation angle in decimal degrees of the rectrangular region in degrees clockwise relative to the Y-axis
grid['zOrig'] = -2.0  # Z location of the grid origin in km relative to the sea-level. Nagative value means above the sea-level; Positive values for below the sea-level;
grid['xNum'] = 250  # number of grid nodes in the X direction
grid['yNum'] = 250  # number of grid nodes in the Y direction
grid['zNum'] = 60  # number of grid nodes in the Z direction
grid['dgrid'] = 0.4  # grid spacing in kilometers


# %% Initialize MALMI
coseismiq = MALMI(seismic, tt, grid, control)
gc.collect()


# %% Format input data set
coseismiq.format_ML_inputs()
gc.collect()


# %% Run ML models to get continuous phase probabilities
input_MLmodel = '/home/peidong/xresearch/code/EQTransformer/ModelsAndSampleData/EqT_model.h5'  # path to a trained EQT model
overlap = 0.8  # overlap rate of time window for generating probabilities. e.g. 0.6 means 60% of time window are overlapped
coseismiq.generate_prob(input_MLmodel, overlap)
gc.collect()


# %% Detect locatable events from continuous phase probabilities
twind_srch = None  # time window length in second where events will be searched in this range
twlex = 1.0  # time in second for extend the time window, roughly equal to the width of P- or S-probability envelope
P_thrd = 0.05  # probability threshold for detecting P-phases/events from the ML-predicted phase probabilities
S_thrd = 0.05  # probability threshold for detecting S-phases/events from the ML-predicted phase probabilities
nsta_thrd = 3  # minimal number of stations triggered during the specified event time period
npha_thrd = 6  # minimal number of phases triggered during the specified event time period
coseismiq.event_detect_ouput(twind_srch, twlex, P_thrd, S_thrd, nsta_thrd, npha_thrd)
gc.collect()


# %% Migration location for each event
probthrd = 0.01  # if maximum value of the input phase probabilites is larger than this threshold, the input trace will be normalized (to 1)
coseismiq.migration(probthrd)
gc.collect()


# %% Generate waveform plots for each event
coseismiq.rsprocess_view()
gc.collect()


# %% Delete the input continuous seismic data for ML models for saving disk space
coseismiq.clear_interm()
gc.collect()


