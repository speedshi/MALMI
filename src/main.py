#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:41:56 2021

@author: Peidong Shi
@email: speedshi@hotmail.com

MALMI main function - building an integrated framework.
Coordinate convention: X-axis -> East; Y-axis -> North; Z-axis -> Depth (vertical-down).
"""


import os
import gc
import copy
import shutil
import glob
import datetime
import numpy as np
import obspy
from xstation import load_station
from xcatalog import load_catalog, catalog2dict
        

class MALMI:

    def __init__(self, seismic, tt=None, grid=None, control={}, detect={}, MIG={}):
        """
        Initilize global input and output paramaters, configure MALMI.
        Parameters
        ----------
        seismic parameters (For input seismic data set):
            seismic['dir']: str, required.
                path to raw continuous seismic data.
            seismic['instrument_code']: list of str, optional.
                the used instruments code of the input seismic data,
                default is ["HH", "BH", "EH", "SH", "HG", "HN"].
                This instrument_code list specify the priority code list, 
                will try to load only 1 instrument code data for the colocated stations (having the same station name);
                if instrument_code is None, will try to load all avaliable data;
            seismic['datastru']: str, optional, default: 'AIO'.
                the input seismic data file structure.
                'AIO' : continuous seismic data are organized All In One folder;
                'SDS' : continuous seismic data are organized in SeisComP Data Structure;
                'EVS' : event segments;
            seismic['date']: datetime.date, optional, default is None.
                When seisdatastru is 'SDS', seismic data set are processed daily, 
                therefore this parameter is used to specify the date for processing.
            seismic['stainvf']: str, required.
                station inventory file including path. The data format should be 
                recognizable by ObsPy, such as:
                    FDSNWS station text format: *.txt,
                    FDSNWS StationXML format: *.xml.
                or a simply CSV file using ',' as the delimiter in which the first row 
                is column name and must contain: 'network', 'station', 'latitude', 
                'longitude', 'elevation'. Latitude and longitude are in decimal degree 
                and elevation in meters relative to the sea-level (positive for up).
            seismic['freqband']: list of folat, optional
                frequency range in Hz for filtering seismic data for ML inputs, 
                such as [3, 45] meaning filter seismic data to 3-45 Hz before input to ML models.
                default is None, means no filtering.
            seismic['split']: boolen or dict, default is False.
                whether to split the input continous data into unmasked traces without gaps.
                seismic['split']['mask_value']: float, int or None
                    input continous seismic data of the specified value will be recognized as gap, 
                    and will be masked and used to split the traces.
                    This is good for filtering, because filter the contious data with 
                    0 (for example) filled gap will produce glitches. It is recommand
                    to filter the data before merge the seismic data.
                seismic['split']['minimal_continous_points'] : int
                    this specifies that at least certain continuous points having the mask_value
                    will be recognized as gap.
        
        traveltime parameters (For loading or generating traveltime tables):
            tt['vmodel']: str, optional, default is None.
                filename including path of the velocity model.    
                None means directly load existing traveltime table of NonLinLoc format.
            tt['dir']: str, optional, default is './data/traveltime'.
                path to travetime data directory or for saving generated traveltime tabels.
            tt['ftage']: str, optional, default is 'layer'. 
                traveltime data set filename tage as used for NonLinLoc,
                used at the file root name.
            tt['build'] : boolen, default is True.
                whether to build the traveltime table.
        
        grid parameters (Primarily used in migration):
            Note all input stations must lie within the grid.
            If load existing traveltime tables of NonLinLoc format, 
            grid is then not needed, use None instead.
            grid['LatOrig']: latitude in decimal degrees of the origin point of
                             the rectangular migration region. 
                             Required. (float, min:-90.0, max:90.0)
            grid['LongOrig']: longitude in decimal degrees of the origin point of
                              the rectangular migration region.
                              Required. (float, min:-180.0, max:180.0)
            grid['rotAngle']: rotation angle in decimal degrees of the rectrangular region 
                              in degrees clockwise relative to the Y-axis.
                              Default is 0.0. (float, min:-360.0, max:360.0)
                              Not used currently!
            grid['xOrig']: X location of the grid origin in km relative to the 
                           geographic origin (positive: east).
                           optional, default value is 0.0. Type: [float].
            grid['yOrig']: Y location of the grid origin in km relative to the 
                           geographic origin (positive: north).
                           optional, default value is 0.0. Type: [float].
            grid['zOrig']: Z location of the grid origin in km relative to the 
                           sea-level. Nagative value means above the sea-level; 
                           Positive values for below the sea-level;
                           Note grid model should include all stations, thus
                           grid['zOrig'] should <= -1 * max(station_evelation/1000.0);
                           Required. (float)
            grid['xNum']: number of grid nodes in the X direction. Required. (int, >=2)
            grid['yNum']: number of grid nodes in the Y direction. Required. (int, >=2)
            grid['zNum']: number of grid nodes in the Z direction. Required. (int, >=2)
            grid['dgrid']: grid spacing in kilometers. Required. (float, >0)

        control parameters (For control outputs and processing):
            control['dir_output'] : str, optional, default: './data'.
                directory for outputs.
            control['n_processor'] : int, optional, default: 1.
                number of CPU processors for parallel processing.
            control['plot_map'] : boolean, optional, default: True.
                whether plot the station basemap.
                
        detection parameters (For detecting events):
            detect['twind_srch'] : float, optional, the default is None.
                time window length in second where events will be searched in this range.
                How to determine this parameter:
                Upper-value estimation: maximum P-S traveltime difference between 
                different stations for the whole imaging area.
                If None, then automatically determine it from traveltime table.
            detect['twlex'] : float, optional
                time window length in second for extending the output time range, 
                usually set to be 1-2 second. The default is 1.0 second.
            detect['P_thrd'] : float, optional
                probability threshold for detecting P-phases/events from the ML-predicted 
                phase probabilities. The default is 0.1.
            detect['S_thrd'] : float, optional
                probability threshold for detecting S-phases/events from the ML-predicted 
                phase probabilities. The default is 0.1.
            detect['nsta_thrd'] : int, optional
                minimal number of stations triggered during the specified event time period.
                The default is 3.
            detect['npha_thrd'] : int, optional
                minimal number of phases triggered during the specified event time period.
                The default is 4.
            detect['nsta_min_percentage'] : float, optional
                Minimal decimal percentage of total available stations for triggering (ranging from 0. to 1.0).
                The default is None.
            detect['outseis'] : boolen, optional
                whether to output raw seismic data segments for the detectd events.
                The default is True.
            detect['output_allsta'] : boolen, optional
                If true: output data (probability and seismic) of all available stations;
                If false: output data of triggered stations only;
                The default is True.
            
        migration parameters (For using migration to locate events):
            MIG['probthrd'] : float, optional
                probability normalization threshold. If maximum value of the input 
                phase probabilites is larger than this threshold, the input trace 
                will be normalized (to 1). The default is 0.001.
            MIG['ppower'] : float, optional
                element wise power over the phase probabilities before stacking, 
                if None, no powering is applied, and use original data for stacking.
                The default is 4.
            MIG['output_migv'] : boolen, optional
                specify whether to output the migrated data volume for each detected event.
                The migrated data volume could be large if the migration grid is large;
                If so to save disk space, set MIG['output_migv'] = False.
            MIG['migv_4D'] : boolen, optional
                specify to calculate 3D (xyz) or 4D (xyzt) stacking matrix, 
                default is false , i.e. 3D.

        Returns
        -------
        None.

        """
        
        from traveltime import build_traveltime
        from xcoordinate import get_regioncoord
        from utils_plot import plot_basemap
        
        # set default parameters----------------------------------------------- 
        if 'instrument_code' not in seismic:
            seismic['instrument_code'] = ["HH", "BH", "EH", "SH", "HG", "HN", "FP"]
        
        if 'datastru' not in seismic:
            seismic['datastru'] = 'AIO'
        
        if 'date' not in seismic:
            seismic['date'] = None
        
        if 'freqband' not in seismic:
            seismic['freqband'] = None
            
        if 'split' not in seismic:
            seismic['split'] = False
        
        if (tt is not None):
            if ('vmodel' not in tt):
                tt['vmodel'] = None
        
            if ('dir' not in tt):
                tt['dir'] = './data/traveltime'
        
            if ('ftage' not in tt):
                tt['ftage'] = 'layer'
        
            if ('build' not in tt):
                tt['build'] = True
        
            tt['hdr_filename'] = 'header.hdr'  # travetime data set header filename
        
        if (grid is not None):
            if ('xOrig' not in grid):
                grid['xOrig'] = 0.0
            
            if ('yOrig' not in grid):
                grid['yOrig'] = 0.0
            
            if ('rotAngle' not in grid):
                grid['rotAngle'] = 0.0
        
        if control is None:
            control = {}
        
        if 'dir_output' not in control:
            control['dir_output'] = './data'
        
        if 'n_processor' not in control:
            control['n_processor'] = 1
            
        if 'plot_map' not in control:
            control['plot_map'] = True
            
        if detect is None:
            detect = {}
        
        if 'twind_srch' not in detect:
            detect['twind_srch'] = None
        
        if 'twlex' not in detect:
            detect['twlex'] = 1.0
        
        if 'P_thrd' not in detect:
            detect['P_thrd'] = 0.1
        
        if 'S_thrd' not in detect:
            detect['S_thrd'] = 0.1
        
        if 'nsta_thrd' not in detect:
            detect['nsta_thrd'] = 3
            
        if 'npha_thrd' not in detect:    
            detect['npha_thrd'] = 4
        
        if 'nsta_min_percentage' not in detect:
            detect['nsta_min_percentage'] = None
        
        if 'outseis' not in detect:
            detect['outseis'] = True
        
        if 'output_allsta' not in detect:
            detect['output_allsta'] = True
            
        if MIG is None:
            MIG = {}
            
        if 'probthrd' not in MIG:
            MIG['probthrd'] = min(detect['P_thrd'], detect['S_thrd'])
        
        if 'ppower' not in MIG:
            MIG['ppower'] = 4
            
        if 'output_migv' not in MIG:
            MIG['output_migv'] = False
        
        if 'migv_4D' not in MIG:    
            MIG['migv_4D'] = False
        #----------------------------------------------------------------------
        
        self.seisdatastru = copy.deepcopy(seismic['datastru'])
        self.dir_seismic = copy.deepcopy(seismic['dir'])
        self.instrument_code = copy.deepcopy(seismic['instrument_code'])  # the instrument codes of the input seismic data
        self.seisdate = copy.deepcopy(seismic['date'])  # date of seismic data
        self.freqband = copy.deepcopy(seismic['freqband'])  # frequency band in Hz for filtering seismic data
        self.n_processor = copy.deepcopy(control['n_processor'])  # number of threads for parallel processing
        if (self.seisdatastru == 'AIO') or (self.seisdatastru == 'EVS'):
            # input seismic data files are stored simply in one folder or event segments input
            # get the foldername of the input seismic data, used as the identifer of the input data set
            if self.dir_seismic[-1] == '/':
                fd_seismic = self.dir_seismic.split('/')[-2]
            else:
                fd_seismic = self.dir_seismic.split('/')[-1]
        elif self.seisdatastru == 'SDS':
            # input seismic data files are organized in SDS
            # use the date as the identifer of the input data set
            fd_seismic = datetime.datetime.strftime(self.seisdate, "%Y%m%d")
        # elif self.seisdatastru == 'EVS':
        #     # input seismic data are cutted event segments
        #     # each event files are stored in one folder
        #     fd_seismic = 'events_input'
        else:
            raise ValueError('Unrecognized input for: seisdatastru! Can\'t determine the structure of the input seismic data files!')
        
        self.dir_projoutput = control['dir_output']  # project output direcotroy
        self.dir_CML = os.path.join(self.dir_projoutput, 'data_ML')  # directory for common ML outputs
        self.dir_ML = os.path.join(self.dir_CML, fd_seismic)  # directory for ML outputs of the currrent processing dataset
        self.dir_mseed = os.path.join(self.dir_ML, 'mseeds')  # directory for outputting seismic data for EQT, NOTE do not add '/' at the last part
        self.dir_hdf5 = self.dir_mseed + '_processed_hdfs'  # path to the hdf5 and csv files
        self.dir_EQTjson = os.path.join(self.dir_ML, 'json')  # directory for outputting station json file for EQT
        self.dir_prob = os.path.join(self.dir_ML, 'prob_and_detection')  # output directory for ML probability outputs
        
        self.dir_MIG = os.path.join(self.dir_projoutput, 'data_MIG')  # directory for common migration outputs
        self.dir_migration = os.path.join(self.dir_MIG, fd_seismic)  # directory for migration outputs of the current processing dataset
        self.fld_prob = 'prob_evstream'  # foldername of the probability outputs of different events
        self.dir_lokiprob = os.path.join(self.dir_migration, self.fld_prob)  # directory for probability outputs of different events in SEED format
        self.dir_lokiseis = os.path.join(self.dir_migration, 'seis_evstream')  # directory for raw seismic outputs of different events in SEED format
        
        # make sure the project output directory exist
        if not os.path.exists(self.dir_projoutput):
            os.makedirs(self.dir_projoutput, exist_ok=True)
        
        # read in station invertory and obtain station information
        self.stainv = load_station(seismic['stainvf'], outformat='obspy')  # obspy station inventory 
        
        # config travel-time dataset and migration output directories
        if (tt is None) and (grid is None):
            # grid and travel-time information are only required during the migration process
            # if migration part is not performed, then no need to sep up grid and travel-time information
            print('Skip grid set up and skip travel-time table building!')
        else:
            # set up grid and traveltime tables
            
            if (tt is not None) and (tt['build']):
                # build or check traveltime data set
                grid = build_traveltime(grid, tt, self.stainv)
            
            self.grid = copy.deepcopy(grid)
            if self.grid is not None:
                region1, mgregion = get_regioncoord(self.grid, self.stainv, 0.05, consider_mgregion=True)  # obtain migration region lon/lat range
                self.grid['mgregion'] = copy.deepcopy(mgregion)  # migration region lat/lon range, [lon_min, lon_max, lat_min, lat_max] in degree
                self.grid['pltregion'] = copy.deepcopy(region1)  # plot region lat/lon range in format of [lon_min, lon_max, lat_min, lat_max] in degree
        
            # plot stations and migration region for checking
            if control['plot_map']:
                fname1 = os.path.join(control['dir_output'], "basemap_stations_mgarea.png")
                if not os.path.isfile(fname1):
                    plot_basemap(region1, self.stainv, mgregion, fname1, True, '30s')
        
            # travetime dataset related information
            self.dir_tt = copy.deepcopy(tt['dir'])  # path to travetime data set
            self.tt_precision = 'single'  # persicion for traveltime data set, 'single' or 'double'
            self.tt_hdr_filename = copy.deepcopy(tt['hdr_filename'])  # travetime data set header filename
            self.tt_ftage = copy.deepcopy(tt['ftage'])  # traveltime data set filename tage
            if self.dir_tt.split('/')[-1] == '':
                tt_folder = self.dir_tt.split('/')[-2]
            else:
                tt_folder = self.dir_tt.split('/')[-1]
        
            # config the final migration output directories
            self.fld_migresult = 'result_MLprob_{}'.format(tt_folder)  # foldername of the final migration results
            self.dir_lokiout = os.path.join(self.dir_migration, self.fld_migresult)  # path for loki final outputs
        
            # create the final migration output directories
            if not os.path.exists(self.dir_lokiout):
                os.makedirs(self.dir_lokiout, exist_ok=True)
        
        self.detect = detect.copy()  # detection parameters
        self.MIG = MIG.copy()  # migration parameters
        self.seismic = seismic.copy()  # seismic related parameters
        
        return


    def format_ML_inputs(self):
        """
        Format input data set for ML models.
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        
        from ioseisdata import seisdata_format_4ML
        
        print('MALMI starts to format input data set for ML models:')
        DFMT = {}
        DFMT['seisdatastru_input'] = self.seisdatastru
        DFMT['dir_seismic_input'] = self.dir_seismic
        DFMT['dir_seismic_output'] = self.dir_mseed
        DFMT['seismic_date'] = self.seisdate
        DFMT['stainv'] = self.stainv
        DFMT['instrument_code'] = self.instrument_code
        DFMT['freqband'] = self.freqband
        DFMT['split'] = self.seismic['split']
        
        seisdata_format_4ML(DFMT=DFMT)
        gc.collect()
        print('MALMI_format_ML_inputs complete!')
        return


    def generate_prob(self, ML):
        """
        Generate event and phase probabilities using ML models.
        Parameters
        ----------
        ML : dict, machine-learning related parameters.
        ML['engine'] : str
            machine learning engine to generate continous phase probabilities;
            can be 'EQTransformer', 'seisench';
        ML['model'] : str
            path to a pre-trained ML model for original EQTransformer, 
            or a pretrained model name from seisbench in form of 'modelname.datasetname',
            such as 'PhaseNet.stead';
        ML['overlap'] : float, default: 0.5
            overlap rate of time window for generating probabilities. 
            e.g. 0.6 means 60% of time window are overlapped.
        ML['number_of_cpus'] : int, default: 2
            Number of CPUs used for the parallel preprocessing and feeding of 
            data for prediction.
            If no GPU to use, this value shoud not be set too large, 
            otherwise prediction tends to be very slow.
            I have tested using a number of 96 on a server with only cpus avaliable,
            this makes the prediction process 20 times slower than using just 1 cpus.

        Returns
        -------
        None.

        """
        
        # set default parameters-----------------------------------------------
        if 'engine' not in ML:
            ML['engine'] = 'EQTransformer'
        
        if 'overlap' not in ML:
            ML['overlap'] = 0.5
        
        if 'number_of_cpus' not in ML:
            ML['number_of_cpus'] = 2
        #----------------------------------------------------------------------
        
        print('MALMI starts to generate event and phase probabilities using ML models:')
        
        if ML['engine'].lower() == 'eqtransformer':
            from ioformatting import stainv2json
            from EQTransformer.utils.hdf5_maker import preprocessor
            from EQTransformer.utils.plot import plot_data_chart
            from EQTransformer.core.predictor import predictor
            
            if self.seismic['datastru'] == 'EVS':
                # input are event segments
                
                # get the folder name of each event and sorted
                event_folders = sorted([fdname for fdname in os.listdir(self.dir_seismic) if os.path.isdir(os.path.join(self.dir_seismic, fdname))])
                hdf5_evdirs = []
                prob_evdirs = []
                
                for ievfd in event_folders:
                    # create hdf5 data for ML inputs--------------------------------------
                    # create station jason file for EQT if it does not exist
                    dir_EQTjson_ev = os.path.join(self.dir_EQTjson, ievfd)
                    stations_json = os.path.join(dir_EQTjson_ev, "station_list.json")  # station JSON file
                    mseed_directory_ev = os.path.join(self.dir_mseed, ievfd)  # mseed directory for each event
                    if not os.path.exists(stations_json):
                        stainv2json(stainfo=self.stainv, mseed_directory=mseed_directory_ev, dir_json=dir_EQTjson_ev)
            
                    preproc_dir_ev = os.path.join(self.dir_ML, "preproc_overlap{}".format(ML['overlap']), ievfd)  # path of the directory where will be located the summary files generated by preprocessor step
                    hdf5_dir_ev = os.path.join(self.dir_hdf5, ievfd)
                    preprocessor(preproc_dir=preproc_dir_ev, mseed_dir=mseed_directory_ev, output_dir=hdf5_dir_ev,
                                 stations_json=stations_json, overlap=ML['overlap'], 
                                 n_processor=1)
                    gc.collect()
                    
                    # show data availablity for each station-------------------------------
                    file_pkl = os.path.join(preproc_dir_ev, 'time_tracks.pkl')
                    time_interval = 1  # Time interval in hours for tick spaces in xaxes
                    plot_data_chart(time_tracks=file_pkl, time_interval=time_interval, dir_output=preproc_dir_ev)
                    
                    hdf5_evdirs.append(hdf5_dir_ev)
                    prob_evdirs.append(os.path.join(self.dir_prob, ievfd))
            
                # generate event and phase probabilities-------------------------------
                predictor(input_dir=hdf5_evdirs, input_model=ML['model'], output_dir=prob_evdirs,
                          output_probabilities=True, estimate_uncertainty=False,
                          detection_threshold=min(self.detect['P_thrd'], self.detect['S_thrd']), 
                          P_threshold=self.detect['P_thrd'], S_threshold=self.detect['S_thrd'], 
                          keepPS=False, number_of_cpus=ML['number_of_cpus'],
                          number_of_plots=100, plot_mode='time_frequency')
                gc.collect()
            
            else:
                # create hdf5 data for ML inputs--------------------------------------
                # create station jason file for EQT if it does not exist
                stations_json = os.path.join(self.dir_EQTjson, "station_list.json")  # station JSON file
                if not os.path.exists(stations_json):
                    stainv2json(self.stainv, self.dir_mseed, self.dir_EQTjson)
                
                preproc_dir = self.dir_ML + "/preproc_overlap{}".format(ML['overlap'])  # path of the directory where will be located the summary files generated by preprocessor step
                preprocessor(preproc_dir=preproc_dir, mseed_dir=self.dir_mseed, 
                             stations_json=stations_json, overlap=ML['overlap'], 
                             n_processor=1)
                gc.collect()
                
                # show data availablity for each station-------------------------------
                file_pkl = preproc_dir + '/time_tracks.pkl'
                time_interval = 1  # Time interval in hours for tick spaces in xaxes
                plot_data_chart(time_tracks=file_pkl, time_interval=time_interval, dir_output=preproc_dir)
                
                # generate event and phase probabilities-------------------------------
                predictor(input_dir=self.dir_hdf5, input_model=ML['model'], output_dir=self.dir_prob,
                          output_probabilities=True, estimate_uncertainty=False,
                          detection_threshold=min(self.detect['P_thrd'], self.detect['S_thrd']), P_threshold=self.detect['P_thrd'], S_threshold=self.detect['S_thrd'], 
                          keepPS=False, number_of_cpus=ML['number_of_cpus'],
                          number_of_plots=100, plot_mode='time_frequency')
                gc.collect()
        elif ML['engine'].lower() == 'seisbench':
            from xseisbench import seisbench_geneprob
            sbppara = {}
            sbppara['model'] = ML['model']
            sbppara['P_thrd'] = self.detect['P_thrd']
            sbppara['S_thrd'] = self.detect['S_thrd']
            sbppara['dir_in'] = self.dir_mseed
            sbppara['dir_out'] = self.dir_prob
            if self.seismic['datastru'] == 'EVS':
                sbppara['evsegments'] = True
            else:
                sbppara['evsegments'] = False
            seisbench_geneprob(spara=sbppara)
        else:
            raise ValueError('Unrecoginzed input for ML[\'engine\']: {}!'.format(ML['engine']))
        
        print('MALMI_generate_prob complete!')
        return

            
    def event_detect_ouput(self):
        """
        event detection based on the ML predicted event probabilites
        and output the corresponding phase probabilites of the detected events.
        Parameters
        ----------
        Returns
        -------
        None.

        """
        
        print('MALMI starts to detect events based on the ML predicted phase probabilites and output the corresponding phase probabilites of the detected events:')
        
        # from event_detection import eqt_arrayeventdetect
        from event_detection import phasedetectfprob, arrayeventdetect
        from utils_dataprocess import maxP2Stt
        
        if self.detect['twind_srch'] is None:
            self.detect['twind_srch'], _, _ = maxP2Stt(self.dir_tt, self.tt_hdr_filename, self.tt_ftage, self.tt_precision)
        
        if self.instrument_code is not None:
            seismic_channels = [iinstru+'?' for iinstru in self.instrument_code]
        else:
            seismic_channels = None
        
        if self.seismic['datastru'] == 'EVS':
            # get the folder name of each event and sorted
            event_folders = sorted([fdname for fdname in os.listdir(self.dir_seismic) if os.path.isdir(os.path.join(self.dir_seismic, fdname))])
            
            for ievfd in event_folders:
                dir_prob_ev = os.path.join(self.dir_prob, ievfd)
                event_info = phasedetectfprob(dir_probinput=dir_prob_ev, P_thrd=self.detect['P_thrd'], S_thrd=self.detect['S_thrd'])
            
                if self.detect['outseis']:
                    dir_seisdataset = os.path.join(self.dir_mseed, ievfd)
                else:
                    dir_seisdataset = None

                arrayeventdetect(event_info=event_info, 
                                 twind_srch=self.detect['twind_srch'], twlex=self.detect['twlex'], 
                                 nsta_thrd=self.detect['nsta_thrd'], npha_thrd=self.detect['npha_thrd'], 
                                 dir_output=self.dir_lokiprob, dir_output_seis=self.dir_lokiseis, dir_seisdataset=dir_seisdataset, 
                                 seismic_channels=seismic_channels, output_allsta=self.detect['output_allsta'])
        else:
            # eqt_arrayeventdetect(self.dir_prob, self.dir_lokiprob, sttd_max, twlex, d_thrd, nsta_thrd, spttdf_ssmax)
            event_info = phasedetectfprob(self.dir_prob, self.detect['P_thrd'], self.detect['S_thrd'])
            gc.collect()
                
            if self.detect['outseis']:
                dir_seisdataset = self.dir_mseed
            else:
                dir_seisdataset = None
                
            if self.detect['nsta_min_percentage'] is not None:
                num_avasta = len([fdname for fdname in os.listdir(self.dir_prob) if os.path.isdir(os.path.join(self.dir_prob, fdname))])  # get the total number of available stations
                nsta_thrd_new = int(np.ceil(self.detect['nsta_min_percentage'] * num_avasta))
                npha_thrd_new = int(np.ceil(2 * nsta_thrd_new * 0.7))  # at least 70% percent have both P- and S-phases
                if nsta_thrd_new > self.detect['nsta_thrd']:
                    self.detect['nsta_thrd'] = nsta_thrd_new  # reset using minimal decimal percentage of total available stations
                if npha_thrd_new > self.detect['npha_thrd']:
                    self.detect['npha_thrd'] = npha_thrd_new  # reset using minimal decimal percentage of total available phases
                
            arrayeventdetect(event_info=event_info, twind_srch=self.detect['twind_srch'], twlex=self.detect['twlex'], 
                             nsta_thrd=self.detect['nsta_thrd'], npha_thrd=self.detect['npha_thrd'], 
                             dir_output=self.dir_lokiprob, dir_output_seis=self.dir_lokiseis, dir_seisdataset=dir_seisdataset, 
                             seismic_channels=seismic_channels, output_allsta=self.detect['output_allsta'])
            gc.collect()
        
        print('MALMI_event_detect_ouput complete!')
        return


    def migration(self):
        """
        Perform migration based on input phase probabilites

        Parameters
        ----------

        Returns
        -------
        None.

        """
        
        from loki.loki import Loki
        
        print('MALMI starts to perform migration:')
        
        inputs = {}
        inputs['model'] = self.tt_ftage  # traveltime data set filename tage
        if self.n_processor < 1:
            inputs['npr'] = (os.cpu_count()-2)  # number of cores to run
        else:
            inputs['npr'] = self.n_processor  # number of cores to run
        inputs['normthrd'] = self.MIG['probthrd']  # if maximum value of the input phase probabilites is larger than this threshold, the input trace will be normalized (to 1)
        inputs['ppower'] = self.MIG['ppower']  # compute array element wise power over the input probabilities before stacking
        inputs['output_migv'] = self.MIG['output_migv']  # specify whether to output the migrated data volume for each event
        inputs['migv_4D'] = self.MIG['migv_4D']  # speify to calculate 3D or 4D stacking matrix
        if (len(self.stainv[0][0].channels) > 0):
            # have channel information
            inputs['station_idmode'] = 'network.station.location.instrument'
        else:
            # no channel information
            inputs['station_idmode'] = 'network.station'
        comp = ['P','S']  # when input data are probabilities of P- and S-picks, comp must be ['P', 'S']
        extension = '*'  # seismic data filename for loading, accept wildcard input, for all data use '*'
        
        l1 = Loki(data_path=self.dir_lokiprob, output_path=self.dir_lokiout, db_path=self.dir_tt, hdr_filename=self.tt_hdr_filename, mode='locator')
        l1.location(extension, comp, self.tt_precision, **inputs)
        gc.collect()
        print('MALMI_migration complete!')
        return
    
        
    def rsprocess_view(self, getMLpick=True, PLT={}):
        """
        Visualize some results.
        
        Parameters
        ----------
        getMLpick : boolen, optional
            whether to extract ML picks according to theretical arrivaltimes.
            The default value is True.
        PLT : dict, optional
            contains parameters to plot the seismic waveforms (overlapped with ML probabilites 
            and theoretical arrivaltimes) of each event.
            The default value is {}. If it is None, then not plotting waveforms.
            PLT['component'] : list of str, specify the seismic data components to be plotted,
                               e.g. ['Z', 'N', 'E'], ['Z', '1', '2'], ['Z', 'E'];
        
        Returns
        -------
        None.

        """
        
        from ioformatting import read_arrivaltimes, get_MLpicks_ftheart
        from utils_plot import seischar_plot
        
        print('MALMI starts to post-process, visualize the outputted results:')
    
        # obtain the data folder name of each event, each folder contain the results for a particular event
        evdir = sorted([fdname for fdname in os.listdir(self.dir_lokiout) if os.path.isdir(os.path.join(self.dir_lokiout, fdname))])
    
        # loop over each event, extract the ML picks and visualize the data
        for iefd in evdir:
            # get the input and output foldername for each event
            dir_seis_ev = os.path.join(self.dir_lokiseis, iefd)  # seismic data folder of the current event
            dir_prob_ev = os.path.join(self.dir_lokiprob, iefd)  # ML probability folder of the current event
            dir_output_ev = os.path.join(self.dir_lokiout, iefd)  # migration results folder of the current event
            
            # extract the ML picks according to theoretical arrivaltimes
            if getMLpick:
                snr_para = {}
                snr_para['fband'] = self.seismic['freqband']
                get_MLpicks_ftheart(dir_prob=dir_prob_ev, dir_io=dir_output_ev, maxtd_p=1.3, maxtd_s=1.3, 
                                    P_thrd=self.detect['P_thrd'], S_thrd=self.detect['S_thrd'], 
                                    thephase_ftage='.phs', ofname=None, dir_seis=dir_seis_ev, snr_para=snr_para)
            
            # plot waveforms overlapped with ML probabilites and theoretical arrivaltimes
            if PLT is not None:
                file_arrvt_list = glob.glob(dir_output_ev+'/*.phs', recursive=True)
                if len(file_arrvt_list) == 1:
                    file_arrvt = file_arrvt_list[0]
                    arrvtt = read_arrivaltimes(file_arrvt)
                else:
                    arrvtt = None
                if 'component' in PLT:
                    comp = PLT['component']
                else:
                    comp = None
                seischar_plot(dir_seis=dir_seis_ev, dir_char=dir_prob_ev, dir_output=dir_output_ev, 
                              figsize=(12, 12), comp=comp, dyy=1.8, fband=self.freqband, 
                              normv=self.MIG['probthrd'], ppower=self.MIG['ppower'], tag=None, staname=None, 
                              arrvtt=arrvtt, timerg=None, dpi=300, figfmt='png', process=None, plotthrd=0.001, 
                              linewd=1.5, problabel=False, yticks='auto', ampscale=1.0)
    
        gc.collect()
        print('MALMI_rsprocess_view complete!')
        return

    
    def clear_interm(self, CL={}):
        """
        Clear some gegerated data set such as seismic data or porbability data
        for saving disk space.
        Parameters
        ----------
        CL : dict, parameters.
        CL['mseed'] : boolean
            whether to delete the mseed directory which are the formateed 
            continuous seismic data set for ML inputs.
        CL['hdf5_seis'] : boolean
            whether to delete the hdf5 directory which are formatted overlapping 
            data segments of seismic data set for ML predictions.
        CL['hdf5_prob'] : boolean
            whether to delete the continuous probability hdf5 files.
        CL['migration_volume'] : boolen
            whether to delete the final migration volume of each event.
        """
        
        if CL is None:
            CL = {}
            
        if 'mseed' not in CL:
            CL['mseed'] = True
            
        if 'hdf5_seis' not in CL:
            CL['hdf5_seis'] = True
            
        if 'hdf5_prob' not in CL:
            CL['hdf5_prob'] = True
            
        if 'migration_volume' not in CL:
            CL['migration_volume'] = False
        
        print('MALMI starts to clear some intermediate results for saving disk space:')
        if CL['mseed']:
            try:
                shutil.rmtree(self.dir_mseed)  # remove the mseed directory which are the formated continuous seismic data set for ML inputs
            except Exception as emsg:
                print(emsg)
        
        if CL['hdf5_seis']:
            try:
                shutil.rmtree(self.dir_hdf5)  # remove the hdf5 directory which are formatted overlapping data segments of seismic data set for ML_EQT predictions
            except Exception as emsg:
                print(emsg)
        
        if CL['hdf5_prob']:
            # remove the generated continuous probability hdf5 files
            prob_h5files = glob.glob(self.dir_prob + '/**/*.hdf5', recursive=True)
            for ipf in prob_h5files:
                os.remove(ipf)
        
        if CL['migration_volume']:
            # remove the final migration volume of each event
            migration_volume_npy = glob.glob(self.dir_lokiout + '/**/corrmatrix_*.npy', recursive=True)
            for imigf in migration_volume_npy:
                os.remove(imigf)
        
        gc.collect()        
        print('MALMI_clear_interm complete!')
        return
    
        
    def phase_associate(self, ASSO=None):
        """
        Associate the picked phases for event location.

        Parameters
        ----------
        ASSO : dict, optional
            phase association parameters. The default is None.
            ASSO['engine'] : str, can be 'EQT'
                the phase associate approach, default is 'EQT'.
            ASSO['window'] : float, default is 15.
                The length of time window used for association in second.
            ASSO['Nmin'] : int, default is 3.
                The minimum number of stations used for the association.
            ASSO['starttime'] : datetime.datetime, default is None.
                Start of a time period of interest.
            ASSO['endtime'] : datetime.datetime, default is None.
                End of a time period of interest.
        Returns
        -------
        None.

        """
        
        dtfmt = '%Y-%m-%d %H:%M:%S.%f'  # to produce 'YYYY-MM-DD hh:mm:ss.f'
        
        if ASSO is None:
            ASSO = {}
        
        if 'engine' not in ASSO:
            ASSO['engine'] = 'EQT'
            
        if 'window' not in ASSO:
            ASSO['window'] = 15
        
        if 'Nmin' not in ASSO:
            ASSO['Nmin'] = 3
        
        if ('starttime' not in ASSO) or (ASSO['starttime'] is None):
            starttime = (datetime.datetime.combine(self.seisdate, datetime.time.min)).strftime(dtfmt)
            
        if ('endtime' not in ASSO) or (ASSO['endtime'] is None):
            endtime = (datetime.datetime.combine(self.seisdate, datetime.time.max)).strftime(dtfmt)
        
        self.dir_asso = os.path.join(self.dir_ML, 'association')  # output directory for phase association outputs    
        os.makedirs(self.dir_asso)
        
        if ASSO['engine'] == 'EQT':
            from EQTransformer.utils.associator import run_associator as associator
            associator(input_dir=self.dir_prob, output_dir=self.dir_asso,
                       start_time= starttime, end_time=endtime,  
                       moving_window=ASSO['window'], pair_n=ASSO['Nmin'])
        
        return


    def event_location(self, LOC):
        """
        Perform event location from phase association results.

        Parameters
        ----------
        LOC : dict
            Parameters regarding to event location.
            LOC['engine'] : str, default is 'NonLinLoc'
                the event location approach, 
                can be: 'NonLinLoc'.
            LOC['dir_tt'] : str, default is None.
                the directory of traveltime date set for event location.
                If None, using the default traveltime directory of MALMI.

        Returns
        -------
        None.

        """
        
        
        from obspy.core.event import read_events
        
        if 'engine' not in LOC:
            LOC['engine'] = 'NonLinLoc'
            
        if 'dir_tt' not in LOC:
            LOC['dir_tt'] = None
        
        self.dir_evloc = os.path.join(self.dir_ML, 'event_location')  # output directory for event locations   
        
        if LOC['engine'] == 'NonLinLoc':
            
            # prepare the input picking files for NonLinLoc
            self.dir_NLLocobs = os.path.join(self.dir_evloc, 'NLLocobs')
            os.makedirs(self.dir_NLLocobs)
            
            # read the phase association results
            file_association = os.path.join(self.dir_asso, 'associations.xml')
            pickcatlog = read_events(file_association)
            
            # output the NonLinLoc Phase file format 
            filename_obs = os.path.join(self.dir_NLLocobs, 'NLLoc.obs')
            pickcatlog.write(filename_obs, format="NLLOC_OBS")
        
            # generate NonLinLoc InputControlFile
            if LOC['dir_tt'] == None:
                dir_tt = self.dir_tt
                #self.grid
            else:
                dir_tt = LOC['dir_tt']
                
        
        return
    
    
    def get_catalog(self, CAT={}):
        """
        Get earthquake catalog from the processing results.

        Parameters
        ----------
        CAT : dict
            parameters controlling the process of retriving catalog.
            CAT['dir_dateset'] : str
                database directory, i.e. the path to the parent folder of MALMI catalog and phase file.
                (corresponding to the 'dir_MIG' folder in the MALMI main.py script).
            CAT['cata_fold'] : str,
                Catalog-file parent folder name.
                (corresponding to the 'fld_migresult' folder in the MALMI main.py script).
            CAT['dete_fold'] : str
                Detection-file parent folder name.
                (corresponding to the 'fld_prob' folder in the MALMI main.py script).
            CAT['extract'] : str, default is None.
                The filename including path of the catalog to load.
                If None, will extrace catalog from MALMI processing result database 'CAT['dir_dateset']'.
                If a str will load existing extracted catalog.
            CAT['search_fold'] : list of str, optional
                The MALMI result folders which contains catalog files.
                Will search catalog files from this fold list.
                (This corresponds to the 'fd_seismic' folder).
                The default is None, which means all avaliable folders in 'dir_dateset'.
            CAT['dir_output'] : str
                directory for outputting catalogs.
                default value is related to the current project directory.
            CAT['fname'] : str
                the output catalog filename.
                The default is 'MALMI_catalog_original'.
            CAT['fformat'] : str
                the format of the output catalog file, can be 'pickle' and 'csv'.
                The default is 'pickle'.
            CAT['rmrpev'] : dict, 
                whether to remove the repeated events in the catalog, and the related parameters;
                set to None or False if you do not want to remove.
            CAT['evselect'] : dict, default is None
                parameters controlling the selection of events from original catalog,
                i.e. quality control of the orgiginal catalog.
                type 1:
                    CAT['evselect']['thrd_cmax'] : float, required
                        threshold of minimal coherence, e.g. 0.036
                    CAT['evselect']['thrd_cstd'] : float
                        threshold of maximum standard variance of stacking volume, e.g. 0.119
                    CAT['evselect']['thrd_stanum'] : int, default is None
                        threshold of minimal number of triggered stations.
                    CAT['evselect']['thrd_phsnum'] : int, default is None
                        threshold of minimal number of triggered phases.
                    CAT['evselect']['thrd_llbd'] : float, default is 0.002    
                        lat/lon in degree for excluding migration boundary.
                        e.g. latitude boundary is [lat_min, lat_max], event coordinates 
                        must then within [lat_min+thrd_llbd, lat_max-thrd_llbd].
                    CAT['evselect']['latitude'] : list of float
                        threshold of latitude range in degree, e.g. [63.88, 64.14].
                        default values are determined by 'thrd_llbd' and 'mgregion'.
                    CAT['evselect']['longitude'] : list of float
                        threshold of longitude range in degree, e.g. [-21.67, -21.06].
                        default values are determined by 'thrd_llbd' and 'mgregion'.
                    CAT['evselect']['thrd_depth'] : list of float, default is None
                        threshold of depth range in km, e.g. [-1, 12].
                type 2: 
                    key values keep consistent with catalog dict, selecting event
                    according to the setted lower and higher bounday. For example:
                    CAT['evselect']['coherence_max']=[0.05, np.inf]: events with stacked coherency >= 0.05;
                    CAT['evselect']['asso_station_all']=[3, np.inf]: events with associated stations >= 3;
                    CAT['evselect']['asso_station_PS']=[2, np.inf]: events with associated stations having both P and S phases >= 2;
                    CAT['evselect']['asso_phase_all']=[4, np.inf]: events with associated phases >= 4;
                    CAT['evselect']['rms_pickarvt']=[-np.inf, 1.5]: events with rms error (between picked arrivaltime and theoretical arrivaltime) <= 1.5 second;
                    CAT['evselect']['mae_pickarvt']=[-np.inf, 1.5]: events with mean-absolute-error (between picked arrivaltime and theoretical arrivaltime) <= 1.5 second;
                    CAT['evselect']['latitude']=[10, 11]: events within latitude range between 10 and 11, can autolatically set using CAT['evselect']['thrd_llbd'];
                    CAT['evselect']['longitude']=[50, 51.2]: events within longitude range between 50 and 51.2, can autolatically set using CAT['evselect']['thrd_llbd'];
                    CAT['evselect']['depth_km']=[0, 10]: events within depth range between 0 and 10 km;

        Returns
        -------
        catalog : dict
            original earthquake catalog.

        """
        
        from xcatalog import retrive_catalog_from_MALMI_database
        
        if 'extract' not in CAT:
            CAT['extract'] = None
        
        if 'search_fold' not in CAT:
            CAT['search_fold'] = None
        
        if 'dir_output' not in CAT:
            CAT['dir_output'] = os.path.join(self.dir_projoutput, 'catalogs')
            
        if 'fname' not in CAT:
            CAT['fname'] = 'MALMI_catalog_original'
            
        if 'fformat' not in CAT:
            CAT['fformat'] = 'pickle'
        
        if ('rmrpev' not in CAT) or (CAT['rmrpev'] is True):
            CAT['rmrpev'] = {}
        
        if isinstance(CAT['rmrpev'], dict):
            if 'thrd_time' not in CAT['rmrpev']:
                CAT['rmrpev']['thrd_time'] = 0.5
            
            if 'thrd_hdis' not in CAT['rmrpev']:
                CAT['rmrpev']['thrd_hdis'] = 5
             
            if 'thrd_depth' not in CAT['rmrpev']:
                CAT['rmrpev']['thrd_depth'] = 5
        
        if 'evselect' not in CAT:
            CAT['evselect'] = None
        
        if (CAT['evselect'] is not None) and ('thrd_llbd' not in CAT['evselect']):
            CAT['evselect']['thrd_llbd'] = 0.002
        
        if (CAT['evselect'] is not None) and ('latitude' not in CAT['evselect']):
            if (CAT['evselect']['thrd_llbd'] is not None):    
                CAT['evselect']['latitude'] = [self.grid['mgregion'][2]+CAT['evselect']['thrd_llbd'], self.grid['mgregion'][3]-CAT['evselect']['thrd_llbd']]
            else:
                CAT['evselect']['latitude'] = None
        
        if (CAT['evselect'] is not None) and ('longitude' not in CAT['evselect']):
            if (CAT['evselect']['thrd_llbd'] is not None):
                CAT['evselect']['longitude'] = [self.grid['mgregion'][0]+CAT['evselect']['thrd_llbd'], self.grid['mgregion'][1]-CAT['evselect']['thrd_llbd']]    
            else:
                CAT['evselect']['longitude'] = None
        
        if (CAT['evselect'] is not None) and ('thrd_cmax' in CAT['evselect']) and ('thrd_cstd' not in CAT['evselect']):
            CAT['evselect']['thrd_cstd'] = None
        
        if (CAT['evselect'] is not None) and ('thrd_cmax' in CAT['evselect']) and ('thrd_stanum' not in CAT['evselect']):
                CAT['evselect']['thrd_stanum'] = None
            
        if (CAT['evselect'] is not None) and ('thrd_cmax' in CAT['evselect']) and ('thrd_phsnum' not in CAT['evselect']):    
            CAT['evselect']['thrd_phsnum'] = None    
        
        if (CAT['evselect'] is not None) and ('thrd_cmax' in CAT['evselect']) and ('thrd_depth' not in CAT['evselect']):    
            CAT['evselect']['thrd_depth'] = None
        
        if 'dir_dateset' not in CAT:
            CAT['dir_dateset'] = self.dir_MIG
        
        if 'cata_fold' not in CAT:
            CAT['cata_fold'] = self.fld_migresult
        
        if 'dete_fold' not in CAT:
            CAT['dete_fold'] = self.fld_prob
        
        if 'evidtag' not in CAT:
            CAT['evidtag'] = 'malmi'
        
        catalog = retrive_catalog_from_MALMI_database(CAT)
        
        return catalog
    
    
    def catalog_relocate(self, RELOC={}):
        """
        Obtain relative relocated catalog.

        Parameters
        ----------
        RELOC : dict
            parameters related to event relocation.
            RELOC['catalog'] : str or obspy catalog object or dict.
                the input catalog that need to be relocated; 
                str type: the filename of the catalog;
                obspy catalog object: the catalog of obspy format;
                dict type: the catalog of python dictory format;
                Default is the filename of the quality-controlled catalog.
            RELOC['engine'] : str, default is 'rtdd'
                the event relative relocation approach, 
                can be: 'rtdd'.
            RELOC['dir_output'] : str
                directory for outputting files.
                default value is related to the current project directory.

        Returns
        -------
        None.

        """
        
        from xevrelocation import event_reloc
        
        if 'catalog' not in RELOC:
            RELOC['catalog'] = os.path.join(self.dir_projoutput, 'catalogs', 'MALMI_catalog_QC.pickle')
            
        if 'engine' not in RELOC:
            RELOC['engine'] = 'rtdd'
        
        if 'dir_output' not in RELOC:
            RELOC['dir_output'] = os.path.join(self.dir_projoutput, 'catalogs')
        
        # load the catalog to be relocated
        if isinstance(RELOC['catalog'], str):
            # input is the filename of the catalog
            RELOC['catalog'] = load_catalog(RELOC['catalog'], outformat='dict')
        elif isinstance(RELOC['catalog'], dict):
            # input is the catalog of simple dict format
           pass
        elif isinstance(RELOC['catalog'], obspy.core.event.Catalog):
            # input is the catalog of obspy catalog object
            RELOC['catalog'] = catalog2dict(RELOC['catalog'])
        else:
            raise ValueError('Wrong input for MAGNI[\'catalog\']!')
        
        RELOC['stainv'] = self.stainv

        if 'channel_codes' not in RELOC:
            RELOC['channel_codes'] = ['HHE', 'HHN', 'HHZ']
        
        event_reloc(RELOC)
        
        return
    
    
    def para_testing(self):
        """
        Test various parameters such as triggered station number and phase number
        for running MALMI on continuous data.
        Tests are usually perform on smaller dataset such as one day's data,
        once parameters are optimised and determined through testing, they 
        can be directly applied to continous data.

        Returns
        -------
        None.

        """
        
        from xparameters import staphs_trigger_ana
        
        # statistical analysis of the number of station and phase triggered
        file_detection = os.path.join(self.dir_lokiprob, 'event_station_phase_info.txt')
        dir_out = os.path.join(self.dir_migration, 'tested_figure')
        if not os.path.exists(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        staphs_trigger_ana(file_detection=file_detection, dir_out=dir_out)
        
        return
    
    
    def estmate_magnitue(self, MAGNI):
        """
        Determine the magnitude for events in the input catalog.

        Parameters
        ----------
        MAGNI : dict
            contains input parameters.
            MAGNI['engine'] : str
                specify the method for determining event magnitude; can be:
                'relative' : relative amplitude ratio method, will need another 
                             reference catalog (at least have one matched event) 
                             which have magnitude information.
            MAGNI['catalog'] : str or obspy catalog object or dict.
                            the input catalog that need magnitude determination; 
                            str type: the filename of the catalog;
                            obspy catalog object: the catalog of obspy format;
                            dict type: the catalog of python dictory format; 
            MAGNI['catalog_ref'] : str or obspy catalog object or dict.
                            the input reference catalog that have magnitude determination; 
                            str type: the filename of the catalog;
                            obspy catalog object: the catalog of obspy format;
                            dict type: the catalog of python dictory format;
            MAGNI['stations'] : str or obspy station inventory object or dict, optional;
                            the station inventory; 
                            str type: the filename of the station inventory;
                            obspy inventory object: the inventory of obspy format;
                            dict type: the inventory of python dictory format;
            other parameters check 'estimate_magnitude' for detail;
        Returns
        -------
        None.

        """
        
        from xmagnitude import estimate_magnitude
        from xstation import stainv2stadict
        
        MAG = {}
        
        if 'engine' not in MAGNI:
            MAG['engine'] = 'relative'
        else:
            MAG['engine'] = MAGNI['engine']
        
        if MAG['engine'] == 'relative':
            # input catalog
            if isinstance(MAGNI['catalog'], str):
                # input is the filename of the catalog
                MAG['catalog'] = load_catalog(MAGNI['catalog'], outformat='dict')
            elif isinstance(MAGNI['catalog'], dict):
                # input is the catalog of simple dict format
                MAG['catalog'] = MAGNI['catalog']
            elif isinstance(MAGNI['catalog'], obspy.core.event.Catalog):
                # input is the catalog of obspy catalog object
                MAG['catalog'] = catalog2dict(MAGNI['catalog'])
            else:
                raise ValueError('Wrong input for MAGNI[\'catalog\']!')
        
            # reference catalog
            if isinstance(MAGNI['catalog_ref'], str):
                # input is the filename
                MAG['catalog_ref'] = load_catalog(MAGNI['catalog_ref'], outformat='dict')
            elif isinstance(MAGNI['catalog_ref'], dict):
                # input is the catalog of simple dict format
                MAG['catalog_ref'] = MAGNI['catalog_ref']
            elif isinstance(MAGNI['catalog_ref'], obspy.core.event.Catalog):
                # input is the catalog of obspy catalog object
                MAG['catalog_ref'] = catalog2dict(MAGNI['catalog_ref'])
            else:
                raise ValueError('Wrong input for MAGNI[\'catalog_ref\']!')
        
            # station information
            if 'stations' not in MAGNI:
                MAG['stations'] = stainv2stadict(self.stainv)
            else:
                if isinstance(MAGNI['stations'], str):
                    # input is the filename
                    MAG['stations'] = load_station(MAGNI['stations'], outformat='dict')
                elif isinstance(MAGNI['stations'], dict):
                    # input of simple dict format
                    MAG['stations'] = MAGNI['stations']
                elif isinstance(MAGNI['stations'], obspy.core.inventory.inventory.Inventory):
                    # input is of obspy inventory object
                    MAG['stations'] = stainv2stadict(MAGNI['stations'])
                else:
                    raise ValueError('Wrong input for MAGNI[\'stations\']!')
        
            if 'match_thrd_time' not in MAGNI:
                MAG['match_thrd_time'] = 2.5
            else:
                MAG['match_thrd_time'] = MAGNI['match_thrd_time']
        
            if 'mgcalpara' not in MAGNI:
                MAG['mgcalpara'] = None
            else:    
                MAG['mgcalpara'] = MAGNI['mgcalpara']
        
        catalog_new = estimate_magnitude(MAGNI=MAG)
        
        return catalog_new


