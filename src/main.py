#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:41:56 2021

@author: Peidong Shi
@email: speedshi@hotmail.com

MALMI main function - building a whole framework.
Coordinate convention: X-axis -> East; Y-axis -> North; Z-axis -> Depth (vertical-down).
"""


import os
import gc
import copy
import shutil
import glob
import datetime
        

class MALMI:

    def __init__(self, seismic, tt, grid=None, control=None, detect=None, MIG=None):
        """
        Initilize global input and output paramaters, configure MALMI.
        Parameters
        ----------
        seismic parameters (For input seismic data set):
            seismic['dir']: str, required.
                path to raw continuous seismic data.
            seismic['channels']: list of str, optional, default: ["*HE", "*HN", "*HZ"].
                the used channels of the input seismic data.
            seismic['datastru']: str, optional, default: 'AIO'.
                the input seismic data file structure.
                'AIO' : continuous seismic data are organized All In One folder;
                'SDS' : continuous seismic data are organized in SeisComP Data Structure.
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
        
        traveltime parameters (For loading or generating traveltime tables):
            tt['vmodel']: str, optional, default is None.
                filename including path of the velocity model.    
                None means directly load existing traveltime table of NonLinLoc format.
            tt['dir']: str, optional, default is './data/traveltime'.
                path to travetime data directory or for saving generated traveltime tabels.
            tt['ftage']: str, optional, default is 'layer'. 
                traveltime data set filename tage as used for NonLinLoc,
                used at the file root name.
        
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
            grid['zOrig']: Z location of the grid origin in km relative to the 
                           sea-level. Nagative value means above the sea-level; 
                           Positive values for below the sea-level;
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

        Returns
        -------
        None.

        """
        
        from traveltime import build_traveltime
        from ioformatting import read_stationinfo
        from xcoordinate import get_lokicoord
        from utils_plot import plot_basemap
        
        # set default parameters----------------------------------------------- 
        if 'channels' not in seismic:
            seismic['channels'] = ["*HE", "*HN", "*HZ"]
        
        if 'datastru' not in seismic:
            seismic['datastru'] = 'AIO'
        
        if 'date' not in seismic:
            seismic['date'] = None
        
        if 'freqband' not in seismic:
            seismic['freqband'] = None
        
        if 'vmodel' not in tt:
            tt['vmodel'] = None
        
        if 'dir' not in tt:
            tt['dir'] = './data/traveltime'
        
        if 'ftage' not in tt:
            tt['ftage'] = 'layer'
        
        tt['hdr_filename'] = 'header.hdr'  # travetime data set header filename
        
        if (grid is not None) and ('rotAngle' not in grid):
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
        
        if 'outseis' not in detect:
            detect['outseis'] = True
        
        if 'output_allsta' not in detect:
            detect['output_allsta'] = True
            
        if MIG is None:
            MIG = {}
            
        if 'probthrd' not in MIG:
            MIG['probthrd'] = 0.001
        
        if 'ppower' not in MIG:
            MIG['ppower'] = 4
        #----------------------------------------------------------------------
        
        # make sure output directory exist
        if not os.path.exists(control['dir_output']):
            os.makedirs(control['dir_output'], exist_ok=True)
        
        # read in station invertory and obtain station information
        self.stainv = read_stationinfo(seismic['stainvf'])  # obspy station inventory 
        
        # build travel-time data set
        grid = build_traveltime(grid, tt, self.stainv)
        self.grid = copy.deepcopy(grid)
        sta_inv1, region1, mgregion = get_lokicoord(tt['dir'], tt['hdr_filename'], 0.05, consider_mgregion=True)  # obtain migration region lon/lat range
        self.grid['mgregion'] = mgregion  # migration region lat/lon range, [lon_min, lon_max, lat_min, lat_max] in degree
        
        # plot stations and migration region for checking
        if control['plot_map']:
            fname1 = os.path.join(control['dir_output'], "basemap_stations_mgarea.png")
            if not os.path.isfile(fname1):
                plot_basemap(region1, sta_inv1, mgregion, fname1, False, '30s')
        
        self.seisdatastru = copy.deepcopy(seismic['datastru'])
        self.dir_seismic = copy.deepcopy(seismic['dir'])
        self.seismic_channels = copy.deepcopy(seismic['channels'])  # the channels of the input seismic data
        self.seisdate = copy.deepcopy(seismic['date'])  # date of seismic data
        self.freqband = copy.deepcopy(seismic['freqband'])  # frequency band in Hz for filtering seismic data
        
        if self.seisdatastru == 'AIO':
            # input seismic data files are stored simply in one folder
            # get the foldername of the input seismic data, used as the identifer of the input data set
            if self.dir_seismic[-1] == '/':
                fd_seismic = self.dir_seismic.split('/')[-2]
            else:
                fd_seismic = self.dir_seismic.split('/')[-1]
        elif self.seisdatastru == 'SDS':
            # input seismic data files are organized in SDS
            # use the date as the identifer of the input data set
            fd_seismic = datetime.datetime.strftime(self.seisdate, "%Y%m%d")
        else:
            raise ValueError('Unrecognized input for: seisdatastru! Can\'t determine the structure of the input seismic data files!')
        
        self.dir_projoutput = control['dir_output']  # project output direcotroy
        self.dir_CML = os.path.join(self.dir_projoutput, 'data_ML')  # directory for common ML outputs
        self.dir_MIG = os.path.join(self.dir_projoutput, 'data_MIG')  # directory for common migration outputs
        
        self.dir_ML = os.path.join(self.dir_CML, fd_seismic)  # directory for ML outputs of the currrent processing dataset
        self.dir_prob = os.path.join(self.dir_ML, 'prob_and_detection')  # output directory for ML probability outputs
        self.dir_migration = os.path.join(self.dir_MIG, fd_seismic)  # directory for migration outputs of the current processing dataset
        self.n_processor = copy.deepcopy(control['n_processor'])  # number of threads for parallel processing
        
        self.dir_tt = copy.deepcopy(tt['dir'])  # path to travetime data set
        self.tt_precision = 'single'  # persicion for traveltime data set, 'single' or 'double'
        self.tt_hdr_filename = copy.deepcopy(tt['hdr_filename'])  # travetime data set header filename
        self.tt_ftage = copy.deepcopy(tt['ftage'])  # traveltime data set filename tage
        if self.dir_tt.split('/')[-1] == '':
            tt_folder = self.dir_tt.split('/')[-2]
        else:
            tt_folder = self.dir_tt.split('/')[-1]

        self.dir_mseed = os.path.join(self.dir_ML, 'mseeds')  # directory for outputting seismic data for EQT, NOTE do not add '/' at the last part
        self.dir_hdf5 = self.dir_mseed + '_processed_hdfs'  # path to the hdf5 and csv files
        self.dir_EQTjson = os.path.join(self.dir_ML, 'json')  # directory for outputting station json file for EQT
        self.fld_prob = 'prob_evstream'  # foldername of the probability outputs of different events
        self.dir_lokiprob = os.path.join(self.dir_migration, self.fld_prob)  # directory for probability outputs of different events in SEED format
        self.dir_lokiseis = os.path.join(self.dir_migration, 'seis_evstream')  # directory for raw seismic outputs of different events in SEED format
        self.fld_migresult = 'result_MLprob_{}'.format(tt_folder)  # foldername of the final migration results
        self.dir_lokiout = os.path.join(self.dir_migration, self.fld_migresult)  # path for loki final outputs
        
        # create output directories
        if not os.path.exists(self.dir_lokiout):
            os.makedirs(self.dir_lokiout, exist_ok=True)
        
        self.detect = detect.copy()  # detection parameters
        self.MIG = MIG.copy()  # migration parameters
        
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
        
        from ioformatting import stainv2json
        from ioseisdata import format_AIO, format_SDS
        
        print('MALMI starts to format input data set for ML models:')
        
        if self.seisdatastru == "AIO":
            # input seismic data files are stored simply in one folder
            # suitable for formatting small data set
            seisdate = format_AIO(dir_seismic=self.dir_seismic, seismic_channels=self.seismic_channels, 
                                  dir_output=self.dir_mseed, freqband=self.freqband)
            if self.seisdate is None:
                # retrive date of seismic data, NOTE might not be accurate 
                # and if data longer than one day, this parameters is 
                self.seisdate = seisdate
            
        elif self.seisdatastru == 'SDS':
            # input seismic data files are organized in SDS
            # suitable for formatting large or long-duration data set
            format_SDS(seisdate=self.seisdate, stainv=self.stainv, 
                       dir_seismic=self.dir_seismic, seismic_channels=self.seismic_channels, 
                       dir_output=self.dir_mseed, freqband=self.freqband)
            
        else:
            raise ValueError('Unrecognized input for: seisdatastru! Can\'t determine the structure of the input seismic data files!')
        
        # create station jason file for EQT------------------------------------
        stainv2json(self.stainv, self.dir_mseed, self.dir_EQTjson)
        gc.collect()
        print('MALMI_format_ML_inputs complete!')
        return


    def generate_prob(self, ML):
        """
        Generate event and phase probabilities using ML models.
        Parameters
        ----------
        ML : dict, machine-learning related parameters.
        ML['model'] : str
            path to a trained ML model.
        ML['overlap'] : float, default: 0.5
            overlap rate of time window for generating probabilities. 
            e.g. 0.6 means 60% of time window are overlapped.
        ML['number_of_cpus'] : int, default: 5
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
        if 'overlap' not in ML:
            ML['overlap'] = 0.5
        
        if 'number_of_cpus' not in ML:
            ML['number_of_cpus'] = 4
        #----------------------------------------------------------------------
        
        from EQTransformer.utils.hdf5_maker import preprocessor
        from EQTransformer.utils.plot import plot_data_chart
        from EQTransformer.core.predictor import predictor
        
        print('MALMI starts to generate event and phase probabilities using ML models:')
        # create hdf5 data for EQT inputs--------------------------------------
        stations_json = self.dir_EQTjson + "/station_list.json"  # station JSON file
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
                  detection_threshold=0.1, P_threshold=0.1, S_threshold=0.1, 
                  keepPS=False, number_of_cpus=ML['number_of_cpus'],
                  number_of_plots=100, plot_mode='time_frequency')
        gc.collect()
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
        
        # from event_detection import eqt_arrayeventdetect
        from event_detection import eqt_eventdetectfprob, arrayeventdetect
        from utils_dataprocess import maxP2Stt
        
        print('MALMI starts to detect events based on the ML predicted phase probabilites and output the corresponding phase probabilites of the detected events:')
        # eqt_arrayeventdetect(self.dir_prob, self.dir_lokiprob, sttd_max, twlex, d_thrd, nsta_thrd, spttdf_ssmax)
        event_info = eqt_eventdetectfprob(self.dir_prob, self.detect['P_thrd'], self.detect['S_thrd'])
        gc.collect()
        if self.detect['twind_srch'] is None:
            self.detect['twind_srch'], _, _ = maxP2Stt(self.dir_tt, self.tt_hdr_filename, self.tt_ftage, self.tt_precision)
            
        if self.detect['outseis']:
            dir_seisdataset = self.dir_mseed
        else:
            dir_seisdataset = None
        arrayeventdetect(event_info, self.detect['twind_srch'], self.detect['twlex'], 
                         self.detect['nsta_thrd'], self.detect['npha_thrd'], 
                         self.dir_lokiprob, self.dir_lokiseis, dir_seisdataset, self.seismic_channels, self.detect['output_allsta'])
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
        comp = ['P','S']  # when input data are probabilities of P- and S-picks, comp must be ['P', 'S']
        extension = '*'  # seismic data filename for loading, accept wildcard input, for all data use '*'
        
        l1 = Loki(self.dir_lokiprob, self.dir_lokiout, self.dir_tt, self.tt_hdr_filename, mode='locator')
        l1.location(extension, comp, self.tt_precision, **inputs)
        gc.collect()
        print('MALMI_migration complete!')
        return
    
        
    def rsprocess_view(self, getMLpick=True, plotwaveforms=True):
        """
        Visualize some results.
        
        Parameters
        ----------
        getMLpick : boolen, optional
            whether to extract ML picks according to theretical arrivaltimes.
            The default value is True.
        plotwaveforms : boolen, optional
            whether to plot the seismic waveforms (overlapped with ML probabilites 
            and theoretical arrivaltimes) of each event.
            The default value is True.
        
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
                get_MLpicks_ftheart(dir_prob_ev, dir_output_ev, maxtd_p=3.0, maxtd_s=3.0, 
                                    P_thrd=self.detect['P_thrd'], S_thrd=self.detect['S_thrd'], 
                                    thephase_ftage='.phs', ofname=None)
            
            # plot waveforms overlapped with ML probabilites and theoretical arrivaltimes
            if plotwaveforms:
                file_arrvt_list = glob.glob(dir_output_ev+'/*.phs', recursive=True)
                if len(file_arrvt_list) == 1:
                    file_arrvt = file_arrvt_list[0]
                    arrvtt = read_arrivaltimes(file_arrvt)
                else:
                    arrvtt = None
                seischar_plot(dir_seis_ev, dir_prob_ev, dir_output_ev, figsize=(12, 12), 
                              comp=['Z','N','E'], dyy=1.8, fband=[2, 30], normv=self.MIG['probthrd'], 
                              ppower=self.MIG['ppower'], tag=None, staname=None, arrvtt=arrvtt)
    
        gc.collect()
        print('MALMI_rsprocess_view complete!')
        return

    
    def clear_interm(self, CL=None):
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
        """
        
        if CL is None:
            CL = {}
            
        if 'mseed' not in CL:
            CL['mseed'] = True
            
        if 'hdf5_seis' not in CL:
            CL['hdf5_seis'] = True
            
        if 'hdf5_prob' not in CL:
            CL['hdf5_prob'] = True
        
        print('MALMI starts to clear some intermediate results for saving disk space:')
        if CL['mseed']:
            shutil.rmtree(self.dir_mseed)  # remove the mseed directory which are the formated continuous seismic data set for ML inputs
        
        if CL['hdf5_seis']:
            shutil.rmtree(self.dir_hdf5)  # remove the hdf5 directory which are formatted overlapping data segments of seismic data set for ML_EQT predictions
        
        if CL['hdf5_prob']:
            # remove the generated continuous probability hdf5 files
            prob_h5files = glob.glob(self.dir_prob + '/**/*.hdf5', recursive=True)
            for ipf in prob_h5files:
                os.remove(ipf)
        
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
            LOC['engine'] : str, option: 'NonLinLoc'
                the event location approach, default is 'NonLinLoc'.
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
            CAT['extract'] : boolen, default is 'True'
                whether to extract the catalog from processing results.
            CAT['dir_output'] : str
                directory for outputting catalogs.
                default value is related to the current project directory.
            CAT['fname'] : str
                the output catalog filename.
                The default is 'MALMI_catalog_original.pickle'.
            CAT['evselect'] : dict
                parameters controlling the selection of events from original catalog,
                i.e. quality control of the orgiginal catalog.
                CAT['evselect']['rmrpev'] : boolen, default is 'True'
                    whether to remove the repeated events in the original catalog.
                CAT['evselect']['fname'] : str
                    the output filename of the quality-controlled catalog.
                    The default is 'MALMI_catalog_QC.pickle'.
                CAT['evselect']['thrd_cmax'] : float, default is 0.036
                    threshold of minimal coherence.
                CAT['evselect']['thrd_cstd'] : float, default is 0.119
                    threshold of maximum standard variance of stacking volume.
                CAT['evselect']['thrd_stanum'] : int, default is None
                    threshold of minimal number of triggered stations.
                CAT['evselect']['thrd_phsnum'] : int, default is None
                    threshold of minimal number of triggered phases.
                CAT['evselect']['thrd_llbd'] : float, default is 0.002    
                    lat/lon in degree for excluding migration boundary.
                    e.g. latitude boundary is [lat_min, lat_max], event coordinates 
                    must then within [lat_min+thrd_llbd, lat_max-thrd_llbd].
                CAT['evselect']['thrd_lat'] : list of float
                    threshold of latitude range in degree, e.g. [63.88, 64.14].
                    default values are determined by 'thrd_llbd' and 'mgregion'.
                CAT['evselect']['thrd_lon'] : list of float
                    threshold of longitude range in degree, e.g. [-21.67, -21.06].
                    default values are determined by 'thrd_llbd' and 'mgregion'.
                CAT['evselect']['thrd_depth'] : list of float, default is None
                    threshold of depth range in km, e.g. [-1, 12].
                

        Returns
        -------
        catalog : dict
            original earthquake catalog.

        """
        
        from ioformatting import retrive_catalog
        import pickle
        from catalogs import catalog_rmrpev, catalog_select
        
        if 'extract' not in CAT:
            CAT['extract'] = True
            
        if 'dir_output' not in CAT:
            CAT['dir_output'] = os.path.join(self.dir_projoutput, 'catalogs')
            
        if 'fname' not in CAT:
            CAT['fname'] = 'MALMI_catalog_original.pickle'
            
        if 'evselect' not in CAT:
            CAT['evselect'] = None
            
        if (CAT['evselect'] is not None) and ('rmrpev' not in CAT['evselect']):
            CAT['evselect']['rmrpev'] = True
            
        if (CAT['evselect'] is not None) and ('fname' not in CAT['evselect']):
            CAT['evselect']['fname'] = 'MALMI_catalog_QC.pickle'
            
        if (CAT['evselect'] is not None) and ('thrd_cmax' not in CAT['evselect']):    
            CAT['evselect']['thrd_cmax'] = 0.036  
        
        if (CAT['evselect'] is not None) and ('thrd_cstd' not in CAT['evselect']):
            CAT['evselect']['thrd_cstd'] = 0.119
            
        if (CAT['evselect'] is not None) and ('thrd_stanum' not in CAT['evselect']):
            CAT['evselect']['thrd_stanum'] = None
            
        if (CAT['evselect'] is not None) and ('thrd_phsnum' not in CAT['evselect']):    
            CAT['evselect']['thrd_phsnum'] = None
        
        if (CAT['evselect'] is not None) and ('thrd_llbd' not in CAT['evselect']):
            CAT['evselect']['thrd_llbd'] = 0.002
        
        if (CAT['evselect'] is not None) and ('thrd_lat' not in CAT['evselect']):    
            CAT['evselect']['thrd_lat'] = [self.grid['mgregion'][2]+CAT['evselect']['thrd_llbd'], self.grid['mgregion'][3]-CAT['evselect']['thrd_llbd']]
        
        if (CAT['evselect'] is not None) and ('thrd_lon' not in CAT['evselect']):
            CAT['evselect']['thrd_lon'] = [self.grid['mgregion'][0]+CAT['evselect']['thrd_llbd'], self.grid['mgregion'][1]-CAT['evselect']['thrd_llbd']]    
        
        if (CAT['evselect'] is not None) and ('thrd_depth' not in CAT['evselect']):    
            CAT['evselect']['thrd_depth'] = None
            
        # catalog name
        if not os.path.exists(CAT['dir_output']):
            os.makedirs(CAT['dir_output'], exist_ok=True)
        cfname = os.path.join(CAT['dir_output'], CAT['fname'])
        
        if CAT['extract']:
            # extract catlog from processing results
            catalog = retrive_catalog(dir_dateset=self.dir_MIG, cata_ftag='catalogue', dete_ftag='event_station_phase_info.txt', 
                                      cata_fold=self.fld_migresult, dete_fold=self.fld_prob)
        
            # save the extracted original catalog
            with open(cfname, 'wb') as handle:
                pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # directly load existing catalog
            with open(cfname, 'rb') as handle:
                catalog = pickle.load(handle)
        
        if CAT['evselect'] is not None:
            # select events from the original catalog using quality control parameters
            
            # remove repeated events
            if CAT['evselect']['rmrpev']:
                catalog_QC = catalog_rmrpev(catalog, 0.5, 5, 5, evkp='coherence_max')
            
            # quality control
            catalog_QC = catalog_select(catalog_QC, thrd_cmax=CAT['evselect']['thrd_cmax'], 
                                        thrd_stanum=CAT['evselect']['thrd_stanum'], 
                                        thrd_phsnum=CAT['evselect']['thrd_phsnum'], 
                                        thrd_lat=CAT['evselect']['thrd_lat'], 
                                        thrd_lon=CAT['evselect']['thrd_lon'], 
                                        thrd_cstd=CAT['evselect']['thrd_cstd'], 
                                        thrd_depth=CAT['evselect']['thrd_depth'])
        
            # save the catalog after quality control
            cfname_QC = os.path.join(CAT['dir_output'], CAT['evselect']['fname'])
            with open(cfname_QC, 'wb') as handle:
                pickle.dump(catalog_QC, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return catalog
    
    
    
    
    