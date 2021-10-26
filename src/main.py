#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:41:56 2021

@author: Peidong Shi
@email: speedshi@hotmail.com

MALMI main function - building a whole framework.
"""


import os
import gc
import copy
import shutil
import glob
        

class MALMI:

    def __init__(self, dir_seismic, dir_output, file_station, dir_tt, tt_ftage='layer', n_processor=1):
        """
        Initilize global input and output paramaters, configure MALMI.
        Parameters
        ----------
        dir_seismic : str
            path to raw continuous seismic data.
        dir_output : str
            path for outputs.
        file_station : str
            station metadata file including path. The data format should be 
            recognizable by ObsPy, such as:
                FDSNWS station text format: *.txt,
                FDSNWS StationXML format: *.xml.
        dir_tt : str
            path to travetime data set.
        tt_ftage : str, optional
            traveltime data set filename tage. The default is 'layer'.
        n_processor : int, default: 1
            number of CPU processors for parallel processing.

        Returns
        -------
        None.

        """
        
        self.dir_seismic = copy.deepcopy(dir_seismic)
        
        # get the foldername of the input seismic data, used as the identifer of the input data set
        if self.dir_seismic[-1] == '/':
            fd_seismic = self.dir_seismic.split('/')[-2]
        else:
            fd_seismic = self.dir_seismic.split('/')[-1]
            
        self.dir_ML = dir_output + "/data_EQT/" + fd_seismic  # directory for ML outputs
        self.dir_prob = self.dir_ML + '/prob_and_detection'  # output directory for ML probability outputs
        self.dir_migration = dir_output + '/data_loki/' + fd_seismic  # directory for migration outputs
        self.n_processor = copy.deepcopy(n_processor)  # number of threads for parallel processing
        
        self.dir_tt = copy.deepcopy(dir_tt)  # path to travetime data set
        self.tt_precision = 'single'  # persicion for traveltime data set, 'single' or 'double'
        self.tt_hdr_filename = 'header.hdr'  # travetime data set header filename
        self.tt_ftage = copy.deepcopy(tt_ftage)  # traveltime data set filename tage
        if self.dir_tt.split('/')[-1] == '':
            tt_folder = self.dir_tt.split('/')[-2]
        else:
            tt_folder = self.dir_tt.split('/')[-1]

        self.dir_mseed = self.dir_ML + "/mseeds"  # directory for outputting seismic data for EQT, NOTE do not add '/' at the last part
        self.dir_hdf5 = self.dir_mseed + '_processed_hdfs'  # path to the hdf5 and csv files
        self.dir_EQTjson = self.dir_ML + "/json"  # directory for outputting station json file for EQT
        self.dir_lokiprob = self.dir_migration + '/prob_evstream'  # directory for probability outputs of different events in SEED format
        self.dir_lokiseis = self.dir_migration + '/seis_evstream'  # directory for raw seismic outputs of different events in SEED format
        self.dir_lokiout = self.dir_migration + '/result_MLprob_{}'.format(tt_folder)  # path for loki final outputs
        
        self.file_station = copy.deepcopy(file_station)  # path to the station metadata to get station invertory
        

    def format_ML_inputs(self, seismic_channels=["*HE", "*HN", "*HZ"], seisdatastru='AIOFD'):
        """
        Format input data set for ML models.
        Parameters
        ----------
        seismic_channels : list of str, default: ["*HE", "*HN", "*HZ"]
            specify the channels of the input seismic data.
        seisdatastru : str
            specify the seismic data file structure.
            'AIOFD' : continuous seismic data are organized All In One FolDer;
            'SDS' : continuous seismic data are organized in SeisComP Data Structure.
        
        Returns
        -------
        None.

        """
        
        from ioformatting import read_seismic_fromfd, stream2EQTinput, stainv2json
        
        print('MALMI starts to format input data set for ML models:')
        
        self.seismic_channels = copy.deepcopy(seismic_channels)  # the channels of the input seismic data
        
        # read in all continuous seismic data in the input folder as an obspy stream
        stream = read_seismic_fromfd(self.dir_seismic)
        
        # output to the seismic data format that QET can handle 
        stream2EQTinput(stream, self.dir_mseed, self.seismic_channels)
        del stream
        gc.collect()
        
        # create station jason file for EQT------------------------------------
        stainv2json(self.file_station, self.dir_mseed, self.dir_EQTjson)
        gc.collect()
        print('MALMI_format_ML_inputs complete!')

    
    def generate_prob(self, input_MLmodel, overlap=0.5):
        """
        Generate event and phase probabilities using ML models.
        Parameters
        ----------
        input_MLmodel : str
            path to a trained EQT model.
        overlap : float, default: 0.5
            overlap rate of time window for generating probabilities. 
            e.g. 0.6 means 60% of time window are overlapped.

        Returns
        -------
        None.

        """
        
        from EQTransformer.utils.hdf5_maker import preprocessor
        from EQTransformer.utils.plot import plot_data_chart
        from EQTransformer.core.predictor import predictor
        
        print('MALMI starts to generate event and phase probabilities using ML models:')
        # create hdf5 data for EQT inputs--------------------------------------
        stations_json = self.dir_EQTjson + "/station_list.json"  # station JSON file
        preproc_dir = self.dir_ML + "/preproc_overlap{}".format(overlap)  # path of the directory where will be located the summary files generated by preprocessor step
        # generate hdf5 files
        preprocessor(preproc_dir=preproc_dir, mseed_dir=self.dir_mseed, 
                     stations_json=stations_json, overlap=overlap, 
                     n_processor=1)
        gc.collect()
        
        # show data availablity for each station-------------------------------
        file_pkl = preproc_dir + '/time_tracks.pkl'
        time_interval = 1  # Time interval in hours for tick spaces in xaxes
        plot_data_chart(time_tracks=file_pkl, time_interval=time_interval, dir_output=preproc_dir)
        gc.collect()
        
        # generate event and phase probabilities-------------------------------
        predictor(input_dir=self.dir_hdf5, input_model=input_MLmodel, output_dir=self.dir_prob,
                  output_probabilities=True, estimate_uncertainty=False,
                  detection_threshold=0.1, P_threshold=0.1, S_threshold=0.1, 
                  keepPS=False, number_of_cpus=self.n_processor,
                  number_of_plots=100, plot_mode='time_frequency')
        gc.collect()
        print('MALMI_generate_prob complete!')

            
    def event_detect_ouput(self, twind_srch=None, twlex=1.0, P_thrd=0.1, S_thrd=0.1, nsta_thrd=3, npha_thrd=4, outseis=True):
        """
        event detection based on the ML predicted event probabilites
        and output the corresponding phase probabilites of the detected events.
        Parameters
        ----------
        twind_srch : float, optional
            time window length in second where events will be searched in this range.
            How to determine this parameter:
            Conservative estimation: maximum P-S traveltime difference between 
            different stations for the whole imaging area.
            If None, then automatically determine it from traveltime table.
        twlex : float, optional
            time window length in second for extending the output time range, 
            usually set to be 1-2 second. The default is 1.0 second.
        P_thrd : float, optional
            probability threshold for detecting P-phases/events from the ML-predicted 
            phase probabilities. The default is 0.1.
        S_thrd : float, optional
            probability threshold for detecting S-phases/events from the ML-predicted 
            phase probabilities. The default is 0.1.
        nsta_thrd : int, optional
            minimal number of stations triggered during the specified event time period.
            The default is 3.
        npha_thrd : int, optional
            minimal number of phases triggered during the specified event time period.
            The default is 4.
        outseis : boolen, optional
            whether to output raw seismic data segments for the detectd events.
            The default is True.

        Returns
        -------
        None.

        """
        
        # from event_detection import eqt_arrayeventdetect
        from event_detection import eqt_eventdetectfprob, arrayeventdetect
        from utils_dataprocess import maxP2Stt
        
        self.P_thrd = P_thrd
        self.S_thrd = S_thrd
        
        print('MALMI starts to detect events based on the ML predicted phase probabilites and output the corresponding phase probabilites of the detected events:')
        # eqt_arrayeventdetect(self.dir_prob, self.dir_lokiprob, sttd_max, twlex, d_thrd, nsta_thrd, spttdf_ssmax)
        event_info = eqt_eventdetectfprob(self.dir_prob, self.P_thrd, self.S_thrd)
        gc.collect()
        if twind_srch is None:
            twind_srch, _, _ = maxP2Stt(self.dir_tt, self.tt_hdr_filename, self.tt_ftage, self.tt_precision)
            
        if outseis:
            dir_seismic = self.dir_seismic
        else:
            dir_seismic = None
        arrayeventdetect(event_info, twind_srch, twlex, nsta_thrd, npha_thrd, self.dir_lokiprob, self.dir_lokiseis, dir_seismic, self.seismic_channels)
        gc.collect()
        print('MALMI_event_detect_ouput complete!')


    def migration(self, probthrd=0.001):
        """
        Perform migration based on input phase probabilites

        Parameters
        ----------
        probthrd : float, optional
            probability normalization threshold. If maximum value of the input 
            phase probabilites is larger than this threshold, the input trace 
            will be normalized (to 1). The default is 0.001.

        Returns
        -------
        None.

        """
        
        from loki.loki import Loki
        
        print('MALMI starts to perform migration:')
        
        self.probthrd = copy.deepcopy(probthrd)
        self.ppower = 4
        
        inputs = {}
        inputs['model'] = self.tt_ftage  # traveltime data set filename tage
        # inputs['npr'] = self.n_processor  # number of cores to run
        inputs['npr'] = (os.cpu_count()-2)  # number of cores to run
        inputs['normthrd'] = self.probthrd  # if maximum value of the input phase probabilites is larger than this threshold, the input trace will be normalized (to 1)
        inputs['ppower'] = self.ppower  # compute array element wise power over the input probabilities before stacking
        comp = ['P','S']  # when input data are probabilities of P- and S-picks, comp must be ['P', 'S']
        extension = '*'  # seismic data filename for loading, accept wildcard input, for all data use '*'
        
        l1 = Loki(self.dir_lokiprob, self.dir_lokiout, self.dir_tt, self.tt_hdr_filename, mode='locator')
        l1.location(extension, comp, self.tt_precision, **inputs)
        gc.collect()
        print('MALMI_migration complete!')
    
        
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
                                    P_thrd=self.P_thrd, S_thrd=self.S_thrd, 
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
                              comp=['Z','N','E'], dyy=1.8, fband=[2, 30], normv=self.probthrd, 
                              ppower=self.ppower, tag=None, staname=None, arrvtt=arrvtt)
    
        gc.collect()
        print('MALMI_rsprocess_view complete!')

    
    def clear_interm(self):
        
        print('MALMI starts to clear some intermediate results for saving disk space:')
        shutil.rmtree(self.dir_mseed)  # remove the mseed directory which are the formateed continuous seismic data set for ML inputs
        shutil.rmtree(self.dir_hdf5)  # remove the hdf5 directory which are formatted overlapping data segments of seismic data set for ML predictions
        
        # remove the generated continuous probability hdf5 files
        prob_h5files = glob.glob(self.dir_prob + '/**/*.hdf5', recursive=True)
        for ipf in prob_h5files:
            os.remove(ipf)
        
        gc.collect()        
        print('MALMI_clear_interm complete!')
        
        
