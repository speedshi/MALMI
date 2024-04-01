
'''
Calculate the magnitude of an quake event.
'''


import yaml
from ioformatting import csv2dict
from qm_moment import dispf2mw
import numpy as np
from qm_processing import remove_outliers_iqr, remove_outliers_zscore


def xmag_input(file_para):
    # load and check the input parameters

    if isinstance(file_para, str):
        # load paramters
        with open(file_para, 'r') as file:
            paras = yaml.safe_load(file)
    elif isinstance(file_para, dict):
        # input is a dictionary containing the parameters
        paras = file_para
    else:
        raise ValueError("Parameter file should be a string or a parameter dictionary.")

    # check the parameters
    if 'method' not in paras:
        paras['method'] = 'Mw'

    if 'network_magnitude' not in paras:
        paras['network_magnitude'] = {}
        paras['network_magnitude']['remove_outlier'] = 'none'
        paras['network_magnitude']['method'] = 'median'

    return paras




def get_magnitude(stream, ev_xyz, station, picks, file_para):

    sx, sy, sz = ev_xyz  # event location in meter

    # load paramters
    paras = xmag_input(file_para=file_para)

    # load station parameter file
    station_para = csv2dict(file_csv=paras['station']['file'])
    station_para['location'] = np.where(np.isnan(station_para['location']), '', station_para['location'])
        
    stamag = {}
    for jsta in range(len(station_para['station'])):
        sid = f"{station_para['network'][jsta]}.{station_para['station'][jsta]}.{station_para['location'][jsta]}.{station_para['instrument'][jsta]}"
        if sid not in stamag:
            stamag[sid] = {}
            stamag[sid]['scale_factor'] = {}
            stamag[sid]['integrate'] = {}
        stamag[sid]['scale_factor'][station_para['component'][jsta]] = station_para['scale_factor'][jsta]
        stamag[sid]['integrate'][station_para['component'][jsta]] = station_para['integrate'][jsta]

    staid_list = list(stamag.keys())
    nsta = len(staid_list)  # total number of stations for magnitude estimation
    magnitude_station = {}  # magnitude for each station
    magnitude_list = []

    if paras['method'].lower() == 'mw':
        # moment magnitude
        magnitude_type = 'Mw'

        for ii in range(nsta):  # loop over each station_id
            staid_c = staid_list[ii]
            staid = f"{staid_c.split('.')[0]}.{staid_c.split('.')[1]}.{staid_c.split('.')[2]}"

            if staid in picks:  # proceed if we have picks for the station
            
                ist = stream.select(id=staid_c+'*')  # select data for this station

                if ist.count() >= 3:  # proceed if we have three components
                    for itr in ist:  # loop over each trace/component
                        # correct the instrument sensitivity by multiplying the scale factor
                        itr.data = itr.data * stamag[staid_c]['scale_factor'][itr.stats.channel[-1]]

                    # process seismic data before magnitude estimation
                    if isinstance(paras['filter_1'], list):
                        # apply the first filter
                        ist.detrend('demean')
                        ist.detrend('simple')
                        if (not isinstance(paras['filter_1'][0], (int,float))) and isinstance(paras['filter_1'][1], (int,float)):
                            ist.filter('lowpass', freq=paras['filter_1'][1], corners=2, zerophase=True)
                        elif (not isinstance(paras['filter_1'][1], (int,float))) and isinstance(paras['filter_1'][0], (int,float)):
                            ist.filter('highpass', freq=paras['filter_1'][0], corners=2, zerophase=True)
                        elif isinstance(paras['filter_1'][0], (int,float)) and isinstance(paras['filter_1'][1], (int,float)):
                            ist.filter('bandpass', freqmin=paras['filter_1'][0], freqmax=paras['filter_1'][1], corners=2, zerophase=True)
                        else:
                            raise ValueError(f"Invalid input for fband: {paras['filter_1']}!")
                        ist.detrend('demean')  # remove mean, i.e. removing DC component

                    # get displacement measurements from the recordings
                    for itr in ist:
                        for _ in range(stamag[staid_c]['integrate'][itr.stats.channel[-1]]):
                            itr.integrate()  # integrate

                    if isinstance(paras['filter_2'], list):
                        # apply the first filter
                        ist.detrend('demean')
                        ist.detrend('simple')
                        if (not isinstance(paras['filter_2'][0], (int,float))) and isinstance(paras['filter_2'][1], (int,float)):
                            ist.filter('lowpass', freq=paras['filter_2'][1], corners=2, zerophase=True)
                        elif (not isinstance(paras['filter_2'][1], (int,float))) and isinstance(paras['filter_2'][0], (int,float)):
                            ist.filter('highpass', freq=paras['filter_2'][0], corners=2, zerophase=True)
                        elif isinstance(paras['filter_2'][0], (int,float)) and isinstance(paras['filter_2'][1], (int,float)):
                            ist.filter('bandpass', freqmin=paras['filter_2'][0], freqmax=paras['filter_2'][1], corners=2, zerophase=True)
                        else:
                            raise ValueError(f"Invalid input for fband: {paras['filter_2']}!")
                        ist.detrend('demean')  # remove mean, i.e. removing DC component  

                    # calculate distance from source to station, in meter
                    aidx = np.where(station.id == staid_c)[0]
                    rx, ry, rz = station.x[aidx], station.y[aidx], station.z[aidx]  # receiver location in meter
                    dist = np.sqrt((sx-rx)**2 + (sy-ry)**2 + (sz-rz)**2)  # in meter

                    mag_temp = []
                    for iph in picks[staid]:
                        tt_start = picks[staid][iph] - paras[iph]['t_before'] / ist[0].stats.sampling_rate
                        tt_end = picks[staid][iph] + paras[iph]['t_after'] / ist[0].stats.sampling_rate
                        itrace = ist.slice(starttime=tt_start, endtime=tt_end).copy()  # traces around pick
                        itrace.merge(method=1, fill_value=0)  # merge data
                        
                        if itrace.count() == 3:
                            mag_temp.append(dispf2mw(traces=itrace, rho=paras['density'], vel=paras[iph]['velocity'], radipat=paras[iph]['radipat'], dist=dist))    
                        elif itrace.count() > 3:
                            raise ValueError(f"Erorr: More than 3 components for station {staid}!")

                    if len(mag_temp) > 0:
                        mag_temp = np.nanmax(mag_temp)
                        magnitude_station[staid_c] = mag_temp
                        magnitude_list.append(mag_temp)
        
        if (paras['network_magnitude']['remove_outlier'].lower() == 'iqr'):
            magnitude_list = remove_outliers_iqr(data=magnitude_list, thrdr=1.5)
        elif (paras['network_magnitude']['remove_outlier'].lower() == 'zscore'):
            magnitude_list = remove_outliers_zscore(data=magnitude_list, threshold=3)
        elif (paras['network_magnitude']['remove_outlier'].lower() == 'none'):
            pass
        else:
            raise ValueError(f"Wrong input for network_magnitude:remove_outlier: {paras['network_magnitude']['remove_outlier']}!")

        if paras['network_magnitude']['method'].lower() == 'median':
            magnitude = np.nanmedian(magnitude_list)
        elif paras['network_magnitude']['method'].lower() == 'mean':
            magnitude = np.nanmean(magnitude_list)
        elif paras['network_magnitude']['method'].lower() == 'max':
            magnitude = np.nanmax(magnitude_list)
        elif paras['network_magnitude']['method'].lower() == 'min':
            magnitude = np.nanmin(magnitude_list)
        else:
            raise ValueError(f"Wrong input for network_magnitude:method: {paras['network_magnitude']['method']}!")

    else:
        raise ValueError(f"Wrong input for method: {paras['method']}!")

    return magnitude, magnitude_station, magnitude_type



