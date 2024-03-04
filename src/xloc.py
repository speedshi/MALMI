

import yaml
import numpy as np
from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import differential_evolution
import pickle


# define the objective function
def objfun(xyz, it0, cf_station, cf, cf_starttime, data_sampling_rate, traveltime, paras):
    x, y, z = xyz
    va = 0
    for jjsta in cf_station:
        for jjph in paras['phase']:
            va += cf[jjsta][jjph]((it0 + traveltime.tt_model[jjsta][jjph](x,y,z) - cf_starttime[jjsta][jjph])*data_sampling_rate + 1)

    return -va

def objfun_test(xyz, it0, cf_station, cf, cf_starttime, data_sampling_rate, traveltime, paras):
    x, y, z = xyz
    va = 0
    for jjsta in cf_station:
        for jjph in paras['phase']:
            # jjtraveltime = 
            va += cf[jjsta][jjph]((it0 + traveltime.tt_model[jjsta][jjph](x,y,z) - cf_starttime[jjsta][jjph])*data_sampling_rate + 1)

    return -va

def xloc_input(file_para):

    if isinstance(file_para, str):
        # load paramters
        with open(file_para, 'r') as file:
            paras = yaml.safe_load(file)
    elif isinstance(file_para, dict):
        # input is a dictionary containing the parameters
        paras = file_para
    else:
        raise ValueError("Parameter file should be a string or a parameter dictionary.")

    # check input parameters
    if 'method' not in paras:
        raise ValueError("phassoc parameter file should contain 'method' setting.")
    else:
        if not paras['method'].lower() in ['xmig']:
            raise ValueError("'xloc:method' should be either 'xmig'.")

    if paras['method'].lower() == 'xmig':
        if 'cfun_interpolation' not in paras:
            paras['cfun_interpolation'] = 'linear'
        else:
            if not paras['cfun_interpolation'].lower() in ['linear', 'zero', 'slinear', 'quadratic', 'cubic']:
                raise ValueError(f"Invalid interpolation method: {paras['cfun_interpolation']}")

        if 'cfun_normalization' not in paras:
            paras['cfun_normalization'] = {'P': None, 'S': None}

    if 'phase' not in paras:
        paras['phase'] = ['P', 'S']
    else:
        for ph in paras['phase']:
            if not ph in {'P', 'S'}:
                raise ValueError(f"Invalid phase name: {ph}")

    if 'time_step' not in paras:
        paras['time_step'] = 1
    elif not isinstance(paras['time_step'], int):
        raise ValueError("time_step should be an integer.")
    else:
        if paras['time_step'] < 1:
            raise ValueError("time_step should be a positive integer.")

    return paras


def xmig(data, traveltime, region, paras):
    """
    Locate source using migration-based method (back-projection).
    """

    data = data.merge(method=1, fill_value=0, interpolation_samples=-1)  # merge traces with the same id

    # check data sampling rate, all traces must have the same sampling rate
    sampling_rates = set(itr.stats.sampling_rate for itr in data)
    if len(sampling_rates) != 1:
        raise ValueError("All traces must have the same sampling rate.")
    data_sampling_rate = list(sampling_rates)[0]  # data sampling rate in Hz

    # get the earliest starttime and the latest endtime for input data
    starttime = min(itr.stats.starttime for itr in data)
    endtime = max(itr.stats.endtime for itr in data)

    # determine the starttime and endtime for searching event origin time
    t0_start = starttime - traveltime.tt_max  # starttime for searching event origin time
    t0_end = endtime - traveltime.tt_min  # endtime for searching event origin time

    # obtain the characteristic function for each station and each phase
    # characteristic function is a dictionary with the following structure:
    # cf = {'station1': {'phase1': function1, 'phase2': function2, ...}, 
    #       'station2': {'phase1': function1, 'phase2': function2, ...}, 
    #        ...}
    cf = {}
    cf_starttime = {}
    for itr in data:
        istaid = itr.id[:-1]  # station id: "network_code.station_code.location_code.instrument_code"
        iphase = itr.id[-1].upper()  # phase name: "P" or "S"
        if istaid not in cf:
            cf[istaid] = {}
            cf_starttime[istaid] = {}
        assert(iphase not in cf[istaid])
        assert(iphase not in cf_starttime[istaid])
        cf_starttime[istaid][iphase] = itr.stats.starttime

        # normalize the trace (characteristic function)
        if paras['cfun_normalization'][iphase] is not None:
            icfmax = np.max(itr.data)
            if icfmax >= paras['cfun_normalization'][iphase]:
                itr.data /= icfmax  # maximum value normalize to 1

        # construct the characteristic function for the station and the phase
        cf[istaid][iphase] = interp1d(x=np.arange(1, len(itr.data)+1), y=itr.data, 
                                      kind=paras['cfun_interpolation'],
                                      bounds_error=False, fill_value=0, assume_sorted=True)

    # calculate searching origin times
    t0_step = paras['time_step'] / data_sampling_rate  # time step in seconds
    t0s = np.arange(t0_start, t0_end+t0_step, t0_step)  # searching origin time in UTCDateTime
    nt0s = len(t0s)  # number of searching origin times
    res_v = np.zeros((nt0s, 1))  # migration value over time
    res_x = np.zeros((nt0s, 1))  # x coordinates over time
    res_y = np.zeros((nt0s, 1))  # y coordinates over time
    res_z = np.zeros((nt0s, 1))  # z coordinates over time

    # start location
    cf_station = list(cf.keys())
    if traveltime.tt_type == 'function':
        # traveltimes are expressed as functions
        # use global optimization method for event location

        bounds = [(region.x_min, region.x_max), (region.y_min, region.y_max), (region.z_min, region.z_max)]

        for ii, it0 in enumerate(t0s):
            # loop over each searching origin time
            result = differential_evolution(func=objfun, args=(it0, cf_station, cf, cf_starttime, data_sampling_rate, traveltime, paras), 
                                            bounds=bounds, popsize=20, workers=1, vectorized=False,
                                            maxiter=10000, tol=1e-3, mutation=(0.5, 1.9), recombination=0.7, seed=42)
            res_x[ii], res_y[ii], res_z[ii] = result.x 
            res_v[ii] = -result.fun

    elif traveltime.tt_type == 'table':
        # traveltimes are expressed as tables
        # use grid search method

        for ii, it0 in enumerate(t0s):
            # loop over each searching origin time

            # calculate the migration value for each grid point
            va = 0
            for jjsta in cf_station:
                for jjph in paras['phase']:
                    va += cf[jjsta][jjph](( (it0-cf_starttime[jjsta][jjph]) + traveltime.tt_model[jjsta][jjph])*data_sampling_rate + 1)

            imax = np.unravel_index(np.argmax(va), va.shape)
            res_x[ii], res_y[ii], res_z[ii] = region.x[imax], region.y[imax], region.z[imax]
            res_v[ii] = va[imax]

    else:
        raise ValueError(f"Invalid traveltime type: {traveltime.tt_type}!")


    
        
    np.savez("results.npz", t0s=t0s, res_v=res_v, res_x=res_x, res_y=res_y, res_z=res_z)
    exit()    


    return


def location_agg(data, file_parameter, traveltime, region):
    """
    Input:
        data: data object, should be obspy stream;
        file_parameter: parameter file, str or dict object;
        traveltime: traveltime class object;
        region: region class object;

    """

    if isinstance(file_parameter, str):
        # load paramters from file
        paras = xloc_input(file_para=file_parameter)
    elif isinstance(file_parameter, dict):
        # input is a dictionary containing the parameters
        paras = file_parameter

    if paras['method'] == 'xmig':
        xmig(data=data, traveltime=traveltime, region=region, paras=paras)
    else:
        raise ValueError(f"Invalid method for location_agg: {paras['method']}")

    return


