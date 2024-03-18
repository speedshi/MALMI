

import yaml
import numpy as np
from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import differential_evolution
import pickle
import os
from numba import jit, prange
import bottleneck as bn


# define the objective function
def objfun(xyz, it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras):
    x, y, z = xyz
    va = 0
    for jjsta in cf_station:
        for jjph in paras['phase']:
            va += cf[jjsta][jjph]((it0 + traveltime.tt_model[jjsta][jjph](x,y,z) - cf_starttime_s[jjsta][jjph])*data_sampling_rate + 1)

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
            if paras['cfun_interpolation'] is not None:
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
        
    if 'save_result' not in paras:
        paras['save_result'] = {}
    else:
        if 'save_loc' not in paras['save_result']:
            paras['save_result']['save_loc'] = False

        if 'data_dim' not in paras['save_result']:
            paras['save_result']['data_dim'] = 1
        else:
            if paras['save_result']['data_dim'] not in [1, 2, 3, 4]:
                raise ValueError("Invalid input for 'save_result:data_dim'.")

    return paras


def xmig(data, traveltime, region, paras, dir_output):
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
    cf_starttime = {}  # in UTCDateTime
    cf_starttime_s = {}  # in seconds relative to t0_start
    for itr in data:
        istaid = itr.id[:-1]  # station id: "network_code.station_code.location_code.instrument_code"
        iphase = itr.id[-1].upper()  # phase name: "P" or "S"
        if istaid not in cf:
            cf[istaid] = {}
            cf_starttime[istaid] = {}
            cf_starttime_s[istaid] = {}
        assert(iphase not in cf[istaid])
        assert(iphase not in cf_starttime[istaid])
        assert(iphase not in cf_starttime_s[istaid])
        cf_starttime[istaid][iphase] = itr.stats.starttime
        cf_starttime_s[istaid][iphase] = itr.stats.starttime - t0_start

        # normalize the trace (characteristic function)
        if paras['cfun_normalization'][iphase] is not None:
            icfmax = np.max(itr.data)
            if icfmax >= paras['cfun_normalization'][iphase]:
                itr.data /= icfmax  # maximum value normalize to 1

        # construct the characteristic function for the station and the phase
        if paras['cfun_interpolation'] is None:
            # use the original data
            cf[istaid][iphase] = itr.data
        else:
            # use interpolation function to construct the characteristic function
            cf[istaid][iphase] = interp1d(x=np.arange(1, len(itr.data)+1), y=itr.data, 
                                          kind=paras['cfun_interpolation'],
                                          bounds_error=False, fill_value=0, assume_sorted=True)

    # calculate searching origin times
    t0_len = t0_end - t0_start  # time length in second for searching origin time
    t0_step = paras['time_step'] / data_sampling_rate  # time step in seconds, note need to convert from sample to second
    t0_s = np.arange(0, t0_len+t0_step, t0_step)  # searching origin time in seconds relative to the t0_start
    # UTCDatetime of origin time will be: t0_start + t0_s
    nt0 = len(t0_s)  # number of searching origin times
    if paras['save_result']['data_dim'] == 1:
        d_shape = (nt0,)
    elif paras['save_result']['data_dim'] == 2:
        raise ValueError("Not implemented yet!")
    elif paras['save_result']['data_dim'] == 3:
        d_shape = (region.nx, region.ny, region.nz)
    elif paras['save_result']['data_dim'] == 4:
        d_shape = (nt0, region.nx, region.ny, region.nz)
    else:
        raise ValueError("Invalid input for 'save_result:data_dim'.")
    output_v = np.zeros(d_shape)  # migration values over time
    output_x = np.zeros((nt0,))  # x coordinates over time
    output_y = np.zeros((nt0,))  # y coordinates over time
    output_z = np.zeros((nt0,))  # z coordinates over time

    # start location
    cf_station = list(cf.keys())
    NCFS = len(cf_station)  # number of stations that have characteristic function
    NPHS = len(paras['phase'])  # number of seismic phases

    if traveltime.tt_type == 'function':
        # traveltimes are expressed as functions
        # use global optimization method for event location

        if paras['save_result']['data_dim'] != 1:
            raise ValueError("Migration data dimension must be 1!")

        bounds = [(region.x_min, region.x_max), (region.y_min, region.y_max), (region.z_min, region.z_max)]

        for ii, it0 in enumerate(t0_s):
            # loop over each searching origin time
            result = differential_evolution(func=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                                            bounds=bounds, popsize=20, workers=1, vectorized=False,
                                            maxiter=1000, tol=1e-2, mutation=(0.5, 1.9), recombination=0.7, seed=42)
            output_x[ii], output_y[ii], output_z[ii] = result.x 
            output_v[ii] = -result.fun  # need to implement, to do ......

    elif traveltime.tt_type == 'table':
        # traveltimes are expressed as tables
        # use grid search method

        for ii, it0 in enumerate(t0_s):
            # loop over each searching origin time

            va = np.zeros((region.nx, region.ny, region.nz))  # 3D array of (X, Y, Z)

            # calculate the migration value for each grid point  
            for jjsta in cf_station:
                # loop over each station
                for jjph in paras['phase']:
                    # loop over each phase
                    va += cf[jjsta][jjph](( (it0-cf_starttime_s[jjsta][jjph]) + traveltime.tt_model[jjsta][jjph])*data_sampling_rate + 1)

            # find the maximum value along XYZ and its index
            max_indices = np.unravel_index(np.argmax(va, axis=None), shape=va.shape)  # index of the maximum value: (ix, iy, iz)
            output_x[ii], output_y[ii], output_z[ii] = region.x[max_indices], region.y[max_indices], region.z[max_indices]

            if paras['save_result']['data_dim'] == 1:
                # save the maximum value along XYZ, i.e., output_v[T]
                output_v[ii] = va[max_indices]
            elif paras['save_result']['data_dim'] == 2:
                # save the maximum value along XY, YZ, XZ profiles, i.e., output_v[T, ny*nz, nx*nz, nx*ny]
                # maximum value profiles at the source location along X-, Y-, Z-axis
                raise ValueError("Not implemented yet!")
            elif paras['save_result']['data_dim'] == 3:
                # save the maximum value along time axis, i.e., output_v[X,Y,Z]
                np.maximum(output_v, va, out=output_v)
            elif paras['save_result']['data_dim'] == 4:
                # save full migration values over time and space, i.e., output_v[T, X, Y, Z]
                output_v[ii] = va
            else:
                raise ValueError("Invalid input for 'save_result:data_dim'.")

    else:
        raise ValueError(f"Invalid traveltime type: {traveltime.tt_type}!")

    output_t0 = np.array([t0_start+it0_s for it0_s in t0_s])  # origin times in UTCDateTime
    
    # save results  
    if paras['save_result']['save_loc']:
        fname = os.path.join(dir_output, "migration_results.npz")
        np.savez(fname, output_t0=output_t0, output_v=output_v, 
                 output_x=output_x, output_y=output_y, output_z=output_z)

    return output_x, output_y, output_z, output_t0


def location_agg(data, file_parameter, traveltime, region, dir_output="./"):
    """
    Input:
        data: data object, should be obspy stream;
        file_parameter: parameter file, str or dict object;
        traveltime: traveltime class object;
        region: region class object;
        dir_output: output directory, str;

    """

    if isinstance(file_parameter, str):
        # load paramters from file
        paras = xloc_input(file_para=file_parameter)
    elif isinstance(file_parameter, dict):
        # input is a dictionary containing the parameters
        paras = file_parameter

    if paras['method'] == 'xmig':
        output_x, output_y, output_z, output_t0 = xmig(data=data, traveltime=traveltime, region=region, paras=paras, dir_output=dir_output)
    else:
        raise ValueError(f"Invalid method for location_agg: {paras['method']}")

    return np.max(output_x), np.max(output_y), np.max(output_z), np.max(output_t0)


