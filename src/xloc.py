

import yaml
import numpy as np
from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import differential_evolution, minimize, brute
from scipy import optimize
import os
from numba import jit
import multiprocessing as mp
from multiprocessing.pool import Pool
import itertools


def _parti(rng, num):
    # partition the 'rng' into 'num' smaller parts
    # rng: [min, max]

    rstep = (rng[1] - rng[0]) / num
    rpart = [[rng[0]+ir*rstep, rng[0]+(ir+1)*rstep] for ir in range(num)]

    return rpart


def _migv_point(xrange, yrange, zrange, trange, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras, ntgrid, region):

    dxr = xrange[1] - xrange[0]
    dyr = yrange[1] - yrange[0]
    dzr = zrange[1] - zrange[0]

    if (dxr > ntgrid * region.dx) and (dyr > ntgrid * region.dy) and (dzr > ntgrid * region.dz):
        # calculate the minimal and maximum traveltime using the traveltime table
        ttcal_tab = 1
        grid_indices, _, _, _ = region.mesh3D_xyz_subgrid_index(x_bound=xrange, y_bound=yrange, z_bound=zrange, 
                                                                dnx=1, dny=1, dnz=1)

    va = 0
    # calculate the migration value for each grid point  
    for jjsta in cf_station:
        # loop over each station
        for jjph in paras['phase']:
            # loop over each phase

            # calculate the minimal and maximum traveltime for the given station and phase
            if ttcal_tab == 1:
                tt_min = np.min(traveltime.tt_tab[jjsta][jjph][grid_indices], axis=None)
                tt_max = np.max(traveltime.tt_tab[jjsta][jjph][grid_indices], axis=None)
            else:
                tt_min, tt_max = traveltime.get_minmaxtt_fun_staphs(xrange=xrange, 
                                                                    yrange=yrange, 
                                                                    zrange=zrange, 
                                                                    station=jjsta, 
                                                                    phase=jjph, 
                                                                    nx=ntgrid, 
                                                                    ny=ntgrid, 
                                                                    nz=ntgrid)
            
            at_min_sample = (trange[0] + tt_min - cf_starttime_s[jjsta][jjph])*data_sampling_rate + 1  # the earliest arrivaltime in sample
            at_max_sample = (trange[1] + tt_max - cf_starttime_s[jjsta][jjph])*data_sampling_rate + 1  # the latest arrivaltime in sample
            
            va += np.max(cf[jjsta][jjph](np.arange(at_min_sample, at_max_sample+1)))

    return va


# define the objective function for optimization for (x, y, z)
def objfun(xyz, it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras):
    x, y, z = xyz
    va = 0
    for jjsta in cf_station:
        for jjph in paras['phase']:
            va += cf[jjsta][jjph]((it0 + traveltime.tt_fun[jjsta][jjph](x,y,z) - cf_starttime_s[jjsta][jjph])*data_sampling_rate + 1)

    return -va


# define the objective function for optimization for (t, x, y, z)
def objfunt(txyz, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras):
    it0, x, y, z = txyz
    va = 0
    for jjsta in cf_station:
        for jjph in paras['phase']:
            va += cf[jjsta][jjph]((it0 + traveltime.tt_fun[jjsta][jjph](x,y,z) - cf_starttime_s[jjsta][jjph])*data_sampling_rate + 1)

    return -va


# define that calculation of migration at each origin time (it0)
def migration_ti(ii, t0_s, region, paras, traveltime, cf, cf_station, cf_starttime_s, data_sampling_rate, wdps, nwdps):

    # This function will be executed by each process
    it0= t0_s[ii]  # the ii-th origin time

    x_bound = [region.x_min, region.x_max]
    y_bound = [region.y_min, region.y_max]
    z_bound = [region.z_min, region.z_max]

    for jnx, jny, jnz, jw in zip(paras['loc_grid']['dnx'], paras['loc_grid']['dny'], paras['loc_grid']['dnz'], range(nwdps)):
        
        grid_indices, x_index, y_index, z_index = region.mesh3D_xyz_subgrid_index(x_bound=x_bound, y_bound=y_bound, z_bound=z_bound, dnx=jnx, dny=jny, dnz=jnz)

        x_sub = region.x[x_index]  # 1D array of x coordinates
        y_sub = region.y[y_index]  # 1D array of y coordinates
        z_sub = region.z[z_index]  # 1D array of z coordinates

        va = np.zeros((len(x_sub),len(y_sub),len(z_sub)))  # 3D array of (X, Y, Z)

        # calculate the migration value for each grid point  
        for jjsta in cf_station:
            # loop over each station
            for jjph in paras['phase']:
                # loop over each phase
                atidx_3d = ((it0-cf_starttime_s[jjsta][jjph]) + traveltime.tt_tab[jjsta][jjph][grid_indices])*data_sampling_rate + 1
                # wwg_range = np.arange(-wdps[jjph][jw], wdps[jjph][jw]+1)
                # vtemp_3d = np.max(np.array([cf[jjsta][jjph](atidx_3d+wwg) for wwg in wwg_range]), axis=0)
                # va += vtemp_3d
                vtemp_3d = np.full_like(atidx_3d, -np.inf)
                for wwg in range(-wdps[jjph][jw], wdps[jjph][jw]+1):
                    vtemp_3d = np.maximum(cf[jjsta][jjph](atidx_3d+wwg), vtemp_3d)
                va += vtemp_3d

        # find the maximum value along XYZ and its index
        max_indices = np.unravel_index(np.argmax(va, axis=None), shape=va.shape)  # index of the maximum value: (ix, iy, iz)
        va_max = va[max_indices]
        va_min = np.nanmin(va, axis=None)

        # update bounds for x, y, z
        va_thd = paras['loc_grid']['sigma'] * (va_max - va_min) + va_min
        above_threshold = (va >= va_thd)
        xb1, xb2 = np.where(np.any(above_threshold, axis=(1,2)))[0][[0, -1]]
        if xb1 == xb2:  # single point scenario, extend outward of 1 point
            xb1 = max(0, xb1-1)
            xb2 = min(len(x_sub)-1, xb2+1)
        yb1, yb2 = np.where(np.any(above_threshold, axis=(0,2)))[0][[0, -1]]
        if yb1 == yb2:
            yb1 = max(0, yb1-1)
            yb2 = min(len(y_sub)-1, yb2+1)
        zb1, zb2 = np.where(np.any(above_threshold, axis=(0,1)))[0][[0, -1]]
        if zb1 == zb2:
            zb1 = max(0, zb1-1)
            zb2 = min(len(z_sub)-1, zb2+1)
        x_bound = [x_sub[xb1], x_sub[xb2]]
        y_bound = [y_sub[yb1], y_sub[yb2]]
        z_bound = [z_sub[zb1], z_sub[zb2]]

        # check if early-stop the grid loop
        if va_max < paras['loc_grid']['early_stop']:
            # trigger early stop
            # break the grid loop

            if wdps[jjph][jw] != wdps[jjph][-1]:
                # recalculate the output migration value
                if paras['save_result']['data_dim'] == 1:
                    # only need to calculate the maxima point
                    # calculate the migration value for each grid point 
                    va_max = 0 
                    for jjsta in cf_station:
                        # loop over each station
                        for jjph in paras['phase']:
                            # loop over each phase
                            atidx_3d = ((it0-cf_starttime_s[jjsta][jjph]) + traveltime.tt_tab[jjsta][jjph][max_indices])*data_sampling_rate + 1
                            # wwg_range = np.arange(-wdps[jjph][jw], wdps[jjph][jw]+1)
                            # vtemp_3d = np.max(np.array([cf[jjsta][jjph](atidx_3d+wwg) for wwg in wwg_range]), axis=0)
                            # va += vtemp_3d
                            vtemp_3d = np.full_like(atidx_3d, -np.inf)
                            for wwg in range(-wdps[jjph][jw], wdps[jjph][jw]+1):
                                vtemp_3d = np.maximum(cf[jjsta][jjph](atidx_3d+wwg), vtemp_3d)
                            va += vtemp_3d
                else:
                    # calculate the migration value for each grid point  
                    va[:,:,:] = 0
                    for jjsta in cf_station:
                        # loop over each station
                        for jjph in paras['phase']:
                            # loop over each phase
                            atidx_3d = ((it0-cf_starttime_s[jjsta][jjph]) + traveltime.tt_tab[jjsta][jjph][grid_indices])*data_sampling_rate + 1
                            # wwg_range = np.arange(-wdps[jjph][jw], wdps[jjph][jw]+1)
                            # vtemp_3d = np.max(np.array([cf[jjsta][jjph](atidx_3d+wwg) for wwg in wwg_range]), axis=0)
                            # va += vtemp_3d
                            vtemp_3d = np.full_like(atidx_3d, -np.inf)
                            for wwg in range(-wdps[jjph][jw], wdps[jjph][jw]+1):
                                vtemp_3d = np.maximum(cf[jjsta][jjph](atidx_3d+wwg), vtemp_3d)
                            va += vtemp_3d

            break
    
    # compile return results
    if paras['save_result']['data_dim'] == 1:
        mig_v = va_max
    else:
        mig_v = va
    mig_x = x_sub[max_indices[0]]
    mig_y = y_sub[max_indices[1]]
    mig_z = z_sub[max_indices[2]]

    return ii, mig_v, mig_x, mig_y, mig_z



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
            if not paras['cfun_interpolation'].lower() in ['linear', 'zero', 'slinear', 'quadratic', 'cubic', 'none']:
                raise ValueError(f"Invalid interpolation method: {paras['cfun_interpolation']}")
            if paras['cfun_interpolation'].lower() == 'none':
                paras['cfun_interpolation'] = None

        if 'cfun_normalization' not in paras:
            paras['cfun_normalization'] = {'P': None, 'S': None}

    if 'phase' not in paras:
        paras['phase'] = ['P', 'S']
    else:
        for ph in paras['phase']:
            if not ph in {'P', 'S'}:
                raise ValueError(f"Invalid phase name: {ph}")

    if 'origin_time' not in paras:
        paras['origin_time']['step'] = 1
        paras['origin_time']['start_shift'] = 0
        paras['origin_time']['end_shift'] = 0
    else:
        if paras['origin_time']['step'] <= 0:
            raise ValueError("origin_time:step should be a positive number.")

    if 'migration_engine' not in paras: paras['migration_engine'] = 'grid'

    if 'multiprocessing' not in paras:
        paras['multiprocessing'] = {}
        paras['multiprocessing']['processes'] = 1
        paras['multiprocessing']['chunksize'] = None
    else:
        if 'processes' not in paras['multiprocessing']:
            paras['multiprocessing']['processes'] = 1
        else:
            if not isinstance(paras['multiprocessing']['processes'], int):
                raise ValueError("multiprocessing:processes should be an integer.")
            if paras['multiprocessing']['processes'] < 1:
                # Use the number of available CPU cores
                paras['multiprocessing']['processes'] = mp.cpu_count()
            if (isinstance(paras['multiprocessing']['chunksize'], str)) and (paras['multiprocessing']['chunksize'].lower() == 'none'):
                paras['multiprocessing']['chunksize'] = None

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
            if (paras['save_result']['data_dim'] not in [1, 4]) and (paras['mp_cores'] != 1):
                raise ValueError("When save_result:data_dim is not 1 or 4, mp_cores must be 1!")
            
    if 'loc_grid' not in paras:
        paras['loc_grid'] = None
    else:
        if 'dnx' not in paras['loc_grid']: paras['loc_grid']['dnx'] = [1]
        if 'dny' not in paras['loc_grid']: paras['loc_grid']['dny'] = [1]
        if 'dnz' not in paras['loc_grid']: paras['loc_grid']['dnz'] = [1]
        if 'atscale' not in paras['loc_grid']: paras['loc_grid']['atscale'] = [1]
        if 'early_stop' not in paras['loc_grid']: paras['loc_grid']['early_stop'] = -1
        if 'local_opt' not in paras['loc_grid']: paras['loc_grid']['local_opt'] = 'L-BFGS-B'

        if ((len(paras['loc_grid']['dnx']) != len(paras['loc_grid']['dny'])) or 
            (len(paras['loc_grid']['dnx']) != len(paras['loc_grid']['dnz'])) or
            (len(paras['loc_grid']['dnx']) != len(paras['loc_grid']['atscale']))):
            raise ValueError("loc_grid: dnx, dny, dnz must have the same length.")

        if not all(isinstance(ii, int) and ii>0 for ii in paras['loc_grid']['dnx']):
            raise ValueError("loc_grid: dnx should be a list of positive integers.")
        
        if not all(isinstance(ii, int) and ii>0 for ii in paras['loc_grid']['dny']):
            raise ValueError("loc_grid: dny should be a list of positive integers.")
        
        if not all(isinstance(ii, int) and ii>0 for ii in paras['loc_grid']['dnz']):
            raise ValueError("loc_grid: dnz should be a list of positive integers.")
        
        if not all(ii>=0 for ii in paras['loc_grid']['atscale']):
            raise ValueError("loc_grid: atscale should be a list of non-negative floats.")

        if (paras['save_result']['data_dim'] != 1):
            if (len(paras['loc_grid']['dnx']) > 1):
                raise ValueError("Migration data dimension must be 1 for loc_grid: dnx, dny, dnz contains more than 1 element!")
            if paras['loc_grid']['local_opt'].lower() != 'none':
                raise ValueError("Cannot use local_opt if output migration data dimension is not 1.")

    return paras


def xmig(data, traveltime, region, paras, dir_output, velocity_model=None):
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

    # determine the starttime and endtime for searching event origin time (in UTCDateTime)
    t0_start = starttime - traveltime.tt_max + paras['origin_time']['start_shift'] / data_sampling_rate  # starttime for searching event origin time
    t0_end = endtime - traveltime.tt_min + paras['origin_time']['end_shift'] / data_sampling_rate  # endtime for searching event origin time

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
    t0_step = paras['origin_time']['step'] / data_sampling_rate  # time step in seconds, note need to convert from sample to second
    t0_s = np.arange(0, t0_len+t0_step, t0_step)  # searching origin time in seconds relative to the t0_start
    # UTCDatetime of origin time will be: t0_start + t0_s
    nt0 = len(t0_s)  # number of searching origin times
    if paras['save_result']['data_dim'] == 1:
        d_shape = (nt0,)
    elif paras['save_result']['data_dim'] == 2:
        raise ValueError("Not implemented yet!")
    elif paras['save_result']['data_dim'] == 3:
        raise ValueError("Not implemented yet!")
        d_shape = (region.nx, region.ny, region.nz)  # NOTE we cannot do this, because we need to have time axis to find event t0
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
    # NCFS = len(cf_station)  # number of stations that have characteristic function
    # NPHS = len(paras['phase'])  # number of seismic phases

    if paras['migration_engine'].lower() == 'optimize':
        # traveltimes are expressed as functions
        # use global optimization method for event location

        if paras['save_result']['data_dim'] != 1:
            raise ValueError("Migration data dimension must be 1!")

        # bounds for the event location (x, y, z)
        bounds = [(region.x_min, region.x_max), 
                  (region.y_min, region.y_max), 
                  (region.z_min, region.z_max)]
        
        # initial guess for the event location
        x0 = np.array([(region.x_min+region.x_max)*0.5,
                       (region.y_min+region.y_max)*0.5, 
                       (region.z_min+region.z_max)*0.5])
        
        if paras['migration_engine'].lower() == 'differential_evolution':
            for ii, it0 in enumerate(t0_s):
                # loop over each searching origin time
                result = differential_evolution(func=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                                                bounds=bounds, popsize=20, workers=1, vectorized=True, init='sobol', x0=x0,
                                                maxiter=1000, tol=1e-2, mutation=(0.5, 1.0), recombination=0.7, seed=42)
                output_x[ii], output_y[ii], output_z[ii] = result.x 
                output_v[ii] = -result.fun  
        elif paras['migration_engine'].lower() == 'minimize':
            for ii, it0 in enumerate(t0_s):
                # loop over each searching origin time
                result = minimize(fun=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                                  bounds=bounds, x0=x0, method='COBYLA', # COBYLA, SLSQP, Nelder-Mead, CG, L-BFGS-B
                                  options={'maxiter': 1000, 'xatol': 0.8, 'xtol': 0.01, 'gtol': 0.0001,'disp': False, 'adaptive': True},)
                output_x[ii], output_y[ii], output_z[ii] = result.x 
                output_v[ii] = -result.fun
        elif paras['migration_engine'].lower() == 'brute':
            rranges = (slice(region.x_min, region.x_max, complex(paras['loc_grid']['nx'])), 
                       slice(region.y_min, region.y_max, complex(paras['loc_grid']['ny'])), 
                       slice(region.z_min, region.z_max, complex(paras['loc_grid']['nz'])))
            for ii, it0 in enumerate(t0_s):
                # loop over each searching origin time
                result = brute(func=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                               ranges=rranges, finish=optimize.fmin, workers=1, full_output=True)
                output_x[ii], output_y[ii], output_z[ii] = result[0]
                output_v[ii] = -result[1]
        elif paras['migration_engine'].lower() == 'dual_annealing':
            for ii, it0 in enumerate(t0_s):
                # loop over each searching origin time
                result = optimize.dual_annealing(func=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                                                 bounds=bounds, x0=x0, maxiter=1000, seed=42)
                output_x[ii], output_y[ii], output_z[ii] = result.x 
                output_v[ii] = -result.fun
        elif paras['migration_engine'].lower() == 'direct':
            for ii, it0 in enumerate(t0_s):
                # loop over each searching origin time
                result = optimize.direct(func=objfun, args=(it0, cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                                         bounds=bounds, eps=0.1, maxiter=1000, len_tol=1e-2, vol_tol=1e-6)
                output_x[ii], output_y[ii], output_z[ii] = result.x
                output_v[ii] = -result.fun
        else:
            raise ValueError(f"Invalid migration engine: {paras['migration_engine']}!")

        output_t0 = np.array([t0_start+it0_s for it0_s in t0_s])  # origin times in UTCDateTime

    elif paras['migration_engine'].lower() == 'grid':
        # traveltimes are expressed as tables
        # use grid search method

        wdps_d = [np.sqrt((region.dx*kx*0.5)**2 + (region.dy*ky*0.5)**2 + (region.dz*kz*0.5)**2) for kx, ky, kz in 
                zip(paras['loc_grid']['dnx'], paras['loc_grid']['dny'], paras['loc_grid']['dnz'])]
        wdps = {}
        for kphs in paras['phase']:
            wdps[kphs] = np.round((np.array(wdps_d) / velocity_model.velmin[kphs] * data_sampling_rate + 0.5*paras['origin_time']['step']) * np.array(paras['loc_grid']['atscale'])).astype(int)
        nwdps = len(wdps_d)
        print(f"wdps: {wdps}")

        # Create a pool of processes
        if (paras['multiprocessing']['chunksize'] is not None) and (paras['multiprocessing']['chunksize'] < 1):
            chunksize = int(nt0/paras['multiprocessing']['processes'])+1
        else:
            chunksize = paras['multiprocessing']['chunksize']

        # before send to the pool, we need to take out the traveltime function which are not picklable
        # this must be addressed in the future, how to make it picklable
        traveltime_tt_fun = traveltime.tt_fun
        traveltime.tt_fun = None

        # print(f"processes: {paras['multiprocessing']['processes']}")
        # print(f"chunksize: {chunksize}")
        with Pool(processes=paras['multiprocessing']['processes']) as pool:
            # Compute migration in parallel

            iresults = pool.starmap_async(migration_ti, [(ii, t0_s, region, paras, traveltime, cf, cf_station, cf_starttime_s, data_sampling_rate, wdps, nwdps) for ii in range(nt0)], chunksize=chunksize)
            for iresults in iresults.get():
                output_x[iresults[0]], output_y[iresults[0]], output_z[iresults[0]] = iresults[2], iresults[3], iresults[4]
                if paras['save_result']['data_dim'] == 1:
                    # save the maximum value along XYZ, i.e., output_v[T]
                    output_v[iresults[0]] = iresults[1]
                elif paras['save_result']['data_dim'] == 2:
                    # save the maximum value along XY, YZ, XZ profiles, i.e., output_v[T, ny*nz, nx*nz, nx*ny]
                    # maximum value profiles at the source location along X-, Y-, Z-axis
                    raise ValueError("Not implemented yet!")
                elif paras['save_result']['data_dim'] == 3:
                    # save the maximum value along time axis, i.e., output_v[X,Y,Z]
                    np.maximum(output_v, iresults[1], out=output_v)
                elif paras['save_result']['data_dim'] == 4:
                    # save full migration values over time and space, i.e., output_v[T, X, Y, Z]
                    output_v[iresults[0]] = iresults[1]
                else:
                    raise ValueError("Invalid input for 'save_result:data_dim'.")
                
            # for iresults in pool.starmap(migration_ti, [(ii, t0_s, region, paras, traveltime, cf, cf_station, cf_starttime_s, data_sampling_rate, wdps, nwdps) for ii in range(nt0)], chunksize=chunksize):
            #     output_x[iresults[0]], output_y[iresults[0]], output_z[iresults[0]] = iresults[2], iresults[3], iresults[4]
            #     if paras['save_result']['data_dim'] == 1:
            #         # save the maximum value along XYZ, i.e., output_v[T]
            #         output_v[iresults[0]] = iresults[1]
            #     elif paras['save_result']['data_dim'] == 2:
            #         # save the maximum value along XY, YZ, XZ profiles, i.e., output_v[T, ny*nz, nx*nz, nx*ny]
            #         # maximum value profiles at the source location along X-, Y-, Z-axis
            #         raise ValueError("Not implemented yet!")
            #     elif paras['save_result']['data_dim'] == 3:
            #         # save the maximum value along time axis, i.e., output_v[X,Y,Z]
            #         np.maximum(output_v, iresults[1], out=output_v)
            #     elif paras['save_result']['data_dim'] == 4:
            #         # save full migration values over time and space, i.e., output_v[T, X, Y, Z]
            #         output_v[iresults[0]] = iresults[1]
            #     else:
            #         raise ValueError("Invalid input for 'save_result:data_dim'.")

        # restore the traveltime function after the pool
        traveltime.tt_fun = traveltime_tt_fun   
        output_t0 = np.array([t0_start+it0_s for it0_s in t0_s])  # origin times in UTCDateTime     

    elif paras['migration_engine'].lower() == 'partition':
        # partition the grid search into multiple smaller parts
        # similar to octree searching, if paras['partition']['number'] = 2 
        # if xrange <= paras['partition']['x_resolution'], stop x partition;
        # if yrange <= paras['partition']['y_resolution'], stop y partition;
        # if zrange <= paras['partition']['z_resolution'], stop z partition;
        # if trange <= paras['partition']['t_resolution'], stop t partition;
        # NOTE paras['partition']['t_resolution'] in samples, NOT IN SECONDS;

        ntgrid = 10  # for estimating the mininum and maximum traveltimes
        x_res = paras['partition']['x_resolution']  # x resolution in meters
        y_res = paras['partition']['y_resolution']  # y resolution in meters
        z_res = paras['partition']['z_resolution']  # z resolution in meters
        t_res = paras['partition']['t_resolution'] / data_sampling_rate  # t resolution in seconds

        # range for the selected best event location range (x, y, z, t)
        xrange_s = [region.x_min, region.x_max]
        yrange_s = [region.y_min, region.y_max]
        zrange_s = [region.z_min, region.z_max]
        trange_s = [t0_s[0], t0_s[-1]]  # searching origin time range in seconds
        
        xr_list = [xrange_s]
        yr_list = [yrange_s]
        zr_list = [zrange_s]
        tr_list = [trange_s]
        va_list = [None]

        evix = 0  # the index of current best estimate of the event in the list

        while (xrange_s[1]-xrange_s[0]>x_res) or (yrange_s[1]-yrange_s[0]>y_res) or (zrange_s[1]-zrange_s[0]>z_res) or (trange_s[1]-trange_s[0]>t_res):
            
            # remove the previous best estimate of the result list
            del xr_list[evix], yr_list[evix], zr_list[evix], tr_list[evix], va_list[evix]

            # partition the grid search in x direction
            if xrange_s[1]-xrange_s[0]>x_res:
                xpar = _parti(rng=xrange_s, num=paras['partition']['number'])
            else:
                xpar = [xrange_s]

            # partition the grid search in y direction
            if yrange_s[1]-yrange_s[0]>y_res:
                ypar = _parti(rng=yrange_s, num=paras['partition']['number'])
            else:
                ypar = [yrange_s]
            
            # partition the grid search in z direction
            if zrange_s[1]-zrange_s[0]>z_res:
                zpar = _parti(rng=zrange_s, num=paras['partition']['number'])
            else:
                zpar = [zrange_s]

            # partition the grid search in t direction
            if trange_s[1]-trange_s[0]>t_res:
                tpar = _parti(rng=trange_s, num=paras['partition']['number'])
            else:
                tpar = [trange_s]

            # Generate all combinations of xpar, ypar, zpar, and tpar
            par_combinations = itertools.product(xpar, ypar, zpar, tpar)

            # Iterate over all combinations and populate lists
            for ixpar, iypar, izpar, itpar in par_combinations:
                xr_list.append(ixpar)
                yr_list.append(iypar)
                zr_list.append(izpar)
                tr_list.append(itpar)
                iva = _migv_point(xrange=ixpar, yrange=iypar, zrange=izpar, trange=itpar, 
                                  cf_station=cf_station, cf=cf, cf_starttime_s=cf_starttime_s, 
                                  data_sampling_rate=data_sampling_rate, traveltime=traveltime, 
                                  paras=paras, ntgrid=ntgrid, region=region)
                va_list.append(iva)                 
            
            # find the current best estimate of the event location range
            evix = np.argmax(va_list)
            xrange_s = xr_list[evix]
            yrange_s = yr_list[evix]
            zrange_s = zr_list[evix]
            trange_s = tr_list[evix]
        
        # determine the final event location results
        output_x = np.array([np.mean(jxr) for jxr in xr_list])  # location in the middle of the range
        output_y = np.array([np.mean(jyr) for jyr in yr_list])
        output_z = np.array([np.mean(jzr) for jzr in zr_list])
        t0_s = np.array([np.mean(jtr) for jtr in tr_list])  # rewrite t0_s
        output_v = np.array(va_list)

        # In t0_s we might have multiple same values, we need to keep only the unique values
        # keep the one that has the maximum migration value (output_v)
        # Sort indices based on t0_s in ascending order and output_v in descending order
        sorted_indices = np.lexsort((-output_v, t0_s))

        # Find indices of unique elements in sorted order
        unique_indices = np.unique(t0_s[sorted_indices], return_index=True)[1]

        # Use indices to extract desired elements from t0_s and output_v
        final_indices = sorted_indices[unique_indices]

        output_x = output_x[final_indices]
        output_y = output_y[final_indices]
        output_z = output_z[final_indices]
        output_v = output_v[final_indices]
        t0_s = t0_s[final_indices]

        output_t0 = np.array([t0_start+it0_s for it0_s in t0_s])  # origin times in UTCDateTime
        
    else:
        raise ValueError(f"Invalid migration_engine: {paras['migration_engine']}!")
    
    # save results  
    if paras['save_result']['save_loc']:
        fname = os.path.join(dir_output, "migration_results.npz")
        np.savez(fname, output_t0=output_t0, output_v=output_v, 
                 output_x=output_x, output_y=output_y, output_z=output_z)

    # get final event location results
    if paras['event_pick'].lower() == 'max':
        evt0_id = np.unravel_index(np.argmax(output_v, axis=None), shape=output_v.shape)  # index of the maximum migration value
        ev_x = np.array([output_x[evt0_id[0]]])  # x coordinate of the event location
        ev_y = np.array([output_y[evt0_id[0]]])  # y coordinate of the event location
        ev_z = np.array([output_z[evt0_id[0]]])  # z coordinate of the event location
        ev_t0 = np.array([output_t0[evt0_id[0]]])  # origin time of the event in UTCDateTime
        ev_it0 = np.array([t0_s[evt0_id[0]]])  # origin time in second (relative to t0_start) of the event
        ev_mig = np.array([output_v[evt0_id]])  # migration value of the event
    else:
        raise ValueError(f"Invalid event_pick: {paras['event_pick']}")

    # polish over txyz using local optimization
    # apply local optimization to polish the final location results
    if paras['loc_grid']['local_opt'].lower() != 'none':
        for ll in range(len(ev_t0)):  # loop over each located event
            rext = 3.0
            bounds = [[ev_it0[ll]-t0_step*rext, ev_it0[ll]+t0_step*rext], 
                      [ev_x[ll]-region.dx*paras['loc_grid']['dnx'][-1]*rext, ev_x[ll]+region.dx*paras['loc_grid']['dnx'][-1]*rext], 
                      [ev_y[ll]-region.dy*paras['loc_grid']['dny'][-1]*rext, ev_y[ll]+region.dy*paras['loc_grid']['dny'][-1]*rext], 
                      [ev_z[ll]-region.dz*paras['loc_grid']['dnz'][-1]*rext, ev_z[ll]+region.dz*paras['loc_grid']['dnz'][-1]*rext]]
            x0 =  np.array([ev_it0[ll], ev_x[ll], ev_y[ll], ev_z[ll]])
            result = minimize(fun=objfunt, args=(cf_station, cf, cf_starttime_s, data_sampling_rate, traveltime, paras), 
                              bounds=bounds, x0=x0, method=paras['loc_grid']['local_opt'],
                              options={'maxiter': 100, 'xatol': 0.8, 'xtol': 0.01, 'gtol': 0.0001, 'ftol': 0.01, 'adaptive': True},)
            ev_it0[ll], ev_x[ll], ev_y[ll], ev_z[ll] = result.x 
            ev_t0[ll] = t0_start + ev_it0[ll]
            ev_mig[ll] = -result.fun

    return ev_x, ev_y, ev_z, ev_t0, ev_mig


def location_agg(data, file_parameter, traveltime, region, dir_output="./", velocity_model=None):
    """
    Input:
        data: data object, should be obspy stream;
        file_parameter: parameter file, str or dict object;
        traveltime: traveltime class object;
        region: region class object;
        dir_output: output directory, str;
        velocity_model: velocity model class object;

    Output:
        ev_x: 1D np.array, x coordinate of the event location;
        ev_y: 1D np.array, y coordinate of the event location;
        ev_z: 1D np.array, z coordinate of the event location;
        ev_t0: 1D np.array, origin time of the event;    

    """

    # load input paramters from file
    paras = xloc_input(file_para=file_parameter)

    if paras['method'] == 'xmig':
        ev_x, ev_y, ev_z, ev_t0, ev_mig = xmig(data=data, traveltime=traveltime, region=region, paras=paras, dir_output=dir_output, velocity_model=velocity_model)
    else:
        raise ValueError(f"Invalid method for location_agg: {paras['method']}")

    return ev_x, ev_y, ev_z, ev_t0, ev_mig


