# perform seismic phase association


import pandas as pd
import numpy as np
import yaml


def asso_input(file_para):

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
        if not paras['method'].upper() in ['SIMPLE']:
            raise ValueError("'phassoc:method' should be either 'simple'.")

    # check required parameters for 'simple' method 
    if paras['method'].upper() == "SIMPLE":
        if "time_span" not in paras:
            raise ValueError("phassoc parameter file should contain 'time_span' setting for the 'simple' phassoc method.")
        else:
            if not isinstance(paras['time_span'], (int, float,)):
                raise ValueError("'phassoc:time_span' should be a number.")
            else:
                if (paras['time_span'] <= 0):
                    raise ValueError("'phassoc:time_span' should be a positive number.")

        if 'time_split' not in paras:
            raise ValueError("phassoc parameter file should contain 'time_split' setting for the 'simple' phassoc method.")
        else:
            if not isinstance(paras['time_split'], (int, float,)):
                raise ValueError("'phassoc:time_split' should be a number.")
            else:
                if (paras['time_split'] <= 0):
                    raise ValueError("'phassoc:time_split' should be a positive number.")

        if 'n_station' not in paras:
            paras['n_station'] = 0
        elif isinstance(paras['n_station'], (int,)):
            assert(paras['n_station']>=0)
        else:
            raise ValueError("'phassoc:n_station' should be a non-negative integer.")

        if 'n_pick_P' not in paras:
            paras['n_pick_P'] = 0
        elif isinstance(paras['n_pick_P'], (int,)):
            assert(paras['n_pick_P']>=0)
        else:
            raise ValueError("'phassoc:n_pick_P' should be a non-negative integer.")

        if 'n_pick_S' not in paras:
            paras['n_pick_S'] = 0
        elif isinstance(paras['n_pick_S'], (int,)):
            assert(paras['n_pick_S']>=0)
        else:
            raise ValueError("'phassoc:n_pick_S' should be a non-negative integer.")
        
        if 'n_pick_all' not in paras:
            paras['n_pick_all'] = 0
        elif isinstance(paras['n_pick_all'], (int,)):
            assert(paras['n_pick_all']>=0)
        else:
            raise ValueError("'phassoc:n_pick_all' should be a non-negative integer.")

    return paras


def asso_simple(pick: pd.DataFrame, paras):

    # sort all picks by peak_time/pick_time in ascending order
    pick = pick.sort_values(by='peak_time', ascending=True).reset_index(drop=True)

    # initialize output
    output = {}
    output['time_range'] = []  # list of time range for each event
    output['pick'] = []  # list of pick dateframe for each event

    npicks = len(pick)  # total number of picks
    ii = 0
    while ii < npicks:
        t_start = pick['peak_time'].iloc[ii]
        t_end_est = t_start + paras['time_span']
        ii_end_act = (pick['peak_time'] <= t_end_est)[::-1].idxmax()  

        # the next pick time should be at least "time_split" away from the last (end) pick
        ii_end = np.searchsorted(pick['peak_time'], pick['peak_time'].iloc[ii_end_act] + paras['time_split'], side='left') - 1

        t_end = pick['peak_time'].iloc[ii_end]
        ev_picks = pick.iloc[ii:ii_end+1].copy().reset_index(drop=True)
        assert(((ev_picks['peak_time'] >= t_start) & (ev_picks['peak_time'] <= t_end)).all())

        # count number of picked stations and picks
        n_station = len(ev_picks['trace_id'].unique())  # total number of stations picked
        n_pick_P = len(ev_picks[ev_picks['phase'] == 'P'])  # total number of P picks
        n_pick_S = len(ev_picks[ev_picks['phase'] == 'S'])  # total number of S picks
        n_pick_all = len(ev_picks)  # total number of all picks
        assert(n_pick_P+n_pick_S == n_pick_all)  # check

        # check if the current detection fullfill the requirement
        if ((n_station >= paras['n_station']) and  # enough station triggered
            (n_pick_P >= paras['n_pick_P']) and  # enough P picks
            (n_pick_S >= paras['n_pick_S']) and  # enough S picks
            (n_pick_all >= paras['n_pick_all'])):  # enough all picks
            # fullfill event requirement, add to output
            output['time_range'].append([t_start, t_end])
            output['pick'].append(ev_picks)
            print(f"Potential events found from {t_start} to {t_end}.")
            print(f"Number of picked stations: {n_station}, P-picks: {n_pick_P}, S-picks: {n_pick_S}, all picks: {n_pick_all}.")
        else:
            print(f"No valid events are found from {t_start} to {t_end_est}.")

        # update index
        ii = ii_end + 1

    return output


def asso(pick: pd.DataFrame, file_para):

    paras = asso_input(file_para=file_para)

    if paras['method'].upper() == 'SIMPLE':
        output = asso_simple(pick=pick, paras=paras)
    else:
        raise ValueError("Unknown phase association method.")

    return output







