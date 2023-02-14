'''
Functions for Seismic Phase Picking.

Autor: Peidong Shi
Contact: speedshi@hotmail.com  or  peidong.shi@sed.ethz.ch
Create time: 20221219

picks: dict, containing picking information,
picks['station_id']['P']: P-wave picked arrivaltime;
picks['station_id']['P_snr']: P-wave pick signal_noise_ratio;
picks['station_id']['S']: S-wave picked arrivaltime;
picks['station_id']['S_snr']: S-wave pick signal_noise_ratio;

arrivals: dict, containing theoretical arrivaltime information,
arrivals['station_id']['P']: P-wave arrivaltime;
arrivals['station_id']['S']: S-wave arrivaltime;
'''


import numpy as np


def prob2pick(trace_prob, pick_thrd):
    '''
    trace_prob: Obspy trace object,
                continuous phase probability of a particular phase (P or S) at a station.
    pick_thrd: float,
               picking threshold.
    '''

    picks = {}

    return picks


def picks_select(picks, arriv_para=None, snr_para=None):
    '''
    INPUT:
        picks: dict, picking information at each station;
            picks['station_id']['P']: P-wave picked arrivaltime;
            picks['station_id']['P_snr']: P-wave pick signal_noise_ratio;
            picks['station_id']['S']: S-wave picked arrivaltime;
            picks['station_id']['S_snr']: S-wave pick signal_noise_ratio;
        arriv_para: dict, arrival related information;
            arriv_para['arrivaltime']: dict, theoretical arrivaltimes;
                arriv_para['arrivaltime']['station_id']['P']: P-phase theoretical arrivaltimes;
                arriv_para['arrivaltime']['station_id']['S']: S-phase theoretical arrivaltimes;
            arriv_para['P_maxtd']: float,
                time duration in second, [P_theoratical_arrt-P_maxtd, P_theoratical_arrt+P_maxtd] 
                is the time range to consider possible ML picks for P-phase.
            arriv_para['S_maxtd']:float,
                time duration in second, [S_theoratical_arrt-S_maxtd, S_theoratical_arrt+S_maxtd] 
                is the time range to consider possible ML picks for S-phase.
        snr_para: dict, signal-to-noise ratio information;
            snr_para['P']: P-phase picking snr threshold;
            snr_para['S']: S-phase picking snr threshold;

    OUTPUT:
        picks_s: dict, selected picking that fillfull the input requirements.
    '''

    picks_s = {}
    phases = ['P', 'S']  # must consistant with picks and arrivaltimes dict

    for ista in list(picks.keys()):  # loop over each station_id
        for iphs in phases:  # loop over each phase
            if iphs in list(picks[ista].keys()):
                select = True
                if arriv_para is not None:
                    tmin = arriv_para['arrivaltime'][ista][iphs] - arriv_para[iphs+'_maxtd']
                    tmax = arriv_para['arrivaltime'][ista][iphs] + arriv_para[iphs+'_maxtd']
                    if (picks[ista][iphs] < tmin) or (picks[ista][iphs] > tmax):
                        # do not pass arrivaltime range requirement
                        select = False

                if snr_para is not None:
                    if (iphs+'_snr' not in picks[ista]) or (picks[ista][iphs+'_snr'] < snr_para[iphs]) or (picks[ista][iphs+'_snr'] == np.inf):
                        # do not pass snr requirement: no picking_snr or low picking snr or picking_snr = inf
                        select = False

                if select:
                    # pass selection criteria, add into list
                    if ista not in picks_s:
                        picks_s[ista] = {}
                    picks_s[ista][iphs] = picks[ista][iphs]
                    picks_s[ista][iphs+'_snr'] = picks[ista][iphs+'_snr']

    return picks_s


