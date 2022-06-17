#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:45:52 2022

@author: shipe
"""


from xrtdd import output_rtddstation, output_rtddeventphase


def event_reloc(RELOC):
    """
    Obtain relative relocated catalog.
    Perform relative relocation or prepare files for relative relocation

    Parameters
    ----------
    RELOC : dict
        parameters related to event relocation.
        RELOC['catalog'] : dict
            the input catalog to be relocated.
        RELOC['engine'] : str, default is 'rtdd'
            the event relative relocation approach, 
            can be: 'rtdd',
        RELOC['dir_output'] : str
            directory for outputting files.

    Returns
    -------
    None.

    """
    
    if RELOC['engine'] == 'rtdd':
        # generate files for rtdd relocation
        output_rtddstation(stainv=RELOC['stainv'], dir_output=RELOC['dir_output'], filename='station_rtdd.csv')
        output_rtddeventphase(catalog=RELOC['catalog'], stainv=RELOC['stainv'], dir_output=RELOC['dir_output'], 
                              filename_event='event_rtdd.csv', filename_phase='phase_rtdd.csv',
                              phaseart_ftage='.MLpicks', station_channel_codes=RELOC['channel_codes'])  # Note must use '.MLpicks' to use ML picks, theoratical arrivaltimes do not work
    else:
        raise ValueError('Incorrect input for RELOC[\'engine\']!')

    return


