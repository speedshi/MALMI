#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:44:06 2022

Initiate the project and create necessary directories that will be used later.

@author: shipe
"""


import os


def malmi_project_init(para: dict,):
    
    if not os.path.exists(para['project_root']):
        os.makedirs(para['project_root'])
        print(f"Project root directory {para['project_root']} created.")

    if para['results']:
        if not os.path.exists(para['results']):
            os.makedirs(para['results'])
            print(f"Results directory {para['results']} created.")
    
    fld_pick = "phase_pick"
    para['phase_pick'] = os.path.join(para['project_root'], fld_pick)
    if not os.path.exists(para['phase_pick']):
        os.makedirs(para['phase_pick'])
        print(f"Phase_pick directory {para['phase_pick']} created.")

    fld_log = "log"
    para['log'] = os.path.join(para['project_root'], fld_log)
    if not os.path.exists(para['log']):
        os.makedirs(para['log'])
        print(f"Log directory {para['log']} created.")

    fld_seismic_raw = "data/seismic_raw"
    para['seismic_raw'] = os.path.join(para['project_root'], fld_seismic_raw)
    if not os.path.exists(para['seismic_raw']):
        os.makedirs(para['seismic_raw'])
        print(f"Seismic_raw directory {para['seismic_raw']} created.")

    return para























