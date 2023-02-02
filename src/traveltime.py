#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:43:45 2022

Functions for generating traveltimes.

@author: shipe
"""


import numpy as np
from ioNLL import gene_NLLinputs
from xstation import get_station_ids
import os
import subprocess
import warnings


def read_NLLvel(vfile):
    """
    Load velocity model of NonLinLoc format.
    The velocity file specifies a constant or gradient velocity layer.
    
    Format of the velocity model file:
        depth Vp_top Vp_grad Vs_top Vs_grad rho_top rho_grad
    
    depth: (float) depth to top of layer (use negative values for layers above z=0)
    Vp_top Vs_top rho_top: (float) P velocity, and S velocity in km/s and density in kg/m**3 at the top of the layer.
    Vp_grad Vs_grad rho_grad: (float) Linear P velocity and S velocity gradients in km/s/km and density gradient in kg/m**3/km increasing directly downwards from the top of the layer.
    Notes:
        1. Multiple layers must be specified in order of increasing depth of top of layer.
        2. The layer with the deepest top extends implicitly to infinite depth.
    
    Returns
    -------
    model: dict, containing info about the velocity model.

    """
    
    vmodel = np.loadtxt(vfile)
    model = {}    
    model['depth_top'] = vmodel[:,0]
    model['Vp_top'] = vmodel[:,1]
    model['Vp_grad'] = vmodel[:,2]
    model['Vs_top'] = vmodel[:,3]
    model['Vs_grad'] = vmodel[:,4]
    model['rho_top'] = vmodel[:,5]
    model['rho_grad'] = vmodel[:,6]
    
    return model


def check_NLLtt(ttfileroot, stainv):
    """
    Check traveltime tables are correct and complete.

    Parameters
    ----------
    ttfileroot : str
        path and file root name of travel-time tables.
    stainv : obspy station inventory object.
        obspy station inventory containing the station information.

    Returns
    -------
    None.

    """
    
    staids, _ = get_station_ids(stainv)

    for ista in staids:        
        phases = ['P', 'S']
        tformat = ['hdr', 'buf']
        for ips in phases:
            for itf in tformat:
                tfile = ttfileroot + '.{}.{}.time.{}'.format(ips, ista, itf)
                if not os.path.isfile(tfile):
                    warnings.warn('Traveltime file: {} does not exist!'.format(tfile))

    return


def build_tthdr(ttdir, ttftage, stainv, filename='header.hdr'):
    """
    To build the header file from NonLinLoc traveltime data set. The header file
    is needed for the LOKI migration module.

    Parameters
    ----------
    ttdir : str
        dirctory of the traveltime data set.
    ttftage : str
        traveltime data set filename tage as used for NonLinLoc,
        used at the file root name.
    stainv : obspy station inventory object.
        obspy station inventory containing the station information.
    filename : str, optional, default is 'header.hdr'.
        filename of the generated hearder file. 

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    # check if the input file already exist or not.
    ofile = os.path.join(ttdir, filename)
    if os.path.isfile(ofile):
        warnings.warn('Header file: {} already exist! Skip header file generation, use the exist header file.'.format(ofile))
    else:
        # compile header file
        staids, stainfo = get_station_ids(stainv)

        ofile = open(ofile, 'a')
        ii = 0
        for ista in staids:
            statthdr = os.path.join(ttdir, ttftage+'.P.{}.time.hdr'.format(ista))
            if os.path.isfile(statthdr):
                f = open(statthdr)
                lines = f.readlines()
                f.close()
                if ii == 0:
                    line0 = lines[0].split()
                    ofile.write('{} {} {}\n'.format(line0[0], line0[1], line0[2]))  # xNum, yNum, zNum
                    ofile.write('{} {} {}\n'.format(line0[3], line0[4], line0[5]))  # xOrig, yOrig, zOrig
                    ofile.write('{} {} {}\n'.format(line0[6], line0[7], line0[8]))  # dx, dy, dz
                    line2 = lines[2].split()
                    ofile.write('{} {}\n'.format(line2[3], line2[5]))  # LatOrig, LongOrig

                ofile.write('{} {} {} {}\n'.format(ista, stainfo[ista]['latitude'], stainfo[ista]['longitude'], stainfo[ista]['elevation']/1000.0))  # station_code, station_lat, station_lon, station_ele
                ofile.flush()
                ii += 1
            else:
                warnings.warn('Traveltime file: {} does not exist!'.format(statthdr))
        ofile.close()
    return


def build_traveltime(grid, tt, stainv):
    """
    To build the traveltime tables for migration engine using NonLinLoc.

    Returns
    -------
    None.

    """

    # format NonLinLoc input file and generate traveltime tables
    if (tt['vmodel'] is not None) and (grid is not None):
        print("Start to compile NonLinLoc input file for generating traveltime tables.")
        if not os.path.exists(tt['dir']):
            os.makedirs(tt['dir'], exist_ok=True)
        vmodel = read_NLLvel(tt['vmodel'])
        inpara = {}
        inpara['VGOUT'] = os.path.join(tt['dir'], tt['ftage'])
        inpara['ttfileroot'] = os.path.join(tt['dir'], tt['ftage'])
        inpara['LatOrig'] = grid['LatOrig']
        inpara['LongOrig'] = grid['LongOrig']
        inpara['rotAngle'] = grid['rotAngle']
        inpara['xOrig'] = grid['xOrig']
        inpara['yOrig'] = grid['yOrig']
        inpara['zOrig'] = grid['zOrig']
        inpara['xNum'] = grid['xNum']
        inpara['yNum'] = grid['yNum']
        inpara['zNum'] = grid['zNum']
        inpara['dgrid'] = grid['dgrid']
        inpara['depth_top'] = vmodel['depth_top']
        inpara['Vp_top'] = vmodel['Vp_top']
        inpara['Vp_grad'] = vmodel['Vp_grad']
        inpara['Vs_top'] = vmodel['Vs_top']
        inpara['Vs_grad'] = vmodel['Vs_grad']
        inpara['rho_top'] = vmodel['rho_top']
        inpara['rho_grad'] = vmodel['rho_grad']
        inpara['stainv'] = stainv
        inpara['filename'] = os.path.join(tt['dir'], 'nlloc_P_tt.in')
        inpara['ttwaveType'] = 'P'
        gene_NLLinputs(inpara)  # for P traveltimes
        subprocess.run(["Vel2Grid", inpara['filename']])  # call NonLinLoc Vel2Grid program to generate velocity grid
        subprocess.run(["Grid2Time", inpara['filename']])  # call NonLinLoc Grid2Time program to generate P traveltime tables
        inpara['filename'] = os.path.join(tt['dir'], 'nlloc_S_tt.in')
        inpara['ttwaveType'] = 'S'
        gene_NLLinputs(inpara)  # for S traveltimes
        subprocess.run(["Grid2Time", inpara['filename']])  # call NonLinLoc Grid2Time program to generate S traveltime tables
    
    # check traveltime tables are correct and complete
    check_NLLtt(os.path.join(tt['dir'], tt['ftage']), stainv)
        
    # compile traveltime table header file
    build_tthdr(tt['dir'], tt['ftage'], stainv, filename=tt['hdr_filename'])
    
    # retive grid parameter from header file
    if (grid is None):
        grid = header2grid(os.path.join(tt['dir'], tt['hdr_filename']))
    
    return grid
    
    
def header2grid(file_header):
    """
    Retrive the grid information from the header file of traveltime data set.

    Parameters
    ----------
    file_header : str
        filename including full path of the header file.

    Returns
    -------
    grid : dict
        grid parameters.
        
    """
    
    f = open(file_header)
    lines = f.readlines()
    f.close()

    grid = {}
    grid['xNum'] = int(lines[0].split()[0])
    grid['yNum'] = int(lines[0].split()[1])
    grid['zNum'] = int(lines[0].split()[2])
    grid['xOrig'] = float(lines[1].split()[0])
    grid['yOrig'] = float(lines[1].split()[1])
    grid['zOrig'] = float(lines[1].split()[2])
    assert(lines[2].split()[0] == lines[2].split()[1] == lines[2].split()[2])  # currently NonLinLoc requires dx=dy=dz
    grid['dgrid'] = float(lines[2].split()[0]) 
    grid['LatOrig'] = float(lines[3].split()[0])
    grid['LongOrig'] = float(lines[3].split()[1])
    
    return grid



