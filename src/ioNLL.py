#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:21:48 2022

Input and output functions for NonLinLoc.

@author: shipe
"""


import os
import numpy as np
from xstation import get_station_ids


def gene_NLLinputs(inpara):
    """
    Generate NonLinLoc parameter inputs files.

    Parameters
    ----------
    inpara : dict, containing various input parameters.
        inpara['filename']: filename including path of the output text file. 
                            Default is './nlloc_sample.in'. Type: [str].
        inpara['TRANS']: coordinate transformation type, can be 'SIMPLE' or 'TRANS_MERC',
                         Default is 'SIMPLE'. Type: [str].
        inpara['refEllipsoid']: reference ellipsoid name, can be 'WGS-84' 'GRS-80'
                                'WGS-72' 'Australian' 'Krasovsky' 'International'
                                'Hayford-1909' 'Clarke-1880' 'Clarke-1866' 'Airy'
                                'Bessel' 'Hayford-1830' 'Sphere'. Default is 'WGS-84'.
                                Type: [str].
        inpara['LatOrig']: (float, min:-90.0, max:90.0) latitude in decimal degrees 
                           of the rectangular coordinates origin. Required.
        inpara['LongOrig']: (float, min:-180.0, max:180.0) longitude in decimal degrees 
                           of the rectangular coordinates origin. Required.
        inpara['rotAngle']: (float, min:-360.0, max:360.0) rotation angle in decimal degrees
                          of geographic north in degrees clockwise relative to 
                          the rectangular coordinates system Y-axis. Default is 0.0
        inpara['VGOUT']: Output Grid File Root Name, including path. 
                         Default is './model/layer'. Type: [str].
        inpara['xNum']: (integer, min:2) number of grid nodes in the X direction, required;
        inpara['yNum']: (integer, min:2) number of grid nodes in the Y direction, required;
        inpara['zNum']: (integer, min:2) number of grid nodes in the Z direction, required;
        inpara['xOrig']: x location of the grid origin in km relative to the geographic origin (positive: east).
                         Default value is 0.0. Type: [float].
        inpara['yOrig']: y location of the grid origin in km relative to the geographic origin (positive: north).
                         Default value is 0.0. Type: [float].
        inpara['zOrig']: z location of the grid origin in km relative to the sea-level (positive: down).
                         Nagative value means above the sea-level; Positive values for below the sea-level;
                         Required. Type: [float].
        inpara['dgrid']: grid node spacing in kilometers along the X, Y, and Z axes.
                         Currently only accept equal spcaing in the X, Y, and Z directions.
                         Required. Type: [float].
        inpara['depth_top']: depth to top of layer, use negative values for layers above z=0.
                             Required. Type: [float of numpy array].
        inpara['Vp_top']: P velocity in km/s at the top of the layer.
                          Required. Type: [float of numpy array].
        inpara['Vs_top']: S velocity in km/s at the top of the layer.
                          Required. Type: [float of numpy array].
        inpara['rho_top']: density in kg/m**3 at the top of the layer.
                          Required. Type: [float of numpy array].
        inpara['Vp_grad']: Linear P velocity gradients in km/s/km increasing directly downwards
                           from the top of the layer.
                           Type: [float of numpy array].
                           Default value: 0.0, means each layer has constant velocity.
        inpara['Vs_grad']: Linear S velocity gradients in km/s/km increasing directly downwards
                           from the top of the layer.
                           Type: [float of numpy array].
                           Default value: 0.0, means each layer has constant velocity.
        inpara['rho_grad']: Linear density gradients in kg/m**3/km increasing directly downwards
                            from the top of the layer.
                            Type: [float of numpy array].
                            Default value: 0.0, means each layer has constant density.
        inpara['ttfileroot']: path and file root name (no extension) for output 
                              travel-time and take-off angle grids.
                              Type: [str]. Default value: './time/layer'.
        inpara['ttwaveType']: wave type for generating travel-time and take-off angle grids.
                              Type: [str]. Default value: 'P'.
        inpara['stainv']: station inventory object, see in Obspy for detail.
                          Required.
        
    Returns
    -------
    None.

    """
    
    # input parameters
    if 'filename' in inpara:
        filename = inpara['filename']
    else:
        filename = './nlloc.in'
    
    if 'TRANS' in inpara:
        TRANS = inpara['TRANS']
    else:
        TRANS = 'SIMPLE'
    
    if 'refEllipsoid' in inpara:
        refEllipsoid = inpara['refEllipsoid']
    else:
        refEllipsoid = 'WGS-84'
    
    LatOrig = inpara['LatOrig']
    LongOrig = inpara['LongOrig']
    
    if 'rotAngle' in inpara:
        rotAngle = inpara['rotAngle']
    else:
        rotAngle = 0.0
    
    if 'VGOUT' in inpara:    
        VGOUT = inpara['VGOUT']
    else:
        VGOUT = './model/layer'
    
    xNum = inpara['xNum']    
    yNum = inpara['yNum']
    zNum = inpara['zNum']
    
    if 'xOrig' in inpara:
        xOrig = inpara['xOrig']
    else:
        xOrig = 0.0
        
    if 'yOrig' in inpara:
        yOrig = inpara['yOrig']
    else:
        yOrig = 0.0
        
    zOrig = inpara['zOrig']
    
    dx = inpara['dgrid']
    dy = inpara['dgrid']
    dz = inpara['dgrid']
    
    depth_top = inpara['depth_top']
    Vp_top = inpara['Vp_top']
    Vs_top = inpara['Vs_top']
    rho_top = inpara['rho_top']
    NLY = len(depth_top)  # total number of layers
    if 'Vp_grad' in inpara:
        Vp_grad = inpara['Vp_grad']
    else:
        Vp_grad = np.zeros((NLY,),)
    if 'Vs_grad' in inpara:
        Vs_grad = inpara['Vs_grad']
    else:
        Vs_grad = np.zeros((NLY,),)
    if 'rho_grad' in inpara:
        rho_grad = inpara['rho_grad']
    else:
        rho_grad = np.zeros((NLY,),)
    
    if 'ttfileroot' in inpara:
        ttfileroot = inpara['ttfileroot']
    else:
        ttfileroot = './time/layer'
    
    if 'ttwaveType' in inpara:
        ttwaveType = inpara['ttwaveType']
    else:
        ttwaveType = 'P'
    
    stainv = inpara['stainv']  # station inventory
    
    # check if the input file already exist or not.
    if os.path.exists(filename):
        raise ValueError('File: {} already exist!'.format(filename))
    
    ofile = open(filename, 'a')
    
    # Control
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# Generic control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('CONTROL 1 54321\n')
    ofile.flush()
    
    # Geographic Transformation
    if TRANS == 'SIMPLE':
        ofile.write('TRANS    {}    {}    {}    {}\n'.format(TRANS, LatOrig, LongOrig, rotAngle))
    elif TRANS == 'TRANS_MERC':
        ofile.write('TRANS    {}    {}    {}    {}    {}\n'.format(TRANS, refEllipsoid, LatOrig, LongOrig, rotAngle))
    ofile.write('# =============================================================================\n')
    ofile.write('# END of Generic control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('\n')
    ofile.flush()
    
    # For Vel2Grid Program
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# Vel2Grid control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('VGOUT    {}\n'.format(VGOUT))  # Output Grid File Root Name
    ofile.write('VGTYPE    P\n')  # Wave Type
    ofile.write('VGTYPE    S\n')  # Wave Type
    ofile.write('VGGRID    {}    {}    {}    {}    {}    {}    {}    {}    {}    SLOW_LEN\n'.format(
                                                                                          xNum, yNum, zNum,
                                                                                          xOrig, yOrig, zOrig,
                                                                                          dx, dy, dz))  # velocity grid description
    for ily in range(NLY):  # layered velocity model description
        ofile.write('LAYER    {}    {}    {}    {}    {}    {}    {}\n'.format(depth_top[ily],
                                                                               Vp_top[ily], Vp_grad[ily],
                                                                               Vs_top[ily], Vs_grad[ily],
                                                                               rho_top[ily], rho_grad[ily]))
    ofile.write('# =============================================================================\n')
    ofile.write('# END of Vel2Grid control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('\n')
    ofile.flush()
    
    # For Grid2Time Program
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# Grid2Time control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('GTFILES    {}    {}    {}\n'.format(VGOUT, ttfileroot, ttwaveType))
    ofile.write('GTMODE    GRID3D    ANGLES_YES\n')  # time grid modes
    staids, stainfo = get_station_ids(stainv)  # get the unique station id and location information
    for istaid in staids:  # loop over each unique station
        sta_id = istaid
        sta_latitude = stainfo[istaid]['latitude']
        sta_longitude = stainfo[istaid]['longitude']
        sta_elevation = stainfo[istaid]['elevation'] / 1000.0
        if 'depth' in stainfo[istaid]:
            sta_depth = stainfo[istaid]['depth'] / 1000.0
        else:
            sta_depth = 0.0
        ofile.write('GTSRCE    {}    LATLON    {}    {}    {}    {}\n'.format(
                    sta_id, sta_latitude, sta_longitude, sta_depth, sta_elevation))
    ofile.write('GT_PLFD    1.0e-3    2\n')  # Podvin & Lecomte FD params
    ofile.write('# =============================================================================\n')
    ofile.write('# END of Grid2Time control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('\n')
    ofile.flush()
    
    ### TO BE CONTINUE
    
    ofile.close()
    print("NonLinLoc input file compiled at: {}.".format(filename))
    return


def gene_NLLinputs_NLLoc(inpara):
    
    # input parameters
    if 'filename' in inpara:
        filename = inpara['filename']
    else:
        filename = './nlloc.in'
        
    # check if the input file already exist or not.
    if os.path.exists(filename):
        print('Append NLLoc control file statements to NLLInputfile: {}.'.format(filename))
    else:
        print('Create NLLInputfile: {} for NLLoc control file statements.'.format(filename))
    
    ofile = open(filename, 'a')    
    
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# NLLoc control file statements\n')
    ofile.write('# =============================================================================\n')

    ### TO BE CONTINUE

    ofile.write('# =============================================================================\n')
    ofile.write('# END of NLLoc control file statements\n')
    ofile.write('# =============================================================================\n')
    ofile.write('# =============================================================================\n')
    ofile.write('\n')
    ofile.flush()
    
    ofile.close()
    return


