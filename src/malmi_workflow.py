

import os
from xinputs import load_check_input
from xproject_init import malmi_project_init
from obspy.clients.fdsn import Client
from xstation import station as stacls
from quakephase import quakephase
from phassoc import asso
from xvelocity import velocity
from xseismic_loader import xseismic_loader
import sys
from xtraveltime import traveltime
from xregion import region
import numpy as np
from xloc import location_agg
import time
from obspy import Stream


fmt_datetime = "%Y%m%dT%H%M%SS%f"


def _oneflow(stream, paras: dict, station_seis, traveltime_seis, region_monitor):

    # pick and characterize seismic phases
    time_now = time.time()
    print(f"Apply phase picking and characterization to seismic data:")
    output_phasepp = quakephase.apply(data=stream, file_para=paras['phase_pick']['parameter_file'])
    time_pre, time_now = time_now, time.time()
    print(f"Phase picking and characterization finished, use time: {time_now - time_pre} s.")

    # need to revise the channel code for the phase probability
    # should have 3 characters, the first two are the instrument code, the last one is the phase code
    # the instrument code is kept the same as the original seismic data
    for iprob in output_phasepp['prob']:
        if len(iprob.stats.channel) > 3:
            if iprob.stats.channel[-1] in ['P', 'S']:
                # only keep P and S probability
                iinstru = stream.select(network=iprob.stats.network,
                                        station=iprob.stats.station,
                                        location=iprob.stats.location)[0].stats.channel[:2]
                iprob.stats.channel = f"{iinstru}{iprob.stats.channel[-1]}"
            else:
                # note noise (N) and detection (detection) probabilities are removed
                output_phasepp['prob'].remove(iprob)

    # save phase picks and probabilities if required
    if paras['phase_pick']['save_pick']:
        file_pick = os.path.join(paras['dir']['phase_pick'], f"{paras['id']['results']}_pick.csv")
        output_phasepp['pick'].to_csv(file_pick, index=False)
    if paras['phase_pick']['save_prob']:
        file_prob = os.path.join(paras['dir']['phase_pick'], f"{paras['id']['results']}_prob.mseed")
        output_phasepp['prob'].write(file_prob, format='MSEED')

    # associate phase picks to events
    time_now = time.time()
    print(f"Event detection and association:")
    output_asso = asso(pick=output_phasepp['pick'], file_para=paras['phase_asso']['file'])
    time_pre, time_now = time_now, time.time()
    print(f"Event detection and association finished, use time: {time_now - time_pre} s.")
    
    # locate events for each detection time range
    for jtimerrg, jpick in zip(output_asso['time_range'], output_asso['pick']):
        # loop over each detection time range
        # there might be multiple events in one detection time range: short inter-event times
        # and ovelapping events that can not be easily separated.

        jfld_event = f"{jtimerrg[0].strftime(fmt_datetime)}_{jtimerrg[1].strftime(fmt_datetime)}"
        jdir_event = os.path.join(paras['dir']['results'], jfld_event)
        
        if paras['phase_asso']['time_buffer'] == 'auto':
            # automatically determine the time buffer
            jtimediff = jtimerrg[1] - jtimerrg[0]
            paras['phase_asso']['time_buffer'] = np.max([(traveltime_seis.tt_max - traveltime_seis.tt_min) - jtimediff, 0])
        jtt1 = jtimerrg[0] - paras['phase_asso']['time_buffer']  # starttime of the detection range, with buffer
        jtt2 = jtimerrg[1] + paras['phase_asso']['time_buffer']  # endtime of the detection range, with buffer
        jprob = output_phasepp['prob'].slice(starttime=jtt1, endtime=jtt2).copy()
        jstream = stream.slice(starttime=jtt1, endtime=jtt2).copy()  # make a physical copy of the stream

        if not os.path.exists(jdir_event):
            os.makedirs(jdir_event)
            print(f"Event directory {jdir_event} created.")

        if paras['phase_asso']['save_pick']:
            # save phase picks for the current detection range
            jfile_pick = os.path.join(jdir_event, "pick_all.csv")
            jpick.to_csv(jfile_pick, index=False)
        if paras['phase_asso']['save_prob']:
            # save phase probabilities for the current detection range
            jfile_prob = os.path.join(jdir_event, "prob_all.mseed")
            jprob.write(jfile_prob, format='MSEED') 
        if paras['phase_asso']['save_seis']:
            # save raw seismic data for the current detection range
            jfile_seis = os.path.join(jdir_event, "seismic_raw.mseed")
            jstream.write(jfile_seis, format='MSEED')

        # locate the event
        time_now = time.time()
        print(f"Locate events for the detection range: {jtimerrg[0]} to {jtimerrg[1]}")
        jsource_x, jsource_y, jsource_z, jsource_t0 = location_agg(data=jprob.copy(), file_parameter=paras['event_location']['file'], traveltime=traveltime_seis, region=region_monitor, dir_output=jdir_event)
        time_pre, time_now = time_now, time.time()
        print(f"Event location finished, use time: {time_now - time_pre} s.")

        # estimate event magnitude

        # write located event into catalog file


    return




def malmi_workflow(file_parameter: str):
    """
    This function is the integrated workflow of the MALMI. 
    It will read the parameter file and execute the workflow.
    INPUT
        param file_parameter: str, 
        yaml file that contains the parameters for configuring the workflow.
    """

    time_pre = time.time()

    # Save the current sys.stdout
    original_stdout = sys.stdout

    # load and check input parameters
    print(f"Load and check input parameters from {file_parameter}.")
    paras_in = load_check_input(file_para=file_parameter)
    time_now = time.time()
    print(f"Load and check input paramter finished, use time: {time_now - time_pre} s.")

    # setup the project directory
    print(f"Check and setup project directory.")
    paras_in['dir'] = malmi_project_init(para=paras_in['dir'])
    paras_in['id'] = {}
    time_pre, time_now = time_now, time.time()
    print(f"Project directory setup finished, use time: {time_now - time_pre} s.")

    # load station inventory
    print(f"Load station inventory.")  
    stations = stacls(file_station=paras_in['station']['file'])  # station class
    time_pre, time_now = time_now, time.time()
    print(f"Load station inventory finished, use time: {time_now - time_pre} s.")

    # determine the region of interest and projection system: the UTM_zone, etc
    print(f"Load monitoring region information and determine projection system.")
    if 'latitude_min' not in paras_in['region']:
        print(f"Minimum latitude of monitoring region not set, use the minimum latitude from the station inventory.")
        paras_in['region']['latitude_min'] = stations.latitude_min
    if 'latitude_max' not in paras_in['region']:
        print(f"Maximum latitude of monitoring region not set, use the maximum latitude from the station inventory.")
        paras_in['region']['latitude_max'] = stations.latitude_max
    if 'longitude_min' not in paras_in['region']:
        print(f"Minimum longitude of monitoring region not set, use the minimum longitude from the station inventory.")
        paras_in['region']['longitude_min'] = stations.longitude_min
    if 'longitude_max' not in paras_in['region']:
        print(f"Maximum longitude of monitoring region not set, use the maximum longitude from the station inventory.")
        paras_in['region']['longitude_max'] = stations.longitude_max
    if 'depth_min' not in paras_in['region']:
        paras_in['region']['depth_min'] = stations.elevation_max * -1.0
        print(f"Minimum depth of monitoring region not set, use the maximum elevation from the station inventory *-1.0: {paras_in['region']['depth_min']}.")
    region_monitor = region(**paras_in['region'])  # monitoring region class
    print(F"Monitoring region information: {region_monitor}.")
    time_pre, time_now = time_now, time.time()
    print(f"Load monitoring region information and determine projection system finished, use time: {time_now - time_pre} s.")

    # get UTM coordinates of stations, add to the station class
    print(f"Add UTM coordinates to the station inventory.")
    sta_x, sta_y, sta_z = region_monitor.coordsystem.lonlatele2xyz(longitude=stations.longitude, latitude=stations.latitude, elevation=stations.elevation)
    stations.append_xyz(x=sta_x, y=sta_y, z=sta_z)  # z is depth, downward positive
    time_pre, time_now = time_now, time.time()
    print(f"Add UTM coordinates to the station inventory finished, use time: {time_now - time_pre} s.")

    # load velocity model
    print(f"Load velocity model.")
    velocity_model = velocity(file=paras_in['velocity']['file'], file_format=paras_in['velocity']['format'], velocity_type=paras_in['velocity']['type'])
    time_pre, time_now = time_now, time.time()
    print(f"Load velocity model finished, use time: {time_now - time_pre} s.")

    # generate traveltime tables/functions
    print(f"Generate traveltime tables/functions.")
    travelts = traveltime(station=stations, velocity=velocity_model, region=region_monitor,
                          seismic_phase=['P', 'S'], **paras_in['traveltime'])
    time_pre, time_now = time_now, time.time()
    print(f"Generate traveltime tables/functions finished, use time: {time_now - time_pre} s.")
    # return travelts, stations, region_monitor, velocity_model  # for testing

    # set seismic data loader
    if paras_in['seismic_data']['get_data'].upper() == "FDSN":
        # get data from FDSN server
        client = Client(base_url=paras_in['seismic_data']['data_source'], 
                        user=paras_in['seismic_data']['user'], 
                        password=paras_in['seismic_data']['password'])
        setattr(client, 'get_waveforms', client.get_waveforms_bulk)
    elif paras_in['seismic_data']['get_data'].upper() == "AIO":
        # get data from local storage in one folder
        # "AIO" means "All In One folder"
        client = xseismic_loader(load_type=paras_in['seismic_data']['get_data'], 
                                 data_source=paras_in['seismic_data']['data_source'], 
                                 file_exclude=paras_in['seismic_data']['file_exclude'], 
                                 write_loaded_to_exclude=True)
        stream_keep = Stream()
        
    # request seismic data and process
    tt1 = paras_in['seismic_data']['starttime']  # starttime of the first time period
    if tt1 is not None:
        tt2 = paras_in['seismic_data']['starttime'] + paras_in['seismic_data']['processing_time']  # endtime of the first time period
    else:
        tt2 = None

    while (paras_in['seismic_data']['endtime'] is None) or (tt1<paras_in['seismic_data']['endtime']):
        # this is a loop for processing seismic data in different time periods

        # set the name id for the current time period, will update for each loop
        paras_in['id']['results'] = f"{tt1.strftime(fmt_datetime)}_{tt2.strftime(fmt_datetime)}"
        
        file_log = os.path.join(paras_in['dir']['log'], f"{paras_in['id']['results']}.log")
        sys.stdout = open(file_log, "w", buffering=1)
        print("Working on time period: ", tt1, " to ", tt2)

        # add buffer time to the start and end time
        if tt1 is not None:
            tt1_bf = tt1 - paras_in['seismic_data']['buffer_time']  # buffered starttime
        else:
            tt1_bf = None
        if tt2 is not None:
            tt2_bf = tt2 + paras_in['seismic_data']['buffer_time']  # buffered endtime
        else:
            tt2_bf = None

        # compile the data request info
        if paras_in['seismic_data']['get_data'].upper() == "FDSN":
            rinfo = []
            for inet, ista in zip(stations.network, stations.station):
                rinfo.append((inet, ista, "*", "*", tt1_bf, tt2_bf))
            print("Requesting seismic data from stations: ", rinfo)
        else:
            rinfo = {}
            rinfo['load_number'] = paras_in['seismic_data']['load_number']
            rinfo['file_order'] = paras_in['seismic_data']['file_order']
    
        # request seismic data
        time_now = time.time()
        stream = client.get_waveforms(rinfo)
        time_pre, time_now = time_now, time.time()
        print(f"Requesting seismic data finised, use time: {time_now - time_pre} s.")

        if paras_in['seismic_data']['get_data'].upper() == "FDSN":
            stream.trim(starttime=tt1_bf, endtime=tt2_bf)
        else:
            if tt2 is not None:
                stream += stream_keep.copy()
                stream_keep = stream.copy().trim(starttime=tt2-paras_in['seismic_data']['buffer_time'])
            stream.trim(starttime=tt1_bf, endtime=tt2_bf)

        if stream:  
            print("Data available for time period: ", tt1, " to ", tt2)
            print("Start processing...")
            if paras_in['dir']['results_tag'] is None:
                # results are saved separately for each time period
                paras_in['dir']['results'] = os.path.join(paras_in['dir']['project_root'], "results", paras_in['id']['results'])
                if not os.path.exists(paras_in['dir']['results']):
                    os.makedirs(paras_in['dir']['results'])
                    print(f"Results directory {paras_in['dir']['results']} created.")

            # save raw seismic data if required
            if paras_in['seismic_data']['save_raw']:
                file_raw = os.path.join(paras_in['dir']['seismic_raw'], f"{paras_in['id']['results']}_seismic_raw.mseed")
                stream.write(file_raw, format='MSEED')

            # process the current seismic data 
            _oneflow(stream=stream, paras=paras_in, station_seis=stations, traveltime_seis=travelts, region_monitor=region_monitor)

            print(f"Processing finished---------------------------------------------------")
            print("")

        else:
            print("No data available for time period: ", tt1, " to ", tt2)
            print("Skip processing..., move to next time period.")
            print("")

        # update tt1 and tt2
        tt1 = tt2
        if tt1 is not None: tt2 = tt1 + paras_in['seismic_data']['processing_time']
        sys.stdout.close()



            

    # Restore the original sys.stdout
    sys.stdout = original_stdout
    return



