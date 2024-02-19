

import os
from xinputs import load_check_input
from xproject_init import malmi_project_init
from obspy.clients.fdsn import Client
from xstation import load_station
from quakephase import quakephase
from phassoc import asso
from read_velocity import load_model as load_velocity_model
import sys


fmt_datetime = "%Y%m%dT%H%M%SS%f"


def _oneflow(stream, paras: dict, station):

    # pick and characterize seismic phases
    print(f"Apply phase picking and characterization to seismic data:")
    output_phasepp = quakephase.apply(data=stream, file_para=paras['phase_pick']['parameter_file'])

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
    print(f"Event detection and association:")
    output_asso = asso(pick=output_phasepp['pick'], file_para=paras['phase_asso']['file'])
    
    # locate events for each detection time range
    for jtimerrg, jpick in zip(output_asso['time_range'], output_asso['pick']):
        jdir_event = os.path.join(paras['dir']['results'], f"{jtimerrg[0].strftime(fmt_datetime)}_{jtimerrg[1].strftime(fmt_datetime)}")
        
        # if paras['phase_asso']['time_buffer'] == 'auto':
        #     paras['phase_asso']['time_buffer'] = 
        jtt1 = jtimerrg[0] - paras['phase_asso']['time_buffer']
        jtt2 = jtimerrg[1] + paras['phase_asso']['time_buffer']
        jprob = output_phasepp['prob'].slice(starttime=jtt1, endtime=jtt2).copy()
        jstream = stream.slice(starttime=jtt1, endtime=jtt2).copy()

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

    # Save the current sys.stdout
    original_stdout = sys.stdout

    # load and check input parameters
    print(f"Load and check input parameters from {file_parameter}.")
    paras_in = load_check_input(file_para=file_parameter)

    # setup the project directory
    print(f"Check and setup project directory.")
    paras_in['dir'] = malmi_project_init(para=paras_in['dir'])
    paras_in['id'] = {}

    # load station inventory
    print(f"Load station inventory.")
    station = load_station(file_station=paras_in['station']['file'], outformat='dict')

    # load velocity model
    print(f"Load velocity model.")
    velocity_model = load_velocity_model(file_model=paras_in['velocity']['file'], format=paras_in['velocity']['format'])

    # generate traveltime tables/functions

    # get seismic data
    if paras_in['seismic_data']['get_data'].upper() == "FDSN":
        # get data from FDSN server
        client = Client(base_url=paras_in['seismic_data']['data_source'], 
                        user=paras_in['seismic_data']['user'], 
                        password=paras_in['seismic_data']['password'])
        
        tt1 = paras_in['seismic_data']['starttime']
        tt2 = paras_in['seismic_data']['starttime'] + paras_in['seismic_data']['processing_time']
        paras_in['id']['results'] = f"{tt1.strftime(fmt_datetime)}_{tt2.strftime(fmt_datetime)}"

        file_log = os.path.join(paras_in['dir']['log'], f"{paras_in['id']['results']}.log")
        sys.stdout = open(file_log, "w")

        while (paras_in['seismic_data']['endtime'] is None) or (tt1<paras_in['seismic_data']['endtime']):
            print("Working on time period: ", tt1, " to ", tt2)

            # add buffer time to the start and end time
            tt1_ac = tt1 - paras_in['seismic_data']['buffer_time']  # actual start time
            tt2_ac = tt2 + paras_in['seismic_data']['buffer_time']  # actual end time
            
            # compile the bulk request
            bulk = []
            for inet, ista in zip(station['network'], station['station']):
                bulk.append((inet, ista, "*", "*", tt1_ac, tt2_ac))
            print("Requesting seismic data from stations: ", bulk)

            # request seismic data
            stream = client.get_waveforms_bulk(bulk)

            if stream:
                print("Data available for time period: ", tt1, " to ", tt2)
                print("Start processing...")

                stream.trim(starttime=tt1_ac, endtime=tt2_ac)
                

                if paras_in['dir']['results_tag'] is None:
                    # results are saved separately for each time period
                    paras_in['dir']['results'] = os.path.join(paras_in['dir']['project_root'], "results", paras_in['id']['results'])
                    if not os.path.exists(paras_in['dir']['results']):
                        os.makedirs(paras_in['dir']['results'])
                        print(f"Results directory {paras_in['dir']['results']} created.")

                # process the current seismic data 
                _oneflow(stream=stream, paras=paras_in, station=station)

                # save raw seismic data if required
                if paras_in['seismic_data']['save_raw']:
                    file_raw = os.path.join(paras_in['dir']['seismic_raw'], f"{paras_in['id']['results']}_seismic_raw.mseed")
                    stream.write(file_raw, format='MSEED')

                print(f"Processing finished---------------------------------------------")
                print("")

            else:
                print("No data available for time period: ", tt1, " to ", tt2)
                print("Skip processing..., move to next time period.")
                print("")

            # update tt1 and tt2
            tt1 = tt2
            tt2 = tt1 + paras_in['seismic_data']['processing_time']
            sys.stdout.close()

    elif paras_in['seismic_data']['get_data'].upper() == "LOCAL":
        # get data from local storage
        pass
    

            

    # Restore the original sys.stdout
    sys.stdout = original_stdout
    return



