"""
Load parameter file.
Check input parameters are valid.
"""


import yaml
from obspy import UTCDateTime
import os


def load_check_input(file_para):

    # load paramters
    if isinstance(file_para, str):
        with open(file_para, 'r') as file:
            paras = yaml.safe_load(file)
    elif isinstance(file_para, dict):
        paras = file_para
    else:
        raise ValueError("Parameter file should be a string or a parameter dictionary.")

    # check 'dir' setting
    if 'dir' not in paras:
        raise ValueError("Parameter file should contain 'dir' setting.")
    else:
        # check 'dir_project_root' setting
        if 'project_root' not in paras['dir']:
            raise ValueError("Parameter file should contain 'dir:project_root' setting.")
        else:
            if not isinstance(paras['dir']['project_root'], (str,)):
                raise ValueError("'dir_project_root' should be a string.")

        # check 'dir_results_tag' setting
        if 'results_tag' not in paras:
            paras['dir']['results_tag'] = None
            paras['dir']['results'] = None
        else:
            if not isinstance(paras['dir']['results_tag'], (str,)):
                raise ValueError("'dir:results_tag' should be a string.")
            elif paras['dir']['results_tag'].upper() == "NONE":
                paras['dir']['results_tag'] = None
                paras['dir']['results'] = None
            else:
                # input is a valid string
                paras['dir']['results'] = os.path.join(paras['dir']['project_root'], paras['dir']['results_tag'])
            
    # check 'seismic_data' setting
    if 'seismic_data' not in paras:
        raise ValueError("Parameter file should contain 'seismic_data' setting.")
    else:
        assert(isinstance(paras['seismic_data'], (dict,)))
        if "get_data" not in paras['seismic_data']:
            raise ValueError("Parameter file should contain 'seismic_data:get_data' setting.")
        else:
            if not paras['seismic_data']['get_data'].upper() in ['FDSN', 'LOCAL']:
                raise ValueError("'seismic_data:get_data' should be either 'FDSN' or 'LOCAL'.")
        
        if "data_source" not in paras['seismic_data']:
            raise ValueError("Parameter file should contain 'seismic_data:data_source' setting.")
        
        if paras['seismic_data']['get_data'].upper() == 'FDSN':
            if "user" not in paras['seismic_data']:
                paras['seismic_data']['user'] = None
            if "password" not in paras['seismic_data']:
                paras['seismic_data']['password'] = None

            # the following settings are not necessary for FDSN seismic data input
            assert("starttime" in paras['seismic_data'])
            assert("processing_time" in paras['seismic_data'])
            assert("buffer_time" in paras['seismic_data'])

        if "starttime" not in paras['seismic_data']:
            paras['seismic_data']['starttime'] = None
        else:
            paras['seismic_data']['starttime'] = UTCDateTime(paras['seismic_data']['starttime'])
        
        if "endtime" not in paras['seismic_data']:
            paras['seismic_data']['endtime'] = None     
        else:
            paras['seismic_data']['endtime'] = UTCDateTime(paras['seismic_data']['endtime'])

        if "processing_time" not in paras['seismic_data']:
            paras['seismic_data']['processing_time'] = None

        if "buffer_time" not in paras['seismic_data']:
            paras['seismic_data']['buffer_time'] = None

        if "save_raw" not in paras['seismic_data']:
            paras['seismic_data']['save_raw'] = False
        elif isinstance(paras['seismic_data']['save_raw'], (str)):
            if ((paras['seismic_data']['save_raw'].upper() == "TRUE") or 
                (paras['seismic_data']['save_raw'].upper() == "YES")  or 
                (paras['seismic_data']['save_raw'].upper() == "Y")):
                paras['seismic_data']['save_raw'] = True
            else:
                paras['seismic_data']['save_raw'] = False

    # check station setting
    if 'station' not in paras:
        raise ValueError("Parameter file should contain 'station' setting.")
    else:
        if "file" not in paras['station']:
            raise ValueError("Parameter file should contain 'station_file' setting.")

    # check 'velocity' setting
    if 'velocity' not in paras:
        raise ValueError("Parameter file should contain 'velocity' setting.")
    else:
        if "file" not in paras['velocity']:
            raise ValueError("Parameter file should contain 'velocity:file' setting.")
        else:
            if not os.path.isfile(paras['velocity']['file']):
                raise ValueError("Velocity model file: {} does not exist.".format(paras['velocity']['file']))
        if "type" not in paras['velocity']:
            raise ValueError("Parameter file should contain 'velocity:type' setting.")
        else:
            if not isinstance(paras['velocity']['type'], (str,)):
                raise ValueError("'velocity:type' should be a string.")
            else:
                if paras['velocity']['type'].upper() not in ['HOMO', '0D', '1D', '2D', '3D']:
                    raise ValueError("'velocity:type' should be either 'HOMO', '0D', '1D', '2D', '3D'.")
        if "format" not in paras['velocity']:
            raise ValueError("Parameter file should contain 'velocity:format' setting.")
        else:
            if not isinstance(paras['velocity']['format'], (str,)):
                raise ValueError("'velocity:format' should be a string.")
            else:
                if paras['velocity']['format'].upper() not in ['NLL']:
                    raise ValueError("'velocity:format' should be 'NLL'.")

    # chech 'traveltime' setting
    if 'traveltime' not in paras:
        paras['traveltime'] = {}
        paras['traveltime']['request'] = 'function'
    else:
        if 'request' not in paras['traveltime']:
            paras['traveltime']['request'] = 'function'
        else:
            if paras['traveltime']['request'].upper() not in ['TABLE', 'FUNCTION']:
                raise ValueError(f"'traveltime:request' should be either 'table' or 'function'")

    # check 'phase_pick' setting
    if 'phase_pick' not in paras:
        raise ValueError("Parameter file should contain 'phase_pick' setting.")
    else:
        if 'parameter_file' not in paras['phase_pick']:
            raise ValueError("Parameter file should contain 'phase_pick:parameter_file' setting.")
        if 'save_pick' not in paras['phase_pick']:
            paras['phase_pick']['save_pick'] = False
        elif isinstance(paras['phase_pick']['save_pick'], (str)):
            if ((paras['phase_pick']['save_pick'].upper() == "TRUE") or 
                (paras['phase_pick']['save_pick'].upper() == "YES")  or 
                (paras['phase_pick']['save_pick'].upper() == "Y")):
                paras['phase_pick']['save_pick'] = True
            else:
                paras['phase_pick']['save_pick'] = False
        if 'save_prob' not in paras['phase_pick']:
            paras['phase_pick']['save_prob'] = False
        elif isinstance(paras['phase_pick']['save_prob'], (str)):
            if ((paras['phase_pick']['save_prob'].upper() == "TRUE") or 
                (paras['phase_pick']['save_prob'].upper() == "YES") or
                (paras['phase_pick']['save_prob'].upper() == "Y")):
                paras['phase_pick']['save_prob'] = True
            else:
                paras['phase_pick']['save_prob'] = False

    # check 'phase_asso' setting
    if 'phase_asso' not in paras:
        paras['phase_asso'] = None
    else:
        if 'file' not in paras['phase_asso']:
            raise ValueError("Parameter file should contain 'phase_asso:file' setting.")
        
        if "time_buffer" not in paras['phase_asso']:
            paras['phase_asso']['time_buffer'] = 'auto'
        elif isinstance(paras['phase_asso']['time_buffer'], (int, float,)):
            assert(paras['phase_asso']['time_buffer'] >= 0)
        elif isinstance(paras['phase_asso']['time_buffer'], (str,)):
            if paras['phase_asso']['time_buffer'].upper() == "AUTO":
                paras['phase_asso']['time_buffer'] = 'auto'
            else:
                raise ValueError("'phase_asso:time_buffer' should be a non-nagtive number or 'auto'.")
        else:
            raise ValueError("'phase_asso:time_buffer' should be a non-nagtive number or 'auto'.")

        if 'save_pick' not in paras['phase_asso']:
            paras['phase_asso']['save_pick'] = True
        elif isinstance(paras['phase_asso']['save_pick'], (str)):
            if ((paras['phase_asso']['save_pick'].upper() == "TRUE") or 
                (paras['phase_asso']['save_pick'].upper() == "YES")  or 
                (paras['phase_asso']['save_pick'].upper() == "Y")):
                paras['phase_asso']['save_pick'] = True
            else:
                paras['phase_asso']['save_pick'] = False
        if 'save_prob' not in paras['phase_asso']:
            paras['phase_asso']['save_prob'] = True
        elif isinstance(paras['phase_asso']['save_prob'], (str)):
            if ((paras['phase_asso']['save_prob'].upper() == "TRUE") or 
                (paras['phase_asso']['save_prob'].upper() == "YES") or
                (paras['phase_asso']['save_prob'].upper() == "Y")):
                paras['phase_asso']['save_prob'] = True
            else:
                paras['phase_asso']['save_prob'] = False
        if 'save_seis' not in paras['phase_asso']:
            paras['phase_asso']['save_seis'] = True
        elif isinstance(paras['phase_asso']['save_seis'], (str)):
            if ((paras['phase_asso']['save_seis'].upper() == "TRUE") or 
                (paras['phase_asso']['save_seis'].upper() == "YES") or
                (paras['phase_asso']['save_seis'].upper() == "Y")):
                paras['phase_asso']['save_seis'] = True
            else:
                paras['phase_asso']['save_seis'] = False

    return paras




