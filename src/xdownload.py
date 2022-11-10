# Download seismic data, station inventory from online data center
# Author: Peidong Shi
# Contact: speedshi@hotmail.com  or  peidong.shi@sed.ethz.ch
# Create time: 2022-11-09


from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import RectangularDomain, CircularDomain, Restrictions, MassDownloader
from multiprocessing.pool import ThreadPool
import datetime
import os


date_strfmt = "%Y%m%d"


def download_ss(domain_para, time_range, clients=None, dir_para=None, n_processor=None):
    # "domain_para" should either contain: domain_para['minlatitude'], domain_para['maxlatitude'], domain_para['minlongitude'], domain_para['maxlongitude']
    # or domain_para['latitude'], domain_para['longitude'], domain_para['minradius'], domain_para['maxradius'].
    # "time_range" should be datetime.date format specifying the date range (included) to be downloaded,
    # for example: time_range = [datetime.date(2021,1,1), datetime.date(2021,3,31)] for requesting 3 months data.
    # clients=None will use all FDSN implementations known to ObsPy except RASPISHAKE and IRISPH5,
    # otherwise specify the clients, such as clients = ["ETHZ", "GFZ", "IRIS", "ORFEUS", "INGV"].

    if (dir_para is None):
        mseed_storage_root = "waveforms"
        stationxml_storage_root = "stations"
    else:
        if ('mseed_storage_root' not in dir_para):
            mseed_storage_root = "waveforms"
        else:
            mseed_storage_root = dir_para['mseed_storage_root']

        if ('stationxml_storage_root' not in dir_para):
            stationxml_storage_root = "stations"
        else:
            stationxml_storage_root = dir_para['stationxml_storage_root']

    if 'minlatitude' in domain_para:
        # Rectangular domain
        domain = RectangularDomain(minlatitude=domain_para['minlatitude'], maxlatitude=domain_para['maxlatitude'],
                                   minlongitude=domain_para['minlongitude'], maxlongitude=domain_para['maxlongitude'])
    else:
        # Circular domain around a point
        if 'minradius' not in domain_para:
            domain_para['minradius'] = 0
        domain = CircularDomain(latitude=domain_para['latitude'], longitude=domain_para['longitude'],
                                minradius=domain_para['minradius'], maxradius=domain_para['maxradius'])

    mdl = MassDownloader(providers=clients)
    
    assert(len(time_range)==2)
    assert(isinstance(time_range[0], datetime.date) and isinstance(time_range[1], datetime.date))
    assert(time_range[0] <= time_range[1])
    oneday = datetime.timedelta(days=1)
    ndays = (time_range[1] - time_range[0]).days + 1
    date_list = [time_range[0]+datetime.timedelta(days=ii) for ii in range(ndays)]
    assert(date_list[-1]==time_range[1])

    if n_processor is None:
        for idate in date_list:
            print(f'======= Working on date: {idate}.')
            time_it = [UTCDateTime(idate), UTCDateTime(idate+oneday)]
            mseed_storage = os.path.join(mseed_storage_root, idate.strftime(date_strfmt))
            stationxml_storage = os.path.join(stationxml_storage_root, idate.strftime(date_strfmt))
            _get_data(mdl, mseed_storage, stationxml_storage, time_it, domain)
    else:        
        def process(idate):
            print(f'======= Working on date: {idate}.')
            time_it = [UTCDateTime(idate), UTCDateTime(idate+oneday)]
            mseed_storage = os.path.join(mseed_storage_root, idate.strftime(date_strfmt))
            stationxml_storage = os.path.join(stationxml_storage_root, idate.strftime(date_strfmt))
            _get_data(mdl, mseed_storage, stationxml_storage, time_it, domain)
            
        with ThreadPool(n_processor) as p:
            p.map(process, date_list)
    
    return


def _get_data(mdl, mseed_storage, stationxml_storage, time_it, domain):

    restrictions = Restrictions(
            starttime=time_it[0],  # The start time of the data to be downloaded
            endtime=time_it[1],  # The end time of the data
            # chunklength_in_sec=86400,  # Chunk it to have one file per day
            reject_channels_with_gaps=False,
            minimum_length=0.0,
            sanitize=False,
            minimum_interstation_distance_in_m=1.0)

    mdl.download(domain, restrictions, mseed_storage=mseed_storage, stationxml_storage=stationxml_storage)

    return

