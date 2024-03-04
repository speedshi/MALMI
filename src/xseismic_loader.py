

# class to load seismic data
import os
from obspy import read
from obspy import Stream


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def get_filenames(folder_path, file_order=None):
    filenames = []
    filectimes = []
    for filename in os.listdir(folder_path):
        _cfile = os.path.join(folder_path, filename)
        if os.path.isfile(_cfile):
            filenames.append(filename)
            filectimes.append(os.path.getctime(_cfile))

    if isinstance(file_order, (str)):
        if (file_order.lower() == "name"):
            filenames.sort()
        elif (file_order.lower() == "name_r") or (file_order.lower() == "name_reverse"):
            filenames.sort(reverse=True)
        elif (file_order.lower() == "ctime"):
            filenames = [x for _, x in sorted(zip(filectimes, filenames))]
        elif (file_order.lower() == "ctime_r") or (file_order.lower() == "ctime_reverse"):
            filenames = [x for _, x in sorted(zip(filectimes, filenames), reverse=True)]
        else:
            raise ValueError("Unknown file order: ", file_order)
    return filenames


def read_exclude_list(file_path):
    with open(file_path, "r") as f:
        exclude_list = [line.strip() for line in f.readlines()]
    return exclude_list


def write_exclude_list(filename_list, file_path):
    with open(file_path, "a") as f:
        for ifilename in filename_list:
            f.write(ifilename + '\n')
    return


class xseismic_loader:

    def __init__(self, load_type, data_source, file_exclude=None, write_loaded_to_exclude=False):
        self.load_type = load_type
        self.data_source = data_source
        self.file_exclude = file_exclude
        self.write_loaded_to_exclude = write_loaded_to_exclude

    def get_waveforms(self, paras):
        if self.load_type.upper() == "AIO":
            return self.load_seismic_AIO(paras)
        else:
            raise ValueError("Unknown load type: ", self.load_type)

    def load_seismic_AIO(self, paras):
        
        sfiles = get_filenames(folder_path=self.data_source, file_order=paras['file_order'])  # get all files in the directory

        # load filenames that need to exclude
        if self.file_exclude is not None:
            files_exclude_list = read_exclude_list(file_path=self.file_exclude)
        else:
            files_exclude_list = []

        files_included = [itme for itme in sfiles if itme not in files_exclude_list]
        nfiles = len(files_included)
        if (paras['load_number'] is None) or (paras['load_number'] > nfiles):
            NN = nfiles
        else:
            NN = paras['load_number']
        
        stream = Stream()
        files_loaded = []
        for ifile in files_included[:NN]:
            stream += read(os.path.join(self.data_source, ifile))
            files_loaded.append(ifile)
        
        if self.write_loaded_to_exclude:
            write_exclude_list(filename_list=files_loaded, file_path=self.file_exclude)

        return stream
    


