

from traveltime import read_NLLvel
import numpy as np


class velocity:
    """
    Seismic velocity model class.
    """

    def __init__(self, file: str, file_format: str, velocity_type: str):
        self.file = file
        self.file_format = file_format
        self.velocity_type = velocity_type
        self.load_model(file=self.file, file_format=self.file_format, velocity_type=self.velocity_type)
        self.get_vmaxmin()

    def load_model(self, file: str, file_format: str, velocity_type: str):

        if file_format.upper() == "NLL":
            vmodel = read_NLLvel(file)
            self.model = {}
            if velocity_type.upper() in ['0D', 'HOMO']:
                self.model['P'] = vmodel['Vp_top'] * 1000.0  # convert from km/s to m/s
                self.model['S'] = vmodel['Vs_top'] * 1000.0  # convert from km/s to m/s
                assert(len(self.model['P'])==1), "Input velocity model must be homogeneous model. Please check!"
                assert(len(self.model['S'])==1), "Input velocity model must be homogeneous model. Please check!"
            else:
                raise NotImplementedError("Read other types of velocity model not implement yet!")
        else:
            raise ValueError("Unknown velocity format: ", file_format)

        return
    
    def get_vmaxmin(self):

        self.vmin = np.inf  # global maximum velocity
        self.vmax = -np.inf  # glabal minimal velocity
        self.velmax = {}  # maximum velocity of each phase 
        self.velmin = {}  # minimal velocity of each phase
        for ikey in self.model:
            self.vmin = min(np.min(self.model[ikey]), self.vmin)
            self.vmax = max(np.max(self.model[ikey]), self.vmax)
            self.velmax[ikey] = np.max(self.model[ikey])
            self.velmin[ikey] = np.min(self.model[ikey])

        self.vmean = np.mean([self.vmin, self.vmax])

        return

    def __str__(self):
        cstr = ""
        no_show = ['model']
        for key, value in self.__dict__.items():
            if key not in no_show:
                cstr += f"{key}: {value}, "
        return cstr.rstrip(", ")



