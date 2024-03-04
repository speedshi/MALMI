

from traveltime import read_NLLvel


class velocity:
    """
    Seismic velocity model class.
    """

    def __init__(self, file: str, file_format: str, velocity_type: str):
        self.file = file
        self.file_format = file_format
        self.velocity_type = velocity_type
        self.load_model(file=self.file, file_format=self.file_format, velocity_type=self.velocity_type)

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
                raise NotImplementedError("Real other types of velocity model not implement yet!")
        else:
            raise ValueError("Unknown velocity format: ", file_format)

        return

    def __str__(self):
        cstr = ""
        for key, value in self.__dict__.items():
            cstr += f"{key}: {value}, "
        return cstr.rstrip(", ")



