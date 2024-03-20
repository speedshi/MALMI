# Description: This module is used to calculate the seismic traveltime.
# function to calculate the seismic traveltimes.
# returned traveltime information format:
# if tt_type == 'function':
# calculate the traveltime functions and return the functions corresponding to each station.
# tt_model = {'station1': {'phase1': function1, 'phase2': function2, ...}, 
#             'station2': {'phase1': function1, 'phase2': function2, ...}, 
#              ...}
#
# if tt_type == 'table':
# calculate the traveltime tables and return the tables corresponding to each station.
# tt_model = {'station1': {'phase1': np.array, 'phase2': np.array, ...}, 
#             'station2': {'phase1': np.array, 'phase2': np.array, ...}, 
#              ...}
#


import numpy as np


class traveltime:
    """
    Seismic phase traveltime class.
    """

    def __init__(self, station, velocity, region = None, seismic_phase: list = ['P', 'S'], **kwargs):

        self.seismic_phase = seismic_phase
        self.tt_model = {}
        self.tt_type = kwargs.get('type', 'function')
        if self.tt_type == 'function':
            if velocity.velocity_type.upper() in ['0D', 'HOMO']:
                # homogeneous velocity model
                self.vel2fun_homo(station=station, velocity=velocity)
            else:
                raise NotImplementedError("Build traveltime function from a non-homogeneous model is not implemented yet.")
        elif self.tt_type == 'table':
            if velocity.velocity_type.upper() in ['0D', 'HOMO']:
                self.vel2tab_homo(station=station, velocity=velocity, region=region)
            else:
                raise NotImplementedError("Build traveltime table from a non-homogeneous model is not implemented yet.")
        else:
            raise ValueError(f"Invalid type {self.tt_type}.")

        if region is not None:
            # get the maximum and minimum traveltime
            if self.tt_type == 'function':
                self.get_minmaxtt_fun(region=region, station=station, nx=100, ny=100, nz=100)
            elif self.tt_type == 'table':
                self.get_minmaxtt_tab()
            else:
                raise ValueError(f"Invalid type {self.tt_type}.")

    def vel2fun_homo(self, station, velocity):

        for ii, istaid in enumerate(station.id):
            self.tt_model[istaid] = {}
            for iphase in self.seismic_phase:
                self.tt_model[istaid][iphase] = self.create_traveltime_function(station, velocity, ii, iphase)

        return
    
    def create_traveltime_function(self, station, velocity, ii, iphase):
        def caltt(x, y, z):  # input (x, y, z) should be UTM coordinates in meter
            return np.sqrt((x - station.x[ii])**2 + (y - station.y[ii])**2 + (z - station.z[ii])**2) / velocity.model[iphase][0]
        return caltt

    def vel2tab_homo(self, station, velocity, region):
        # calculate traveltime tables from homogeneous (0D) model.

        # Pre-calculate the flattened arrays
        xxx_flat = region.xxx.flatten()
        yyy_flat = region.yyy.flatten()
        zzz_flat = region.zzz.flatten()

        for ii, istaid in enumerate(station.id):  # loop over each station
            self.tt_model[istaid] = {}
            for iphase in self.seismic_phase:  # loop over each phase
                self.tt_model[istaid][iphase] = np.sqrt((xxx_flat - station.x[ii])**2 + (yyy_flat - station.y[ii])**2 + (zzz_flat - station.z[ii])**2) / velocity.model[iphase][0]
                self.tt_model[istaid][iphase] = self.tt_model[istaid][iphase].reshape(region.xxx.shape)
                if self.tt_model[istaid][iphase].size != region.nxyz:
                    raise ValueError("Size of traveltime table and region size should correspond!")

        return

    def get_minmaxtt_fun(self, region, station, nx=100, ny=100, nz=100):
        """
        Get the minimal and the maximal traveltimes for the given monitoring region.
        Traveltime are expressed in function.
        """

        # Create meshgrid directly using np.meshgrid
        xxx, yyy, zzz = np.meshgrid(np.linspace(region.x_min, region.x_max, nx),
                                    np.linspace(region.y_min, region.y_max, ny),
                                    np.linspace(region.z_min, region.z_max, nz),
                                    indexing='ij')

        # Pre-calculate the flattened arrays
        xxx_flat = xxx.flatten()
        yyy_flat = yyy.flatten()
        zzz_flat = zzz.flatten()

        tt_min = np.inf
        tt_max = -np.inf
        for iphase in self.seismic_phase:
            # loop over all phases
            for istaid in station.id:
                # loop over all stations
                ttall_ = self.tt_model[istaid][iphase](xxx_flat, yyy_flat, zzz_flat)
                tt_min = np.nanmin([tt_min, np.min(ttall_)])
                tt_max = np.nanmax([tt_max, np.max(ttall_)])

        self.tt_min = tt_min
        self.tt_max = tt_max
        return

    def get_minmaxtt_tab(self):
        """
        Get the minimal and the maximal traveltimes for the given monitoring region.
        Traveltime are expressed in table.
        """

        tt_min = np.inf
        tt_max = -np.inf
        for iphase in self.seismic_phase:
            # loop over all phases
            for istaid in self.tt_model.keys():
                # loop over all stations
                tt_min = np.nanmin([tt_min, np.min(self.tt_model[istaid][iphase])])
                tt_max = np.nanmax([tt_max, np.max(self.tt_model[istaid][iphase])])

        self.tt_min = tt_min
        self.tt_max = tt_max
        return

    def __str__(self) -> str:
        cstr = ""
        no_show = ['tab', 'tt_model']
        for key, value in self.__dict__.items():
            if key not in no_show:
                cstr += f"{key}: {value}, "
        return cstr.rstrip(", ")

