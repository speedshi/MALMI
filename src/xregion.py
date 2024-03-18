

from xcoordinate import coordsystem
import numpy as np


class region:
    """
    Monitoring region class.
    Note depth is the negative of elevation (in meter), downward for positive.
    """

    def __init__(self, **kwargs):
        """
        Initialize the class.
        """
        self.latitude_min = kwargs.get('latitude_min', None)
        self.latitude_max = kwargs.get('latitude_max', None)
        self.longitude_min = kwargs.get('longitude_min', None)
        self.longitude_max = kwargs.get('longitude_max', None)
        self.depth_min = kwargs.get('depth_min', None)
        self.depth_max = kwargs.get('depth_max', None)
        self.dx = kwargs.get('dx', None)
        self.dy = kwargs.get('dy', None)
        self.dz = kwargs.get('dz', None)

        # get coordinate converting system
        self.coordsystem = coordsystem(elevation_to_depth_scale=-1.0)
        self.coordsystem.compute_crs(longitude=[self.longitude_min, self.longitude_max], 
                                     latitude=[self.latitude_min, self.latitude_max])
        
        # convert the region to the UTM coordinate system
        cps = self.get_corner_points()
        xx, yy = self.coordsystem.lonlat2xy(longitude=[jll[0] for jll in cps], latitude=[jll[1] for jll in cps])
        self.x_min = np.nanmin(xx)
        self.x_max = np.nanmax(xx)
        self.y_min = np.nanmin(yy)
        self.y_max = np.nanmax(yy)
        self.z_min = self.depth_min
        self.z_max = self.depth_max

        if (self.dx is not None) and (self.dy is not None) and (self.dz is not None):
            # get 3D meshgrid of the region in UTM coordinate system
            self.x, self.y, self.z = np.meshgrid(np.arange(self.x_min, self.x_max+1e-6, self.dx),
                                                 np.arange(self.y_min, self.y_max+1e-6, self.dy),
                                                 np.arange(self.z_min, self.z_max+1e-6, self.dz), 
                                                 indexing='ij')
            self.nx = self.x.shape[0]  # number of grid points in x direction
            self.ny = self.y.shape[1]  # number of grid points in y direction
            self.nz = self.z.shape[2]  # number of grid points in z direction
            self.nxyz = self.nx * self.ny * self.nz  # total number of grid points
        
    def __str__(self):
        cstr = ""
        no_show = ['x', 'y', 'z']
        for key, value in self.__dict__.items():
            if key not in no_show:
                cstr += f"{key}: {value}, "
        return cstr.rstrip(", ")

    def get_corner_points(self):
        """
        Get the corner points of the region.
        P4(lon_min, lat_max)-----------------P3(lon_max, lat_max)
        |                                               |
        |                                               |
        |                                               |
        P1(lon_min, lat_min)-----------------P2(lon_max, lat_min)

        Returns:
        --------
        cps: list
            The list of the corner points of the region.
            [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max]]
        """
        cps = []
        cps.append([self.longitude_min, self.latitude_min])
        cps.append([self.longitude_max, self.latitude_min])
        cps.append([self.longitude_max, self.latitude_max])
        cps.append([self.longitude_min, self.latitude_max])
        return cps

