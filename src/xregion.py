

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
            self.mesh3D_xyz(x_min=self.x_min, x_max=self.x_max,
                            y_min=self.y_min, y_max=self.y_max,
                            z_min=self.z_min, z_max=self.z_max,
                            dx=self.dx, dy=self.dy, dz=self.dz)
            self.nx = self.xxx.shape[0]  # number of grid points in x direction
            self.ny = self.yyy.shape[1]  # number of grid points in y direction
            self.nz = self.zzz.shape[2]  # number of grid points in z direction
            self.nxyz = self.nx * self.ny * self.nz  # total number of grid points

    def mesh3D_xyz(self, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz):

        # get 3D meshgrid of the region in UTM coordinate system
        # +1e-6 to ensure the last point can be included
        self.x = np.arange(x_min, x_max+1e-6, dx)  # 1d array of x coordinates 
        self.y = np.arange(y_min, y_max+1e-6, dy)  # 1d array of y coordinates
        self.z = np.arange(z_min, z_max+1e-6, dz)  # 1d array of z coordinates
        self.xxx, self.yyy, self.zzz = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # 3D meshgrid of the region
        return
    
    def mesh3D_xyz_subgrid_index(self, x_bound, y_bound, z_bound, dnx: int, dny: int, dnz: int):
        # get a sub-grid from the 3D meshgrid of the region 

        x_index = np.where((self.x >= x_bound[0]) & (self.x <= x_bound[1]))[0]
        x_index = x_index[::dnx]

        y_index = np.where((self.y >= y_bound[0]) & (self.y <= y_bound[1]))[0]
        y_index = y_index[::dny]

        z_index = np.where((self.z >= z_bound[0]) & (self.z <= z_bound[1]))[0]
        z_index = z_index[::dnz]

        return np.ix_(x_index, y_index, z_index), x_index, y_index, z_index

    def __str__(self):
        cstr = ""
        no_show = ['xxx', 'yyy', 'zzz']
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

