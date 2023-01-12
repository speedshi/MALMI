# MALMI user manual

## Prepare input dataset

**MALMI** generally requires three kinds of input dataset: continuous raw seismic data, station inventory and velocity model (or traveltime tables).  
### (1) Continuous raw seismic data 
*continuous raw data* can be in any format that is recognizable by [ObsPy read](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html). The continuous data files can be organized in two structures: 
1. store all data files in the same folder (suitable for small dateset); 
2. SeisComP Data Structure ([SDS](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)) (suitable for large dateset).  

Simply set the input parameter: seisdatastru as 'AIO' or 'SDS' for these two dataset structures.

### (2) Station inventory 
*station inventory* can be in any format that is recognizable by [ObsPy read_inventory](https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.read_inventory.html) or a simple CSV file. The required infomation of stations are: newwork code, staiton code, latitude, longitude, latitude, elevation.  
If the input is a CSV file, the delimiter must be ',' and the first row is the column name which must contain: 'network', 'station', 'latitude', 'longitude', 'elevation'. Latitude and longitude are in decimal degree and elevation in meters relative to the sea-level (positive for above the sea-level). 

### (3) Velocity model 
*velocity model* is used to generate traveltime tables for migration location (NonLinLoc must be installed beforehand and the 'grid' parameters must be set).  
The text format velocity file can specify a constant or gradient velocity layer (conform with NonLinLoc velocity model format).  
Format of the velocity model file: "depth Vp_top Vp_grad Vs_top Vs_grad rho_top rho_grad"  
- depth: (float) depth to top of layer in km (use negative values for layers above z=0)  
- Vp_top Vs_top rho_top: (float) P velocity, and S velocity in km/s and density in kg/m^3 at the top of the layer.  
- Vp_grad Vs_grad rho_grad: (float) Linear P velocity and S velocity gradients in km/s/km and density gradient in kg/m^3/km increasing directly downwards from the top of the layer.  

Notes:
1. Multiple layers must be specified in order of increasing depth of top of layer.
2. The layer with the deepest top extends implicitly to infinite depth.

Velocity model example (velocity.txt):
```
-0.20 3.32 0.0 1.87 0.0 2.7 0.0
0.80 4.20 0.0 2.36 0.0 2.7 0.0
1.80 5.03 0.0 2.83 0.0 2.7 0.0
2.90 6.00 0.0 3.37 0.0 2.7 0.0
3.80 6.14 0.0 3.45 0.0 2.7 0.0
4.80 6.31 0.0 3.54 0.0 2.7 0.0
5.80 6.47 0.0 3.63 0.0 2.7 0.0
7.60 6.75 0.0 3.79 0.0 2.7 0.0
7.80 6.91 0.0 3.88 0.0 2.7 0.0
9.80 6.97 0.0 3.92 0.0 2.7 0.0
```

Existing traveltime tables of NonLinLoc format can be directly loaded into MALMI. In this way, a velocity model is not needed anymore. Simply set tt['vmodel'] = None.

## Usage 
Fellow the example script: 'run_MALMI.py' to use the code. You could copy it anywhere in the system. Open the file to change the input parameters at your preference. Good Luck!
```bash
cd WHERE_MALMI_IS_STORED
cp run_MALMI.py TO_WHERE_YOU_WANT_TO_USE
python run_MALMI.py
```
