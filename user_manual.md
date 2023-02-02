# MALMI user manual

## Prepare input dataset

**MALMI** generally requires three kinds of input dataset: continuous raw seismic data, station inventory and velocity model (or traveltime tables).  
### (1) Continuous raw seismic data 
*continuous raw data* can be in any format that is recognizable by [ObsPy read](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html). The continuous data files can be organized in two structures: 
1. store all data files in the same folder (suitable for small dateset); 
2. SeisComP Data Structure ([SDS](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)) (suitable for large dateset).  

Simply set the input parameter: seisdatastru as 'AIO' or 'SDS' for these two dataset structures.  
Example seismic data stored in the 'AIO' dataset structure can be seen in 'test/inputs/seismic_data'.

### (2) Station inventory 
*station inventory* can be in any format that is recognizable by [ObsPy read_inventory](https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.read_inventory.html) or a simple CSV file. The required infomation of stations are: newwork code, staiton code, latitude, longitude, latitude, elevation.  

If the input is a CSV file, the delimiter must be ',' and the first row is the column name which must contain: 'network' (network code, such as "IV"), 'station' (station code, such "AQU"), 'latitude', 'longitude', 'elevation'. Latitude and longitude are in decimal degree and elevation in meters relative to the sea-level (positive for above the sea-level). Optional columns are 'location' (station location code, such as "" or "01"), 'instrument' (instrument code, such as "HH" or "SH"), 'component' (channel component codes, such as "ZNE" or "Z12"), 'depth' (the local depth or overburden of the instrument’s location in meters, such as 500.0). Note 'instrument' and 'component' must coexist in the CSV file.     

Example station inventory file in CSV format can be seen in 'test/inputs/station_inventory'.

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

Example layered velocity model file can be found in 'test/inputs/velocity_model'.

## Usage 
Follow the example script: 'run_MALMI.py' to use the code. You could copy it anywhere in the system. Open the file to change the input parameters at your preference. 

Below is a brief instruction on a few important parameters and enssential steps for setting up a MALMI workflow. There are other parameters and steps that can be used, you can check them in 'run_MALMI.py' and 'src/main.py' and feel free to explore more.  
In 'run_MALMI.py' script, you need at least check and modify the following parameters:  
1. since I haven't set up the MALMI installtion script, you need to manually add MALMI scripts into the python path by change this line "sys.path.append('/THIS_SHOULD_BE_YOUR_PATH_TO_THE_MALMI_CODE/MALMI/src')";  
2. set the input seismic data directory: **seismic['dir']**;  
3. set the station inventory file: **seismic['stainvf']**;  
4. set input seismic data structure: **seismic['datastru']**;  
5. set the output directory: **control['dir_output']**;  
6. set the input velocity model for generating traveltime tables: **tt['vmodel']**;   
7. set where to store or find the traveltime tables: **tt['dir']**;  
8. set whether we want to generate the traveltime tables when run MALMI workflow: **tt['build']**. Note we might run MALMI workflow mutiple times (such as for data at different days), but we only need to generate the traveltime table once. Once traveltime tables have been generated, we can just reuse it, so in this situation set **"tt['build']=Flase"**;  
9. set the migration grid parameters: **grid**. The **grid** parameters depend on the monitoring region and location resolution (grid spacing); perform migration location on a large grid (millions of grid points) might be slow;  
10. set the detection parameters: **detect**;  
11. set the migration location parameters: **MIG**;  
12. set the adopted ML model: **ML['model']**;  

A general MALMI workflow would be:
1. setup output directory and traveltime tables: **myworkflow = MALMI(seismic=seismic, tt=tt, grid=grid, control=control, detect=detect, MIG=MIG)**    
2. format input seismic data for ML: **myworkflow.format_ML_inputs()**  
3. generate continous phase probalities: **myworkflow.generate_prob(ML)**  
4. detect events and trim phase probabilites segments: **myworkflow.event_detect_ouput()**  
5. perform migration location for the detected events: **myworkflow.migration()**  
6. plot waveforms and migration profiles of the located events: **myworkflow.rsprocess_view()**  
7. clean some generated file during ML or migration process to save diskspace: **myworkflow.clear_interm()**    
8. extrace the located events information to a single file (catalog): **myworkflow.get_catalog()**

If parameters have been properly setup, run MALMI by using:
```bash
python run_MALMI.py
```
Good Luck!
