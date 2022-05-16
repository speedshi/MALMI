# MALMI (MAchine Learning aided earthquake MIgration location)  
MALMI is developed on a Linux OS system. Therefore, we suggest using MALMI on Linux. For using MALMI on Windows and Max OS systems, we anticipated there might be problems during compiling and running. Please feel free to create "Pull requests" or suggest changes if you solve problems/bugs in Windows and Max OS systems.

## Installation 
We suggest to create and work in a new python environment for MALMI. The installation is done via Anaconda. For more information see [conda](https://docs.conda.io/en/latest/).

Currently **MALMI** utilize [*EQTransformer*](https://github.com/speedshi/EQTransformer) as the ML engine and [*loki*](https://github.com/speedshi/LOKI) as the migration engine. So the two softwares should be installed as well. We will guide you step by step in this section to install all the required packages.

### Create and activate a new environment 
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n malmi python=3.7 obspy spyder==5.0.3 pygmt six~=1.15.0 numpy~=1.19.2
conda activate malmi
```

### Install EQTransformer 
```bash
git clone https://github.com/speedshi/EQTransformer.git
cd WHERE_EQTransformer_IS_STORED
python setup.py install
```

### Install loki (GNU gcc compiler and openmp required)
```bash
git clone https://github.com/speedshi/LOKI.git
cd WHERE_LOKI_IS_STORED
pip install .
```

### Install MALMI 
```bash
git clone https://github.com/speedshi/MALMI.git
```

### Install NonLinLoc if you want to generate travetime tables in MALMI (optional)
Follow [NonLinLoc Home Page](http://alomax.free.fr/nlloc/) for installing the NonLinLoc software. Only *Vel2Grid* and *Grid2Time* programs are used, and remember to put them in a executable path after compiling NonLinLoc.  
Install example:
```bash
wget http://alomax.free.fr/nlloc/soft7.00/tar/NLL7.00_src.tgz
mv NLL7.00_src.tgz WHERE_CODE_IS_STORED
cd WHERE_CODE_IS_STORED
mkdir NLL
tar -xzvf NLL7.00_src.tgz -C ./NLL
cd ./NLL/src
make -R distrib
echo 'export PATH="WHERE_CODE_IS_STORED/NLL/src:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Input Dataset 
**MALMI** generally requires three kinds of input dataset: continuous raw seismic data, station inventory and velocity model (or traveltime tables).  
### Continuous raw seismic data 
*continuous raw data* can be in any format that is recognizable by [ObsPy read](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html). The continuous data files can be organized in two structures: 
1. store all data files in the same folder (suitable for small dateset); 
2. SeisComP Data Structure ([SDS](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)) (suitable for large dateset).  

Simply set the input parameter: seisdatastru as 'AIO' or 'SDS' for these two dataset structures.

### Station inventory 
*station inventory* can be in any format that is recognizable by [ObsPy read_inventory](https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.read_inventory.html) or a simple CSV file. The required infomation of stations are: newwork code, staiton code, latitude, longitude, latitude, elevation.  
If the input is a CSV file, the delimiter must be ',' and the first row is the column name which must contain: 'network', 'station', 'latitude', 'longitude', 'elevation'. Latitude and longitude are in decimal degree and elevation in meters relative to the sea-level (positive for above the sea-level). 

### Velocity model 
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

## Reference 
Please cite the following paper in your documents if you use MALMI in your work.  
Peidong Shi, Francesco Grigoli, Federica Lanza, Gregory C. Beroza, Luca Scarabello, Stefan Wiemer; MALMI: An Automated Earthquake Detection and Location Workflow Based on Machine Learning and Waveform Migration. Seismological Research Letters 2022; doi: [https://doi.org/10.1785/0220220071](https://doi.org/10.1785/0220220071)

BibTex:
```
@article{10.1785/0220220071,
    author = {Shi, Peidong and Grigoli, Francesco and Lanza, Federica and Beroza, Gregory C. and Scarabello, Luca and Wiemer, Stefan},
    title = "{MALMI: An Automated Earthquake Detection and Location Workflow Based on Machine Learning and Waveform Migration}",
    journal = {Seismological Research Letters},
    year = {2022},
    month = {05},
    issn = {0895-0695},
    doi = {10.1785/0220220071},
    url = {https://doi.org/10.1785/0220220071},
    eprint = {https://pubs.geoscienceworld.org/ssa/srl/article-pdf/doi/10.1785/0220220071/5602568/srl-2022071.1.pdf},
}
```

## License 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. For more details, see in the license file.

## Contributing
If you would like to contribute to the project or have any suggestions about the code, please feel free to create Pull Requests, raise issues and contact me.  
If you have any questions about the usage of this package or find bugs in the code, please also feel free to contact me.

## Contact information 
Copyright(C) 2021 Peidong Shi  
Author: Peidong Shi  
Email: peidong.shi@sed.ethz.ch or speedshi@hotmail.com


