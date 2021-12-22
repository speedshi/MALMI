# MALMI (MAchine Learning Migration Imaging)

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

### Install loki
```bash
git clone https://github.com/speedshi/LOKI.git
cd WHERE_LOKI_IS_STORED
pip install .
```

### Install MALMI
```bash
git clone https://github.com/speedshi/MALMI.git
```

## Input Dataset
**MALMI** generally requires three kinds of input dataset: continuous raw seismic data, station inventory and velocity model (or traveltime table).  
### Continuous raw seismic data
*continuous raw data* can be in any format that is recognizable by [ObsPy read](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html). The continuous data files can be organized in two structures: (1) store all data files in the same folder (suitable for small dateset); (2) SeisComP Data Structure ([SDS](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)) (suitable for large dateset). Simply set the input parameter: seisdatastru as 'AIO' or 'SDS' for these two dataset structures.

### Station inventory
*station inventory* can be in any format that is recognizable by [ObsPy read_inventory](https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.read_inventory.html). The required infomation of stations are: newwork code, staiton code, latitude, longitude, latitude, elevation.

### Velocity model
*velocity model* is

## Usage
Fellow the example script: 'run_MALMI.py' to use the code. You could copy it anywhere in the system. Open the file to change the input parameters at your preference. Good Luck!
```bash
cd WHERE_MALMI_IS_STORED
cp run_MALMI.py TO_WHERE_YOU_WANT_TO_USE
python run_MALMI.py
```

## Reference
We are current working on a paper about MALMI. When it is published, we will update the paper information here. Please cite the paper in your documents if you use MALMI. We currently have a manuscript on Archive, will update the information here soon.
We had a talk about the MALMI system and real data applications at AGU 2021 Fall Meeting. Detail information as below:  
Abstract ID: 932446  
Abstract Title: An End-to-end Seismic Catalog Builder From Continuous Seismic Data Based on Machine Learning and Waveform Migration  
Final Paper Number: S31A-06  
Presentation Type: Oral Session  
Session Number and Title: S31A: Decoding Geophysical Signatures With Machine Learning: Novel Methods and Results I Oral (S31A-06)

## License
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. For more details, see in the license file.

## Contact information
Copyright(C) 2021 Peidong Shi
Author: Peidong Shi
Email: peidong.shi@sed.ethz.ch or speedshi@hotmail.com


