# MALMI (MAchine Learning aided earthquake MIgration location)  
MALMI is developed on a Linux OS system. Therefore, we suggest using MALMI on Linux. For using MALMI on Windows and Mac OS systems, we anticipated there might be problems during compiling and running. Please feel free to create "Pull requests" or suggest changes if you solve problems/bugs in Windows and Mac OS systems.

## Installation 
We suggest to create and work in a new python environment for MALMI. The installation is done via Anaconda. For more information see [conda](https://docs.conda.io/en/latest/).

Currently **MALMI** can utilize [*EQTransformer*](https://github.com/speedshi/EQTransformer) and [*SeisBench*](https://github.com/seisbench/seisbench) as the ML engine and [*loki*](https://github.com/speedshi/LOKI) as the migration engine. So the these softwares should be installed as well. We will guide you step by step in this section to install all the required packages.

### Create and activate a new environment 
If you want to use original EQTransformer as ML engine:  
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n malmi python=3.7 obspy spyder==5.0.3 pygmt six~=1.15.0 numpy~=1.19.2 protobuf'<3.20,>=3.9.2'
conda activate malmi
```

If you want to use SeisBench as ML engine:  
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n malmi python=3.9 pygmt
conda activate malmi
```

### Install EQTransformer (to use EQT as ML engine) 
```bash
git clone https://github.com/speedshi/EQTransformer.git
cd WHERE_EQTransformer_IS_STORED
python setup.py install
```

### Install SeisBench (to use SeisBench as ML engine) 
```bash
pip install seisbench
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
Currently only *Vel2Grid* and *Grid2Time* programs are used and remember to put them in a executable path after compiling NonLinLoc.  
There are two ways to install NonLinLoc:

1. Through [NonLinLoc GitHub Page](https://github.com/alomax/NonLinLoc) (Recomended)  
Install example:
```bash
git clone https://github.com/alomax/NonLinLoc.git
cd NonLinLoc/src
mkdir bin   # bin/ is a subdirectory of src/
cmake .
make
echo 'export PATH="WHERE_CODE_IS_STORED/NonLinLoc/src/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

2. Follow [NonLinLoc Home Page](http://alomax.free.fr/nlloc/) for installing the NonLinLoc software.   
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

## Usage 
Follow the [user manual page](https://github.com/speedshi/MALMI/blob/main/user_manual.md) to use MALMI. 

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


