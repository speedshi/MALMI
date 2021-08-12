# MALMI (MAchine Learning Migration Imaging)

## Installation

We suggest to create and work in a new python environment for MALMI. The installation is done via Anaconda. For more information see [conda](https://docs.conda.io/en/latest/).

Currently **MALMI** utilize [*EQTransformer*](https://github.com/speedshi/EQTransformer) as the ML engine and [*loki*](https://github.com/speedshi/LOKI) as the migration engine. So the two softwares should be installed as well. We will guide you step by step in this section to install all the required packages.

### Create and activate a new environment
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n malmi python=3.7 obspy=1.2.2 matplotlib=3.4.2
conda activate malmi
```

### Install EQTransformer
```bash
conda install -c smousavi05 eqtransformer
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

### Usage
Fellow the example script: 'run_MALMI.py' to use the code. You could copy it anywhere in the system. Open the file to change the input parameters at your preference. Good Luck!
```bash
cd WHERE_MALMI_IS_STORED
cp run_MALMI.py TO_WHERE_YOU_WANT_TO_USE
python run_MALMI.py
```

## Reference
We are current working on a paper about MALMI. When it is published, we will update the paper information here. Please cite the paper in your documents if you use MALMI.


## License
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. For more details, see in the license file.

## Contact information
Copyright(C) 2021 Peidong Shi

Author: Peidong Shi

Email: peidong.shi@sed.ethz.ch or speedshi@hotmail.com


