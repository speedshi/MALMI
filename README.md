# MALMI (MAchine Learning Migration Imaging)

## Installation

We suggest to create and work in a new python environment for MALMI. The installation is done via Anaconda. For more information see [conda](https://docs.conda.io/en/latest/).

Currently **MALMI** utilize *EQTransformer* as the ML engine and *loki* as the migration engine. So the two softwares should be installed as well. We will guide you in this section step by step to install all the required packages.

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
cd WHERE_MALMI_IS_STORED
```
Fellow the script: 'run_MALMI.py' to use the code. 

Good Luck!
