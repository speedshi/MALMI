# MALMI (MAchine Learning Migration Imaging)

## Installation

We suggest to create and work in a new environment for MALMI. The installation is done via Anaconda. For more information see [conda](https://docs.conda.io/en/latest/).

Currently **MALMI** utilize EQTransformer as ML engine and loki as migration engine. So the two software should be installed first. We will guide you step by step to install all these packages.

### Create and activate a new environment
```bash
conda create -n malmi python=3.7 tensorflow
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
Fellow the script: 'run_MALMI.py' to use the code. Good Luck!
