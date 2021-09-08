#!/bin/bash

for FOLDER in '20181201' '20181202' '20181203' '20181204' '20181205' 
do
    FPATH="../data/seismic_data_raw/seismic_raw_${FOLDER}"
    echo "Proceed to data: ${FPATH}"
    python run_MALMI_multiple.py "${FPATH}"

done
