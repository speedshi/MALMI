#!/bin/bash

for DATE in '20181201' '20181202' '20181203' '20181204' '20181205' 
do
    echo "Process data for: ${DATE}"
    python run_MALMI_multiple_SDS.py "${DATE}"

done || exit 1
