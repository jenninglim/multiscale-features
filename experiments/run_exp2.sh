#!/bin/bash

screen -AdmS ex2 -t tab0 bash

# Run on Pulsar dataset
screen -S ex2 -X screen -t tab1 bash -lic "conda activate mskernel && python exp2.py -p Pulsar -a PolySel --threads 5 -v 2 -b MMD"
screen -S ex2 -X screen -t tab2 bash -lic "conda activate mskernel && python exp2.py -p Pulsar -a MultiSel --threads 5 -v 2 -b MMD"

# Run on Heart dataset
screen -S ex2 -X screen -t tab3 bash -lic "conda activate mskernel && python exp2.py -p Heart -a PolySel --threads 5 -v 2 -b MMD"
screen -S ex2 -X screen -t tab3 bash -lic "conda activate mskernel && python exp2.py -p Heart -a MultiSel --threads 5 -v 2 -b MMD"

# Run on Wine dataset
screen -S ex2 -X screen -t tab3 bash -lic "conda activate mskernel && python exp2.py -p Wine -a MultiSel --threads 5 -v 2 -b MMD"
screen -S ex2 -X screen -t tab3 bash -lic "conda activate mskernel && python exp2.py -p Wine -a PolySel --threads 5 -v 2 -b MMD"
