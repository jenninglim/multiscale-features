#!/bin/bash

screen -AdmS ex1c -t tab0 bash

screen -S ex1c -X screen -t tab1 bash -lic "conda activate mskernel && python exp1c.py -p Logit -a PolySel --threads 5 -e Inc -v 2"
screen -S ex1c -X screen -t tab2 bash -lic "conda activate mskernel && python exp1c.py -p Logit -a PolySel --threads 5 -e Block -v 2"
screen -S ex1c -X screen -t tab3 bash -lic "conda activate mskernel && python exp1c.py -p Logit -a MultiSel --threads 5 -e Inc -v 2"
screen -S ex1c -X screen -t tab4 bash -lic "conda activate mskernel && python exp1c.py -p Logit -a MultiSel --threads 5 -e Block -v 2"
