#!/bin/bash

screen -AdmS ex1b -t tab0 bash

screen -S ex1b -X screen -t tab1 bash -lic "conda activate mskernel && python exp1b.py -p MS -a PolySel --threads 5 --mmd Inc -v 2"
screen -S ex1b -X screen -t tab2 bash -lic "conda activate mskernel && python exp1b.py -p MS -a PolySel --threads 5 --mmd Lin -v 2"
screen -S ex1b -X screen -t tab3 bash -lic "conda activate mskernel && python exp1b.py -p MS -a MultiSel --threads 5 --mmd Inc -v 2"
screen -S ex1b -X screen -t tab4 bash -lic "conda activate mskernel && python exp1b.py -p MS -a MultiSel --threads 5 --mmd Lin -v 2"
