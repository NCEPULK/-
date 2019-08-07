#! /bin/bash


for i in 1 2 3 4 5 6 7 8

do 

rm -rf /home/ncepu-lk/smartcityData/pano_train/scene$i
#rm -rf /home/ncepu-lk/smartcityData/pano/scene$i
mkdir /home/ncepu-lk/smartcityData/pano_train/scene$i

done
python move.py

