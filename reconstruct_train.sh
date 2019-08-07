#! /bin/bash



for i in 6
do

rm -rf ~/smartcityData/results/scene$i
mkdir ~/smartcityData/results/scene$i

openMVG_main_SfMInit_ImageListing -i ~/smartcityData/pano/scene$i -f 1 -c 7 \
-o ~/smartcityData/results/scene$i

openMVG_main_ComputeFeatures -i ~/smartcityData/results/scene$i/sfm_data.json -p ULTRA \
-o ~/smartcityData/results/scene$i/matches

openMVG_main_ComputeMatches -i ~/smartcityData/results/scene$i/sfm_data.json \
-o ~/smartcityData/results/scene$i/matches

openMVG_main_IncrementalSfM -i ~/smartcityData/results/scene$i/sfm_data.json \
-m ~/smartcityData/results/scene$i/matches/ -o ~/smartcityData/results/scene$i/reconstruction/

openMVG_main_ConvertSfM_DataFormat -i ~/smartcityData/results/scene$i/reconstruction/sfm_data.bin \
-V -I -E -o ~/smartcityData/results/scene$i/reconstruction/sfm_data.json

done


