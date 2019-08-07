#! /bin/bash



for i in 1 6
do

rm -rf ~/smartcityData/results_train/scene$i
mkdir ~/smartcityData/results_train/scene$i

openMVG_main_SfMInit_ImageListing -i ~/smartcityData/pano_train/scene$i -f 1 -c 7 \
-o ~/smartcityData/results_train/scene$i

openMVG_main_ComputeFeatures -i ~/smartcityData/results_train/scene$i/sfm_data.json -p ULTRA \
-o ~/smartcityData/results_train/scene$i/matches

openMVG_main_ComputeMatches -i ~/smartcityData/results_train/scene$i/sfm_data.json \
-o ~/smartcityData/results_train/scene$i/matches

openMVG_main_IncrementalSfM -i ~/smartcityData/results_train/scene$i/sfm_data.json \
-m ~/smartcityData/results_train/scene$i/matches/ -o ~/smartcityData/results_train/scene$i/reconstruction/

openMVG_main_ConvertSfM_DataFormat -i ~/smartcityData/results_train/scene$i/reconstruction/sfm_data.bin \
-V -I -E -o ~/smartcityData/results_train/scene$i/reconstruction/sfm_data.json

done


