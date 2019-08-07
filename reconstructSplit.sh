#! /bin/bash

cd /home/ncepu-lk/openMVG_build/Linux-x86_64-RELEASE
INPUT_PATH=/home/ncepu-lk/smartcityData/data/scene1
OUT_PATH=/home/ncepu-lk/smartcityData/results/scene1

./openMVG_main_SfMInit_ImageListing -i $INPUT_PATH -f 1 -c 7 \
-o $OUT_PATH

./openMVG_main_ComputeFeatures -i $OUT_PATH/sfm_data.json \
-o $OUT_PATH/matches

./openMVG_main_ComputeMatches -i $OUT_PATH/sfm_data.json \
-o $OUT_PATH/matches

./openMVG_main_IncrementalSfM -i $OUT_PATH/sfm_data.json \
-m $OUT_PATH/matches/ -o $OUT_PATH/reconstruction/

./openMVG_main_ConvertSfM_DataFormat -i $OUT_PATH/reconstruction/sfm_data.bin \
-V -I -E -o $OUT_PATH/reconstruction/sfm_data.json
