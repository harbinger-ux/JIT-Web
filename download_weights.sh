#!/bin/bash

echo "Creating weight directories..."
mkdir -p weights/SPIDER
mkdir -p weights/Mendeley

echo "Downloading SPIDER weights..."
gdown "https://drive.google.com/uc?id=1szaacTdkN5vf7HBTvSVevskCcmK39wus" -O weights/SPIDER/pytorch_model.bin

echo "Downloading Mendeley weights..."
gdown "https://drive.google.com/uc?id=1cHwKVmusakQu8CONa_9meIeKnZdRYou0" -O weights/Mendeley/pytorch_model.bin

echo "Weights downloaded successfully!"