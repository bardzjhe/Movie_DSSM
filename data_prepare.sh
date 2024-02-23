#!/bin/bash
cd preprocess
pip install py27hash



echo "---> Split movielens data ..."
python process.py

# create train and test directories in the current working directory
mkdir -p train/
mkdir -p test/


echo "---> Process train & test data ..."