#!/bin/bash
cd preprocess
pip install py27hash
echo "---> Download movielens 1M data ..."
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
echo "---> Unzip ml-1m.zip ..."
unzip ml-1m.zip
rm ml-1m.zip




echo "---> Process train & test data ..."