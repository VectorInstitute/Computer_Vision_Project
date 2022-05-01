#!/bin/sh

jupyter nbconvert dataLoaderGBPract.ipynb --to python 
ipython ./dataLoaderGBPract.py
