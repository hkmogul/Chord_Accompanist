import numpy as np
import librosa
import sys
import os
import argparse
# use the librosa beat track library to segment a track into its onsets per segment
# ultimately to become the rhythmic data for 
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="File to process", default ="",dest="filename")
parser.parse_args()

if parser.filename is "":
    print("Error. Please add a filename")
    quit()

# read the file
y,sr = librosa.load(parser.filename)
onset_env = librosa.onset.onset_detect(y=y, sr=sr,units='time')