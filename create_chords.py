import numpy as np
import librosa
import os
import sys
import argparse

sys.path.append("libs")
import pitch_estimation
import dataset_utils
import beat_util

parser = argparse.ArgumentParser()
parser.add_argument("-infile", dest = "infile",help='location of audio file')
parser.add_argument("-outfile", dest = "outfile",help="destination file of midi")
args = parser.parse_args()