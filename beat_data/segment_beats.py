# Main file for creating beat segments over a song/folder
import argparse
import pickle
import librosa
import numpy as np
import sys
import os
sys.path.append(os.path.join("..","libs"))
from beat_util import *

parser = argparse.ArgumentParser(description='Generate beat signatures for onsets.')
parser.add_argument('-infile', dest='infile')
parser.add_argument('-outfile', dest='outfile')
args = parser.parse_args()
if args.infile is None:
    print("Enter file pls")
    sys.exit()

segments = segment_onsets(filename=args.infile, group_measures=False, group_num=4)
signatures = []
c = 0
for i in range(len(segments)):
    c+= len(segments[i])
    signatures.append(create_measure_signature(segments[i]))
    print("New signature is \n{}".format(signatures[-1]))


if args.outfile is None:
    print("attempting to stretch the signature to a set amount")
    for sig in signatures:
        print(measure_signature_to_onset_times(sig, duration_ms=500))
    print("Enter outfile too pls")
    sys.exit()
else:
    pickle.dump(signatures, open(args.outfile, "wb"))