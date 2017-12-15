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
parser.add_argument('-tempo', dest='tempo')
parser.add_argument('-timeSig',dest='timeSig')
args = parser.parse_args()
if args.infile is None:
    print("Enter file pls")
    sys.exit()
if args.timeSig is None:
    group_measures = False
    group_num = 1
else:
    group_measures=True
    group_num = int(args.timeSig)
if args.tempo is None:
    segments = segment_onsets(filename=args.infile, group_measures=group_measures, group_num=group_num)
else:
    segments = segment_onsets(filename=args.infile, group_measures=group_measures, group_num=group_num, tempo_est=args.tempo)
signatures = []
c = 0
for i in range(len(segments)):
    c+= len(segments[i])
    signatures.append(create_measure_signature(segments[i]))
    #print("New signature \n{}\n------------".format(signatures[-1]))
print("{} signatures collected".format(c))

if args.outfile is None:
    print("attempting to stretch the signature to a set amount")
    #for sig in signatures:
        #print(measure_signature_to_onset_times(sig, duration_ms=500))
    print("Enter outfile too pls")
    sys.exit()
else:
    pickle.dump(signatures, open(args.outfile, "wb"))