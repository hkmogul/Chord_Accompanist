import glob
import pickle
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# self made libs
from dataset_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-folder", dest = "folder",help='location of pkls')
parser.add_argument("-outfile", dest = "outfile",help="destination file")
args = parser.parse_args()

if args.folder is None or not os.path.exists(args.folder):
    print("None or invalid folder specified. Will find all the pickles")
    folder = os.path.join("**","**")
else:
    folder = args.folder
# get all the pickles!
pkls = glob.glob(os.path.join(folder,"*.pkl"))
lengths = []
chord_seq = np.empty((0)) # check for size 
all_chroma = np.empty((0))
# transitions to each chord label. list of tuples in the form of (initial_chord, transition_chord)
chord_mvs = [] 
vocab = np.array(range(0,len(chord_labels)))
for p in pkls:
    # so we need the chroma and chord_seq objects from the output
    data = pickle.load(open(p, "rb"))
    chroma = data['chroma']
    chord_sequence = data['chord_seq']
    if len(chord_sequence) == 0:
        continue

    if chord_seq.shape[0] == 0:
        chord_seq = np.array(chord_sequence)
    else:
        chord_seq = np.concatenate((chord_seq,np.array(chord_sequence)), axis=0)
    if all_chroma.shape[0] == 0:
        all_chroma = chroma
    else:
        all_chroma = np.concatenate((all_chroma, chroma))
    lengths.append(len(chord_sequence))
    init = chord_sequence[:-1]
    dest = chord_sequence[1:]
    chord_mvs.extend(get_move_list(chord_sequence))

# train a hybrid GMM/HMM
models, transitions, priors = train_gaussian_models(all_chroma,chord_seq, chord_mvs)

