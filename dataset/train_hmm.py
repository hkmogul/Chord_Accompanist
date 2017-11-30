import glob
import pickle
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import randint
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
test_index = randint(0, len(pkls))
# transitions to each chord label. list of tuples in the form of (initial_chord, transition_chord)
chord_mvs = [] 
vocab = np.array(range(0,len(chord_labels)))
count = 0
for p in pkls:
    # so we need the chroma and chord_seq objects from the output
    data = pickle.load(open(p, "rb"))
    chroma = data['chroma']
    for i in range(0, chroma.shape[0]):
        chroma[i,:] = chroma[i,:]/max(chroma[i,:].sum(), 0.00001)
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
    if count == test_index:
        test_chroma = chroma
        test_label = chord_sequence
        print("Test file is {}".format(p))
    count += 1
    

for i in range(0,len(test_label)):
    print("Label: {}\nChroma:\n{}\n-----".format(test_label[i], test_chroma[i,:]))

# train a hybrid GMM/HMM
models, transitions, priors = train_gaussian_models(all_chroma,chord_seq, chord_mvs)

predict = estimate_chords(chroma, models, transitions,priors)
print("Actual labels: \n{}".format(test_label))
print("------\nPredicted labels:\n{}".format(predict))