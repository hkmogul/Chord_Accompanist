import glob
import pickle
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from random import randint
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import sklearn_crfsuite
# self made libs
sys.path.append(os.path.join("..","libs"))

from dataset_utils import *
from hmm_utils import *
sys.path.append(os.path.join("..","pymir"))
sys.path.append(os.path.join("..","pymir","pymir"))
from pymir import Pitch
parser = argparse.ArgumentParser()
parser.add_argument("-folder", dest = "folder",help='location of pkls')
parser.add_argument("-outfile", dest = "outfile",help="destination file")
args = parser.parse_args()
chroma_scale = 100
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
all_filt_chroma = np.empty((0))
test_index = randint(0, len(pkls))
# transitions to each chord label. list of tuples in the form of (initial_chord, transition_chord)
chord_mvs = [] 
vocab = np.array(range(0,len(chord_labels)))
count = 0
features = []
labels = []
for p in pkls:
    # so we need the chroma and chord_seq objects from the output
    data = pickle.load(open(p, "rb"))
    chroma = data['fake_chroma']
    key, isMinor = data['key']
    feature_list = []
    for i in range(0, chroma.shape[0]):
        chroma[i,:] =chroma_scale* chroma[i,:]/max(chroma[i,:].sum(), 0.00001)
        feature_dict = {}
        for j in range(0,chroma.shape[1]):
            feature_dict[str(j)] = chroma[i,j]
        feature_list.append(feature_dict)
    chord_sequence = data['roman_chord']

    if len(chord_sequence) == 0:
        continue
    labelList = []
    for i in chord_sequence:
        labelList.append(i)
    labels.append(labelList)
    features.append(feature_list)
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
        test_transitions = get_move_list(chord_sequence)
        test_key = key
        test_isMinor = isMinor
        test_features = feature_list
        test_labels = labelList
        print("Test file is {}".format(p))
    count += 1

for i in range(test_chroma.shape[0]):
    print("Chord is {}, notes are \n{}\n-----".format(test_labels[i], test_chroma[i,:]))
print("Training CRF")
crf = sklearn_crfsuite.CRF(all_possible_transitions =True, max_iterations=100)
crf.fit(features, labels)
print("Predicting single file")
print("Expected is \n{}".format(test_labels))
print("----\nPredicted is \n{}".format(crf.predict_single(test_features)))