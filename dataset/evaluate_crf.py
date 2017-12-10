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
import matplotlib.pyplot as plt
# self made libs
sys.path.append(os.path.join("..","libs"))

from dataset_utils import *
from hmm_utils import *
from crf_util import *


parser = argparse.ArgumentParser()
parser.add_argument("-folder", dest = "folder",help='location of pkls')
parser.add_argument("-infile", dest = "infile",help="pickled CRF")
args = parser.parse_args()
chroma_scale = 1
if args.folder is None or not os.path.exists(args.folder):
    print("None or invalid folder specified. Will find all the pickles")
    folder = os.path.join("**","**")
else:
    folder = args.folder



# get all the pickles!
pkls = glob.glob(os.path.join(folder,"*.pkl"))

crf = pickle.load(open(args.infile, "rb"))

all_scores = []

for p in pkls:
    data = pickle.load(open(p, "rb"))
    chroma = data['chroma_seq2']
    chord_sequence = data['chord_seq2']

    chord_seq_str = []
    for c in chord_sequence:
        chord_seq_str.append(chord_roman_labels[c])
    feats = mode_variant_feature_dict(chroma)
    title = data['title']
    chord_p = crf.predict_single(feats)
    scores, score_avg = score_chord_accuracy(chord_p, chord_seq_str, threshold=2)
    print("{}:{}".format(title, score_avg))
    all_scores.append(score_avg)

print("-------")
print("Number of songs evaluated: {}".format(len(all_scores)))
print(sum(all_scores)/len(all_scores))