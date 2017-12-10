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
parser.add_argument("-outfile", dest = "outfile",help="destination file")
args = parser.parse_args()
chroma_scale = 1
if args.folder is None or not os.path.exists(args.folder):
    print("None or invalid folder specified. Will find all the pickles")
    folder = os.path.join("**","**")
else:
    folder = args.folder



# get all the pickles!
pkls = glob.glob(os.path.join(folder,"*.pkl"))

lengths = []

test_index = randint(0, len(pkls))
# transitions to each chord label. list of tuples in the form of (initial_chord, transition_chord)

count = 0
features = []
labels = []
chroma_occurence = np.zeros((7,12))
for p in pkls:
    # so we need the chroma and chord_seq objects from the output
    data = pickle.load(open(p, "rb"))
    chroma = data['chroma_seq2']
    chord_sequence = data['chord_seq2']
    if len(chord_sequence) == 0:
        continue

    feats, labelList = create_mode_variant_feature_dict(data)
    features.append(feats)
    labels.append(labelList)
    if count == test_index:
        test_chroma = chroma
        title = data['title']
        print("Test file is {}, chords are \n{}".format(title, chord_sequence))
        test_labels = []
        test_labelInt = chord_sequence
        for i in range(len(chord_sequence)):
            test_labels.append(chord_roman_labels[chord_sequence[i]])

        print("Test file is {}".format(p))
    count += 1


print("Training CRF")
crf = sklearn_crfsuite.CRF(all_possible_transitions =True, max_iterations=100)
crf.fit(features, labels)
print("Predicting single file")
print("Expected is \n{}".format(test_labels))
prediction = crf.predict_single(test_chroma)
print("----\nPredicted is \n{}\n-----".format(prediction))
predictionInt = []
for p in prediction:
    predictionInt.append(chord_roman_labels.index(p))
print("--------")

plt.figure()
plt.plot(test_labelInt,"r--",label="Ground Truth")
plt.plot(predictionInt, "g", label="CRF Prediction")
plt.title("CRF Chord Prediction for Song {}".format(title))
plt.legend()
plt.show()

scores, score_avg = score_chord_accuracy(prediction, test_labels, threshold=1)
plt.figure()
plt.plot(scores)
plt.show()
print(score_avg)