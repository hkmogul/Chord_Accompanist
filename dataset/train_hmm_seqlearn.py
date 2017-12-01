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
# self made libs
from dataset_utils import *
from hmm_utils import *
sys.path.append(os.path.join("..","pymir"))
sys.path.append(os.path.join("..","pymir","pymir"))
import seqlearn


from pymir import Pitch
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
test_index = randint(0, len(pkls))
vocab = np.array(range(0,len(chord_labels)))
count = 0
for p in pkls:
    # so we need the chroma and chord_seq objects from the output
    data = pickle.load(open(p, "rb"))
    alignment = data['alignment']

    if count == test_index:
        test_alignment = data['alignment']
        print("Test file is {}".format(p))
    count += 1
    

for i in range(0,len(test_label)):
    print("Label: {}\nChroma:\n{}".format(chord_labels[test_label[i]], test_chroma[i,:]))
    # use pymir to get a naive filterbank based estimate
    chromaList = list(test_chroma[i,:])
    chName, maxScore = Pitch.getChord(chromaList)
    chNameF, maxScoreF = Pitch.getChord(list(test_filt_chroma[i,:]))
    print("Naive chord estimate: {}".format(chName))
    print("Naive chord estimate (filtered): {}".format(chNameF))
    print("-----")
print("Key is {}, is minor? {}".format(test_key, test_isMinor))
for t in test_transitions:
    print("{} -> {}".format(chord_labels[t[0]],chord_labels[t[1]]))

# train a hybrid GMM/HMM
models, transitions, priors = train_gaussian_models(all_chroma,chord_seq, chord_mvs)

print("Actual labels: \n{}".format(test_label))

sys.exit()
print("Attempting a linear classifier for chord models")
score_list = []
for i in range(10):
    X_train, X_test, y_train,y_test = train_test_split(all_chroma, chord_seq, test_size = 0.25, random_state = int(time.time()))
    clf = svm.SVC(kernel='linear', probability=True, random_state=int(time.time()))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score_list.append(score)
    yp = clf.predict(X_test)
    #print("Y test has {} elements".format(len(y_test)))
    ascore = metrics.accuracy_score(y_test, yp, normalize=False)
    # print("SVC Score is {}".format(score))
    # print("Ascore is {}".format(ascore))
    # print("-----")

print("Average score unfiltered: {}".format(sum(score_list)/len(score_list)))
score_list = []

for i in range(10):
    X_train, X_test, y_train,y_test = train_test_split(all_filt_chroma, chord_seq, test_size = 0.25, random_state = int(time.time()))
    clf = svm.SVC(kernel='linear', probability=True, random_state=int(time.time()))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score_list.append(score)
    yp = clf.predict(X_test)
    #print("Y test has {} elements".format(len(y_test)))
    ascore = metrics.accuracy_score(y_test, yp, normalize=False)
    #print("SVC Score is {}".format(score))
    #print("Ascore is {}".format(ascore))
    #print("-----")

print("Average score filtered: {}".format(sum(score_list)/len(score_list)))
