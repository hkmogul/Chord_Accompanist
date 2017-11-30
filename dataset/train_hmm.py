import glob
import pickle
import os
import sys
import argparse
import seqlearn
import numpy as np
from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import bio_f_score
import matplotlib.pyplot as plt

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
obs = np.empty((0)) # check for size 
states = []
lim = int(len(pkls)*7/8)
#chroma = np.zeros((1,8))

vocab = list(range(0,len(chord_labels)))
count = 0
for p in pkls:
    # chroma_seq, chord_seq = alignment_to_chroma(data['align']) # turn alignment into series of fake chroma corresponding to occurence of chroma in a certain chord
    # # a fake beat synchronous chroma, if you will
    # data['chroma'] = chroma_seq
    # data['chord_seq'] = chord_seq
    # so we need the chroma and chord_seq objects from the output.

    data = pickle.load(open(p, "rb"))
    chroma = data['chroma']
    chord_sequence = data['chord_seq']
    t = data['align']
    if len(chord_sequence) == 0:
        continue

    w,y = zip(*t)
    v,identities = np.unique(w,return_inverse=True)
    X = (identities.reshape(-1, 1) == np.arange(len(vocab))).astype(int)
    if count > lim:
        print("Test file is {}".format(p))
        break
    lengths.append(len(t))
    states = states + list(y)
    if obs.shape[0] == 0:
        obs = X
    else:
        obs = np.concatenate((obs,X), axis=0)
    count += 1
# for i in range(0,chroma.shape[1]):
#     print("Chord is {}. Chroma is {}".format(chord_seq[i], chroma[:,i]))


