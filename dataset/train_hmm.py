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
import dataset_utils

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
def my_imshow(data, **kwargs):
    """Wrapper for imshow that sets common defaults."""
    plt.imshow(data, interpolation='nearest', aspect='auto', 
               origin='bottom', cmap='gray_r', **kwargs)

vocab = list(range(0,len(dataset_utils.chord_labels)))
count = 0
for p in pkls:

    data = pickle.load(open(p, "rb"))
    t = data['align']
    if len(t) == 0:
        continue

    w,y = zip(*t)
    v,identities = np.unique(w,return_inverse=True)
    #print(identities)
    X = (identities.reshape(-1, 1) == np.arange(len(vocab))).astype(int)
    #print(X)
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


print("Obs shape: {}, y shape {}, lengths {}, sum of lengths {}".format(obs.shape, len(states), len(lengths), sum(lengths)))
#create an HMM!
print("Attempting fit with HMM")
clf2 = MultinomialHMM(alpha = 0.5)
clf2.fit(obs,states,lengths)
yp = clf2.predict(X)
print(yp)
print("---\nGround truth:")
print(y)
print("Training StructuredPerceptron")
clf = StructuredPerceptron(max_iter=50, lr_exponent=.001)
clf.fit(obs,states, lengths)
yp = clf.predict(X)
print(yp)
