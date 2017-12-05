import glob
import pickle
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import time
sys.path.append(os.path.join("..","libs"))
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
# dict for mapping key strings to their chroma number 0 = C
features = []
features2 = []
labels = []
def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

for p in pkls:
    data = pickle.load(open(p, "rb"))
    key = data['key'][0]  # don't care about major/minor in this case
    if key is 'unvoiced':
        continue
    ch = data['note_usage']
    # 12 unit array of note usage in melody
    allFloats = True
    normscale = 100 #dilating the data seems to be very helpful
    for c in list(ch):
        allFloats = allFloats and isFloat(c)
    if allFloats:
        ch_n = normscale *ch/max(0.001,ch.sum())
        ch_n = ch_n.astype(np.uint32)
        # get most used note
        mun = np.argmax(ch_n)
        lun = np.argmin(ch_n)
        featvector = []
        featvector = list(ch_n)
        
        features.append(featvector)
        labels.append(key)
    else:
        print("{} has non float chroma???".format(p))

feat = np.array(features) 
y = np.array(labels)



if args.outfile is not None:
    # use all data to generate a pickle of the classifier
    clf = svm.SVC(kernel='linear', probability=True, random_state=int(time.time()))
    clf.fit(feat, y)
    pickle.dump(clf,open(args.outfile, "wb"))
    print("Fit a linear SVC to all data. Pickled to {}".format(args.outfile))
else:
    scores = []
    for i in range(100):
        X_train, X_test, y_train,y_test = train_test_split(feat, y, test_size = 0.1, random_state = int(time.time()))
        clf = svm.SVC(kernel='linear', probability=True, random_state=int(time.time()))
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        yp = clf.predict(X_test)
        # print("Y test has {} elements".format(len(y_test)))
        # ascore = metrics.accuracy_score(y_test, yp, normalize=False)
        # print("SVC Score is {}".format(score))
        # print("Ascore is {}".format(ascore))
        # print("-----")
        scores.append(score)
    print("Average score in 100 iterations was {}".format(sum(scores)/100))
    print(len(y_test))
    print(len(y_train))
