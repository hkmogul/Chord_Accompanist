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
parser = argparse.ArgumentParser()
parser.add_argument("-folder", dest = "folder",help='location of pkls')
parser.add_argument("-outfile", dest = "outfile",help="destination file")
args = parser.parse_args()
if args.outfile is None:
    print("Usage: python train_hmm.py -folder fld -outfile out")
    sys.exit()
if args.folder is None or not os.path.exists(args.folder):
    print("None or invalid folder specified. Will find all the pickles")
    folder = "**\\**"
else:
    folder = args.folder
# get all the pickles!
pkls = glob.glob(folder+"\\*.pkl")
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
labels = []
def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
for p in pkls:
    data = pickle.load(open(p, "rb"))
    key = data['key'][0] # don't care about major/minor in this case
    if key is 'unvoiced':
        continue
    ch = data['midi_ch']
    chordstat = data['chord_stat']
    allFloats = True
    normscale = 100
    for c in list(ch):
        allFloats = allFloats and isFloat(c)
    if allFloats:
        ch_n = normscale *ch/ch.max()
        ch_n = ch_n.astype(np.uint32)
        # get most used note
        mun = np.argmax(ch_n)
        lun = np.argmin(ch_n)
        chord_n = normscale * chordstat/chordstat.max()
        chord_n = chord_n.astype(np.uint32)
        muc = np.argmax(chord_n)
        luc = np.argmin(chord_n)
        featvector = []
        featvector = list(ch_n) + list(chord_n)
        featvector.append(mun)
        featvector.append(muc)
        featvector.append(lun)
        featvector.append(luc)
        featvector.append((mun + 7)%12)
        features.append(featvector)
        labels.append(key)
    else:
        print("{} has non float chroma???".format(p))

feat = np.array(features) 
y = np.array(labels)
X_train, X_test, y_train,y_test = train_test_split(feat, y, test_size = 0.1, random_state = int(time.time()))

# print(len(X_train))
# print(len(X_train[0]))
# for i in range(len(X_train)):
#     print(X_train[i])
#     print("---")

clf = svm.SVC(kernel='linear', probability=True, random_state=int(time.time()))
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
yp = clf.predict(X_test)
ascore = metrics.accuracy_score(y_test, yp, normalize=False)
print("SVC Score is {}".format(score))
print("Ascore is {}".format(ascore))
print("Y test has {} elements".format(len(y_test)))
r_clf = RandomForestClassifier()
r_clf.fit(X_train, y_train)
yp = r_clf.predict(X_test)
rfscore = metrics.accuracy_score(y_test,yp)
rfscore_num =metrics.accuracy_score(y_test,yp, normalize=False)
print("RF CLF score is {}. Number correct = {}".format(rfscore,rfscore_num))

# try using a linear SVC
clf_lin = svm.LinearSVC()
clf_lin.fit(X_train,y_train)
yp = clf_lin.predict(X_test)
linscore = metrics.accuracy_score(y_test, yp)
print("Linear SVC score is {}".format(linscore))