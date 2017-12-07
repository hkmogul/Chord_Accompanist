import numpy as np
import librosa
import os
import sys
import argparse
import pretty_midi
import sklearn_crfsuite
from sklearn import svm
import random
import pickle 
import matplotlib.pyplot as plt
sys.path.append("libs")
import pitch_estimation
import dataset_utils
import beat_util
import synth_utils
import hmm_utils

key_identifier_file = "key_identifier.p"
hmm_file = "hmm.p"
parser = argparse.ArgumentParser()
parser.add_argument("-infile", dest = "infile",help='location of audio file')
parser.add_argument("-onsetData", dest="onsetfile", help="location of pickle file of onset pattern to use")
parser.add_argument("-outfile", dest = "outfile",help="destination file of midi")
parser.add_argument("-ismajor", dest="ismajor", help="True for using major chord progression, False otherwise")
parser.add_argument("-timeSig", dest="timeSig", help="Number of beats per measure")
args = parser.parse_args()
if args.ismajor is None:
    major = True
elif args.ismajor.lower() == "true":
    major = True
    print("Using major chords")
else:
    major = False
    print("Using minor chords")

if args.timeSig is None:
    timeSig = 4
else:
    timeSig = int(args.timeSig)
if args.onsetfile is None:
    onset_sig = np.zeros((1000))
    onset_sig[0] = 1
else:
    onset_signatures = pickle.load(open(args.onsetfile, "rb"))
    onset_sig = random.choice(onset_signatures)

key_identifier = pickle.load(open(key_identifier_file, "rb"))
hmm_data = pickle.load(open(hmm_file,"rb"))

# get pitches, chroma stats, and beat sync chroma of the input file
y,sr = librosa.load(args.infile)
# attempt again, but use the true beat synchronous chroma
print("Predicting chords from true chroma...")
true_beats, true_chroma = pitch_estimation.true_beat_sync_chroma(data=y,fs=sr)
data_usage_vector = true_chroma.sum(axis=0)
print(data_usage_vector.shape)
key = key_identifier.predict(data_usage_vector.reshape(1, -1))
print("Predicted key is {}".format(synth_utils.note_names[key[0]]))
if key[0] != 0:
    print("Rolling chroma to match the key")
    true_chroma = np.roll(true_chroma, key[0], axis=0)
# group by the time signature
true_beats,true_chroma = pitch_estimation.group_beat_chroma(true_beats, true_chroma, group_num=timeSig)
for i in range(true_chroma.shape[0]):
    rank = true_chroma[i,:].flatten()
    rank.sort()
    thresh = rank[-5] # allow only the top 5 notes
    for j in range(len(rank)):
        if true_chroma[i,j] < thresh:
            true_chroma[i,j] = 0
    # renormalize
    true_chroma[i,:] = true_chroma[i,:]/max(0.0001, true_chroma[i,:].sum())

chords = hmm_utils.estimate_chords(true_chroma, hmm_data["models"], hmm_data["transitions"], hmm_data["priors"])
for i in range(len(true_beats)):
    print("Time: {}, chords {}".format(true_beats[i], chords[i]))


onsets = synth_utils.onset_signature_to_onsets(onset_sig, true_beats)
print("Creating MIDI...")
# use chord progression, onset pattern, key, and mode to create MIDI output
mid = synth_utils.create_pretty_midi(chords, true_beats, onsets, key=key, major=major)
mid.write(args.outfile)

