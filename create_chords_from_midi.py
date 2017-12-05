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
import pretty_midi
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
    timeSig = 2
else:
    timeSig = int(args.timeSig)

key_identifier = pickle.load(open(key_identifier_file, "rb"))
hmm_data = pickle.load(open(hmm_file,"rb"))
onset_signatures = pickle.load(open(args.onsetfile, "rb"))
onset_sig = random.choice(onset_signatures)
midiData = pretty_midi.PrettyMIDI(args.infile)
sr = 44100
y = midiData.synthesize(fs=sr)
# get pitches, chroma stats, and beat sync chroma of the input file
print("Estimating pitches...")
pitches = pitch_estimation.calculate_pitches(y,fs=sr)
midi_notes = pitch_estimation.pitches_to_midi(pitches)
print("Converting pitches to chroma...")
beats, all_chroma = pitch_estimation.beat_sync_chroma(data=y, fs=sr,midi= midi_notes)
chroma_stats = pitch_estimation.data_usage_vector(midi_notes)

# use the chroma stats to get key 
key = key_identifier.predict(chroma_stats)
print("Predicted key is {}".format(synth_utils.note_names[key[0]]))
# roll chroma to get the key invariant chroma
rolled_chroma = np.roll(all_chroma, key[0], axis=0)
for i in range(rolled_chroma.shape[0]):
    rolled_chroma[i,:] = rolled_chroma[i,:]/max(0.0001,rolled_chroma[i,:].sum())
plt.figure()

plt.imshow(rolled_chroma.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.show()
print("Predicting chords...")
new_beats,group_chr = pitch_estimation.group_beat_chroma(beats, rolled_chroma,group_num=timeSig)
for i in range(group_chr.shape[0]):
    # rank = group_chr[i,:].flatten()
    # rank.sort()
    # thresh = rank[-3]
    # for j in range(len(rank)):
    #     if group_chr[i,j] < thresh:
    #         group_chr[i,j] = 0
    group_chr[i,:] = group_chr[i,:]/max(0.0001, group_chr[i,:].sum())

plt.imshow(group_chr.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.xlabel("Beat Index")
plt.ylabel("Chroma Index")
plt.title("Grouped Beat Synchronous Chroma. Measure size={}".format(timeSig))
plt.show()
chords = hmm_utils.estimate_chords(group_chr, hmm_data["models"], hmm_data["transitions"], hmm_data["priors"])
for i in range(len(new_beats)):
   print("Time: {}, chords {}".format(new_beats[i], chords[i]))

# get onset times to use
onsets = synth_utils.onset_signature_to_onsets(onset_sig, beats)
print("Creating MIDI...")
# use chord progression, onset pattern, key, and mode to create MIDI output
mid = synth_utils.create_pretty_midi(chords, new_beats, onsets, key=key, major=major)
mid.write(args.outfile)