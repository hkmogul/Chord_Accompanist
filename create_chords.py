import numpy as np
import librosa
import os
import sys
import argparse
import pretty_midi
import sklearn_crfsuite
from sklearn import svm

sys.path.append("libs")
import pitch_estimation
import dataset_utils
import beat_util
import synth_utils
key_identifier_file = "key_identifier.p"
chord_predictor_file = "chord_predictor.p"
parser = argparse.ArgumentParser()
parser.add_argument("-infile", dest = "infile",help='location of audio file')
parser.add_argument("-onsetData", dest="onsetfile", help="location of pickle file of onset pattern to use")
parser.add_argument("-outfile", dest = "outfile",help="destination file of midi")
parser.add_argument("-ismajor", dest="ismajor", help="True for using major chord progression, False otherwise")

args = parser.parse_args()
if args.ismajor is None:
    major = True
elif args.ismajor.lower() is "true":
    major = True
else:
    major = False
key_identifier = pickle.load(open(key_identifier_file, "rb"))
onset_signatures = pickle.load(open(args.onsetfile, "rb"))
# get pitches, chroma stats, and beat sync chroma of the input file
y,sr = librosa.load(args.infile)
pitches = pitch_estimation.calculate_pitches(y,fs=sr)
midi_notes = pitch_estimation.pitches_to_midi(pitches)
beats, all_chroma = pitch_estimation.beat_sync_chroma(data=y, fs=sr, midi_notes)
chroma_stats = pitch_estimation.data_usage_vector(midi_notes)

# use the chroma stats to get key 
key = key_identifier.predict(chroma_stats)

# use beat sync chroma to predict chords

# use beat times to make an onset pattern

# use chord progression, onset pattern, key, and mode to create MIDI output