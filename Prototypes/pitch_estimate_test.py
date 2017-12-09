import numpy as np
import librosa
import librosa.core
import librosa.feature
import matplotlib.pyplot as plt
import matplotlib.style as ms
import time
import sys
import os
from sklearn import svm
import pickle
sys.path.append('..\\')
sys.path.append(os.path.join("..","libs"))
from pitch_estimation import *

key_estimator = pickle.load(open("key_identifier.p","rb"))
y, sr = librosa.load('..\\beat_data\pop.wav')
start = time.time()
tHop = 0.01
pitch = calculate_pitches(y,fs = sr, tHop=tHop)
elp = time.time() - start
print("Elapsed time: {} seconds. Length of piece: {} seconds.".format(elp, y.shape[0]/sr))
plt.plot(pitch)
plt.ylabel("Estimated pitch (Hz)")
plt.show()

midi = pitches_to_midi(pitch)
plt.plot(midi)
plt.ylabel("Estimated MIDI")
plt.show()

beats, chroma = beat_sync_chroma(y, sr, midi=midi,tHop=tHop)

hop_len = 512 # num samples for chromagram
chroma_usage = chroma.sum(axis=0)
normscale = 100 
ch = normscale *chroma_usage/max(0.001,chroma_usage.sum())
chroma_norm = ch.astype(np.uint32)
print(key_estimator.predict([chroma_norm]))

plt.imshow(chroma.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.xlabel("Beat Index")
plt.ylabel("Chroma Index")
plt.title("Beat Synchronous Mock Chroma")
plt.show()

# get raw audio form of beat synchronous chroma

beats, chroma = beat_sync_chroma(data=y, fs=sr)
plt.imshow(chroma.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.show()

onsetStrength = librosa.onset.onset_strength(y=y, sr=sr)
tempo,beats = librosa.beat.beat_track(y=None, sr=sr, onset_envelope=onsetStrength, units='frames')
plt.figure()
for b in beats:
    print(b)
    plt.axvline(x=b,color='k', linestyle='--')

plt.plot(onsetStrength)
plt.show()