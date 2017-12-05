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
y, sr = librosa.load('..\solo\whitneyHouston.mp3')
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
true_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_len)
beat_frames = librosa.core.time_to_frames(beats, sr=sr, hop_length=hop_len)
true_beat_sync_chrom = librosa.util.sync(true_chroma, beat_frames).transpose()
true_ch_usage = true_chroma.transpose().sum(axis=0)
ch = normscale *true_ch_usage/max(0.001,true_ch_usage.sum())
chroma_norm = ch.astype(np.uint32)
print(key_estimator.predict([chroma_norm]))
print(true_beat_sync_chrom.shape)
for i in range(0, true_beat_sync_chrom.shape[0]):
    norm = true_beat_sync_chrom[i,:]
    norm = norm/max(norm.sum(),0.0001)
    # threshold the normalization, then renormalize
    norm_f = norm.flatten()
    norm_f.sort()
    thresh  = norm_f[-2]
    for i in range(12):
        if norm[i] < thresh:
            norm[i] = 0
    norm = norm/max(norm.sum(),0.0001)
    true_beat_sync_chrom[i,:] = norm

plt.imshow(true_beat_sync_chrom.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.xlabel("Beat Index")
plt.ylabel("Chroma Index")
plt.title("Beat Synchronous Chroma")
plt.show()

new_beats,group_chr = group_beat_chroma(beat_frames, true_beat_sync_chrom,group_num=16)

plt.imshow(group_chr.transpose(), interpolation='nearest', aspect='auto', origin='bottom', cmap='gray_r')
plt.xlabel("Beat Index")
plt.ylabel("Chroma Index")
plt.title("Beat Synchronous Chroma Grouped by 4")
plt.show()