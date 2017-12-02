import numpy as np
import librosa
import librosa.core
import matplotlib.pyplot as plt
import matplotlib.style as ms
import time
import sys
import os
sys.path.append('..\\')
sys.path.append(os.path.join("..","libs"))
from pitch_estimation import *

ms.use('seaborn-muted')

y, sr = librosa.load('violin.wav')
start = time.time()
tHop = 0.01
pitch = calculate_pitches(y,fs = sr, tHop=tHop)
elp = time.time() - start
# print("Elapsed time: {} seconds. Length of piece: {} seconds.".format(elp, y.shape[0]/sr))
# plt.plot(pitch)
# plt.ylabel("Estimated pitch (Hz)")
# plt.show()

midi = pitches_to_midi(pitch)
beats, chroma = beat_sync_chroma(y, sr, midi,tHop=tHop)
for i in range(beats.shape[0]):
    print("Beat time: {}".format(beats[i]))
    print(chroma[i])
hop_len = 512 # num samples for chromagram