import numpy as np
import librosa
import librosa.core
import matplotlib.pyplot as plt
import matplotlib.style as ms
import time
import sys
sys.path.append('..\\')
from pitch_estimation import *

ms.use('seaborn-muted')

y, sr = librosa.load('violin.wav')
start = time.time()

pitch = calculate_pitches(y,fs = sr)
elp = time.time() - start
print("Elapsed time: {} seconds. Length of piece: {} seconds.".format(elp, y.shape[0]/sr))
plt.plot(pitch)
plt.ylabel("Estimated pitch (Hz)")
plt.show()