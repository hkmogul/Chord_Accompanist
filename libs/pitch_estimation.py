import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
# function library for pitch and key estimation
def nextpow2(n):
    i = 1
    while i < n:
        i *= 2
    return i
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
def hz_to_midi(val, rnd = True):
    if val == 0:
        return -1
    mid = 69+12*math.log(val/440,2)
    if rnd:
        mid = round(mid)
    return mid

def pitches_to_midi(data, rnd = True):
    mfunc = np.vectorize(hz_to_midi)
    return mfunc(data,rnd=rnd)

def data_usage_vector(midi):
    ''' Extract note usage vector for key estimation '''
    # get pitch of data
    chroma = np.zeros((1,12))
    
    # normalize the octave
    for i in range(midi.shape[0]):
        if midi[i] == -1:
            continue
        midi[i] = midi[i] % 12
        chroma[0,midi[i]] += 1
    return chroma

def beat_sync_chroma(data, fs, midi=None, tHop=0.01):
    if midi is None:
        pitches = calculate_pitches(data=data, fs=fs, tHop=tHop)
        midi = pitches_to_midi(pitches)
    tempo, beats = librosa.beat.beat_track(y=data, sr=fs, units='time')
    end_time = tHop * midi.shape[0]
    pitch_times = list(frange(0,end_time, tHop))
    pitch_index = 0
    # create chroma vector for each beat
    all_chroma = np.zeros((beats.shape[0], 12))
    for i in range(beats.shape[0]):
        while pitch_times[pitch_index] <= beats[i]:
            if midi[pitch_index] == -1:
                # ignore null reads
                pitch_index +=1
                continue
            ch_index = midi[pitch_index] % 12
            all_chroma[i,ch_index] +=1
            pitch_index +=1
    return beats, all_chroma

def true_beat_sync_chroma(data, fs, frame_len=512):
    tempo, beats = librosa.beat.beat_track(y=data, sr=fs, units='frames', hop_length=frame_len, trim=True)
    print("Estimated tempo is {}".format(tempo))
    chroma = librosa.feature.chroma_cqt(y=data, sr=fs, hop_length=frame_len)
    sync_chroma = librosa.util.sync(chroma, beats)
    return librosa.frames_to_time(beats, sr=fs, hop_length=frame_len), sync_chroma.transpose()
# estimate pitch using cepstral domain peak-picking
def calculate_pitches(data, lowest_hz = 40, highest_hz = 600, fs = 220500, tHop = 0.01, tW = 0.025, threshold = 1):
    hopSamples = math.floor(fs*tHop)
    windowSamples = math.floor(fs*tW)
    minPeriodSamples = math.floor(fs * (1/highest_hz))
    maxPeriodSamples = math.floor(fs * (1/lowest_hz))
    nFrames = math.floor((len(data) - maxPeriodSamples - windowSamples)/hopSamples)
    pitch = np.zeros((nFrames))
    hamm = np.hamming(windowSamples)
    nfft = nextpow2(windowSamples)
    for index in range(0,nFrames):
        n0 = index*hopSamples
        n = n0+windowSamples
        section = data[n0:n]
        section = section * hamm

        # cepstrum!
        spectrum = np.fft.fft(section, nfft)
        power_spectrum = abs(spectrum)**2
        cepstrum = np.fft.ifft(power_spectrum, nfft)

        # lifter the cepstrum to remove very high frequencies
        half_cepstrum = cepstrum[0:round(cepstrum.shape[0]/2)]

        Lc = 15
        lifter = np.zeros(half_cepstrum.shape)
        lifter[Lc:half_cepstrum.shape[0]] = 1
        ht_cepstrum = np.real(half_cepstrum * lifter)
        # we want to reject values outside of the min and max
        window = np.indices(ht_cepstrum.shape)
        filt = np.logical_or(window >= minPeriodSamples, window <= maxPeriodSamples).astype(np.uint8)        
        ht_cepstrum = filt * ht_cepstrum
        ht_cepstrum = ht_cepstrum.reshape((ht_cepstrum.shape[1]))
        ht_cepstrum = (ht_cepstrum >= threshold)*ht_cepstrum
        idx = np.argmax(ht_cepstrum)
        if (idx != 0):
            freq = fs/idx
            pitch[index] = freq
        else:
            pitch[index] = 0
    return pitch        

def group_beat_chroma(beat_times, chroma, group_num=4):
    ''' sum beat sync chroma together per beat. returns the new beat times and summed chroma '''
    new_chroma = []
    new_beat_times = []
    for i in range(len(beat_times)):
        if i % group_num == 0:
            # create a new chroma vector to sum other ones with
            chroma_init = chroma[i,:]
            new_chroma.append(chroma_init)
            new_beat_times.append(beat_times[i])
        else:
            new_chroma[-1] = new_chroma[-1] + chroma[i,:]
    chr_array = np.array(new_chroma)
    return new_beat_times, chr_array
    

