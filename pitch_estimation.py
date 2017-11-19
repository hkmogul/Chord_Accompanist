import numpy as np
import math

def nextpow2(n):
    i = 1
    while i < n:
        i *= 2
    return i
def hz_to_midi(val, rnd = True):
    mid = 69+12*math.log(val/440,2)
    if rnd:
        mid = round(mid)
    return mid

def pitches_to_midi(data, rnd = True):
    mfunc = np.vectorize(hz_to_midi)
    return mfunc(data,rnd=rnd)

def calculate_pitches(data, lowest_hz = 40, highest_hz = 900, fs = 220500, tHop = 0.01, tW = 0.025, threshold = 1):
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

        # lifter the cepstrum
        half_cepstrum = cepstrum[0:round(cepstrum.shape[0]/2)]
        Lc = 15
        lifter = np.zeros(half_cepstrum.shape)
        lifter[Lc:half_cepstrum.shape[0]] = 1
        ht_cepstrum = np.real(half_cepstrum * lifter)
        # we want to reject values outside of the min and max
        # ht_cepstrum[0:minPeriodSamples] = 0
        # ht_cepstrum[maxPeriodSamples:] = 0
        count = 0
        foundCandidate = False
        window = np.indices(ht_cepstrum.shape)
        filt = np.logical_or(window >= minPeriodSamples, window <= maxPeriodSamples).astype(np.uint8)        
        ht_cepstrum = filt * ht_cepstrum
        ht_cepstrum = ht_cepstrum.reshape((ht_cepstrum.shape[1]))
        while not foundCandidate and count < ht_cepstrum.shape[0]:
            idx = np.argmax(ht_cepstrum)
            freq = fs/idx
            if ht_cepstrum[idx] >= threshold:
                foundCandidate = True
                print("Freq is {}, corresponding val is {}".format(freq, ht_cepstrum[idx]))

            else:
                ht_cepstrum[idx] = ht_cepstrum.min()
                count += 1
        if foundCandidate:
            pitch[index] = freq
        else:
            pitch[index] = 0
    return pitch        


