# util file for beat segmentation. 
import librosa
import numpy as np
import pretty_midi

signature_len = 1000

def segment_onsets(data=None, sr=None, filename=None, group_measures=False, group_num=4):
    """ Segment an audio file or preloaded data's onsets into each beat

    Args:
        data (np.ndarray): The audio data
        sr (int): 

    Returns:
        list(np.ndarray): List of segmented onsets per beat
    """
    if data is None and sr is None and filename is None:
        raise ValueError("Must specify at least one required argument of data or filename")
        return
    elif data is None and sr is None:
        data,sr = librosa.load(filename, mono=True)
    elif data is None:
        data = librosa.load(filename, sr=sr, mono=True)

    # get onsets
    onset_strength = librosa.onset.onset_strength(y=data, sr=sr)
    tempo_est = librosa.beat.tempo(onset_envelope=onset_strength, sr = sr)
    tempo = tempo_est[0]
    # set trim to true because we want to segment the beats
    t2,beats = librosa.beat.beat_track(onset_envelope=onset_strength, sr = sr, bpm=tempo, trim=True)
    if beats.shape[0] == 0:
        return []
    onsets = librosa.onset.onset_detect(onset_envelope=onset_strength, backtrack=True)
    beat_data = []
    beat_data.append(onsets[0:beats[0]])
    for i in range(0,beats.shape[0]-1):
        n = beats[i]
        n0 = beats[i+1]
        beat_data.append(onsets[n:n0])
    beat_data.append(onsets[beats[-1]:])
    if group_measures:
        new_group = group_onsets(beat_data, group_num)
        return new_group
    else:
        return beat_data

def group_onsets(onsets, group_num=4):
    """
    concatenate onsets in contiguous groups
    """
    new_onsets = []
    for i in range(0, len(onsets)):
        if i % group_num == 0:
            # start the new vector
            on_group = np.copy(onsets[i]) # i trust nothing
        else:
            # concatenate onto on_group
            on_group = np.concatenate((on_group, onsets[i]))

        if i % group_num == group_num -1:
            # finish the on_group
            new_onsets.append(on_group)
            on_group = np.empty((0)) # i really trust nothing
    if on_group.shape[0] != 0:
        # append the last one - this would occur if we have a trailing measure
        new_onsets.append(on_group)
    return new_onsets

def normalize_measure(onset_data):
    """ Normalizes onset times of an onset measure to go from values [0:1]
    Given that these could be in frames or seconds, this brings everything to a unitless sphere

    """
    # we can assume these values are monotonically increasing, so this is just a matter of normalizing
    # subtract by the min value so it starts at 0
    od = onset_data - onset_data.min()
    # do a check on what the max value is in case it is all zeros for some reason
    return od/max(od.max(), 0.0001)
    
def stretch_measure(onset_data, length_data=1000):
    od = np.copy(onset_data)
    if od.max() != 1:
        od = normalize_measure(od)
    return od*length_data
    
def create_measure_signature(onset_measure):
    '''
    Returns a signature_len point vector of impulses that represent the "signature" of the beat section
    '''
    onset_norm = stretch_measure(onset_measure, length_data=1000)
    # quick rounding
    onset_norm = onset_norm.astype(np.uint32)
    impulses = np.zeros((signature_len))
    for i in onset_norm:
        impulses[i] = 1
    return impulses
def measure_signature_to_onset_times(signature, duration_ms = 1000):
    """ convert a time independent onset signature to a vector of onset times"""
    # get locations of nonzero elements
    locs = np.where(signature)[0]
    
    pass
