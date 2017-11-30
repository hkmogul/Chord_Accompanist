import csv
import glob
import pickle
import re 
import os
import numpy as np
from os.path import basename, splitext
import string
import sklearn.mixture
from collections import Counter
def get_move_list(labels):
    '''
    Generate list of tuples of movements
    '''
    init = labels[:-1]
    dest = labels[1:]
    return list(zip(init,dest))

def estimate_chord_transitions(chord_mvs):
    ''' calculate chord movement transitions and priors from transitions '''
    nChordLabels = len(chord_roman_labels)
    transitions = np.zeros((nChordLabels, nChordLabels))
    counter = Counter(chord_mvs)
    for i in range(nChordLabels):
        for j in range(nChordLabels):
            transitions[i,j] = 1 + counter[(i,j)]
    
    priors = np.sum(transitions, axis=1)
    transitions /= priors[:,np.newaxis]
    priors /= np.sum(priors)
    return transitions,priors

def train_gaussian_models(features, labels, chord_mvs):
    nChordLabels = len(chord_roman_labels)
    # a gmm will return the log likelihood of a specific label
    generic = sklearn.mixture.GaussianMixture(n_components=1,covariance_type='full')
    print("Features.shape is {}".format(features.shape))
    generic.fit(features)
    models = []
    for chord_index in range(nChordLabels):
        rows = np.nonzero(labels == chord_index)
        if rows:
            model =  sklearn.mixture.GaussianMixture(n_components=1,covariance_type='full')
            model.fit(features[rows])
            models.append(model)
        else:
            models.append(generic)
    
    transitions, priors = estimate_chord_transitions(chord_mvs)
    return models, transitions, priors

def estimate_chords(chroma, models, transitions, priors):
    scores = np.array([model.score(chroma) for model in models])
    return viterbi(np.exp(scores.transpose()), transitions, priors)

def viterbi(posterior_prob, transition_prob, prior_prob):
    print(posterior_prob.shape)
    nFrames = posterior_prob.shape[0]
    nStates = len(chord_roman_labels)
    traceback = np.zeros((nFrames, nStates), dtype=int)
    # best probability of each state
    best_prob = prior_prob * posterior_prob[0]
    best_prob /= max(0.0001,np.sum(best_prob))
    for f in range(1,nFrames):
        poss_scores = (transition_prob * np.outer(best_prob, posterior_prob[f]))
        traceback[f] = np.argmax(poss_scores, axis=0)
        best_prob = np.max(poss_scores, axis=0)
        best_prob /= max(0.001,np.sum(best_prob))
    path = np.zeros(nFrames, dtype=int)
    path[-1] = np.argmax(best_prob)
    for i in range(nFrames-1, 0,-1):
        path[i-1] = traceback[i, path[i]]
    return path

def get_note_data(filename, distThresh = 1):
    ''' Reads a mel file to get scale numbers, onsets, and interpolation of the duration of the notes '''
    notes = []
    midiSeq = []
    with open(filename,'r') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        lineList = list(reader)
        if len(lineList) == 0:
            return [],[]
        for i in range(0, len(lineList)-1):
            note = {}
            line = lineList[i]
            onset = float(line[0])
            scale_deg = int(line[3]) +1 # add 1 so we can use 0 as no-pitch
            next_onset = float(lineList[i+1][0])
            duration = next_onset - onset
            # default to a threshold if greater, to handle pauses
            duration = min(duration,distThresh)
            note['onset'] = onset
            note['scale_deg'] = scale_deg
            note['duration'] = duration
            note['midi_num'] = int(line[2])
            notes.append(note)
            midiSeq.append(int(line[2]) % 12)
    # by default, say the last note has the same duration as the threshold
    note = {}
    line = lineList[-1]
    note['onset'] = float(line[0])
    note['scale_deg'] = float(line[3]) +1
    note['duration'] = distThresh
    note['midi_num'] = int(line[2])
    midiSeq.append(int(line[2]) % 12)
    notes.append(note)
    return notes,midiSeq

numeral2number = {"VII":14,"VI":13,"IV":11,"V":12,"III":10,"II":9,"I":8,"vii":7,"vi":6,"iv":4,"v":5, "iii":3,"ii":2,"i":1, "none":0}
number2numeral = {v:k for k,v in numeral2number.items()}
def chord_to_number(numeric):
    ''' converts the roman numeral representation of a number to the numerical value. ignores non-chord tone additions
    '''
    # get substring of all chars up to the first non char
    pos = 0
    while pos < len(numeric) and numeric[pos].isalpha():
        pos+=1
    #print(numeric[:pos])
    
    for num in numeral2number:
        if num == numeric[:pos]:
            return numeral2number[num], not num[0].isupper()
    return 0, False

chord_labels = ['none', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b','C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chord_roman_labels = ['none', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
def chord_num_to_index(num, isMinor):
    offset = 1
    if not isMinor:
        offset += 7
    return num+offset
def chord_to_label(numeric, key):
    '''
    Converts a chord in roman numeral format to a non key-specific label of its tonic and major/minor
    Also returns the index that would correspond to this label
    '''
    # get the chord number from the roman numeral
    num,isMinor = chord_to_number(numeric)
    print("Numeral: {}, num:{}, isMinor:{}".format(numeric, num, isMinor))
    if num == 0:
        return chord_labels[0], 0
    # key should already be an int where C = 0.  so all we need to do is offset the chord by  key+1
    chord_index = key+1 + num
    tonic = key %12
    num2 = num%7
    note = chord_num_to_note(num2, isMinor, tonic)
    chord_index = note
    if isMinor:
        offset = 1 # 1 because C = 0 in melody, but 0 corresponds to no note
    else:
        offset = 13 # add 12 to get to the major labels
    chord_index = note %12 + offset
    # print("Minor invariant chord label is {}".format(num2))
    # print("Sanity check: Received chord is {} ({}) parsed to {}".format(numeric, num,number2numeral[num]))
    # print("Key is {} ({})".format(key, num2Key[key]))
    # print("Tonic of key is {}, Chord note is {} which indexes to {}".format(tonic,chord_index, chord_labels[chord_index]))
    # print("-----")
    return chord_labels[chord_index], chord_index

def chord_num_to_note(chord_num, isMinor, tonic):
    # get 1-7 notation of chord, if it is minor, and its tonic to find the root note of this chord
    # for minor, just using Aeolian mode
    if isMinor:
        switcher = {
            1:tonic,
            2:tonic+2,
            3:tonic+3,
            4:tonic+5,
            5:tonic+7,
            6:tonic+8,
            7:tonic+10
        }
    else:
        switcher = {
            1:tonic,
            2:tonic+2,
            3:tonic+4,
            4:tonic+5,
            5:tonic+7,
            6:tonic+9,
            7:tonic+11
        }

    return switcher.get(chord_num, 0)
def get_chord_data(filename,key):
    chords = []
    with open(filename, 'r') as tsv:
        lineList = list(csv.reader(tsv, delimiter='\t'))
        if len(lineList) == 0:
            return
        for i in range(0, len(lineList)-1):
            line = lineList[i]
            chord = {}
            if len(line) < 3:
                continue

            chord['onset'] = float(line[0])
            chord_num, isMinor = chord_to_number(line[2])
            chord['chord'] = chord_num
            chord['chord_str'] = number2numeral[chord_num]
            chord['duration'] =  float(lineList[i+1][0])-chord['onset']
            ch_m, ch_m_i = chord_to_label(line[2], key)
            chord['tonic'] = ch_m
            chord['tonic_i'] = ch_m_i
            chords.append(chord)

        # last chord
        line = lineList[-1]
        chord = {}
        chord['onset'] = float(line[0])
        chord_num, isMinor = chord_to_number(line[2])
        chord['chord'] = chord_num

        chord['chord_str'] = number2numeral[chord['chord']]
        ch_m, ch_m_i = chord_to_label(line[2], key)
        chord['tonic'] = ch_m
        chord['tonic_i'] = ch_m_i
        chord['duration'] = 0
        chords.append(chord)
    return chords

key2Num = {"[C]":0,"[C#]":1,"[Db]":1,"[D]":2, "[D#]":3, "[Eb]":3,"[E]":4,"[F]":5,"[F#]":6,"[Gb]":6,"[G]":7,"[G#]": 8, "[Ab]":8, "[A]":9, "[A#]":10, "[Bb]":10, "[B]":11}
keyList = list(set([v for k,v in key2Num.items()]))
num2Key = {v:k for k,v in key2Num.items()}
def get_key(filename):
    with open(filename,'r') as f:
        for line in f.readlines():
            res = re.findall(r"\[[^\]]*\]",line)
            if len(res) == 0:
                continue
            if res[0] in key2Num:
                key = key2Num[res[0]]
            else:
                key = 'unvoiced'
                print("{}???".format(res[0]))
            major = True
            if len(res) >1:
                if "b" in res[1]:
                    # treat anything that has added flats as minor
                    major = False
            return key, major
    return 'unvoiced',False
def find_corresponding_chord(onset, chords):
    '''
    return index of chord that corresponds to the onset of a note
    '''
    for i in range(0,len(chords)):
        ch = chords[i]
        ch_on = ch['onset']
        ch_off = ch_on + ch['duration']
        if onset >= ch_on and onset <= ch_off:
            return i
    return -1
def find_alignment_index(chord, align):
    ch_on = chord['onset']
    for i in range(0,len(align)):
        on = align[i]['onset']
        if ch_on <= on:
            return i
    return len(align) # corresponds to the chord being appended
def align_chord_notes(chords, notes):
    ''' aligns chords and notes to a state/observation pair tuple
        following notes of RS200, chords have no duration, and should be assumed that they are in play until the next chord
    '''
    if notes is None:
        return []
    alignment = [] 
    
    # ultimately, list of dicts with onset time, note, and chord
    # make an initial pass to pair each note to a corresponding chord
    # then pass through the chords to insert/append/prepend null reads for notes
    for note in notes:
        # find the corresponding chord for that time and duration
        a = {}
        note_onset = note['onset']
        note_num = note['midi_num']
        a['onset'] = note_onset
        ch =0
        ch_in = find_corresponding_chord(note_onset, chords)
        if ch_in != -1:
            chords[ch_in]['used']=True
            ch = chords[ch_in]['chord']

        a['pair'] = [note_num, ch]
        alignment.append(a)
    
    # second pass - go through chords that havent been used and insert them where they belong
    for chord in chords:
        if 'used' in chord:
            continue
        else:
            index = find_alignment_index(chord, alignment)
            a = {}
            a['onset'] = chord['onset']
            a['pair'] = [0,chord['chord']]
            alignment.insert(index, a)

    # one more time, create a list of just the pairs so we have something that can turn into a numpy list
    al = []
    for a in alignment:
        al.append(a['pair'])
    return al

def alignment_to_chroma(al,key=0):
    ''' uses alignment to generate fake chroma for each chord, adding to the corresponding '''
    #chroma_seq = np.zeros((12,8)) # column per chord type, row per note
    first = True
    chord_seq = [] # chord labels
    chroma_seq = np.zeros((1,12))
    for a in al:
        chord = int(a[1])
        note = int(a[0])
        if note == 0:
            # ignore null reads for this
            continue
        elif first:
            chroma_seq = np.zeros((1,12))
            chroma_seq[:,note%12] += 1
            chord_seq.append(chord)
            first = False
            prev_chord = chord
        elif chord == prev_chord:
            chroma_seq[-1,note%12] += 1
        else: # new chord in sequence
            chroma = np.zeros((1,12))
            chroma[:,note%12] += 1
            chroma_seq = np.concatenate((chroma_seq, chroma), axis = 0)
            #print(chroma)
            chord_seq.append(chord)
            prev_chord = chord

    rolled_chroma = np.roll(chroma_seq, key, axis=1)
    return rolled_chroma,chord_seq

def midi_seq_chroma(midiSeq):
    ch = np.zeros((12))
    for n in midiSeq:
        ch[n] += 1
    return ch
def chord_usage_array(chords):
    ch = np.zeros((len(numeral2number)))
    for c in chords:
        chord = c['chord']
        ch[chord] += 1
    return ch

def key_invariant_chord_usage_array(chords):
    ch = np.zeros((len(chord_labels)))
    for c in chords:
        chord = c['tonic_i']
        ch[chord] +=1
    return ch