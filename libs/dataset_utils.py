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

numeral2number = {"VII":14,"VI":13,"IV":11,"V":12,"III":10,"II":9,"I":8,"vii":7,"vi":6,"iv":4,"v":5, "iii":3,"ii":2,"i":1, "none":0}
number2numeral = {v:k for k,v in numeral2number.items()}
chord_roman_labels = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
key2Num = {"[C]":0,"[C#]":1,"[Db]":1,"[D]":2, "[D#]":3, "[Eb]":3,"[E]":4,"[F]":5,"[F#]":6,"[Gb]":6,"[G]":7,"[G#]": 8, "[Ab]":8, "[A]":9, "[A#]":10, "[Bb]":10, "[B]":11}
keyList = list(set([v for k,v in key2Num.items()]))
num2Key = {v:k for k,v in key2Num.items()}
mode_invariant_roman= {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7}


onset_expander = 2

class Note:
    def __init__(self,scale_deg = 0, onset = 0, duration = 0):
        self.scale_deg = scale_deg
        self.onset = onset
        self.duration = duration

class Chord:
    def __init__(self,isMajor = True, onset = 0, duration = 0, tonic = 0, label_index = 0):
        self.isMajor = isMajor
        self.onset = onset
        self.duration = duration
        self.tonic = tonic
        self.label_index = label_index
        self.used = False # for figuring out the alignment

def roman_numeral_to_number(numeric, ignoreCase = True):
    if ignoreCase:
        numeric = numeric.lower()
    for num in numeral2number:
        if numeric == num:
            return numeral2number[num]
    return 0

def get_note_data(filename, distThresh = .99):
    ''' Reads a mel file to get scale numbers, onsets, and interpolation of the duration of the notes '''
    notes = []
    midiSeq = []
    with open(filename,'r') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        lineList = list(reader)
        if len(lineList) == 0:
            return [],[]
        for i in range(0, len(lineList)-1):
            note = Note()
            line = lineList[i]
            note.scale_deg = int(line[3]) 
            note.onset = float(line[1])*onset_expander
            next_onset = float(lineList[i+1][1]) *onset_expander
            duration = next_onset - note.onset -.01
            # default to a threshold if greater, to handle pauses
            note.duration =  min(duration,distThresh)
            notes.append(note)
            midiSeq.append(int(line[2]) % 12)
    # by default, say the last note has the same duration as the threshold
    note = Note()
    line = lineList[-1]
    note.onset = float(line[1])*onset_expander
    note.scale_deg = float(line[3])
    note.duration = distThresh
    midiSeq.append(int(line[2]) % 12)
    notes.append(note)
    return notes,midiSeq

def extract_chord_str(numeric):
    # get substring of all chars up to the first non char
    allowed_chars = ['i','v','I','V']
    orig = numeric
    # first pass to get to a point where there are good chars
    pos = 0
    while pos < len(numeric):
        if numeric[pos] in allowed_chars:
            break
        pos += 1
    numeric = numeric[pos:]
    pos = 0

    while pos < len(numeric):
        if numeric[pos] not in allowed_chars:
            break
        pos+=1
    #print(numeric[:pos])
    numeric = numeric[:pos]

    return numeric
def translate_roman_numeral(numeral):
    # remove any disallowed characters
    num = extract_chord_str(numeral)
    # check if it is in upper case. this will denote major/minor
    isMajor = num.upper() == num
    return mode_invariant_roman[num.upper()], isMajor

def get_chord_label_index(num, isMajor):
    if isMajor:
        offset = 7
    else:
        offset = 0
    return num+offset

def get_chord_data(filename):
    chords = []
    with open(filename, 'r') as tsv:
        lineList = list(csv.reader(tsv, delimiter='\t'))
        if len(lineList) == 0:
            return
        for i in range(0, len(lineList)-1):
            line = lineList[i]
            chord = Chord()
            if len(line) < 3:
                continue
                
            chord.onset = float(line[1]) *onset_expander
            # get mode invariant numeral for the chord (aka, read the chord regardless of upper case or lower case)
            modeInvariantLabel,mode = translate_roman_numeral(line[2])
            chord.tonic = modeInvariantLabel
            chord.isMajor = mode
            chord.label_index = get_chord_label_index(modeInvariantLabel, mode)
            chord.duration = float(lineList[i+1][1])*onset_expander-chord.onset-.01
            chords.append(chord)

        # last chord
        line = lineList[-1]
        chord = Chord()
        if len(line) <= 3:
            return chords
        chord.onset = float(line[1])*onset_expander
        print(line)
        if line[2] != '':
            chord.tonic,chord.isMajor = translate_roman_numeral(line[2])
            chord.label_index = get_chord_label_index(chord.tonic, chord.isMajor)
            chord.duration = 500 # make the last chord last a long time
            chords.append(chord)
    return chords


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
            minor = False
            if len(res) >1:
                if res[1].count('b') > 1:
                    # treat anything that has added flats as minor
                    major = False
                    minor = True
            return key, minor
    return 'unvoiced',False

def find_corresponding_chord(onset, chords):
    '''
    return index of chord that corresponds to the onset of a note
    '''
    for i in range(0,len(chords)):
        ch = chords[i]
        ch_on = ch.onset
        ch_off = ch_on + ch.duration
        if onset >= ch_on and onset <= ch_off:
            return i
    return -1

def find_alignment_index(chord, align):
    ch_on = chord.onset
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
    ch_prev = chords[0].label_index
    for note in notes:
        # find the corresponding chord for that time and duration
        a = {}
        note_onset = note.onset
        note_num = note.scale_deg
        a['onset'] = note_onset
        ch_in = find_corresponding_chord(note_onset, chords)
        if ch_in != -1:
            chords[ch_in].used =True
            ch = chords[ch_in].label_index
            ch_prev = ch
        else:
            ch = ch_prev
            

        a['pair'] = [note_num, ch]
        alignment.append(a)
    
    # second pass - go through chords that havent been used and insert them where they belong
    for chord in chords:
        if chord.used:
            continue
        else:
            index = find_alignment_index(chord, alignment)
            a = {}
            a['onset'] = chord.onset
            a['pair'] = [-1,chord.label_index]
            alignment.insert(index, a)

    # one more time, create a list of just the pairs so we have something that can turn into a numpy list
    al = []
    for a in alignment:
        al.append((a['onset'],a['pair']))
    return al



def alignment_to_chroma(alignment,key=0, allow_self=False):
    ''' uses alignment to generate fake chroma for each chord, adding to the corresponding 
    allow_self = allow self transitions, akak use the time domain to generate the chroma as well
    '''
    #chroma_seq = np.zeros((12,8)) # column per chord type, row per note
    first = True
    chord_seq = [] # chord labels
    chroma_seq = np.zeros((1,12))
    prev_measure = 0
    for align in alignment:
        a = align[1]
        measure_num = align[0]
        chord = a[1]
        note = int(a[0])
        if note == -1:
            # ignore null reads for this
            continue
        elif first:
            chroma_seq = np.zeros((1,12))
            chroma_seq[:,note%12] += 1
            chord_seq.append(chord)
            first = False
            prev_chord = chord
            prev_measure = int(measure_num)
        elif chord == prev_chord and ((allow_self and int(measure_num) == prev_measure) or not allow_self):
            chroma_seq[-1,note%12] += 1
        else: # new chord in sequence
            chroma = np.zeros((1,12))
            chroma[:,note%12] += 1
            chroma_seq = np.concatenate((chroma_seq, chroma), axis = 0)
            #print(chroma)
            chord_seq.append(chord)
            prev_chord = chord
            prev_measure = int(measure_num)

    if key != 0:
        chroma_seq = np.roll(chroma_seq, key, axis=1)
    return chroma_seq,chord_seq

def midi_seq_chroma(midiSeq):
    ch = np.zeros((12))
    for n in midiSeq:
        ch[n] += 1
    return ch


majorFilt = np.array([1,0,1,0,1,1,0,1,0,1,0,1])

# avoid confusion with melodic, natural, and harmonic minor by allowing all sevenths
minorFilt = np.array([1,0,1,1,0,1,0,1,1,0,1,1])
def remove_off_key_tones(chroma, key, isMinor):
    if (isMinor):
        filt = np.roll(minorFilt, key)
    else:
        filt = np.roll(majorFilt, key)
    filtered_chroma = np.copy(chroma)
    for i in range(chroma.shape[0]):
        filtered_chroma[i,:] = chroma[i,:] * filt
    return filtered_chroma
