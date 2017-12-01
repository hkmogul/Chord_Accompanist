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
chord_labels = ['none', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b','C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chord_roman_labels = ['none', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
key2Num = {"[C]":0,"[C#]":1,"[Db]":1,"[D]":2, "[D#]":3, "[Eb]":3,"[E]":4,"[F]":5,"[F#]":6,"[Gb]":6,"[G]":7,"[G#]": 8, "[Ab]":8, "[A]":9, "[A#]":10, "[Bb]":10, "[B]":11}
keyList = list(set([v for k,v in key2Num.items()]))
num2Key = {v:k for k,v in key2Num.items()}
nChordLabels = len(chord_labels)
onset_expander = 4


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
            note = {}
            line = lineList[i]
            onset = float(line[1])*onset_expander
            scale_deg = int(line[3]) +1 # add 1 so we can use 0 as no-pitch
            next_onset = float(lineList[i+1][1]) *onset_expander
            duration = next_onset - onset -.01
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
    note['onset'] = float(line[1])*onset_expander
    note['scale_deg'] = float(line[3]) +1
    note['duration'] = distThresh
    note['midi_num'] = int(line[2])
    midiSeq.append(int(line[2]) % 12)
    notes.append(note)
    return notes,midiSeq


def chord_to_number(numeric):
    ''' converts the roman numeral representation of a number to the numerical value. ignores non-chord tone additions/complexities
    '''
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
    c = 0
    maj = False
    for num in numeral2number:
        if num == numeric[:pos]:
            c = numeral2number[num]
            maj =  not num[0].isupper()
            break
    # # check what got read as a none
    # if c == 0:
    #     print(orig)
    return c, maj


def chord_num_to_index(num, isMinor):
    offset = 1
    if not isMinor:
        offset += 7
    return num+offset
def chord_to_label(numeric, key, isMinor):
    '''
    Converts a chord in roman numeral format to a non key-specific label of its tonic and major/minor
    Also returns the index that would correspond to this label
    '''
    # get the chord number from the roman numeral
    num,isMinor2 = chord_to_number(numeric)
    #print("Numeral: {}, num:{}, isMinor:{}".format(numeric, num, isMinor))
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
def get_chord_data(filename,key, isMinor):
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

            chord['onset'] = float(line[1]) *onset_expander
            chord_num, isMinor = chord_to_number(line[2])
            chord['chord'] = chord_num
            chord['chord_str'] = number2numeral[chord_num]
            chord['duration'] =  float(lineList[i+1][1])*onset_expander-chord['onset']-.01
            ch_m, ch_m_i = chord_to_label(line[2], key, isMinor)
            chord['tonic'] = ch_m
            chord['tonic_i'] = ch_m_i
            chords.append(chord)

        # last chord
        line = lineList[-1]
        chord = {}
        chord['onset'] = float(line[1])*onset_expander
        chord_num, isMinor = chord_to_number(line[2])
        chord['chord'] = chord_num

        chord['chord_str'] = number2numeral[chord['chord']]
        ch_m, ch_m_i = chord_to_label(line[2], key, isMinor)
        chord['tonic'] = ch_m
        chord['tonic_i'] = ch_m_i
        chord['duration'] = 0
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
                if "b" in res[1]:
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
    ch_prev = 0
    for note in notes:
        # find the corresponding chord for that time and duration
        a = {}
        note_onset = note['onset']
        note_num = note['midi_num']
        a['onset'] = note_onset
        ch_in = find_corresponding_chord(note_onset, chords)
        if ch_in != -1:
            chords[ch_in]['used']=True
            ch = chords[ch_in]['tonic_i']
            ch_prev = ch
        else:
            ch = ch_prev
            

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
            a['pair'] = [0,chord['tonic_i']]
            alignment.insert(index, a)

    # one more time, create a list of just the pairs so we have something that can turn into a numpy list
    al = []
    for a in alignment:
        al.append((a['onset'],a['pair']))
    return al

def alignment_to_chroma(al,key=0):
    ''' uses alignment to generate fake chroma for each chord, adding to the corresponding '''
    #chroma_seq = np.zeros((12,8)) # column per chord type, row per note
    first = True
    chord_seq = [] # chord labels
    chroma_seq = np.zeros((1,12))
    prev_measure = 0
    for align in al:
        a = align[1]
        measure_num = align[0]
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
            prev_measure = int(measure_num)
        elif chord == prev_chord and int(measure_num) == prev_measure:
            chroma_seq[-1,note%12] += 1
        else: # new chord in sequence
            chroma = np.zeros((1,12))
            chroma[:,note%12] += 1
            chroma_seq = np.concatenate((chroma_seq, chroma), axis = 0)
            #print(chroma)
            chord_seq.append(chord)
            prev_chord = chord
            prev_measure = int(measure_num)

    chroma_seq = np.roll(chroma_seq, key, axis=1)
    return chroma_seq,chord_seq

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
