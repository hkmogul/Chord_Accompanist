import csv
import glob
import pickle
import re 
import os
import numpy as np
from os.path import basename, splitext
import string
import sys
sys.path.append(os.path.join("..","libs"))
import dataset_utils as dsu

melody_folder = "melody"
chord_folder = "chords"
key_folder = "measures"
dst_folder = "parsed_data"
melody_files = glob.glob("**/*.nlt")
chord_files = glob.glob("**/*.clt")
key_files = glob.glob("**/*.mel")

for kf in key_files:
    title = splitext(basename(kf))[0]
    print("Title: {}".format(title))
    corr_mel = os.path.join(melody_folder, title+'.nlt')
    corr_chord = os.path.join(chord_folder,title + '.clt')
    if not corr_mel in melody_files or not corr_chord in chord_files:
        print(":(")
        continue
    data = {}
    key_data = dsu.get_key(kf)
    title_tru = dsu.get_title(kf)
    data['title'] = title_tru
    # ignore data that doesn't provide a key
    if key_data[0] is 'unvoiced':
        continue
    data['key'] = key_data
    notes, midiSeq = dsu.get_note_data(corr_mel)
    chords = dsu.get_chord_data(corr_chord)
    # ignore data that doesnt have chord changes
    if len(chords) < 3:
        print("Ignoring {}, only has {} chord changes".format(title, len(chords)))
        continue
    
    # align notes and chords
    print("Key is {}, major is {}".format(key_data[0], key_data[1]))
    data['notes'] = notes # midi number, onset/offset time, and scale degree
    data['midi_seq'] = midiSeq # chroma
    data['chords'] = chords # label of tonic and major minor of chords
    numMin, numMaj = dsu.analyze_chords(chords)
    if numMin > numMaj:
        print("Chords vote for minor")
        if key_data[1] != False:
            print("!!!!!!!!")
            key_data = (key_data[0], False)
    elif numMaj > numMin:
        print("Chords vote for major")
        if key_data[1] != True:
            print("!!!!!!!")
            key_data = (key_data[0], True)
    data['key'] = key_data

    alignment = dsu.align_chord_notes(chords, notes, prop ="tonic")
    # mode variant alignment
    alignment_mode = dsu.align_chord_notes(chords,notes, prop="label_index")
    data['alignment'] = alignment
    data['alignment_mode'] = alignment_mode
    chroma_seq, chord_seq = dsu.alignment_to_chroma(alignment, key=0, allow_self=True)
    chroma_seq2, chord_seq2 = dsu.alignment_to_chroma(alignment_mode, key=0, allow_self=False)
    data['chroma_seq'] = chroma_seq
    data['chord_seq'] = chord_seq
    data['chroma_seq2'] = chroma_seq2 # the 2 ones denote labels that include the mode of the chord
    data['chord_seq2'] = chord_seq2
    note_usage = dsu.midi_seq_chroma(midiSeq)
    data['note_usage'] = note_usage
    if data['key'][1]:
        path = os.path.join(dst_folder,'major',title+'.pkl')
    else:
        path = os.path.join(dst_folder,'minor',title+'.pkl')
    with open(path,'wb') as p:
        pickle.dump(data, p)