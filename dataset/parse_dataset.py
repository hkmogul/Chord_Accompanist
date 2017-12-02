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
from dataset_utils import *

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
    data['key'] = get_key(kf)
    # ignore data that doesn't provide a key
    if data['key'][0] is 'unvoiced':
        continue
    notes, midiSeq = get_note_data(corr_mel)

    chords =get_chord_data(corr_chord, data['key'][0], not data['key'][1])
    data['notes'] = notes # midi number, onset/offset time, and scale degree
    data['midi_seq'] = midiSeq # chroma
    data['chords'] = chords # label of tonic and major minor of chords
    alignment = align_chord_notes(chords, notes) # align notes with indices of chord labels
    roman_alignment = align_chord_notes_scale_deg(chords, notes)
    chroma_seq, chord_seq = alignment_to_chroma(alignment,0) # turn alignment into series of fake chroma corresponding to occurence of chroma in a certain chord
    fake_chroma_seq,chord_seq2 = alignment_to_chroma(roman_alignment)
    transition_chroma, transition_chord = alignment_to_chroma(roman_alignment, data['key'][0], allow_self=True)
    # a fake beat synchronous chroma, if you will
    data['alignment'] = alignment
    data['norm_alignment'] = roman_alignment
    data['chroma'] = chroma_seq
    data['chord_seq'] = chord_seq
    data['fake_chroma'] = fake_chroma_seq
    data['roman_chord'] = chord_seq2
    data['midi_ch'] = midi_seq_chroma(midiSeq) # 12 unit array of note usage in melody
    data['chord_stat'] = key_invariant_chord_usage_array(chords)
    if data['key'][1]:
        path = os.path.join(dst_folder,'major',title+'.pkl')
    else:
        path = os.path.join(dst_folder,'minor',title+'.pkl')
    with open(path,'wb') as p:
        pickle.dump(data, p)