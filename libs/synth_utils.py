import pretty_midi
import numpy as np
import sys
import os
import beat_util
np.array([0,0,0,0,0,0,0,0,0,0,0,0])
# triads based on if the key is major or minor
note_names = ["C","C#","D","Eb","E","F","F#","G","G#","A","B"]
chord_triads = {
    ("I",True):np.array([1,0,0,0,1,0,0,1,0,0,0,0]), # major chord
    ("II",True):np.array([0,0,1,0,0,1,0,0,0,1,0,0]), #minor 2 chord
    ("III", True):np.array([0,0,0,0,1,0,0,1,0,0,0,1]), # minor 3 chord
    ("IV", True):np.array([1,0,0,0,0,1,0,0,0,1,0,0]), # major chord
    ("V",True):np.array([0,0,1,0,0,0,0,1,0,0,0,1]), # perfect fifth
    ("VI", True):np.array([1,0,0,0,1,0,0,0,0,1,0,0]), # minor 6th
    ("VII",True):np.array([0,0,1,0,0,0,1,0,0,0,0,1]), # diminished 7th
    ("I",False):np.array([1,0,0,1,0,0,0,1,0,0,0,0]), #minor first
    ("II",False):np.array([0,0,1,0,0,1,0,0,1,0,0,0]), # diminished 2nd
    ("III", False):np.array([0,0,0,0,1,0,0,0,1,0,0,1]), # major third
    ("IV", False):np.array([1,0,0,0,0,1,0,0,1,0,0,0]), # minor fourth
    ("V",False):np.array([0,0,1,0,0,0,0,1,0,0,1,0]), #minor fifth (assume harmonic minor)
    ("VI", False):np.array([1,0,0,1,0,0,0,0,1,0,0,0]), # major 6th
    ("VII",False):np.array([0,0,1,0,0,1,0,0,0,0,1,0]), # major 7th
}

def get_chord_index(onset_time, chord_times):
    index = 0
    while index < len(chord_times) and chord_times[index] <= onset_time:
        index += 1
    return index

def create_pretty_midi(chord_sequences, chord_times, onset_times, instrument_name='Piano', octave=5,key=0,major=True):
    ''' creates a PrettyMIDI object of chords '''
    duration = 0.1
    chord_index =0
    midi_data = pretty_midi.PrettyMIDI()
    inst_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(programm=inst_program)
    for time in onset_times:
        chord_index = get_chord_index(time, chord_times)
        triad_key = (chord_sequences[chord_index], major)
        activations= np.roll(chord_triads[triad_key], key)
        for i in range(12):
            if activations[i] != 0:
                noteName = note_names[i]+str(octave)
                noteNum = pretty_midi.note_name_to_number(noteName)
                note = pretty_midi.Note(velocity=100, pitch=noteNum, start=onset_times[i], end=onset_times[i]+duration)
                instrument.notes.append(note)
    midi_data.instruments.append(instrument)
    return midi_data

def onset_signature_to_onsets(onset_signature, beats):
    onsets = []
    for i in range(len(beats)-1):
        start = beats[i]
        duration = beats[i+1]-beats[i]
        onsets.extend(beat_util.measure_signature_to_onset_times(onset_signature, duration/1000, start/1000))
    return onsets
