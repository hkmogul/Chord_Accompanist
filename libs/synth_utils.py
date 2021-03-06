import pretty_midi
import numpy as np
import sys
import os
import beat_util
import dataset_utils
np.array([0,0,0,0,0,0,0,0,0,0,0,0])
# triads based on if the key is major or minor
note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
chord_triads = {
    ("I",True):np.array(    [1,0,0,0,1,0,0,1,0,0,0,0]), # major chord
    ("II",True):np.array(   [0,0,1,0,0,1,0,0,0,1,0,0]), #minor 2 chord
    ("III", True):np.array( [0,0,0,0,1,0,0,1,0,0,0,1]), # minor 3 chord
    ("IV", True):np.array(  [1,0,0,0,0,1,0,0,0,1,0,0]), # major chord
    ("V",True):np.array(    [0,0,1,0,0,0,0,1,0,0,0,1]), # perfect fifth
    ("VI", True):np.array(  [1,0,0,0,1,0,0,0,0,1,0,0]), # minor 6th
    ("VII",True):np.array(  [0,0,1,0,0,0,1,0,0,0,0,1]), # diminished 7th
    ("I",False):np.array(   [1,0,0,1,0,0,0,1,0,0,0,0]), #minor first
    ("II",False):np.array(  [0,0,1,0,0,1,0,0,1,0,0,0]), # diminished 2nd
    ("III", False):np.array([0,0,0,0,1,0,0,0,1,0,0,1]), # major third
    ("IV", False):np.array( [1,0,0,0,0,1,0,0,1,0,0,0]), # minor fourth
    ("V",False):np.array(   [0,0,1,0,0,0,0,1,0,0,1,0]), #minor fifth (assume harmonic minor)
    ("VI", False):np.array([1,0,0,1,0,0,0,0,1,0,0,0]), # major 6th
    ("VII",False):np.array([0,0,1,0,0,1,0,0,0,0,1,0]), # major 7th
}

def get_chord_index(onset_time, chord_times):
    index = 0
    while index < len(chord_times)-1 and chord_times[index] <= onset_time:
        index += 1
    return index

def create_pretty_midi(chord_sequences, chord_times, onset_times, instrument_name='	Acoustic Guitar (nylon)', octave=4,key=0,major=True):
    ''' creates a PrettyMIDI object of chords '''
    duration = 0.25
    chord_index =0
    midi_data = pretty_midi.PrettyMIDI()
    inst_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=inst_program)
    prev_onset_time = 0
    prev_velocity = 0
    for i in range(len(onset_times)):
        time = onset_times[i]
        if i == len(onset_times)-1:
            duration = 0.25
        else:
            duration = abs((onset_times[i+1] - time)*.5) * .5

        chord_index = get_chord_index(time, chord_times)
        #chord_numeral = dataset_utils.number2numeral[chord_sequences[chord_index]+1].upper()
        #print("Creating chord {}, key is {}".format(chord_numeral, note_names[int(key)]))
        if time -prev_onset_time < 0.1:
            velocity = int(prev_velocity * .75)
        else:
            velocity = 50
        #triad_key = (chord_numeral, major)
        activations = dataset_utils.mode_variant_triads[chord_sequences[chord_index]]
        if key != 0:
            activations= np.roll(activations, key)
        #print(activations)
        for i in range(12):
            if activations[i]== 0:
                continue
            #print(i)
            noteName = note_names[i]+str(octave)
            #print(noteName)
            noteNum = pretty_midi.note_name_to_number(noteName)
            note = pretty_midi.Note(velocity=velocity, pitch=noteNum, start=time, end=time+duration)
            instrument.notes.append(note)
            #print("---")
        #print("---------------")
        prev_velocity = velocity
        prev_onset_time = time

    midi_data.instruments.append(instrument)
    return midi_data

def onset_signature_to_onsets(onset_signature, beats,beat_len=16):
    onsets = []
    for i in range(len(beats)-beat_len):
        start = beats[i]
        duration = beats[i+beat_len]-beats[i]
        onsets.extend(beat_util.measure_signature_to_onset_times(onset_signature, duration, start))
    return onsets

riemann_pairs = {0:2,1:3,2:4,3:5,4:6,5:7,6:0,7:1}
def riemann_transform(chords, allowed_chords):
    chords_r = []
    for i in range(len(chords)):
        if chords[i] not in allowed_chords:
            chords_r.append(riemann_pairs[chords[i]])
        else:
            chords_r.append(chords[i])
    return chords_r
