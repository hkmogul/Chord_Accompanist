import csv
import glob
import pickle
distThresh = 1
notes = []
with open("purple_haze_dt.nlt",'r') as tsv:
    reader = csv.reader(tsv, delimiter='\t')
    lineList = list(reader)
    for i in range(0, len(lineList)-1):
        note = {}
        line = lineList[i]
        onset = float(line[0])
        scale_deg = int(line[3]) +1
        next_onset = float(lineList[i+1][0])
        duration = next_onset - onset
        # default to a threshold if greater, to handle pauses
        duration = min(duration,distThresh)
        note['onset'] = onset
        note['scale_deg'] = scale_deg
        note['duration'] = duration
        notes.append(note)
