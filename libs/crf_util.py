import sklearn_crfsuite
import numpy as np
chroma_scale = 1

from dataset_utils import *

def mode_variant_feature_dict(chroma_seq):
    feature_list = []
    for i in range(0, chroma_seq.shape[0]):
        # normalize chroma
        chroma_seq[i,:] = chroma_seq[i,:]/max(chroma_seq[i,:].sum(), 0.00001)
        feature_dict = {}
        for j in range(0, chroma_seq.shape[1]):
            feature_dict[str(j)] = chroma_seq[i,j]
        feature_list.append(feature_dict)
    return feature_list

def create_mode_variant_feature_dict(data):
    chord_seq = data['chord_seq2']
    chroma_seq = data['chroma_seq2']
    feature_list = []
    labels = []
    for i in range(0, chroma_seq.shape[0]):
        # normalize chroma
        chroma_seq[i,:] = chroma_seq[i,:]/max(chroma_seq[i,:].sum(), 0.00001)
        feature_dict = {}
        for j in range(0, chroma_seq.shape[1]):
            feature_dict[str(j)] = chroma_seq[i,j]
        feature_list.append(feature_dict)
        labels.append(chord_roman_labels[chord_seq[i]])
    return feature_list, labels


def feature_dict_from_chroma(chroma, chord_sequence):
    feature_list = []
    labels = []
    for i in range(0, chroma.shape[0]):
        chroma_copy = np.copy(chroma[i,:])
        chroma[i,:] =chroma_scale* chroma[i,:]/max(chroma[i,:].sum(), 0.00001)
        feature_dict = {}
        chord = chord_sequence[i]
        intI = roman_numeral_to_number(chord_sequence[i])
        if intI > 7:
            intI -= 7
        
        for j in range(0,chroma.shape[1]):
            feature_dict[str(j)] = chroma[i,j]
        feature_list.append(feature_dict)
        labels.append(str(intI))
    if len(feature_list) != len(labels):
        raise ValueError("Something is wrong. ")
    return feature_list, labels