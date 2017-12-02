import sklearn_crfsuite
import numpy as np
chroma_scale = 1

from dataset_utils import *

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