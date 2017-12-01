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
from dataset_utils import *
def get_move_list(labels):
    '''
    Generate list of tuples of movements
    '''
    init = labels[:-1]
    dest = labels[1:]
    return list(zip(init,dest))

def estimate_chord_transitions(chord_mvs):
    ''' calculate chord movement transitions and priors from transitions '''
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