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

def estimate_chord_transitions(chord_mvs, num_chords=7):
    ''' calculate chord movement transitions and initial probabilities from transitions '''
    transitions = np.ones((num_chords, num_chords))
    counter = Counter(chord_mvs)
    for i in range(num_chords):
        for j in range(num_chords):
            transitions[i,j] += counter[(i,j)]
    
    initialProbabilities = np.sum(transitions, axis=1)
    transitions /= initialProbabilities[:,np.newaxis]
    initialProbabilities /= np.sum(initialProbabilities)
    return transitions,initialProbabilities

def train_gaussian_models(features, labels, chord_mvs,num_chords=7):
    generic = sklearn.mixture.GaussianMixture(n_components=1,covariance_type='full')
    generic.fit(features)
    models = []
    print(num_chords)
    for chord_index in range(num_chords):
        rows = np.where(labels == chord_index)
        print(rows[0].shape[0])
        if rows[0].shape[0] != 0:
            model = sklearn.mixture.GaussianMixture(n_components=1,covariance_type='full')
            model.fit(features[rows[0]])
            models.append(model)
        else:
            models.append(generic)
    
    transitions, initialProbabilities = estimate_chord_transitions(chord_mvs, num_chords=num_chords)
    return models, transitions, initialProbabilities

def estimate_chords(chroma, models, transitions, initialProbabilities,num_chords=7):
    scoreList = []
    posterior = np.zeros((chroma.shape[0], num_chords))
    for i in range(chroma.shape[0]):
        for j in range(num_chords):
            posterior[i,j] = models[j].score(chroma[i].reshape(1,-1))
    posterior = np.array(posterior)
    return viterbi(np.exp(posterior), transitions, initialProbabilities)

def viterbi(emission_prob, transition_prob, prior_prob):
    nFrames = emission_prob.shape[0]
    nStates = emission_prob.shape[1]
    traceback = np.zeros((nFrames, nStates), dtype=int)
    # best probability of each state, normalized
    best_prob = prior_prob * emission_prob[0]
    best_prob /= max(0.0001,np.sum(best_prob))
    for f in range(1,nFrames):
        poss_scores = (transition_prob * np.outer(best_prob, emission_prob[f]))
        traceback[f] = np.argmax(poss_scores, axis=0)
        best_prob = np.max(poss_scores, axis=0)
        best_prob /= max(0.001,np.sum(best_prob))
    path = np.zeros(nFrames, dtype=int)
    path[-1] = np.argmax(best_prob)
    for i in range(nFrames-1, 0,-1):
        path[i-1] = traceback[i, path[i]]
    return path