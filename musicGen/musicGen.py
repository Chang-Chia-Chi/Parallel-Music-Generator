import os
import glob
import json
import time
import numpy as np
import multiprocessing as mp
from const_gen import *

music_id = 0

MajorHigh = 1
MajorLow = 2
MinorHigh = 3
MinorLow = 4
MajorChord = 5
MinorChord = 6

major_types = np.array([MajorChord, MajorChord, MajorLow, MajorLow, MajorHigh, 
                        MajorHigh, MajorChord, MajorChord, MajorLow, MajorHigh])
minor_types = np.array([MinorChord, MinorChord, MinorLow, MinorLow, MinorHigh, 
                        MinorHigh, MinorChord, MinorChord, MinorLow, MinorHigh])

major_high = None
major_low = None
minor_high = None
minor_low = None
major_chord = None
minor_chord = None

def load_matrices(npy_folder):
    global major_high
    global major_low
    global minor_high
    global minor_low
    global major_chord
    global minor_chord

    print("Start Loading Matrices...")
    npy_files = [os.path.join(npy_folder, path) for path in os.listdir(npy_folder)]
    for n_file in npy_files:
        if "Major" in n_file and "Chord" in n_file:
            major_chord = np.load(n_file)
        elif "Minor" in n_file and "Chord" in n_file:
            minor_chord = np.load(n_file)
        elif "Major" in n_file and "High" in n_file:
            major_high = np.load(n_file)
        elif "Major" in n_file and "Low" in n_file:
            major_low = np.load(n_file)
        elif "Minor" in n_file and "High" in n_file:
            minor_high = np.load(n_file)
        elif "Minor" in n_file and "Low" in n_file:
            minor_low = np.load(n_file)        

    print("Complete Reading All Matrices")

def matrix2prob(matrix):
    new_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row_sum = sum(matrix[i])
        if row_sum == 0:
            new_matrix[i] = 1 / matrix.shape[1]
        else:
            new_matrix[i, :] = matrix[i, :] / row_sum
    return new_matrix
    
def matrices2probMP():
    global major_high
    global major_low
    global minor_high
    global minor_low
    global major_chord
    global minor_chord

    print("Start Converting Matrices to Prob Matrices...")
    processes = []
    matrices = [major_high, major_low, minor_high, minor_low, major_chord, minor_chord]
    
    pool = mp.Pool(processes=6)
    prob_matrices = pool.map(matrix2prob, matrices)
    major_high = prob_matrices[0].copy()
    major_low = prob_matrices[1].copy()
    minor_high = prob_matrices[2].copy()
    minor_low = prob_matrices[3].copy()
    major_chord = prob_matrices[4].copy()
    minor_chord = prob_matrices[5].copy()    

    print("Complete Converting Matrices to Prob Matrices")
    pool.close()

def matrices2probSeq():
    global major_high
    global major_low
    global minor_high
    global minor_low
    global major_chord
    global minor_chord

    print("Start Converting Matrices to Prob Matrices...")
    matrices = [major_high, major_low, minor_high, minor_low, major_chord, minor_chord]
    prob_matrices = []
    for mat in matrices:
        new_matrix = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            row_sum = sum(mat[i])
            if row_sum == 0:
                new_matrix[i] = 1 / mat.shape[1]
            else:
                new_matrix[i, :] = mat[i, :] / row_sum

        prob_matrices.append(new_matrix)

    major_high = prob_matrices[0].copy()
    major_low = prob_matrices[1].copy()
    minor_high = prob_matrices[2].copy()
    minor_low = prob_matrices[3].copy()
    major_chord = prob_matrices[4].copy()
    minor_chord = prob_matrices[5].copy()

    print("Complete Converting Matrices to Prob Matrices")

def get_next_note(prev_tone, prev_dur, m_type, matrix):
    if m_type in [1, 2, 3, 4]: # melodic line
        if prev_tone == -1:
            curr_tone = np.random.choice(12, p = [0.5, 0, 0.1, 0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0])
            curr_tone = np.random.randint(OCTAVE_SAPN) * 12 + curr_tone
            curr_dur = np.random.randint(0, NUM_DURATION)
        else: # by markov matrix
            row = prev_tone * NUM_DURATION + prev_dur
            curr_note = np.random.choice(matrix.shape[1], p=matrix[row])
            curr_tone = int(curr_note/NUM_DURATION)
            curr_dur = curr_note % NUM_DURATION
    else: # chord
        if prev_tone == -1:
            if m_type == 5:
                mid = 3
            elif m_type == 6:
                mid = 4

            curr_tone = np.random.choice([7 + mid * 144, mid + 7*12, mid * 12 + 7 * 144]) + CHORD_BASE
        else:
            row = prev_tone - CHORD_BASE
            curr_tone = np.random.choice(matrix.shape[1], p=matrix[row])
            curr_tone += CHORD_BASE
        
        curr_dur = np.random.randint(4, NUM_DURATION)

    return (int(curr_tone), int(curr_dur))

def music_gentype(matrix, m_type):
    np.random.seed() # to prevent sub-process having same seed as main process
    music_gen = []
    prev_tone = -1
    prev_dur = -1
    beats_gen = 0
    while beats_gen < NUM_BEATS:
        curr_tone, curr_dur = get_next_note(prev_tone, prev_dur, m_type, matrix)
        beats_gen += (curr_dur + 1)
        prev_tone = curr_tone
        prev_dur = curr_dur
        if beats_gen > NUM_BEATS: # If beats generated exceed limited length, chop off
            curr_dur -= beats_gen - NUM_BEATS
        music_gen.append([curr_tone, curr_dur])
    return music_gen

def gen_wrapper(matrices, func):
    def wrapper(m_type):
        return func(matrices[m_type - 1], m_type)
    return wrapper

def music_genMP(matrices, tune):
    if tune == 1:
        music_types = minor_types
    elif tune == 2:
        music_types = major_types

    pool = mp.Pool(processes=10)
    args = [(matrices[m_type - 1], m_type) for m_type in music_types]
    musics = pool.starmap(music_gentype, args)
    pool.close()

    return musics

def music_genSeq(matrices, tune):
    if tune == 1:
        music_types = minor_types
    elif tune == 2:
        music_types = major_types
    
    musics = []
    for type in music_types:
        wrap_fun = gen_wrapper(matrices, music_gentype)
        musics.append(wrap_fun(type))
    
    return musics

def get_music(tune):
    n_folder_name = "matrix_npy"
    current_path = os.getcwd()
    npy_folder = os.path.join(current_path, n_folder_name)
    load_matrices(npy_folder)
    matrices2probMP()

    matrices = [major_high, major_low, minor_high, minor_low, major_chord, minor_chord]
    # print("Start sequential music generation")
    # t_s = time.time()
    # musics = music_genSeq(matrices, tune)
    # print("Complete sequential music generation")
    # print("Time of sequential music generation: {:.2f} s".format(time.time() - t_s))

    print("Start parallel music generation")
    t_s = time.time()
    musics = music_genMP(matrices, tune)
    print("Complete parallel music generation")
    print("Time of parallel music generation: {:.2f} s".format(time.time()- t_s))
    return json.dumps({'id':music_id, 'music':musics})

# interface for websocket
def main(tune):
    return get_music(tune)

#main(tune=1)