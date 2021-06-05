import os
import glob
import pickle
import numpy as np
from constant import *

def txt2pickle(txt_folder, pickle_folder):
    os.chdir(txt_folder)
    txt_files = glob.glob("*.txt")
    # Pickle read and write in binary mode
    for txt in txt_files:
        with open(txt, 'rb') as t_file:
            txt_content = t_file.read()
            txt_name = os.path.split(txt)[-1][:-4]
            pickle_name = txt_name + ".pickle"
            pickle_path = os.path.join(pickle_folder, pickle_name)
            with open(pickle_path, 'wb') as p_file:
                pickle.dump(txt_content, p_file)
    return
    
def txt2npy(txt_folder, pickle_folder):
    os.chdir(txt_folder)
    txt_files = glob.glob("*.txt")
    for txt in txt_files:
        with open(txt, 'r') as t_file:
            t_lines = t_file.readlines()
            txt_name = os.path.split(txt)[-1][:-4]
            if "Chord" in txt_name:
                matrix = np.zeros([NUM_CHORD, NUM_CHORD])
            else:
                matrix = np.zeros([NUM_NOTE, NUM_NOTE])
            row = 0
            for line in t_lines:
                line_values = line.split()
                col = 0
                for value in line_values:
                    matrix[row][col] = value
                    col += 1
                row += 1
            npy_name = txt_name + ".pickle"
            npy_path = os.path.join(pickle_folder, npy_name)
            np.save(npy_path, matrix)

def main(to_pickle:bool):
    t_folder_name = "matrix_result"
    current_path = os.getcwd()
    txt_folder = os.path.join(current_path, t_folder_name)
    if to_pickle:
        p_folder_name = "matrix_pickle"
        pickle_folder = os.path.join(current_path, p_folder_name)
        if not os.path.isdir(pickle_folder):
            os.mkdir(pickle_folder)
        txt2pickle(txt_folder, pickle_folder)
    else:
        n_folder_name = "matrix_npy"
        npy_folder = os.path.join(current_path, n_folder_name)
        if not os.path.isdir(npy_folder):
            os.mkdir(npy_folder)
        txt2npy(txt_folder, npy_folder)
        

if __name__ == "__main__":
    main(False)

