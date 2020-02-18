from numpy import savez_compressed
import matplotlib.pyplot as plt
import  numpy as np
import os
import scipy.sparse


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
input_dir = dir_path + "HiCNN2_package/matrices/predicted_fullMats/"

dir_above =  os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"  # get the parent directory



for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        input_mat = os.path.join(input_dir, filename)
        filename = os.path.splitext(filename)[0]            # filename without .npy
        print(filename)

        matrix = np.load(input_mat)
        matrix = scipy.sparse.csc_matrix(matrix)
        
        out_file = dir_above + "tmp/imputation_output/" + filename

        
        scipy.sparse.save_npz(out_file, matrix)


