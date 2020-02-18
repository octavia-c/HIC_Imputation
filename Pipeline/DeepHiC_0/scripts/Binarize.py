import os
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt

binary_cutoff = 0.85
input_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/output/GM12878/'
output_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/output/DeepHiC_50_binary_' + str(binary_cutoff) + '/'

def heatmap(output_name, matrix, lim):
    plt.xlim(0, lim)
    plt.ylim(lim, 0)
    plt.imshow(matrix, cmap='pink', interpolation='nearest')
    plt.savefig(output_name + ".png")

def to_binary(output_folder, filename, matrix, binary_cutoff):
    tri =  np.triu(matrix, k=0)
    vector = tri[np.triu_indices(len(matrix))]
    q = np.quantile(vector, binary_cutoff)
    vector= 1 * (vector > q)
    vector = scipy.sparse.csr_matrix(vector)
    #print(vector)
    heatmap(output_folder + filename, tri[0:500, 0:500], 500)
    return vector

def to_binary_for_folder(input_folder, output_folder, binary_cutoff):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pca_matrix = None
    file_counter = 0
    for file in os.listdir(input_folder):
        file_counter = file_counter + 1
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".npz"):
            print("Sample " + str(file_counter) + ": " + filename + " binarization")
            matrix = scipy.sparse.load_npz(input_folder + filename)
            matrix = matrix.todense()
            vector = to_binary(output_folder, filename, matrix, binary_cutoff)

            binary_path = os.path.join(output_folder, "binary_matrix_" + str(binary_cutoff))
            if not os.path.exists(binary_path):
                os.makedirs(binary_path)
            scipy.sparse.save_npz(binary_path + "/" + filename + "_binary.npz", vector)

            if(pca_matrix==None):
                pca_matrix=vector
            else:
                pca_matrix = scipy.sparse.vstack([pca_matrix, vector])

    pca_path = os.path.join(output_folder, "pca")
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    scipy.sparse.save_npz(pca_path+"/PCA.npz", pca_matrix)

to_binary_for_folder(input_folder, output_folder, binary_cutoff)
