import numpy as np
import scipy.sparse
import os
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--output', '-o', required = True)
args = parser.parse_args()

input_folder = args.input
output_folder = args.output


def write_labels(output_file, list) :
    with open(output_file + ".txt", 'w') as filehandle:
        for listitem in list:
            filehandle.write('%s\n' % listitem)


def buildPCAmatrix(directory, output_file, labels_file) :
    matrix_PCA = None
    rownames = []
    i=0
    for filename in os.listdir(directory):
        i=i+1
        print(i)
        if filename.endswith(".npz"):
            print(os.path.join(directory, filename))
            sparse_matrix = scipy.sparse.load_npz(os.path.join(directory, filename))
            matrix = sparse_matrix.todense()
            matrix = np.array(matrix)



            flat_mat = matrix.flatten()
            flat_mat = scipy.sparse.csc_matrix(flat_mat)

            rownames.append(os.path.splitext(filename)[0])

            if matrix_PCA is None:
                matrix_PCA = flat_mat

            else:
                matrix_PCA = scipy.sparse.vstack([matrix_PCA, flat_mat])



    print("writing labels in file")
    write_labels(labels_file, rownames)
    print("writing matrix to file")
    scipy.sparse.save_npz(output_file, matrix_PCA)
    print(matrix_PCA.shape)
    return matrix_PCA


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = output_folder + "/PCA_matrix"
labels_file = output_folder + "/PCA_labels"


matrix_PCA = buildPCAmatrix(input_folder, output_file, labels_file)