import scipy.sparse
from scipy import ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse as ap


def uniform_filter(matrix):
    return ndimage.uniform_filter(matrix,size=3)

def uniform_filter_threshold(matrix, threshold):
    filtered = ndimage.uniform_filter(matrix,size=3)
    threshold_indices = filtered < threshold
    filtered[threshold_indices] = 0
    return filtered

def random_walk_filter(matrix, p):
    new_matrix = np.eye(len(matrix))
    I = np.eye((len(matrix)))
    #p=0.03
    terminate = False
    i = 0
    while((not terminate) and i < 30):
        print("iteration: " + str(i))
        i+=1
        oldcopy = np.array(new_matrix, copy=True)
        new_matrix = np.add((1-p)*np.dot(new_matrix,matrix),p*I)
        if(np.linalg.norm(new_matrix-oldcopy)<= 0.000001):
            terminate = True
    return new_matrix

def norm_rows(matrix):
    new_matrix = np.zeros((len(matrix), len(matrix)))
    for i in range(len(matrix)):
        rowsum = np.sum(matrix[i,])
        for j in range(len(matrix)):
            if(rowsum!=0):
                new_matrix[i,j]=matrix[i,j]/rowsum
    return new_matrix

def heatmap(output,matrix,lim):
    plt.xlim(0, lim)
    plt.ylim(lim, 0)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.savefig(output+".png")


def uniform_for_folder(threshold, folder,output,replace = False):
    if not os.path.exists(output):
        os.makedirs(output)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            if(not os.path.exists(output+filename) or replace):
                print(filename)
                matrix = scipy.sparse.load_npz(folder+filename)
                matrix = matrix.todense()
                matrix = uniform_filter_threshold(matrix, threshold)
                heatmap(output+filename,matrix[0:500,0:500],500)
                matrix = scipy.sparse.csr_matrix(matrix)
                scipy.sparse.save_npz(output+filename, matrix)

def rw_for_folder(folder,output,replace = False, p):
    if not os.path.exists(output):
        os.makedirs(output)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            if(not os.path.exists(output + filename) or replace):
                print(filename)
                matrix = scipy.sparse.load_npz(folder+filename)
                matrix = matrix.todense()
                matrix = np.array(matrix)
                matrix = norm_rows(matrix)
                matrix = random_walk_filter(matrix, p)
                heatmap(output+filename,matrix[0:500,0:500],500)
                matrix = scipy.sparse.csr_matrix(matrix)
                scipy.sparse.save_npz(output+filename, matrix)

def to_binary(output,filename,matrix, quant_q):
    tri =  np.triu(matrix, k=0)
    vector = tri[np.triu_indices(len(matrix))]
    q = np.quantile(vector, quant_q)
    vector= 1 * (vector > q)
    vector = scipy.sparse.csr_matrix(vector)
    print(vector)
    tri = 1*(tri>q)
    print("Tri")
    print(tri)
    heatmap(output + filename, tri[0:500, 0:500], 500)
    return vector

def to_binary_for_folder(folder, output, quant_q):
    if not os.path.exists(output):
        os.makedirs(output)
    pca_matrix = None
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            print(filename)
            matrix = scipy.sparse.load_npz(folder + filename)
            matrix = matrix.todense()
            vector = to_binary(output,filename,matrix, quant_q)

            if(pca_matrix==None):
                pca_matrix=vector
            else:
                pca_matrix = scipy.sparse.vstack([pca_matrix, vector])
    pca_path = os.path.join(output,"pca")
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    scipy.sparse.save_npz(pca_path+"/PCA.npz", pca_matrix)

def write_labels(output_file, list) :
    with open(output_file + ".txt", 'w') as filehandle:
        for listitem in list:
            filehandle.write('%s\n' % listitem)

def buildPCAmatrix(directory, output_file):
    matrix_PCA = None
    rownames = []
    for filename in os.listdir(directory):
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
    scipy.sparse.save_npz(output_file, matrix_PCA)
    write_labels(output_file, rownames)
    print(matrix_PCA.shape)
    return matrix_PCA


parser = ap.ArgumentParser()
parser.add_argument('--uniform', '-u', required = True)
parser.add_argument('--threshold', '-t', default=0.1, type = float) #uniform threshold
parser.add_argument('--input', '-i', required = True) # raw matrices
parser.add_argument('--output', '-o', required = True)
parser.add_argument('--rwprob', '-p', default=0.03, type = float) #restart probability
parser.add_argument('--binarization', '-b', action="store_true")
parser.add_argument('--quantile', '-q', default=0.85, type = float)
args = parser.parse_args()

uniform = args.uniform
input_raw = args.input # raw data
output_folder = args.output
threshold = args.threshold
restart_prob = args.rwprob
binarization = False
quantile = args.quantile
#uniform = output_folder + "/uniform/" #output mit / oder ohne
rw = output_folder
binary = output_folder + "/binary/"

uniform_for_folder(threshold, input_raw, uniform)
rw_for_folder(uniform,rw, restart_prob)
if(binarization):
    to_binary_for_folder(rw,binary, quantile)
