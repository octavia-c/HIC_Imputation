import os
import numpy as np
import scipy.sparse
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--output', '-o', required = True)
args = parser.parse_args()

input_folder = args.input
output_folder = args.output

#input_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/GM12878/predict/'
#output_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/output/GM12878/'

print("Data preparation with parameters: " + input_folder + " and " + output_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

i = 0
for subfolder in os.scandir(input_folder):
    i = i + 1
    subpath = subfolder.path
    sample = subfolder.name
    print("Processing sample " + str(i) + ": " + sample)
    sparse_matrix = np.load(os.path.join(subpath,"predict_chr1_40kb.npz"))
    sparse_matrix = sparse_matrix['deephic']
    sparse_matrix = scipy.sparse.csr_matrix(sparse_matrix)
    print(sparse_matrix)
    scipy.sparse.save_npz(os.path.join(output_folder, sample + ".npz"), sparse_matrix)
    print("Done sample " + str(i) + ": " + sample)