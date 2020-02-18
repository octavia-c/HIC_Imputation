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

#input_folder = '/nfs/proj/scHiC_imputation/contact_maps/'
#output_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/GM12878/mat/'

print("Data preparation with parameters: " + input_folder + " and " + output_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def processFolder(input_folder,  output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".npz"):
            sample_path = os.path.join(input_folder, filename)
            matrix = scipy.sparse.load_npz(sample_path)
            matrix = matrix.todense()
            matrix = np.array(matrix)
            non_zero_lines = []
            for i in range(len(matrix)):
                rowsum = np.sum(matrix[i,])
                if (rowsum > 0):
                    non_zero_lines.append(i)
            compact = np.array(non_zero_lines)

            sample = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder,sample)
            print(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.savez_compressed(os.path.join(output_path,"chr1_40kb.npz"), hic=matrix, compact=compact)
            np.savez_compressed(os.path.join(output_path,"chr1_10kb.npz"), hic=matrix, compact=compact)

processFolder(input_folder, output_folder)