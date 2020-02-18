import os
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--output', '-o', required = True)
args = parser.parse_args()

input_folder = args.input
output_folder = args.output

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def detect_signal_percentages(input_folder, output_folder):
    print("Detect Signal Percentage")
    file_counter = 0
    percentages = []
    for file in os.listdir(input_folder):
        file_counter = file_counter + 1
        if file.endswith(".npz"):
            sample = os.path.splitext(file)[0]
            print("Sample " + str(file_counter) + ": " + sample + " detect signal percentage")
            matrix = scipy.sparse.load_npz(os.path.join(input_folder, file))
            matrix = matrix.todense()
            nonzeros = np.count_nonzero(matrix)
            total = np.shape(matrix)[0] * np.shape(matrix)[1]
            nonzeros_percentage = nonzeros / total
            percentages.append(nonzeros_percentage)
    plt.hist(percentages, color='blue', edgecolor='black')
    plt.title('Histogram of Nonzero Percentages Before Binarization')
    plt.xlabel('Nonzero Percentage')
    plt.ylabel('Sample')
    plt.savefig(output_folder + "/signal_percentages.png")
    return percentages

def heatmap(output_name, matrix, lim):
    plt.xlim(0, lim)
    plt.ylim(lim, 0)
    plt.imshow(matrix, cmap='pink', interpolation='nearest')
    plt.savefig(output_name + ".png")

def to_binary(output_folder, filename, matrix, lowest_percentage):
    tri =  np.triu(matrix, k=0)
    vector = tri[np.triu_indices(len(matrix))]
    q = np.quantile(vector, 1-lowest_percentage)
    vector= 1 * (vector > q)
    vector = scipy.sparse.csr_matrix(vector)
    #print(vector)
    #heatmap(output_folder + filename, tri[0:500, 0:500], 500)
    return vector

def to_binary_for_folder(input_folder, output_folder, lowest_percentage):
    print("Binarization")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pca_matrix = None
    file_counter = 0
    for file in os.listdir(input_folder):
        file_counter = file_counter + 1
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            sample = os.path.splitext(filename)[0]
            print(sample)
            print("Sample " + str(file_counter) + ": " + sample + " binarization")
            matrix = scipy.sparse.load_npz(os.path.join(input_folder, filename))
            matrix = matrix.todense()
            vector = to_binary(output_folder, filename, matrix, lowest_percentage)

            binary_path = os.path.join(output_folder, "binary_matrix_personalized")
            if not os.path.exists(binary_path):
                os.makedirs(binary_path)
            scipy.sparse.save_npz(binary_path + "/" + sample + "_binary.npz", vector)

            if(pca_matrix==None):
                pca_matrix=vector
            else:
                pca_matrix = scipy.sparse.vstack([pca_matrix, vector])

    pca_path = os.path.join(output_folder, "pca")
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    scipy.sparse.save_npz(pca_path+"/PCA.npz", pca_matrix)

signal_percentages = detect_signal_percentages(input_folder, output_folder)
lowest_percentage = min(signal_percentages)
print("Lowest percentage: " + str(lowest_percentage))
to_binary_for_folder(input_folder, output_folder, lowest_percentage)
