import os
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt

input_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/output/GM12878/'
output_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/output/DeepHiC_50_binary_personalized_selected/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def compute_signal_percentage(filepath):
    matrix = scipy.sparse.load_npz(filepath)
    matrix = matrix.todense()
    nonzeros = np.count_nonzero(matrix)
    total = np.shape(matrix)[0] * np.shape(matrix)[1]
    nonzeros_percentage = nonzeros / total
    return nonzeros_percentage

def detect_signal_percentages(input_folder, output_folder):
    print("Detect Signal Percentage")
    file_counter = 0
    percentages = []
    for file in os.listdir(input_folder):
        file_counter = file_counter + 1
        if file.endswith(".npz"):
            sample = os.path.splitext(file)[0]
            print("Sample " + str(file_counter) + ": " + sample + " detect signal percentage")
            nonzeros_percentage = compute_signal_percentage(os.path.join(input_folder, file))
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

def to_binary(output_folder, filename, matrix, nonzero_percentage):
    tri =  np.triu(matrix, k=0)
    vector = tri[np.triu_indices(len(matrix))]
    q = np.quantile(vector, 1-nonzero_percentage)
    vector= 1 * (vector > q)
    vector = scipy.sparse.csr_matrix(vector)
    #print(vector)
    heatmap(output_folder + filename, tri[0:500, 0:500], 500)
    return vector

def to_binary_for_files_over_median(input_folder, output_folder, median_percentage):
    print("Binarization")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pca_matrix = None
    file_counter = 0
    for file in os.listdir(input_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            sample = os.path.splitext(filename)[0]
            print(sample)
            percentage = compute_signal_percentage(os.path.join(input_folder, filename))
            if(percentage > median_percentage):
                file_counter = file_counter + 1
                print("Sample " + str(file_counter) + ": " + sample + " binarization")
                matrix = scipy.sparse.load_npz(os.path.join(input_folder, filename))
                matrix = matrix.todense()
                vector = to_binary(output_folder, filename, matrix, median_percentage)
                binary_path = os.path.join(output_folder, "binary_matrix_personalized_selected")
                if not os.path.exists(binary_path):
                    os.makedirs(binary_path)
                scipy.sparse.save_npz(binary_path + "/" + sample + "_binary.npz", vector)
    print("Done binarization for " + str(file_counter) + " files")

signal_percentages = detect_signal_percentages(input_folder, output_folder)
median_percentage = np.median(signal_percentages)
print("Median percentage: " + str(median_percentage))
to_binary_for_files_over_median(input_folder, output_folder, median_percentage)
