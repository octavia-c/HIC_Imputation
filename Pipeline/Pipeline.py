import subprocess
import os
import argparse as ap
import shutil

parser = ap.ArgumentParser()
parser.add_argument('--input_folder', '-i', required = True)
parser.add_argument('--method' , '-m',  choices=['RW', 'HiCNN2', 'DeepHiC'], required = True)
parser.add_argument('--chr_length', '-c', default = 0, type = int)
parser.add_argument('--num_clusters', '-k', required = True)
parser.add_argument('--output_folder', '-o', required = True)
parser.add_argument('--quantile', '-q', default=0.85, type = float)
parser.add_argument('--threshold', '-t', default=0.1, type = float)
parser.add_argument('--restart_probability', '-p', default = 0.03, type = float)

group = parser.add_mutually_exclusive_group()
group.add_argument('--binarization', '-b', action="store_true")
group.add_argument('--personalized_binarization', '-pb', action="store_true")
group.add_argument('--personalized_selective_binarization', '-psb', action="store_true")

args = parser.parse_args()
input_folder = args.input_folder
method = args.method
output_folder = args.output_folder
binarization = args.binarization
personalized_binarization = args.personalized_binarization
personalized_selective_binarization = args.personalized_selective_binarization
chrLength = args.chr_length
k = args.num_clusters
quantile = args.quantile
threshold = args.threshold
restart_probability = args.restart_probability


if(method == 'HiCNN2' and chrLength == 0):
    raise NameError('Chromosome Length -c is required for method HiCNN2')


if os.path.exists(output_folder):
    try:
        shutil.rmtree(output_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
os.makedirs(output_folder)

dataset = 'dataset'


dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
os.chdir(dir_path)

tmp_output =  dir_path + '/tmp/'
if os.path.exists(tmp_output):
    try:
        shutil.rmtree(tmp_output)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
os.makedirs(tmp_output)

imputation_output = tmp_output + '/imputation_output/'
os.makedirs(imputation_output)
binarization_output = tmp_output + '/binarization_output/'
os.makedirs(binarization_output)
pca_input = imputation_output

print("------------------\nPIPELINE STARTED\n------------------")
print("Imputation method: " + method)
print("Binarization: " + str(binarization))
print("Personalized Binarization: " + str(personalized_binarization))
print("Personalized Selective Binarization: " + str(personalized_selective_binarization))
print("------------------\n")

if(method == "RW"):
    print("------------------\nRW\n------------------\n")
    uniform_output = dir_path + "/RW/uniform_output/"
    if os.path.exists(uniform_output):
        try:
            shutil.rmtree(uniform_output)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(uniform_output)
    subprocess.call(["python", dir_path + "/RW/RW.py", "--input", input_folder, "--output", imputation_output, "--uniform", uniform_output, "-p", str(restart_probability), "-t", str(threshold)])
elif (method == "HiCNN2"):
    print("------------------\nHiCNN2\n------------------\n")
    subprocess.call(["python", dir_path + "/HiCNN2/HiCNN2.py", input_folder, str(chrLength)])
elif (method == "DeepHiC"):
    print("------------------\nDeepHiC\n------------------\n")
    subprocess.call(
        ["python", dir_path + "/DeepHiC/scripts/DeepHiC_RUN.py", "--input", input_folder, "--output", imputation_output, "--dataset", dataset, "--deephic", dir_path + "/DeepHiC"])
if(binarization):
    print("------------------\nBinarization\n------------------\n")
    subprocess.call(["python", dir_path + "/binarization/Binarize.py", "--input", imputation_output, "--output", binarization_output, "-q", str(quantile)])
    pca_input = binarization_output + "/binary_matrix/"
if(personalized_binarization):
    print("------------------\nPersonalized Binarization\n------------------\n")
    subprocess.call(["python", dir_path + "/binarization/Binarize_personalized.py", "--input", imputation_output, "--output", binarization_output])
    pca_input = binarization_output + "/binary_matrix_personalized/"
if (personalized_selective_binarization):
    print("------------------\nPersonalized Selective Binarization\n------------------\n")
    subprocess.call(["python", dir_path + "/binarization/Binarize_personalized_selected.py", "--input", imputation_output, "--output", binarization_output])
    pca_input = binarization_output + "/binary_matrix_personalized/selected"
os.chdir(dir_path)
print("------------------\nPCA\n------------------\n")
print("Creating big matrix for PCA")
subprocess.call(["python", dir_path + "/PCA_matrix.py", "--input", pca_input, "--output", tmp_output])
print("Perfoming PCA")
subprocess.call(["python", dir_path + "/PCA_server.py", "--input", tmp_output, "--output", output_folder, "-k", str(k)])
print("------------------\nPIPELINE DONE")
print("Check results in " + output_folder + "\n------------------")






