import subprocess
import argparse as ap
import os
import shutil


parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--dataset', '-d', required = True)
parser.add_argument('--sampling_rate', '-s', type = int, default=16)
parser.add_argument('--output', '-o', required = True)
parser.add_argument('--deephic' , required = True)
args = parser.parse_args()

input_folder = args.input
dataset = args.dataset
sampling_rate = args.sampling_rate
output_folder = args.output
deephic = args.deephic

# DON'T CHANGE
dataset_name = dataset
deephic_folder = deephic + '/DeepHiC'
preparation_folder = deephic_folder + '/data/' + dataset_name + '/mat/'
predict_folder = deephic_folder + '/data/' + dataset_name + '/predict/'
scripts_folder = deephic + '/scripts/'

if os.path.exists(deephic_folder + '/data/' + dataset_name):
    try:
        shutil.rmtree(deephic_folder + '/data/' + dataset_name)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
os.makedirs(deephic_folder + '/data/' + dataset_name)


print("Data preparation")
subprocess.call(["python", scripts_folder + "DeepHiC_preparation.py", "--input", input_folder, "--output", preparation_folder])

print("Generate Submatrices")
subprocess.call(["python", scripts_folder + "DeepHiC_generateSubmatrices.py", "--input", preparation_folder, "--deephic", deephic_folder, "--sampling_rate",  str(sampling_rate)])

print("Prediction")
subprocess.call(["python", scripts_folder + "DeepHiC_predict.py", "--input", preparation_folder, "--deephic", deephic_folder, "--sampling_rate", str(sampling_rate)])

print("Output processing")
subprocess.call(["python", scripts_folder + "DeepHiC_outputProcessing.py", "--input", predict_folder, "--output", output_folder])

