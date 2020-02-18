import os
import subprocess
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--deephic', '-d', required = True)
parser.add_argument('--sampling_rate', '-s', required = True, type = int)
args = parser.parse_args()

input_folder = args.input
deephic_folder = args.deephic
sampling_rate = args.sampling_rate

#input_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/GM12878/mat/'
#deephic_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC'
#sampling_rate=50

print("Prediction with parameters: " + input_folder + ", " + deephic_folder + " and " + str(sampling_rate))

sampling_rates = [16, 25, 50, 100]
if(sampling_rate not in sampling_rates):
    raise NameError("Unknown sampling rate: " + str(sampling_rate) + ". Only 16, 25, 50, 100 allowed.")

os.chdir(deephic_folder)

i = 0
for subfolder in os.scandir(input_folder):
    i = i + 1
    subpath = subfolder.path
    sample = subfolder.name
    print("Starting sample " + str(i) + ": " + sample)
    subprocess.call(["python", "data_predict.py", "-lr", "40kb", "-ckpt", "save/deephic_raw_" + str(sampling_rate) +  ".pth", "-c", sample])
    print("Done sample " + str(i) + ": " + sample)
