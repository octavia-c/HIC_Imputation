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
#sampling_rate = 50

print("Generate Submatrices with parameters: " + input_folder + ", " + deephic_folder + " and " + str(sampling_rate))


sampling_rate_TO_lrc = {16:100, 25:80, 50:50, 100:25}
if(sampling_rate not in sampling_rate_TO_lrc.keys()):
    raise NameError("Unknown sampling rate: " + str(sampling_rate) + ". Only 16, 25, 50, 100 allowed.")
lrc = sampling_rate_TO_lrc.get(sampling_rate)

os.chdir(deephic_folder)

i = 0
for subfolder in os.scandir(input_folder):
    i = i + 1
    subpath = subfolder.path
    sample = subfolder.name
    print("Starting sample " + str(i) + ": " + sample)
    subprocess.call(["python", "data_generate.py", "-hr", "10kb", "-lr", "40kb", "-s", "one", "-chunk", "40", "-stride", "40", "-bound", "201", "-scale", "1", "-c", sample, '-lrc', str(lrc)])
    print("Done sample " + str(i) + ": " + sample)
