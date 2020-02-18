import subprocess
import os
import sys
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

shutil.rmtree(dir_path + "HiCNN2_package/read_files/")
shutil.rmtree(dir_path + "HiCNN2_package/matrices/")

input_dir = sys.argv[1]
chr_len = sys.argv[2]


#generate read files
print("generating read files")
subprocess.call(["python", dir_path + "transform_reads_pipeline.py", input_dir])

#generate subMats
print("generating submatrices")
subprocess.call(["python", dir_path + "generate_subMats_pipeline.py", chr_len])


#predict subMats
print("predicting submatrices")
subprocess.call(["python", dir_path + "predict_subMats_pipeline.py"])

#merge subMats
print("merging submatrices")
subprocess.call(["python", dir_path + "merge_subMats_pipeline.py", chr_len])

#convert npy matrices to npz
print("Converting matrices to npz format")
subprocess.call(["python", dir_path + "npy_to_npz_pipeline.py", chr_len])

