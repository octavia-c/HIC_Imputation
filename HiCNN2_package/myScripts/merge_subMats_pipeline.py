import subprocess
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

chr_len = sys.argv[1]

pred_subMats_dir = dir_path + "HiCNN2_package/matrices/predicted_subMats/"
index_dir = dir_path + "HiCNN2_package/matrices/subMats/" # contains the raw subMats and the respective index files
script_path = dir_path + "HiCNN2_package/combine_subMats.py"

pred_Mat_dir = dir_path + "HiCNN2_package/matrices/predicted_fullMats/"

for filename in os.listdir(pred_subMats_dir):
    if filename.endswith(".submats_predicted.npy"):
        filepath = os.path.join(pred_subMats_dir, filename) # path to predicted submatrix file
        print(filename)
        filename = os.path.splitext(filename)[0]            # filename without .npy
        filename = filename.split(".submats_predicted")[0]
        index_filepath = index_dir + filename + ".index.npy"

        predicted_mat_file = pred_Mat_dir + filename + ".predictedMat"

        subprocess.call(["python", script_path, filepath, index_filepath, str(chr_len), "40000", predicted_mat_file])





#example
#"python combine_subMats.py data/chr15.subMats_HiCNN23_16.npy data/chr15.index.npy 102531392 10000 data/chr15.predictedMat"

#"data/chr15.subMats_HiCNN23_16.npy" is from step (5).
#"data/chr15.index.npy" is from step(4).
#"102531392" is the chromosome length.
#"10000" is the resolution.
#"data/chr15.predictedMat" is the predicted high-resolution matrix for one chromosome (chr15).