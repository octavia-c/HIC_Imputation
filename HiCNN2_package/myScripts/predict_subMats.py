import subprocess
import os


subMats_dir = "/nfs/home/students/fmolnar/HiCNN2_package/matrices/subMats/"
pred_subMats_dir = "/nfs/home/students/fmolnar/HiCNN2_package/matrices/predicted_subMats/"
script_path = "/nfs/home/students/fmolnar/HiCNN2_package/HiCNN2_predict.py"
model_path = "/nfs/home/students/fmolnar/HiCNN2_package/checkpoint/model_HiCNN21_16.pt"

for filename in os.listdir(subMats_dir):
    if filename.endswith(".submats.npy"):
        filepath = os.path.join(subMats_dir, filename)
        print(filepath)
        filename = os.path.splitext(filename)[0] # the .subMats and .index files should not still have the .reads ending
        subprocess.call(["python", script_path, "-f1", filepath, "-f2", os.path.join(pred_subMats_dir, (filename + "_predicted")), 
                        "-mid", "1", "-m", model_path, "-r", "16"])





#example
# python HiCNN2_predict.py -f1 subMats/GSM2109888_1_oocyte_NSN-40kb.subMats.npy 
# -f2 subMats/GSM2109888_1_oocyte_NSN-40kb_high_res -mid 1 -m checkpoint/model_HiCNN21_16.pt -r 16

#"python HiCNN2_predict.py -f1 data/chr15.subMats.npy -f2 data/chr15.subMats_HiCNN21_16 -mid 1 -m checkpoint/model_HiCNN21_16.pt -r 16"
#
#	"-f1" is followed by the input file generated in step (4). 
#	"-f2" is followed by the output file. 
#	"-mid 3" means that we are using HiCNN2-3. 
#	"-m" indicates the best model we want to use. We provide 6 checkpoint files in the "checkpoint" folder. The checkpoint files are named with the format "model_HiCNN2*_#.pt", where "*" may be 1/2/3 representing the three architectures and "#" may be 8/16/25 representing the three down sampling ratios (1/8, 1/16, and 1/25).
