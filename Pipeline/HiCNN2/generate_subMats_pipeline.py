import subprocess
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

reads_dir = dir_path + "HiCNN2_package/read_files/"
chr_len = sys.argv[1]
resolution = 40000

subMats_dir = dir_path + "HiCNN2_package/matrices/subMats/"
if not os.path.exists(subMats_dir):
    os.makedirs(subMats_dir)

script_path = dir_path + "HiCNN2_package/get_HiCNN2_input.py"

for filename in os.listdir(reads_dir):
    if filename.endswith(".reads"):
        filepath = os.path.join(reads_dir, filename)
        print(filepath)
        filename = os.path.splitext(filename)[0] # the .subMats and .index files should not still have the .reads ending
        subprocess.call(["python", script_path, filepath, str(chr_len), str(resolution), 
                        os.path.join(subMats_dir, (filename + ".submats")), os.path.join(subMats_dir, (filename + ".index"))])



#example
#python get_HiCNN2_input.py read_files/GSM2109888_1_oocyte_NSN-40kb.reads 195471971 40000 
#subMats/GSM2109888_1_oocyte_NSN-40kb.subMats subMats/GSM2109888_1_oocyte_NSN-40kb.index

