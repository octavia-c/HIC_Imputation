import os 
import sys

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

input_dir   = sys.argv[1]
output_dir  = dir_path + "HiCNN2_package/read_files/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

quality_check = open("/nfs/proj/scHiC_imputation/quality_check.txt", "r") 


for filename in os.listdir(input_dir):
    file = open(input_dir + filename)
    filename = filename.split(".csv")[0]
    outputfile = open(output_dir + filename + ".reads", "w")
    
    firstline = True
    for line in file:
        if(firstline):	# skip the header line
            firstline = False
            continue
        line = line.split(",")

        if line[0] == "1" and line[1] == "1":
            for i in range(int(line[6])):
                outputfile.write(line[2] + " " + line[4] + "\n")

    firstline = True
    file.close()
    outputfile.close()
    