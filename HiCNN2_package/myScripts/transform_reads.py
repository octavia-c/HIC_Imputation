quality_check = open("/nfs/proj/scHiC_imputation/quality_check.txt", "r") 
  

counter = 0
for path in quality_check: 
    counter = counter+1
    print(counter)
    print(path)
    path = path.rstrip()
    file = open("/nfs/proj/scHiC_imputation/GSE80006/" + path)
    filename = path.split(".csv")[0]
    outputfile = open("/nfs/home/students/fmolnar/HiCNN2_package/read_files/" + filename + ".reads", "w")
    
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
    