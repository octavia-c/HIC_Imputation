import subprocess

# can be changed
dataset = 'GM12878'
sampling_rate = 16
dataset_name = dataset + "_" + str(sampling_rate)
input_folder = '/nfs/proj/scHiC_imputation/contact_maps/'
output_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/output/' + dataset_name + '/'
scripts_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/scripts/'

# DON'T CHANGE
deephic_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC'
preparation_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/' + dataset_name + '/mat/'
predict_folder = '/nfs/home/students/ciora/HiCluster/DeepHiC/DeepHiC/data/' + dataset_name + '/predict/'


print("Data preparation")
subprocess.call(["python", scripts_folder + "DeepHiC_preparation.py", "--input", input_folder, "--output", preparation_folder])

print("Generate Submatrices")
subprocess.call(["python", scripts_folder + "DeepHiC_generateSubmatrices.py", "--input", preparation_folder, "--deephic", deephic_folder, "--sampling_rate",  str(sampling_rate)])

print("Prediction")
subprocess.call(["python", scripts_folder + "DeepHiC_predict.py", "--input", preparation_folder, "--deephic", deephic_folder, "--sampling_rate", str(sampling_rate)])

print("Output processing")
subprocess.call(["python", scripts_folder + "DeepHiC_outputProcessing.py", "--input", predict_folder, "--output", output_folder])

