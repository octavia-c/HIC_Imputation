# HIC_Imputation


The tool HIC Imputation was developed by Florian Molnar, Octavia-Andreea Ciora and Kim Le for the De novo Endophenotyping project under the supervision of Olga Lazareva. This project was created as part of the Systems Biomedicine lecture at the Chair of Experimental Bioinformatics.

HIC Imputation uses Hi-C contact matrices as an input and performs imputation from low to high resolution data. On the imputed matrices is a Principal Component Analysis (PCA) performed which is the basis for the k-means clustering. 


Usage:

  $ python Pipeline.py -i <input_path> -o <output_path>
  -m (DeepHiC|HiCNN2|RW) -k <n_clusters> [-t] [-p] 
  [-c <chr_len>] [-b] [-q] [-pb] [-psb]
  
 
 ``` 
 Command  Description
 -i       Path to the raw contact matrices (or to the read files for HiCNN2)
 -o       Path to the output directory
 -m       Requested imputation method (DeepHiC, HiCNN2 or RW)
 -k       Number of clusters for k-means
 -t       Uniform threshold value (default: 0.1)
 -p       Restart probability (default: 0.03)
 -c       Length of the chromosome (needed for HiCNN2)
 -b       Binarizes the data after imputation
 -q       Binarization quantile (default: 0.85)
 -pb      Applies personalized binarization to the data after imputation
 -psb     Applies personalized selective binarization to the data after imputation 
```
