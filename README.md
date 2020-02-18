# HIC_Imputation


The HIC Imputation tool was developed by Florian Molnar, Kim Le and Octavia-Andreea Ciora for the De novo Endophenotyping project under the supervision of Olga Lazareva. This project was created as part of the Systems Biomedicine lecture at the Chair of Experimental Bioinformatics, TUM.

### Pipeline:

HIC Imputation aims to identify and cluster the cell-types of Hi-C data based on the chromatin structure. It takes Hi-C contact matrices (Random Walk, DeepHiC) or read files (HiCNN2) as input. There are three methods available for performing the imputation from low to high resolution data. The user can choose between Random Walk, HiCNN2 and DeepHiC. The binarization is an optional data postprocessing step, where the user can choose between binarization, personalized binarization and selective personalized binarization. The imputed matrices are then used for a Principal Component Analysis (PCA), followed by k-means clustering using the first 20 (or all) principal components. The pipeline outputs a PCA plot, the k-means clustering, a k-means elbow plot and a PCA scree plot.

<img width="922" alt="Pipeline" src="https://user-images.githubusercontent.com/51077615/74770461-9b24ca00-528c-11ea-847e-1f0196db06d9.png">

### Dependencies:

HiC Imputation pipeline is written in python3. Following depenedcies are necessary for running the tool:

Python 3.6
pytorch 1.1.0
torchvision 0.3.0
numpy 1.16.4
scipy 1.3.0
pandas 0.24.2
scikit-learn 0.21.2
matplotlib 3.1.0
tqdm 4.32.2
visdom 0.1.8.8

### Usage:

```
$ python Pipeline.py -i <input_path> -o <output_path> -m <DeepHiC|HiCNN2|RW> -k <n_clusters> 
[-t] [-p] [-c <chr_len>] [-b|-pb|-psb] [-q]
  ```
  
 
 ``` 
 Command  Description
 -i       Path to the raw contact matrices (or to the read files for HiCNN2)
 -o       Path to the output directory
 -m       Requested imputation method (DeepHiC, HiCNN2 or RW)
 -k       Number of clusters for k-means
 -t       Uniform threshold value (only for RW, default: 0.1) 
 -p       Restart probability (only for RW, default: 0.03)
 -c       Length of the chromosome (needed for HiCNN2)
 -b       Binarizes the data after imputation
 -pb      Applies personalized binarization to the data after imputation
 -psb     Applies personalized selective binarization to the data after imputation 
 -q       Binarization quantile (only if -b is given, default: 0.85)
```
