# HIC_Imputation

The HIC Imputation tool was developed by Florian Molnar, Kim Le and Octavia-Andreea Ciora for the De novo Endophenotyping project under the supervision of Olga Lazareva. This project was created as part of the Systems Biomedicine lecture at the Chair of Experimental Bioinformatics, TUM.

This project is based on the following imputation methods:
* **Random Walk:** Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., Sejnowski, T. J., Dixon, J. R., & Ecker, J. R. (2019). Robust single-cell Hi-C clustering by convolution- and random-walk–based imputation. Proceedings of the National Academy of Sciences, 116(28), 14011–14018. https://doi.org/10.1073/pnas.1901423116
* **HiCNN2:** Liu, T., & Wang, Z. (2019). HiCNN2: Enhancing the Resolution of Hi-C Data Using an Ensemble of Convolutional Neural Networks. Genes, 10(11), 862. https://doi.org/10.3390/genes10110862
* **DeepHiC:** Hong, H., Jiang, S., Li, H., Quan, C., Zhao, C., Li, R., Bo, X. (2019). DeepHiC: A Generative Adversarial Network for Enhancing Hi-C Data Resolution. https://doi.org/10.1101/718148

### Pipeline:

HIC Imputation aims to identify cell-types and cluster of Hi-C data based on the chromatin structure of each sample. It takes Hi-C contact matrices (Random Walk, DeepHiC) or read files (HiCNN2) as input. There are three methods available for performing the imputation from low to high resolution data. The user can choose between Random Walk, HiCNN2 and DeepHiC. The binarization is an optional data postprocessing step, where the user can choose between binarization, personalized binarization and selective personalized binarization. The imputed matrices are then used for a Principal Component Analysis (PCA), followed by k-means clustering using the first 20 (or all) principal components. The pipeline outputs a PCA, a k-means, an elbow and a PCA scree plot, as well as the imputed matrices.

<img width="922" alt="Pipeline" src="https://user-images.githubusercontent.com/51077615/74770461-9b24ca00-528c-11ea-847e-1f0196db06d9.png">

### Dependencies:

HiC Imputation pipeline is written in python3. Following dependencies are necessary for running the tool:

* Python 3.6
* pytorch 1.1.0
* torchvision 0.3.0
* numpy 1.16.4
* scipy 1.3.0
* pandas 0.24.2
* scikit-learn 0.21.2
* matplotlib 3.1.0
* tqdm 4.32.2
* visdom 0.1.8.8

### Usage:

```
$ python Pipeline.py -i <input_path> -o <output_path> -m <DeepHiC|HiCNN2|RW> -k <n_clusters> 
[-t] [-p] [-c <chr_len>] [-b|-pb|-psb] [-q]
  ```
  
 
 ``` 
 Mandatory Command  Description
 -i                 Path to the input directory containing raw contact matrices (or read files for HiCNN2)
 -o                 Path to the output directory
 -m                 Requested imputation method (DeepHiC, HiCNN2 or RW)
 -k                 Number of clusters for k-means
 -c                 Length of the chromosome (only for HiCNN2)
Optional Command   Description
 -t                 Uniform threshold value (only for RW, default: 0.1) 
 -p                 Restart probability (only for RW, default: 0.03)
 -b                 Binarizes the data after imputation
 -pb                Applies personalized binarization to the data after imputation
 -psb               Applies personalized selective binarization to the data after imputation 
 -q                 Binarization quantile (only if -b is given, default: 0.85)
```

### Input:
The pipeline takes Hi-C data as input. The files should be named according to the corresponding cell-type (e.g. typeA_1, typeA_2, typeB_1, typeB_2). The input format depends on the chosen method for the imputation step. If Random Walk or DeepHiC are used, the pipeline requires contact matrices as .npz sparse matrices. For HiCNN2, the Hi-C read files are needed.

### Output:
The output folder is defined by the user. The pipeline outputs a PCA plot, the k-means clustering plot, a k-means elbow plot and a PCA scree plot aiming to make the interpretation of results easier. The imputed matrices can be found in the pipeline folder under /tmp/imputation_output and /tmp/binarization_output (in case binarization is performed).
