from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse
import argparse as ap
from sklearn.decomposition import PCA

parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--output', '-o', required = True)
args = parser.parse_args()

input_folder = args.input
output_folder = args.output


def colorList(rownames):
    colors = []
    for name in rownames:
        if name.startswith("NSN"):
            colors.append("red")
        elif name.startswith("SN"):
            colors.append("blue")
        elif name.startswith("zigM"):
            colors.append("yellow")
        elif name.startswith("zigP"):
            colors.append("green")

    return colors

def performPCA(matrix, rownames):
    scaler = StandardScaler()
    scaler.fit(matrix)

    scaled_data = scaler.transform(matrix)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(output_folder + "/Scree_plot.png")

    # the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, index=rownames, columns=labels)

    colors = colorList(rownames)
    labels = [i.split("_")[0] for i in rownames]

    plt.scatter(pca_df.PC1, pca_df.PC2, color=colors)
    plt.title('DeepHiC')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))

#    for sample in pca_df.index:
#        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

    plt.savefig(output_folder + "/PCA.png")




matrix_PCA = scipy.sparse.load_npz(input_folder + "/PCA_matrix.npz")

rownames = [] # read in the label file
with open(input_folder + "/PCA_labels.txt", 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        rownames.append(currentPlace)



matrix_PCA = matrix_PCA.todense()

performPCA(matrix_PCA, rownames)