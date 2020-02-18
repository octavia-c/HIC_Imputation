from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse
import argparse as ap


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



parser = ap.ArgumentParser()
parser.add_argument('--input', '-i', required = True)
parser.add_argument('--output', '-o', required = True)
parser.add_argument('--numClusters', '-k', required = True)
args = parser.parse_args()

input_folder = args.input
output_folder = args.output
numClusters = int(args.numClusters)

def performPCA(matrix, rownames, output_dir, numClusters):

    print("Normalizing")
    scaler = StandardScaler()
    scaler.fit(matrix)

    scaled_data = scaler.transform(matrix)

    print("Started fitting the data")
    pca = PCA(n_components= min(20, len(rownames)))
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    print("Start Plotting")

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    print("Making screeplot")
    make_screeplot(pca, output_dir)

    print("Making k-means elbow plot")
    elbowplot_kmeans(pca_data, output_dir, len(rownames)) # len(rownames) = num samples

    print("Making k-means Plot")
    perform_kmeans(pca_data, output_dir, numClusters, len(rownames))

    # the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, columns=labels)

    rownames = [name.split("_")[0] for name in rownames]
    df_rownames = pd.DataFrame({'target':rownames})

    finalDf = pd.concat([pca_df, df_rownames], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1 - {0}%'.format(per_var[0]), fontsize = 15)
    ax.set_ylabel('PC2 - {0}%'.format(per_var[1]), fontsize = 15)
    ax.set_title('HiCNN2', fontsize = 20)

    targets = set(rownames)
    cmap = get_cmap(len(rownames)+1)

    i = 0
    for target in targets:
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                   , finalDf.loc[indicesToKeep, 'PC2']
                   , c = cmap(i)
                   , s = 30)
        i = i + 1
    ax.legend(targets)
    ax.grid()


    plt.savefig(output_dir + "PCA_HiCNN2_updated.png")



def make_screeplot(pca , output_dir):
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(output_dir + "Scree_plot_updated.png")
    plt.clf()



def elbowplot_kmeans(pca_data, output_dir, n_samples):
    ks = range(1, min(n_samples, 10))
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
    
        # Fit model to samples
        model.fit(pca_data)
    
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
    

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')

    plt.savefig(output_dir + "elbowplot_kmeans.png")
    plt.clf()



def perform_kmeans(pca_data, output_dir, numClusters, n_samples):
    #Set a 4 KMeans clustering
    kmeans = KMeans(n_clusters = numClusters)

    #Compute cluster centers and predict cluster indices
    X_clustered = kmeans.fit_predict(pca_data)

    #Define our own color map
    LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b', 3: 'y'}
    label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

    # Plot the scatter digram
    plt.figure(figsize = (7,7))
    plt.scatter(pca_data[:,0],pca_data[:,1], c= label_color, alpha=0.5)
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.title("K-means on the first " + str(min(20, n_samples)) + " PCs")

    
    plt.savefig(output_dir + "PCA_HiCNN2_kmeans_2clusters.png")
    plt.clf()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



matrix_path = input_folder + "/PCA_matrix.npz"
label_path = input_folder + "/PCA_labels.txt"

print("Reading in data")

matrix_PCA = scipy.sparse.load_npz(matrix_path)

rownames = [] # read in the label file
with open(label_path, 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        rownames.append(currentPlace)



matrix_PCA = matrix_PCA.todense()

print("Starting PCA")

performPCA(matrix_PCA, rownames, output_folder, numClusters)
