import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
sns.set_theme(style='darkgrid')

plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'lines.markersize': 4.33,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1,
        'legend.facecolor': 'white',
        'legend.fontsize': 12,
        'legend.loc': 'lower center',
        'legend.borderaxespad': 0.,
        'legend.handletextpad': 0.05,
        'legend.columnspacing': 0.5,
        'figure.figsize': (5, 5),
    }
)

np.random.seed(0)

def generate_data(N, radii, spread_param):
    data = np.zeros((N*len(radii), 2))
    for i in range(len(radii)):
        theta = np.random.uniform(0, 2*np.pi, N)
        r = np.random.normal(radii[i], spread_param[i], N)
        data[i*N:(i+1)*N, 0] = r*np.cos(theta)
        data[i*N:(i+1)*N, 1] = r*np.sin(theta)

    return data

def visualize_data(data, save=False):
    plt.scatter(data[:, 0], data[:, 1], label='Input data')
    plt.axis('equal')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, 1.035))
    plt.tight_layout()
    if save: plt.savefig('input_data.svg')
    plt.show()

def kmeans_clustering(data, save=False):
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(data)
    labels = kmeans.labels_

    # Visualize the clustering result
    palette = sns.color_palette("Set2", 3)
    for i in range(3):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=palette[i], label=f'Cluster {i}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', linewidths=0.5, s=40, label='centroids')
    plt.axis('equal')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.035))
    plt.tight_layout()
    if save: plt.savefig('kmeans_clustering.svg')
    plt.show()

def spectral_laplacian(data, save=False):
    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0).fit(data)
    connectivity_matrix = kneighbors_graph(data, n_neighbors=10, include_self=True)
    # connectivity_matrix = rbf_kernel(data, gamma=.1)
    affinity_matrix = 0.5 * (connectivity_matrix + connectivity_matrix.T)

    degrees = np.sum(affinity_matrix, axis=1)
    degree_matrix = np.diagflat(degrees)
    # compute unnormalized laplacian
    laplacian = degree_matrix - affinity_matrix

    # check if laplacian is symmetric
    assert np.allclose(laplacian, laplacian.T), "Laplacian matrix is not symmetric"

    d_inv_sqrt = np.diagflat(1 / np.sqrt(degrees))
    laplacian_sym = np.eye(affinity_matrix.shape[0]) - d_inv_sqrt @ affinity_matrix @ d_inv_sqrt

    plt.spy(laplacian, markersize=0.1, label=r"$L = D - \frac{ 1 }{ 2 } (S + S^T)$")
    plt.grid(False)
    plt.legend(bbox_to_anchor=(0.5, 1.035))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save: plt.savefig('laplacian_matrix.svg')
    plt.show()

    # spectral embedding
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    idx = np.argsort(eigvals)
    eigvecs = np.array(eigvecs[:, idx])
    
    # visualize the eigenvectors
    plt.plot(eigvecs[:, 0], eigvecs[:, 1], ".", label="Spectral embeddings")
    plt.axis('equal')
    plt.xlabel(r'$\mathbf{v}_1$')
    plt.ylabel(r'$\mathbf{v}_2$')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, 1.035))
    plt.tight_layout()
    if save: plt.savefig('spectral_embedding.svg')
    plt.show()

    spectral_kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(eigvecs[:, :2])
    labels = spectral_kmeans.labels_

    palette = sns.color_palette("Set2", 3)
    for i in range(3):
        plt.scatter(eigvecs[labels == i, 0], eigvecs[labels == i, 1], color=palette[i], label=f'Cluster {i}')
    plt.axis('equal')
    plt.xlabel(r'$\mathbf{v}_1$')
    plt.ylabel(r'$\mathbf{v}_2$')
    plt.grid(True)
    plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.035))
    plt.tight_layout()
    if save: plt.savefig('spectral_clustering.svg')
    plt.show()

    # Visualize the clustering result
    palette = sns.color_palette("Set2", 3)
    for i in range(3):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=palette[i], label=f'Cluster {i}')
    plt.scatter(spectral_kmeans.cluster_centers_[:, 0], spectral_kmeans.cluster_centers_[:, 1], c='red', marker='X', linewidths=0.5, s=40, label='centroids')
    plt.axis('equal')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid(True)
    plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.035))
    plt.tight_layout()
    if save: plt.savefig('spectral_clustering_data.svg')
    plt.show()


if __name__ == '__main__':

    N = 300 # number of points per circle
    radii = [1, 2.5, 6] # radii of circles
    spread_param = [0.2, 0.1, 0.5] # spread parameter for each circle

    to_save = True

    data = generate_data(N, radii, spread_param)
    visualize_data(data, save=to_save)
    kmeans_clustering(data, save=to_save)
    spectral_laplacian(data, save=to_save)
    

    
