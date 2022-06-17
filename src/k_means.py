import numpy as np


def euclidean_distance(x, y):
    """
    :param x: D-dimensional vector
    :param y: D-dimensional vector
    :return: dist - scalar value
    """

    return np.linalg.norm(x - y)


def cost_function(X, K, ind_samples_clusters, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: cost - a scalar value
    """

    X_ = X.reshape(X.shape[0], X.shape[1], 1)  # X reshaped for broadcasting in norm
    centroids_ = centroids.T.reshape(1, centroids.shape[1], centroids.shape[0])  # centroids reshaped for broadcasting in norm
    norms = np.linalg.norm(X_ - centroids_, axis=1)
    J = np.sum(ind_samples_clusters * norms**2)

    return J


def closest_centroid(sample, centroids):
    """
    :param sample: a data point x_n (of dimension D)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: idx_closest_cluster, that is, the index of the closest cluster
    """
    
    distances = np.linalg.norm(sample.reshape(1, len(sample)) - centroids, axis=1)
    idx_closest_cluster = np.argmin(distances ** 2)

    return idx_closest_cluster


def assign_samples_to_clusters(X, K, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    """

    N = X.shape[0]

    ind_samples_clusters = np.zeros((N, K))

    for n in range(N):
        ind_samples_clusters[n, closest_centroid(X[n], centroids)] = 1

    assert np.min(ind_samples_clusters) == 0 and np.max(ind_samples_clusters == 1), "These must be one-hot vectors"

    return ind_samples_clusters


def recompute_centroids(X, K, ind_samples_clusters):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :return: centroids - means of clusters, shape: (K, D)
    """

    ind_samples_clusters_ = ind_samples_clusters.reshape(ind_samples_clusters.shape[0], ind_samples_clusters.shape[1], 1)
    X_ = X.reshape(X.shape[0], 1, X.shape[1])
    nominator = np.sum(ind_samples_clusters_ * X_, axis=0)
    denominator = np.sum(ind_samples_clusters_, axis=0)
    centroids = nominator / denominator

    return centroids


def kmeans(X, K, max_iter):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter:
    :return: ind_samples_clusters - indicator variables for all data points, shape: (N, K)
            centroids - means of clusters, shape: (K, D)
            cost - an array with values of cost over iteration
    """

    n, d = X.shape

    # Init centroids
    rnd_points = np.random.randint(low=0, high=n, size=K)
    centroids = X[rnd_points, :]
    eps = 1e-6

    print(f'Init centroids: {centroids}')

    cost = []
    for it in range(max_iter):    
        # Assign samples to the clusters
        ind_samples_clusters = assign_samples_to_clusters(X, K, centroids)
        J = cost_function(X, K, ind_samples_clusters, centroids)
        cost.append(J)
        
        # Calculate new centroids from the clusters
        centroids = recompute_centroids(X, K, ind_samples_clusters)
        J = cost_function(X, K, ind_samples_clusters, centroids)
        cost.append(J)
        
        if it > 0 and np.abs(cost[-1] - cost[-2]) < eps:
            print(f'Iteration {it+1}. Algorithm converged.')
            print(f'New centroids: {centroids}')
            break
    
    return ind_samples_clusters, centroids, cost

