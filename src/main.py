import numpy as np
import matplotlib.pyplot as plt
from k_means import kmeans
from em import em


RANDOM_STATE = 25  #


def load_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content] 

    lst_content = sum(content, [])
    x_str = lst_content[0::3]
    y_str = lst_content[1::3]
    data_labels = np.array(lst_content[2::3])

    x = np.array([float(i) for i in x_str]) 
    y = np.array([float(i) for i in y_str])
    return x, y, data_labels


def plot_original_data(x, y, data_labels):
    labels = ['Head', 'Ear_left', 'Ear_right', 'Noise']

    fig = plt.figure()
    fig.suptitle('Original data')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x')
    ax.set(ylabel='y')
        
    for i in range(4):
        lbl = labels[i]
        x_ = x[np.where(data_labels == lbl)[0]]
        y_ = y[np.where(data_labels == lbl)[0]]
        
        ax.scatter(x_, y_)
        
    plt.show()


def plot_mickey_mouse(X, K, ind_samples_clusters, centroids):
    x, y = X[:, 0], X[:, 1]
    clusters = np.argmax(ind_samples_clusters, axis=1)  # will work both for K-means and EM
    
    fig = plt.figure()
    fig.suptitle('Mickey mouse')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x')
    ax.set(ylabel='y')
        
    for i in range(K):
        x_ = x[np.where(clusters == i)[0]]
        y_ = y[np.where(clusters == i)[0]]
        
        ax.scatter(x_, y_)
        
    for i in range(K):
        ax.scatter(centroids[i, 0], centroids[i, 1], c='k', s=100)

    plt.show()


def plot_cost(algorithm, cost, algorithm_name, title, legend, ylabel):
    x = np.linspace(1, len(cost) / 2 if algorithm == 'kmeans' else len(cost), len(cost))
    plt.plot(x, cost, label=f'{legend} curve')
    plt.title(f'{algorithm_name}: {title} over iterations')
    plt.xlabel('iterations')
    plt.ylabel(f'{ylabel}')
    plt.legend()
    plt.show()


def task_kmeans(X):
    """
    :param X: data for clustering, shape: (N, D), N=500, D = 2
    :return:
    """

    K = 3
    max_iter = 50
    ind_samples_clusters, centroids, cost = kmeans(X, K, max_iter)

    plot_cost('kmeans', cost, 'k-means', 'Cost function', 'loss', 'cost')
    plot_mickey_mouse(X, K, ind_samples_clusters, centroids)


def task_em(X):
    """
    :param X: data for clustering, shape: (N, D), N=500, D = 2
    :return:
    """

    K = 3
    max_iter = 100
    means, soft_clusters, log_likelihood = em(X, K, max_iter)

    plot_cost('em', log_likelihood, 'expectation-maximization', 'Log likelihood', 'log likelihood', 'log likelihood')
    plot_mickey_mouse(X, K, soft_clusters, means)


def main():
    np.random.seed(RANDOM_STATE)

    filename = 'mouse.txt'
    x, y, data_labels = load_data(filename)
    
    plot_original_data(x, y, data_labels)

    X_mouse = np.array([x, y]).T
    print('X_mouse shape: ', X_mouse.shape)

    # ----- Task K-Means
    print('--- Task K-Means ---')
    task_kmeans(X_mouse)

    # ----- Task EM
    print('--- Task EM ---')
    task_em(X_mouse)


if __name__ == '__main__':
    main()
