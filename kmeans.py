import numpy as np
import matplotlib.pyplot as plt
from util import get_clouds
from sklearn.utils import shuffle


def cost_kmean(centers, X, responsibilities):
    '''
    Calculates cost of kmeans, 
    centers = k x num_features
    points = n x num_features
    responsibilities k x n 
    '''
    args = np.argmax(responsibilities, axis=0)
    # args gives us the centerpoint that x[i] belongs to
    cost = 0
    for i in range(len(X)):
        arg = args[i]
        cost += responsibilities[arg, i] * \
            np.linalg.norm((centers[arg] - X[i, :]), ord=2)
    return cost


def cluster_responsibilities(centers, X, beta):
    '''
    Calculates cluster_responsibilities for soft_kmeans algo
    centers in k x num_features
    X in n x num_features
    returns responsibilities in k x n
    '''
    K = len(centers)
    N = len(X)
    responsibilities = np.zeros((K, N))

    for n in xrange(N):
        responsibilities_n = np.exp(-beta *
                                    np.linalg.norm((centers - X[n, :]), ord=2, axis=1))
        responsibilities_n = responsibilities_n/responsibilities_n.sum()
        responsibilities[:, n] = responsibilities_n
    return responsibilities


def calculate_centers(X, responsibilities):
    '''
    calculate new weighted centers
    takes responsibilities in k x n
    takes X in n x num_features
    '''
    K = len(responsibilities)
    N = len(X)
    num_of_features = X.shape[1]
    centers = np.zeros((K, num_of_features))
    for k in xrange(K):
        r_sum = responsibilities[k, :].sum()
        r_X_sum = responsibilities[k, :].dot(X)
        centers[k, :] = r_X_sum/r_sum
    return centers


class soft_kmeans:
    def __init__(self, k):
        self.k = k

    def fit(self, X, beta, eps):
        '''
        takes X as samples x num_features
        calculates responsibilities and saves it inside the model
        '''
        centers = X[np.random.choice(len(X), 4)]
        responsibilities = np.random.rand(self.k, len(X))
        cost_old = 0
        cost_new = cost_kmean(centers, X, responsibilities)

        while (cost_old - cost_new)**2 > eps:
            cost_old = cost_new
            # calculate cluster responsibilities
            responsibilities = cluster_responsibilities(centers, X, beta)
            centers = calculate_centers(X, responsibilities)
            cost_new = cost_kmean(centers, X, responsibilities)

        self.print_cluster(X, responsibilities)

    def print_cluster(self, X, responsibilities):
        point_groups = np.argmax(responsibilities, axis=0)
        group_legends = []
        for k in range(self.k):
            indices = (point_groups == k)
            plt.scatter(X[indices, 0], X[indices, 1], label="group%d" % (k+1))
        plt.legend()
        plt.show()


def main():
    X = get_clouds()
    X = shuffle(X)

    # show data
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    model = soft_kmeans(4)
    model.fit(X, beta=1, eps=1e-3)


main()
