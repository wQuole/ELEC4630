import numpy as np
import matplotlib.pyplot as plt
from utils import plot_faces
from utils import load_data
from utils import get_label

TRAINING = "dataset/training"
TESTING = "dataset/testing"


class PCA():
    def __init__(self):
        self.average_face = None
        self.covariance_matrix = None
        self.principal_components = None
        self.norm_factor = None
        self.img_height = 0
        self.img_width = 0

    def fit(self, X, number_of_components):
        N, H, W = X.shape
        self.img_height = H
        self.img_width = W

        X = X.reshape(N, -1)
        self.average_face = X.mean(axis=0)
        X -= self.average_face
        covariance_matrix = np.dot(X, X.T) / N
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        indices = np.argsort(eigenvalues)[::-1]
        print(f"\nEigenvectors ({len(eigenvectors)}):\n{eigenvectors}"
              f"\nEigenvalues ({len(eigenvalues)}):\n{eigenvalues}")
        eigenvectors, eigenvalues = eigenvectors[:, indices], eigenvalues[indices]

        eigenvectors = eigenvectors[:, :number_of_components]
        eigenvalues = eigenvalues[: number_of_components]

        compact = np.dot(X.T, eigenvectors)
        compact /= np.linalg.norm(compact, axis=0)

        self.norm_factor = np.sqrt(eigenvalues.reshape(1, -1))
        self.principal_components = compact

        return compact, eigenvectors, eigenvalues

    def transform(self, X):
        # Project data to eigenspace
        X = X.reshape(X.shape[0], -1)
        X -= self.average_face
        X = np.dot(X, self.principal_components) / self.norm_factor
        return X

    def inverse_transform(self, X):
        X = np.dot(X * self.norm_factor, self.principal_components.T)
        X += self.average_face
        X = X.reshape((X.shape[0], self.img_height, self.img_width))
        return X




if __name__ == '__main__':
    data = load_data(TRAINING)
    names = list(data.keys())
    X_train = np.array(list(data.values()), dtype=np.float64)
    N, H, W = X_train.shape
    #plot_faces(X_train, titles=names, height=H, width=W, n_row=1, n_col=6)

    pca = PCA()
    num_of_comp = 5
    pc, e_vector, e_value = pca.fit(X_train, num_of_comp)
    print(f"\ne_vector.shape --> {e_vector.shape}")
    rec = pca.inverse_transform(e_vector)
    #plot_faces(rec, names, H, W, 1, num_of_comp)

    eigenfaces = pc.T.reshape((num_of_comp, H, W))
    eigenfaces_titles = ["Eigenface_%d" % e for e in range(N)]
    #plot_faces(eigenfaces, eigenfaces_titles, H, W, 1, N)

    # TESTING
    test_data = load_data(TESTING)
    test_names = list(test_data.keys())
    X_test = np.array(list(test_data.values()), dtype=np.float64)
    N, H, W = X_test.shape
    plot_faces(X_test, titles=get_label(test_names), height=H, width=W, n_row=4, n_col=8)

    transformed = pca.transform(X_test)
    print(f"\nTransformed.shape --> {transformed.shape}")

    reconstructed = pca.inverse_transform(transformed)
    #plot_faces(reconstructed, titles=test_names, height=H, width=W, n_row=4, n_col=8)
    preds = predict(e_vector, transformed)

    actual = get_label(test_names)
    acc = get_accuracy(actual, preds)
    print(f"Accuracy: {acc}%")
    plot_faces(reconstructed, titles=preds, height=H, width=W, n_row=4, n_col=8)


