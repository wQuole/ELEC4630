import utils
import numpy as np

TRAINING = "dataset/training"
TESTING = "dataset/testing"


class PCA:
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
        # Project data back to "imagespace"
        X = np.dot(X * self.norm_factor, self.principal_components.T)
        X += self.average_face
        X = X.reshape((X.shape[0], self.img_height, self.img_width))
        return X


if __name__ == '__main__':
    data = utils.load_data(TRAINING)
    names = list(data.keys())
    X_train = np.array(list(data.values()), dtype=np.float64)
    N, H, W = X_train.shape
    #utils.plot_faces(X_train, titles=names, height=H, width=W, n_row=1, n_col=6)

    pca = PCA()
    num_of_comp = 6
    pc, e_vector, e_value = pca.fit(X_train, num_of_comp)
    rec = pca.inverse_transform(e_vector)
    #utils.plot_faces(rec, utils.get_label(names), H, W, 1, num_of_comp, save=False)

    eigenfaces = pc.T.reshape((num_of_comp, H, W))
    eigenfaces_titles = ["Eigenface %i" % e for e in range(1,N+1)]
    #utils.plot_faces(eigenfaces, eigenfaces_titles, H, W, 1, num_of_comp, save=False)

    # TESTING
    test_data = utils.load_data(TESTING)
    test_names = list(test_data.keys())
    X_test = np.array(list(test_data.values()), dtype=np.float64)
    N, H, W = X_test.shape
    #utils.plot_faces(X_test, titles=utils.get_label(test_names), height=H, width=W, n_row=3, n_col=11, save=False)

    transformed = pca.transform(X_test)
    reconstructed = pca.inverse_transform(transformed)
    #utils.plot_faces(reconstructed, titles=test_names, height=H, width=W, n_row=4, n_col=8)

    preds = utils.predict(e_vector, transformed)
    actual = utils.get_label(test_names)
    acc = utils.get_accuracy(actual, preds)
    print(f"Accuracy: {acc}%")
    utils.plot_pred(reconstructed, titles=preds, labels=actual, height=H, width=W, n_row=3, n_col=11, save=1)


