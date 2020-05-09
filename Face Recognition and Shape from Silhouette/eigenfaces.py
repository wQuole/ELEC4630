import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

TRAINING = "dataset/training"
TESTING = "dataset/testing"


def plot_faces(images, titles, height, width, n_row, n_col):
    plt.figure(figsize=(3 * n_col, 3 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap='gray')
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()


def load_data(path):
    images = {}
    for root, dirs, files in sorted(os.walk(path)):
        for file in sorted(files):
            filepath = root + os.sep + file
            filename = file
            if filename.endswith(".bmp"):
                im = io.imread(filepath, as_gray=True)
                images[filename] = im
    return images


def fit(X, m, N):
    print(f"X.shape --> {X.shape}")
    X = X.reshape(N, -1)
    print(f"X.shape --> {X.shape}")
    avg = np.mean(X, axis=0)
    #avg = avg.reshape(X.shape[0], 1)
    normalized = X - avg
    covariance_matrix = np.cov(np.transpose(normalized))
    print(f"normalized --> {normalized.shape}")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvectors = sorted(eigenvectors, reverse=True)
    m_eigenvectors = eigenvectors[0:m, :]
    eigenfaces = np.dot(m_eigenvectors, normalized.T)
    weights = np.dot(normalized.T, eigenfaces.T)
    return avg, weights, eigenfaces


def predict(sample, avg, trained_weights, eigenfaces, N):
    sample = sample.reshape(N, -1)
    normalized_sample = sample - avg
    prediction_weight = np.dot(normalized_sample.T, eigenfaces.T)
    index = np.argmin(np.linalg.norm(prediction_weight - trained_weights, axis=1))
    return index


def reconstruct_image_from_pca(cov_matrix, components, psi, height, width, i_image):
    weights = np.dot(cov_matrix, components.T)
    print(f"Weights --> {weights.shape}") # Weights --> (6, 6)
    c_vector = np.dot(weights[i_image, :], components)
    recovered = (psi + c_vector).reshape(height, width)
    return recovered


if __name__ == '__main__':
    # Load dataset
    X_train = load_data(TRAINING)
    names = list(X_train.keys())
    images = np.array(list(X_train.values()), dtype=np.float64)
    N, H, W = images.shape
    plot_faces(images, titles=names, height=H,  width=W, n_row=1, n_col=6)

    # Principal Component Analysis
    n_pc = 6
    a, w, e = fit(images, n_pc, N)

    # TESTING
    X_test = load_data(TESTING)
    labels = list(X_train.keys())
    data = np.array(list(X_test.values()), dtype=np.float64)
    _N, _H, _W = data.shape

    test_sample = data[0]
    predicted_index = predict(test_sample, a, w, e, _N)
    print(predicted_index)