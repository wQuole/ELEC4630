import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

TRAINING = "dataset/training"
TESTING = "dataset/testing"


"""It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(3 * n_col, 3 * n_row))
    #plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap='gray')
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


def pca(X, number_of_components):
    psi = np.mean(X, axis=0)    # average face
    covariance_matrix = X - psi
    U, S, V = np.linalg.svd(covariance_matrix)  # Singular Value Decomposition
    components = V[:number_of_components]
    projections = U[:, :number_of_components] * S[:number_of_components]

    return projections, components, psi, covariance_matrix


def reconstruct_image_from_pca(cov_matrix, components, psi, height, width, i_image):
    weights = np.dot(psi, cov_matrix.T)
    print(f"Weights --> {weights.shape}")
    eigenvector = np.dot(weights[i_image, :], components)
    recovered = (cov_matrix + eigenvector).reshape(height, width)
    return recovered


if __name__ == '__main__':
    # Load dataset
    X_train = load_data(TESTING)
    names = list(X_train.keys())
    images = np.asarray(list(X_train.values()))
    N, H, W = images.shape
    plot_portraits(images, titles=names, h=H, w=W, n_row=4, n_col=8)

    # Principal Component Analysis
    n_components = H*W
    X = images.reshape(N, H*W)
    # P = Projections, C = Components, M = psi, Y = covariance_matrix

    P, C, M, Y = pca(X, n_components)


    eigenfaces = C.reshape((n_components, H, W))
    eigenfaces_titles = ["Eigenface_%d" % e for e in range(N)]
    plot_portraits(eigenfaces, eigenfaces_titles, H, W, 4, 8)

    plt.imshow(M.reshape(H, W), cmap='gray')
    plt.show()
    #
    # recovered_images = [reconstruct_image_from_pca(Y, C, M, W, H, i) for i in range(N)]
    # plot_portraits(recovered_images, names, H, W, 4, 4)



