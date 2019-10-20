import numpy as np
from PIL import Image
from PIL.BmpImagePlugin import BmpImageFile


def Moore_Penrose(X):
    def pseudo_inv(X, delta):
        n_rows, n_columns = X.shape
        if n_rows > n_columns:
            X_inv = np.linalg.inv(
                X.T @ X - (delta ** 2 * np.eye(n_columns))) @ X.T
        else:
            X_inv = X.T @ np.linalg.inv(
                X @ X.T - (delta ** 2 * np.eye(n_rows)))
        return X_inv

    delta = 100
    eps = 1e-12
    diff = 1

    X_inv_1 = np.zeros(1)
    while diff > eps:
        print(delta)
        X_inv_1 = pseudo_inv(X, delta)
        delta /= 2.
        X_inv_2 = pseudo_inv(X, delta)
        diff = np.linalg.norm(X_inv_1 - X_inv_2)

    return X_inv_1


def Moore_Penrose_Analitic(X):
    X_pseudoinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    return X_pseudoinv


def Moore_Penrose_SVD(X):
    n, m = X.shape
    U, i, V = np.linalg.svd(X, full_matrices=True)
    for k in range(0, i.size):
        if i[k] != 0:
            i[k] = 1.0/i[k]
    S = np.diag(i)

    r = i.size
    zerro = np.zeros((n-r, m))
    S = np.vstack((S, zerro))
    X_pseudoinv = np.matmul(V, S.T)
    X_pseudoinv = np.matmul(X_pseudoinv, U.T)

    return X_pseudoinv


def Greville(X):
    vec = X[0].reshape(-1, 1)
    eps = np.float32(1e-9)

    if np.abs(np.dot(vec.T, vec)) < eps:
        X_pseudo_inv = vec
    else:
        X_pseudo_inv = vec / np.dot(vec.T, vec)

    eps = 1e-9
    n_rows, n_columns = X.shape

    for i in range(1, n_rows):
        vec = X[i].reshape(-1, 1)
        Z = np.eye(n_columns) - X_pseudo_inv @ X[:i]
        norm = np.dot(np.dot(vec.T, Z), vec)
        if np.abs(norm) < eps:
            Z = X_pseudo_inv @ X_pseudo_inv.T
            norm = np.dot(np.dot(vec.T, Z), vec) + 1.
        X_pseudo_inv -= np.dot(np.dot(np.dot(Z, vec),
                                      vec.T), X_pseudo_inv) / norm
        X_pseudo_inv = np.column_stack((X_pseudo_inv, (np.dot(Z, vec) / norm)))

    return X_pseudo_inv


def image_to_array(BmpImageFile):
    (im_width, im_height) = BmpImageFile.size
    return np.array(BmpImageFile.getdata()).reshape(
        (im_height, im_width)).astype(np.float64)


def array_to_image(ImageArray):
    return Image.fromarray(ImageArray.astype(np.uint8), mode='L')


def prepare_XY():
    x_image = Image.open('./images/x2.bmp')
    y_image = Image.open('./images/y2.bmp')
    return (image_to_array(x_image), image_to_array(y_image))


def runMethod(method_func, output_name):
    X, Y = prepare_XY()
    # M = method_func(X)
    # Z = np.eye(X.shape[0]) - X @ M
    # V = np.random.rand(Y.shape[0], M.shape[1])
    # A = Y @ M + V @ Z.T
    A = np.matmul(Y, method_func(X))

    Y_res = A @ X
    mse = (np.square(Y - Y_res)).mean()

    print("MSE ", output_name, ': ', mse)
    image = array_to_image(Y_res)
    image.save('./results/'+output_name+'.bmp')

    return Y_res


def find_diff_points(a, b):
    eps = 1e-9
    diff = np.abs(a-b)
    for i in range(0, diff.shape[0]):
        for j in range(0, diff.shape[1]):
            if diff[i][j] > eps:
                diff[i][j] = 255.0
    return diff


def main():
    gr = runMethod(Greville, "greville")
    mp = runMethod(Moore_Penrose_SVD, "moore_penrose")

    diff = find_diff_points(gr, mp)
    array_to_image(diff).save('./results/diff.bmp')


main()
