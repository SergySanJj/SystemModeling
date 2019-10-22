import numpy as np
import numpy.linalg as linalg
from PIL import Image
from PIL.BmpImagePlugin import BmpImageFile
import os

img_path = os.getcwd() + "\\images\\"


def Moore_Penrose_step(X, delta):
    n, m = X.shape
    if n > m:
        X_inv = linalg.inv(X.T @ X - (delta ** 2 * np.eye(m))) @ X.T
    else:
        X_inv = X.T @ linalg.inv(X @ X.T - (delta ** 2 * np.eye(n)))
    return X_inv


def Moore_Penrose_Limit(X):
    delta = 255.0
    eps = 1e-24
    diff = 1.

    X_curr = Moore_Penrose_step(X, delta)
    X_prev = Moore_Penrose_step(X, delta)
    while diff > eps:
        X_curr = X_prev
        delta = delta/1.61803398875
        X_prev = Moore_Penrose_step(X, delta)
        diff = linalg.norm(X_curr - X_prev)

    return X_curr


def MP_SVD(X):
    u, s, vt = linalg.svd(X, full_matrices=False)

    def rev(i):
        if (i < 1e-9):
            return 0.0
        return 1/i
    s = np.vectorize(rev)(s)

    res = np.matmul(vt.T, np.multiply(s[..., np.newaxis], u.T))
    return res


def Greville(X):
    eps = 1e-12
    cur_col = X[0].reshape(-1, 1)

    if cur_col.T @ cur_col < eps:
        X_pseudo_inv = cur_col
    else:
        X_pseudo_inv = cur_col / cur_col.T @ cur_col

    n_rows, n_columns = X.shape

    for i in range(1, n_rows):
        cur_col = X[i].reshape(-1, 1)
        Z = np.eye(n_columns) - X_pseudo_inv @ X[:i]
        norm = cur_col.T @ Z @ cur_col
        if np.abs(norm) < eps:
            Z = X_pseudo_inv @ X_pseudo_inv.T
            norm = cur_col.T @ Z @ cur_col + 1.

        X_pseudo_inv -= Z @ cur_col @ cur_col.T@ X_pseudo_inv / norm
        X_pseudo_inv = np.hstack((X_pseudo_inv, Z @ cur_col / norm))

    return X_pseudo_inv


def image_to_array(BmpImageFile):
    (im_width, im_height) = BmpImageFile.size
    return (np.array(BmpImageFile.getdata()).reshape(
        (im_height, im_width)).astype(np.float64))


def array_to_image(ImageArray):
    return Image.fromarray(ImageArray.astype(np.uint8), mode='L')


def prepare_XY():
    # x2 y2
    x_image = Image.open(img_path+'x2.bmp')
    n, m = x_image.size
    y_image = Image.open(img_path+'y2.bmp')
    return (np.vstack((image_to_array(x_image), np.ones((1, n)))), image_to_array(y_image))


def runMethod(method_func, output_name):
    X, Y = prepare_XY()
    A = np.matmul(Y, method_func(X))

    Y_res = A @ X
    mse = (np.square(Y - Y_res)).mean()

    print("MSE ", output_name, ': ', mse)
    image = array_to_image(Y_res)
    image.save(img_path+output_name+'.bmp')

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
    mp = runMethod(MP_SVD, "moore_penrose_svd")
    mp = runMethod(Moore_Penrose_Limit, "moore_penrose_limit")

    diff = find_diff_points(gr, mp)
    array_to_image(diff).save(img_path+'diff.bmp')


def Z(X, method_func):
    n, m = X.shape
    return np.eye(n) - X @ method_func(X)


def run_method(method, method_name):
    X, Y = prepare_XY()
    n, m = X.shape

    V = np.random.rand(Y.shape[0], method(X).shape[1])
    nV, mV = V.shape

    ZZ = Z(X, method).T
    YY = Y @ method(X)
    A = YY + V @ ZZ.T

    array_to_image(A).save(img_path+method_name+'_A_matrix.bmp')

    Y_res = A @ X
    image = array_to_image(Y_res)
    image.save(img_path+method_name+'.bmp')

    print("MSE ", method_name, ': ', (np.square(Y - Y_res)).mean())
    return Y_res


mp = run_method(Moore_Penrose_Limit, "moore_penrose")
mg = run_method(Greville, "greville")

# main()
