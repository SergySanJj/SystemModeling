import numpy as np
from PIL import Image
from PIL.BmpImagePlugin import BmpImageFile


def Moore_Penrose_Analytic(X):
    X_pseudoinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    return X_pseudoinv


def MP_SVD(X):
    u, s, vt = np.linalg.svd(X, full_matrices=False)

    def rev(i):
        if (i < 1e-9):
            return 0.0
        return 1/i
    s = np.vectorize(rev)(s)

    res = np.matmul(vt.T, np.multiply(s[..., np.newaxis], u.T))
    return res


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
    mp = runMethod(MP_SVD, "moore_penrose_svd")
    mp = runMethod(Moore_Penrose_Analytic, "moore_penrose_analytic")

    diff = find_diff_points(gr, mp)
    array_to_image(diff).save('./results/diff.bmp')


main()
