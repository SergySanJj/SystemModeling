import numpy as np


def main():
    # Setting initial parameters
    y_func = np.loadtxt('y1.txt', dtype=np.float64)

    delta = np.float64(0.2)
    T = 50.
    time = np.arange(0, T + delta, delta)
    beta = np.array([0.1, 10., 21.], dtype=np.float64)  # [c3, m1, m3]
    eps = 1e-9
    _, n = y_func.shape

    current = 1.0
    previous = 0
    iteration_num = 0

    while np.abs(previous - current) > eps:

        rigidity_param = np.array(
            [0.14, 0.3, beta[0], 0.12], dtype=np.float64)  # [c1, c2, c3, c4]
        weight = np.array([beta[1], 28., beta[2]],
                          dtype=np.float64)  # [m1, m2, m3]

        # calculating A
        A = get_A(beta)
        iteration_num += 1

        # derivative dA/d beta^T
        dA = get_dA(A, beta)

        # Runge-Kutta method
        left_int_part = 0.
        right_int_part = 0.
        new_identification_beta = 0.

        U = np.zeros((6, 3))
        y_vec = np.copy(y_func[:, 0].reshape(-1, 1))

        for i in range(1, n):
            delta_U = get_new_delta(dA, y_vec)

            U += Runge_Kutta_step(lambda x: A@x + delta_U, U, delta)

            # Calculate new y
            y_vec += Runge_Kutta_step(lambda x: A@x, y_vec, delta)

            left_int_part += U.T @ U
            right_int_part += U.T @ (y_func[:, i].reshape(-1, 1) - y_vec)

            new_identification_beta += (y_func[:, i].reshape(-1, 1) -
                                        y_vec).T @ (y_func[:, i] - y_vec.reshape(-1))

        dBeta = np.linalg.inv(left_int_part * delta) @ (right_int_part * delta)
        beta += dBeta.reshape(-1)

        previous = current
        current = new_identification_beta * delta

        print(' {0} : {1:.15f}'.format(iteration_num, current[0]))

    print('Beta', beta)


def Runge_Kutta_step(f, x, delta):
    k1 = delta * f(x)
    k2 = delta * f(x + k1/2.)
    k3 = delta * f(x + k2/2.)
    k4 = delta * f(x + k3)
    return (k1 + 2. * k2 + 2. * k3 + k4)/6.


def get_new_delta(dA, y_vec):
    return np.column_stack(((dA[1] @ y_vec).reshape(-1),
                            (dA[2] @ y_vec).reshape(-1), (dA[3] @ y_vec).reshape(-1)))


def get_A(beta):
    # [c1, c2, c3, c4]
    rigidity_param = np.array(
        [0.14, 0.3, beta[0], 0.12], dtype=np.float64)

    # [m1, m2, m3]
    weight = np.array([beta[1], 28., beta[2]],
                      dtype=np.float64)

    # calculating A
    A = np.zeros((6, 6), dtype=np.float64)
    A[0, 1] = 1.
    A[1, 0] = -(rigidity_param[0] + rigidity_param[1]) / weight[0]
    A[1, 2] = rigidity_param[1] / weight[0]
    A[2, 3] = 1.
    A[3, 0] = rigidity_param[1] / weight[1]
    A[3, 2] = -(rigidity_param[1] + rigidity_param[2]) / weight[1]
    A[3, 4] = rigidity_param[2] / weight[1]
    A[4, 5] = 1.
    A[5, 2] = rigidity_param[2] / weight[2]
    A[5, 4] = -(rigidity_param[3] + rigidity_param[2]) / weight[2]

    return A


def get_dA(A, beta):
    # [c1, c2, c3, c4]
    rigidity_param = np.array(
        [0.14, 0.3, beta[0], 0.12], dtype=np.float64)

    # [m1, m2, m3]
    weight = np.array([beta[1], 28., beta[2]],
                      dtype=np.float64)

    dA = [np.zeros_like(A),
          np.zeros_like(A),
          np.zeros_like(A),
          np.zeros_like(A)]

    dA[1][3, 2] = -1. / weight[1]
    dA[1][3, 4] = 1. / weight[1]
    dA[1][5, 2] = 1. / weight[2]
    dA[1][5, 4] = -1. / weight[2]

    dA[2][1, 0] = (rigidity_param[1] + rigidity_param[0]) / \
        (weight[0] * weight[0])
    dA[2][1, 2] = -(rigidity_param[1]) / (weight[0] * weight[0])

    dA[3][5, 2] = -(rigidity_param[2]) / (weight[2] * weight[2])
    dA[3][5, 4] = (rigidity_param[3] + rigidity_param[2]) / \
        (weight[2] * weight[2])

    return dA


main()
