import numpy as np

# Setting initial parameters
y_func = np.loadtxt('y1.txt', dtype=np.float64)

dt = np.float64(0.2)
T = 50.
time = np.arange(0, T + dt, dt)
beta = np.array([0.1, 10., 21.], dtype=np.float64)  # [c3, m1, m3]
eps = 1e-9
abs_eps = 1e-12
_, n = y_func.shape


identification_beta = 1
prev_ident_beta = 0
iteration_num = 0

while (identification_beta > eps) and (np.abs(prev_ident_beta - identification_beta) > abs_eps):

    rigidity_param = np.array(
        [0.14, 0.3, beta[0], 0.12], dtype=np.float64)  # [c1, c2, c3, c4]
    weight = np.array([beta[1], 28., beta[2]],
                      dtype=np.float64)  # [m1, m2, m3]

    # calculating A
    A = np.zeros((6, 6), dtype=np.float64)
    iteration_num += 1
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

    # derivative dA/d beta^T
    dA1 = np.zeros_like(A)
    dA2 = np.zeros_like(A)
    dA3 = np.zeros_like(A)

    dA1[3, 2] = -1. / weight[1]
    dA1[3, 4] = 1. / weight[1]
    dA1[5, 2] = 1. / weight[2]
    dA1[5, 4] = -1. / weight[2]

    dA2[1, 0] = (rigidity_param[1] + rigidity_param[0]) / \
        (weight[0] * weight[0])
    dA2[1, 2] = -(rigidity_param[1]) / (weight[0] * weight[0])

    dA3[5, 2] = -(rigidity_param[2]) / (weight[2] * weight[2])
    dA3[5, 4] = (rigidity_param[3] + rigidity_param[2]) / \
        (weight[2] * weight[2])

    # Runge-Kutta method
    left_int_part = 0.
    right_int_part = 0.
    temp_iden_beta = 0.

    U = np.zeros((6, 3))
    y_vec = np.copy(y_func[:, 0].reshape(-1, 1))

    for i in range(1, n):
        # Update U
        delta_U = np.column_stack(((dA1 @ y_vec).reshape(-1),
                                   (dA2 @ y_vec).reshape(-1), (dA3 @ y_vec).reshape(-1)))

        K1 = dt * ((A @ U) + delta_U)
        K2 = dt * ((A @ (U + .5 * K1)) + delta_U)
        K3 = dt * ((A @ (U + .5 * K2)) + delta_U)
        K4 = dt * ((A @ (U + K3)) + delta_U)
        U += (1. / 6. * (K1 + 2. * K2 + 2. * K3 + K4))

        # Calculate new y
        k1 = dt * (A @ y_vec)
        k2 = dt * (A @ (y_vec + .5 * k1))
        k3 = dt * (A @ (y_vec + .5 * k2))
        k4 = dt * (A @ (y_vec + k3))
        y_vec += 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        left_int_part += U.T @ U
        right_int_part += U.T @ (y_func[:, i].reshape(-1, 1) - y_vec)

        temp_iden_beta += (y_func[:, i].reshape(-1, 1) -
                           y_vec).T @ (y_func[:, i] - y_vec.reshape(-1))

    dBeta = np.linalg.inv(left_int_part ) @ (right_int_part )
    beta += dBeta.reshape(-1)

    prev_ident_beta = identification_beta
    identification_beta = temp_iden_beta * dt

    print('Iteration {0}. Quality score of beta identification {1:.15f}'.format(iteration_num,
                                                                                identification_beta[0]))


print('Beta', beta)
