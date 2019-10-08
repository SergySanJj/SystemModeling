import matplotlib.pyplot as plt
import numpy as np


def MSE(data, appr, N):
    return np.sum((data-appr)**2)/N


def save_plot(time, data, filename):
    plt.rc('figure', figsize=(20.0, 10.0))
    plt.clf()
    plt.grid(True)
    plt.plot(t, data)
    plt.savefig('images/' + filename + '.png')


def furr(data, N):
    frequency = np.zeros((N))
    for i in range(N):
        sin_freq = 0
        cos_freq = 0
        for j in range(N):
            sin_freq += data[j] * np.sin(2. * np.pi * i * j / float(N))
            cos_freq += data[j] * np.cos(2. * np.pi * i * j / float(N))
        sin_freq /= float(N)
        cos_freq /= float(N)
        frequency[i] = np.sqrt(sin_freq ** 2 + cos_freq ** 2)

    return frequency


def get_valuables(data, N):
    valuables = []
    for i in range(3, N // 2):
        if np.max(data[i - 3:i + 3]) == data[i]:
            valuables.append(i)
            print("Amplitude: ", data[i])

    return valuables


with open('f11.txt') as file:
    data = np.array([float(val) for val in file.read().split()])

T = 5.
dt = 0.01
t = np.arange(0, T + dt, dt)
N = t.shape[0]
save_plot(t, data, 'data')

frequency = furr(data, N)
save_plot(t, frequency, 'freq')

valuables = get_valuables(frequency, N)
main_frequency = valuables[0] / T
print('Main frequency: ', main_frequency)

c = np.array([np.sum(data * t ** 3), np.sum(data * t ** 2), np.sum(data * t),
              np.sum(data * np.sin(2. * np.pi * main_frequency * t)), np.sum(data)])

A = np.zeros((c.shape[0], c.shape[0]))

functions = np.array([t ** 3, t ** 2, t,
                      np.sin(2. * np.pi * main_frequency * t), np.ones(N)])

A = np.matmul(functions, functions.T)

a = np.matmul(np.linalg.inv(A), c.T)
print("Coeffs: ", a)

approximated_func = np.dot(a, functions)
save_plot(t, approximated_func, 'appr')
print("Mean squared error: ", MSE(data, approximated_func, N))
