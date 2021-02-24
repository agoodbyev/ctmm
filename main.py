import numpy as np
import matplotlib.pyplot as plt
import time


def generate_matrix(n):
    A = np.random.uniform(-10, 10, (n, n))
    eps = 1e-2
    while abs(np.linalg.det(A)) < eps:
        A = (rand.sample((n, n)) - 0.5) * 20
    # print(np.linalg.det(A))
    return A


def solve(n):
    average = 0
    for i in range(3):
        start_time = time.time()
        f = np.random.uniform(-10, 10, (n, 1))
        A = generate_matrix(n)
        Ainv = np.linalg.inv(A)
        u = np.dot(Ainv, f)
        # print(u)
        # print(np.dot(A,u))
        # print(f)
        average += (time.time() - start_time) / 3
    return average


x = [i for i in range(5, 101, 5)]
y = [solve(a) for a in x]
#print(x)
#print(y)
fig = plt.figure()
plt.plot(x, y)
plt.grid(True)
plt.show()
