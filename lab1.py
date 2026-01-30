import random

import numpy as np


def solve(A_inv: np.ndarray, x: np.ndarray, i: int) -> np.ndarray | None:
    n = A_inv.shape[0]
    l = np.matmul(A_inv, x)
    l_i = l[i - 1]
    if l_i == 0:
        return None

    l[i - 1] = -1
    l_caret = l * (-1 / l_i)
    q = np.identity(n)
    for j in range(n):
        q[j][i - 1] = l_caret[j]

    result = np.zeros(shape=(n, n))
    for j in range(n):
        for k in range(n):
            result[j][k] = q[j][i - 1] * A_inv[i - 1][k]
            if j != i - 1:
                result[j][k] += A_inv[j][k]

    return result


def main():
    for i in range(10):
        n = 5
        i = random.randint(1, n)
        A = 100 * np.random.random((n, n))
        A_inv = np.linalg.inv(A)
        x = 100 * np.random.random(n)
        A_changed = np.array(A)
        for j in range(len(x)):
            A_changed[j][i - 1] = x[j]
        result = solve(A_inv, x, i)
        if result is None:
            print(f"Matrix A_changed =\n{A_changed}\nis not inverseable")
        else:
            print(np.around(np.matmul(A_changed, result), 4))


if __name__ == '__main__':
    main()
