import math
import pprint

import numpy as np

from lab3 import BasicSolution

EPS = 1e-9


def dual_simplex_method(c: list, A: list[list], b: list, basis: list[int]) -> BasicSolution | None:
    c = np.copy(np.array(c))
    A = np.copy(np.array(A))
    b = np.copy(np.array(b))
    basis = list(basis)
    n = len(c)

    while True:
        basis.sort()
        A_b = A[:, basis]
        A_b_inv = np.linalg.inv(A_b)
        c_b = c[basis]
        y = c_b @ A_b_inv
        basic_vars_of_pseudo_plan = A_b_inv @ b
        pseudo_plan = np.zeros(n)
        for i in range(len(pseudo_plan)):
            if i in basis:
                pseudo_plan[i] = basic_vars_of_pseudo_plan[0]
                basic_vars_of_pseudo_plan = np.delete(basic_vars_of_pseudo_plan, 0)
            else:
                pseudo_plan[i] = 0

        is_optimal = True
        j_k = -1
        for idx, ae in enumerate(pseudo_plan):
            if ae < -EPS:
                j_k = idx
                is_optimal = False
                break

        if is_optimal:
            return BasicSolution(pseudo_plan, basis)

        k = basis.index(j_k)
        delta_y = A_b_inv[k]
        mu = np.zeros(n)
        sigma = np.full(n, math.inf)
        is_inconsistent = True

        for j in range(n):
            mu[j] = delta_y @ A[:, j]
            if mu[j] < -EPS:
                is_inconsistent = False
                sigma[j] = (c[j] - A[:, j] @ y) / mu[j]

        if is_inconsistent:
            print("mu >= 0 => the problem is inconsistent :(")
            return None

        sigma_0 = math.inf
        j_0 = 0
        for j, sigma_j in enumerate(sigma):
            if j in basis or mu[j] >= -EPS:
                continue
            if sigma_j < sigma_0:
                j_0 = j
                sigma_0 = sigma_j

        basis[k] = j_0


def task1():
    c = [1, 1, 0, 0, 0]
    A = [
        [2, 1, -1, 0, 0],
        [-2, -1, 0, -1, 0],
        [-1, -2, 0, 0, -1]
    ]
    b = [3, -6, -6]
    basis = [0, 2, 3]
    opt_plan = dual_simplex_method(c, A, b, basis)
    if opt_plan is None:
        return
    else:
        print("--- Optimal plan ---")
        pprint.pprint(opt_plan.x)


def task2():
    c = [-4, -3, -7, 0, 0]
    A = [
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ]
    b = [-1, -1.5]
    basis = [3, 4]
    opt_plan = dual_simplex_method(c, A, b, basis)
    if opt_plan is None:
        return
    else:
        print("--- Optimal plan ---")
        pprint.pprint(opt_plan.x)


def main():
    task1()
    task2()


if __name__ == "__main__":
    main()
