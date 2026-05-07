import math

import numpy as np

from lab1 import solve
from lab2 import LPProblem
from lab3 import BasicSolution


def make_extended_matrix(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    result = np.vstack((A, Q))
    result = np.hstack((result, np.vstack((np.zeros(shape=(m, n)), np.identity(n) * -1))))
    result = np.hstack((result, np.vstack((np.zeros(shape=(m, m)), np.transpose(A)))))
    result = np.hstack((result, np.vstack((np.zeros(shape=(m, m)), np.transpose(A) * -1))))
    return result


def make_auxiliary_problem(A_tilde: np.ndarray, b_tilde: np.ndarray, m: int, n: int) -> LPProblem:
    n_vars = 2 * n + 2 * m
    n_art = n + m

    c_aux = np.concatenate([
        np.zeros(n_vars),
        np.full(n_art, -1.0)
    ])

    A_aux = np.hstack([
        A_tilde,
        np.eye(n_art)
    ])

    return LPProblem(
        c=c_aux,
        A=A_aux,
        b=b_tilde,
        ops=["="] * (n + m),
        bounds=[(0, None)] * (n_vars + n_art),
        maximize=True
    )


def is_forbidden(j: int, basis: list[int], n: int) -> bool:
    if j < n:
        return (n + j) in basis
    elif j < 2 * n:
        return (j - n) in basis
    return False


def modified_simplex_main_phase(
        problem: LPProblem,
        basic_solution: BasicSolution,
        n: int
) -> BasicSolution | None:
    A_b_inv = None
    k = None
    c, A, b, offset = problem.canonicalize()

    A_b = A[:, basic_solution.basis]
    while True:
        c_b = c[basic_solution.basis]
        if A_b_inv is None:
            A_b_inv = np.linalg.inv(A_b)
        else:
            A_b_inv = solve(A_b_inv, A[:, basic_solution.basis[k]], k + 1)

        u = c_b @ A_b_inv
        delta = c - u @ A

        is_optimal = True
        j_0 = None
        for i, item in enumerate(delta):
            if item > 0 and i not in basic_solution.basis and not is_forbidden(i, basic_solution.basis, n):
                is_optimal = False
                j_0 = i
                break

        if is_optimal:
            return basic_solution

        z = A_b_inv @ A[:, int(j_0)]

        theta: list[float] = []
        for i, z_i in enumerate(z):
            if z_i > 0:
                theta.append(basic_solution.x[basic_solution.basis[i]] / z_i)
            else:
                theta.append(math.inf)

        theta_0 = min(theta)

        if math.isinf(theta_0):
            return None

        k = theta.index(theta_0)
        j_asterisk = basic_solution.basis[k]
        basic_solution.basis[k] = j_0
        basic_solution.x[j_0] = theta_0
        basic_solution.x[j_asterisk] = 0
        for i in range(len(basic_solution.basis)):
            j_i = basic_solution.basis[i]
            if i != k:
                basic_solution.x[j_i] = basic_solution.x[j_i] - theta_0 * z[i]


def solve_quadratic_problem(Q: np.ndarray, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    m, n = A.shape
    n_vars = 2 * n + 2 * m
    n_art = n + m

    A_tilde = make_extended_matrix(Q, A)
    b_tilde = np.hstack((b, -c))

    x_tilde = np.zeros(n_vars + n_art)
    x_tilde[n_vars:] = b_tilde

    aux_basis = [2 * m + 2 * n + i for i in range(m + n)]
    aux_lp = make_auxiliary_problem(A_tilde, b_tilde, m, n)
    aux_basic_plan = BasicSolution(list(x_tilde), aux_basis)
    aux_optimal_plan = modified_simplex_main_phase(aux_lp, aux_basic_plan, n)
    if aux_optimal_plan is None:
        print("Целевой функционал не ограничен (такого быть не может вообще-то)")
        return None

    if np.any(aux_optimal_plan.x[n_vars:] > 0):
        print("Задача несовместна")
        return None

    return aux_optimal_plan.x[:n]


def main():
    Q = np.array([
        [2, -2],
        [-2, 4]
    ])

    A = np.array([
        [1, 1],
        [-1, 2]
    ])

    b = np.array([2, 2])
    c = np.array([-2, -6])
    print(solve_quadratic_problem(Q, A, b, c))


if __name__ == '__main__':
    main()
