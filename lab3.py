import math

import numpy as np

from lab2 import LPProblem
from lab1 import solve


class BasicSolution:
    def __init__(self, x: list, basis: list):
        # basis is ZERO INDEXED
        self.x = np.array(x, dtype=float)
        self.basis = np.array(sorted(basis), dtype=int)

    def __str__(self):
        return f"x: {self.x}\nB: {self.basis}"


def simplex_main_phase(problem: LPProblem, basic_solution: BasicSolution) -> BasicSolution | None:
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
            if item > 0:
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

        print("\n---- New Basic Solution ----")
        print(basic_solution)


def main():
    # c = [1, 1, 0, 0, 0]
    # A = [[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]]
    # b = [1, 3, 2]
    # ops = ['=', '=', '=']
    # bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
    c = [1, 0, 0, 0]
    A = [[1, -1, 1, 0], [-1, 1, 0, 1]]
    b = [1, 2]
    ops = ['=', '=', '=']
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

    lp = LPProblem(c, A, b, ops, bounds, maximize=True)
    # ZERO INDEXED
    optimal_plan = simplex_main_phase(lp, BasicSolution([1, 0, 0, 3], [0, 3]))
    if optimal_plan is None:
        print("The objective function is unbounded above on the feasible set")
    else:
        print(f"Optimal plan: {optimal_plan}")


if __name__ == "__main__":
    main()
