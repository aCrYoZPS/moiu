from lab2 import LPProblem
from lab1 import solve
import numpy as np
import math


class BasisPlan:
    def __init__(self, x: list, basis: list):
        # basis is ZERO INDEXED
        self.x = np.array(x, dtype=float)
        self.basis = np.array(sorted(basis), dtype=int)

    def print(self):
        print(f"x: {self.x}")
        print(f"B: {self.basis}")


def simplex_main_phase(problem: LPProblem, basis_plan: BasisPlan) -> np.ndarray | None:
    A_b_inv = None
    k = None
    c, A, b, offset = problem.canonicalize()

    print(A)
    A_b = A[:, basis_plan.basis]
    while True:
        c_b = c[basis_plan.basis]
        if A_b_inv is None:
            print(A_b)
            A_b_inv = np.linalg.inv(A_b)
        else:
            A_b_inv = solve(A_b_inv, A[:, basis_plan.basis[k]], k + 1)
        print(f"c_b: {c_b}")
        print(f"A_b_inv: {A_b_inv}")
        u = c_b * A_b_inv
        delta = c - u * A
        is_optimal = True
        j_0 = None
        for item in delta:
            if item > 0:
                is_optimal = False
                if j_0 is None:
                    j_0 = item

        if is_optimal:
            return basis_plan.x

        z = A_b_inv * A[:, j_0]

        theta: list[float] = []
        for i, z_i in enumerate(z):
            if (z_i > 0):
                theta.append(basis_plan.x[basis_plan.basis[i]]/z_i)
            else:
                theta.append(math.inf)

        theta_0 = min(theta)

        if math.isinf(theta_0):
            return None

        k = theta.index(theta_0)
        basis_plan.basis[k] = j_0

        for i in range(len(basis_plan.basis)):
            if i == k:
                continue
            j_i = basis_plan.basis[i]
            basis_plan.x[j_i] = basis_plan.x[j_i] - theta_0 * z[i]

        print("\n--- New Basis Plan ---")
        basis_plan.print()


def main():
    c = [1, 1, 0, 0, 0]
    A = [[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]]
    b = [1, 3, 2]
    ops = ['=', '=', '=']
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

    lp = LPProblem(c, A, b, ops, bounds, maximize=True)
    # ZERO INDEXED
    optimal_plan = simplex_main_phase(lp, BasisPlan([0, 0, 1, 3, 2], [2, 3, 4]))
    print(optimal_plan)


if __name__ == "__main__":
    main()
