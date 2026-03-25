from lab2 import LPProblem
from lab3 import BasicSolution, simplex_main_phase
import numpy as np


def simplex_starting_phase(problem: LPProblem) -> BasicSolution | None:
    A = np.copy(problem.A)
    b = np.copy(problem.b)
    c = np.copy(problem.c)

    m, n = np.shape(A)

    for i, b_i in enumerate(b):
        if b_i < 0:
            b[i] = -b_i
            A[i] = np.vectorize(lambda x: -x)(A[i])

    c_tilde = np.append(np.zeros((1, n)), np.array([-1] * m))
    A_tilde = np.hstack((A, np.identity(m)))
    x_tilde = np.append(np.zeros((1, n)), b)
    aux_lp = LPProblem(c_tilde, A_tilde, b, ["="]*m, [(0, None)]*(m+n), True)
    basis = [i for i in range(n, n+m)]
    aux_basic_plan = BasicSolution(x_tilde, basis)

    print(aux_lp)

    # aux_optimal_plan = simplex_main_phase(aux_lp, aux_basic_plan)
    #
    # print(aux_optimal_plan)


def main():
    c = [1, 1, 0, 0, 0]
    A = [[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]]
    b = [-1, 3, -2]
    ops = ['=', '=', '=']
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

    lp = LPProblem(c, A, b, ops, bounds, maximize=True)
    simplex_starting_phase(lp)


if __name__ == "__main__":
    main()
