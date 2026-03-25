import numpy as np

from lab2 import LPProblem
from lab3 import BasicSolution, simplex_main_phase


def simplex_starting_phase(problem: LPProblem) -> tuple[LPProblem, BasicSolution] | None:
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
    aux_lp = LPProblem(c_tilde, A_tilde, b, ["="] * m, [(0, None)] * (m + n), True)
    basis = [i for i in range(n, n + m)]
    aux_basic_plan = BasicSolution(x_tilde, basis)

    print(aux_lp)

    aux_optimal_plan = simplex_main_phase(aux_lp, aux_basic_plan)
    for i in range(n, n + m):
        if aux_optimal_plan.x[i] != 0:
            print(f"x_tilde_{i} != 0 => the problem is inconsistent :(")
            return None

    x = x_tilde[:n]
    has_artificial_var_in_basis = False
    k = 0
    j_k = 0
    for i, j_i in enumerate(aux_optimal_plan.basis):
        if j_i > j_k:
            j_k = j_i
            k = i

        if j_i >= n:
            has_artificial_var_in_basis = True
            break

    while has_artificial_var_in_basis:
        A_tilde_b_inv = np.linalg.inv(A_tilde[:, aux_optimal_plan.basis])
        found_non_zero_l_j_k = False
        for j in range(n):
            if j in aux_optimal_plan.basis:
                continue
            A_j = A_tilde[:, [j]]
            l_j = A_tilde_b_inv @ A_j
            if l_j[k] != 0:
                found_non_zero_l_j_k = True
                aux_optimal_plan.basis[k] = j

        if not found_non_zero_l_j_k:
            i = j_k - n
            A = np.delete(A, [i], axis=0)
            A_tilde = np.delete(A_tilde, [i], axis=0)
            b = np.delete(b, [i]),
            aux_optimal_plan.basis = np.delete(aux_optimal_plan.basis, [k])

            has_artificial_var_in_basis = False
            k = 0
            j_k = 0
            for i, j_i in enumerate(aux_optimal_plan.basis):
                if j_i > j_k:
                    j_k = j_i
                    k = i

                if j_i >= n:
                    has_artificial_var_in_basis = True
                    break

    new_problem = LPProblem(c, A, b, ["="] * len(b), [(0, None)] * len(c), True)
    return (new_problem, BasicSolution(x, aux_optimal_plan.basis))


def main():
    c = [1, 0, 0]
    A = [[1, 1, 1], [2, 2, 2]]
    b = [0, 0]
    ops = ['=', '=']
    bounds = [(0, None), (0, None), (0, None)]

    lp = LPProblem(c, A, b, ops, bounds, maximize=True)
    new_problem, basic_plan = simplex_starting_phase(lp)
    print("---- Simplex method starting phase returned ----")
    print("---- New Problem ----")
    print(new_problem)
    print("---- Basic Plan ----")
    print(basic_plan)


if __name__ == "__main__":
    main()
