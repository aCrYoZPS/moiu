import math

import numpy as np

from lab2 import LPProblem
from lab3 import simplex_main_phase, BasicSolution
from lab4 import simplex_starting_phase
from lab5 import dual_simplex_method

EPS = 1e-9

ILPProblem = LPProblem


def simplex_method(problem: LPProblem) -> BasicSolution | None:
    prob, basic_solution = simplex_starting_phase(problem)
    return simplex_main_phase(prob, basic_solution)


def frac(x: float) -> float:
    return x - math.floor(x)


def format_equation(row: np.ndarray, rhs: float, precision: int = 3) -> str:
    terms = []
    for j, coef in enumerate(row):
        if abs(coef) < EPS:
            continue
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.{precision}f}*x_{j + 1}")

    eq = " ".join(terms) if terms else "0"
    if eq.startswith("+ "):
        eq = eq[2:]

    return f"{eq} = {rhs:.{precision}f}"


def format_system(A: np.ndarray, b: np.ndarray, ops: list[str] | None = None, precision: int = 3) -> str:
    m, n = np.shape(A)
    ops = ops if ops is not None else ["="] * m

    headers = [f"x_{j + 1}" for j in range(n)]
    cells = [[f"{A[i, j]:.{precision}f}" for j in range(n)] for i in range(m)]
    widths = [max(len(headers[j]), max(cells[i][j].__len__() for i in range(m))) for j in range(n)]
    rhs_cells = [f"{b[i]:.{precision}f}" for i in range(m)]
    rhs_width = max(len(cell) for cell in rhs_cells)

    prefix_width = max(len(f"[{i + 1}]") for i in range(m)) + 2

    header_line = " " * prefix_width + "  ".join(f"{headers[j]:>{widths[j]}}" for j in range(n))
    row_lines = [
        f"[{i + 1}]".ljust(prefix_width) + "  ".join(f"{cells[i][j]:>{widths[j]}}" for j in range(n))
        + f"   {ops[i]}   {rhs_cells[i]:>{rhs_width}}"
        for i in range(m)
    ]

    return "\n".join([header_line] + row_lines)


def make_gomory_cuts(problem: ILPProblem) -> tuple[ILPProblem, BasicSolution]:
    problem, sol, resume = make_gomory_cut(problem, None, True)
    while resume:
        try:
            problem, sol, resume = make_gomory_cut(problem, sol)
        except Exception as ex:
            print(ex)
            break

    return problem, sol


def make_gomory_cut(problem: ILPProblem,
                    old_solution: BasicSolution | None,
                    first_pass: bool = False) -> tuple[ILPProblem, BasicSolution, bool] | None:
    c, A, b, offset = problem.canonicalize()
    new_problem = ILPProblem(c, A, b, ["="] * len(b), [(0, None)] * len(c), True)

    if first_pass:
        solution = simplex_method(new_problem)
        if solution is None:
            return None
    else:
        solution = dual_simplex_method(c, A, b, old_solution.basis)
        if solution is None:
            return None

    k, j_k = None, None
    for idx in solution.basis:
        x_i = solution.x[idx]
        if EPS < frac(x_i) < 1 - EPS:
            j_k = idx
            break

    if j_k is None:
        return problem, solution, False

    k = list(solution.basis).index(j_k)

    m, n = np.shape(A)

    basis = solution.basis
    non_basis = sorted(set(range(n)).difference(set(solution.basis)))

    basis = list(basis)
    non_basis = list(non_basis)

    A_b = A[:, basis]
    A_n = A[:, non_basis]
    Q = np.linalg.inv(A_b) @ A_n
    l = Q[k]
    t = len(l)

    new_col = np.zeros(m + 1)
    new_col[m] = -1
    new_row = np.zeros(n)
    for p in range(t):
        new_row[non_basis[p]] = frac(l[p])

    cut_b = frac(solution.x[j_k])
    new_b = list(np.append(b, cut_b))
    new_c = list(np.append(c, 0))
    new_A = np.column_stack((np.vstack((A, new_row)), new_col))
    new_basis = basis + [n]
    new_x = np.append(solution.x, -cut_b)

    print(f"Cut (condition [{m + 1}]): {format_equation(new_A[-1], cut_b)}")

    return ILPProblem(new_c, new_A, new_b, ["="] * (m + 1), [(0, None)] * (n + 1), True), BasicSolution(new_x,
                                                                                                        new_basis), True


def main():
    c = [0, 1, 0, 0]
    A = [
        [3, 2, 1, 0],
        [-3, 2, 0, 1]
    ]
    b = [6, 0]
    ops = ["=", "="]
    bounds = [(0, None), (0, None), (0, None), (0, None)]
    problem = ILPProblem(c, A, b, ops, bounds, True)
    new_problem, sol = make_gomory_cuts(problem)

    fc, fA, fb, foffset = new_problem.canonicalize()
    print("Resulting problem A*x = b:")
    print(format_system(fA, fb))
    print(f"Solution: {sol}")


if __name__ == '__main__':
    main()
