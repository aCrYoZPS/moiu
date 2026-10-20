import numpy as np

from io_lab5 import BipartileGraph, find_max_matching


def reachable_from_s(g: BipartileGraph) -> set[int]:
    visited = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in g.neighbours[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)

    return visited


def hungarian_algorithm(c: np.ndarray) -> list[tuple[int, int]]:
    n = int(c.shape[0])
    alpha = np.zeros(n)
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = min(c[:, i])
    while True:
        j_eq: list[tuple[int, int]] = []
        j_lt: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(n):
                if alpha[i] + beta[j] < c[i][j]:
                    j_lt.append((i, j))
                elif alpha[i] + beta[j] == c[i][j]:
                    j_eq.append((i, j))

        neighbours: dict[int, list[int]] = {}
        for i, j in j_eq:
            if i in neighbours:
                neighbours[i].append(n + j)
            else:
                neighbours[i] = [n + j]

        g = BipartileGraph(list(range(n)), list(range(n, 2 * n)), neighbours)

        m, ext_g = find_max_matching(g)

        if len(m) == n:
            return [(i, j - n) for i, j in m]

        reachable = [v - 1 for v in reachable_from_s(ext_g) if v != 0]

        i_star: list[int] = []
        j_star: list[int] = []
        for v in reachable:
            if v >= n:
                j_star.append(v - n)
            else:
                i_star.append(v)

        alpha_hat = np.zeros(n)
        beta_hat = np.zeros(n)
        for i in range(n):
            if i in i_star:
                alpha_hat[i] = 1
            else:
                alpha_hat[i] = -1

            if i in j_star:
                beta_hat[i] = -1
            else:
                beta_hat[i] = 1

        theta = np.inf

        for i in range(n):
            for j in range(n):
                if i in i_star and j not in j_star:
                    theta = min(theta, (c[i][j] - alpha[i] - beta[j]) / 2)

        for i in range(n):
            alpha[i] = alpha[i] + theta * alpha_hat[i]
            beta[i] = beta[i] + theta * beta_hat[i]


def main():
    c = np.array(
        [
            [7, 2, 1, 9, 4],
            [9, 6, 9, 5, 5],
            [3, 8, 3, 1, 8],
            [7, 9, 4, 2, 2],
            [8, 4, 7, 4, 8],
        ]
    )

    print(sorted(hungarian_algorithm(c)))


if __name__ == "__main__":
    main()
