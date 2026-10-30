import itertools
from collections import deque

import numpy as np


def s_t_path(c: np.ndarray) -> list[int]:
    n = c.shape[0]
    if n == 1:
        return []

    s = 0
    t = n - 1
    queue = deque([s])
    parent: dict[int, int | None] = {s: None}
    while queue:
        u = queue.popleft()
        if u == t:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in range(n):
            if c[u][v] > 0:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)

    return []


def min_capacity_on_path(c: np.ndarray, path: list[int]) -> int:
    min_cap = np.inf
    for i, j in itertools.pairwise(path):
        min_cap = min(min_cap, c[i][j])

    return min_cap


def flow_add(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    n = f1.shape[0]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            res[i][j] = max(0, f1[i][j] - f1[j][i] + f2[i][j] - f2[j][i])

    return res


def max_flow(c: np.ndarray) -> np.ndarray:
    n, _ = c.shape

    f = np.zeros((n, n))
    c_f = np.zeros((n, n))
    while True:
        for i in range(n):
            for j in range(n):
                c_f[i][j] = c[i][j] - f[i][j] + f[j][i]

        g_f_st_path = s_t_path(c_f)
        if len(g_f_st_path) == 0:
            return f

        theta = min_capacity_on_path(c_f, g_f_st_path)

        f_p = np.zeros((n, n))
        for i, j in itertools.pairwise(g_f_st_path):
            f_p[i][j] = theta

        f = flow_add(f, f_p)

        for i, j in itertools.pairwise(g_f_st_path):
            c_f[i][j] -= theta
            c_f[j][i] += theta


def main():
    c = np.array([
        [0, 3, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ])

    print(max_flow(c))
    pass


if __name__ == '__main__':
    main()
