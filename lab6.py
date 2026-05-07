import numpy as np


class BasicTransportSolution:
    def __init__(self, x: np.ndarray, basis: list[tuple[int, int]]):
        self.x = x
        self.B = basis

    def __str__(self):
        return f"x: {self.x}\nB: {self.B}"


def northwest_corner(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> BasicTransportSolution:
    a, b = np.copy(a), np.copy(b)
    m, n = c.shape
    x = np.zeros(shape=(m, n))
    B: list[tuple[int, int]] = []
    i = 0
    j = 0
    while True:
        B.append((i, j))
        x[i][j] = min(a[i], b[j])
        a[i] = a[i] - x[i][j]
        b[j] = b[j] - x[i][j]

        if a[i] == 0 and i < m - 1:
            i += 1
        elif a[i] > 0 and b[j] == 0 and j < n - 1:
            j += 1
        else:
            return BasicTransportSolution(x, B)


def compute_potentials(c: np.ndarray, sol: BasicTransportSolution) -> tuple[np.ndarray, np.ndarray]:
    m, n = c.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for (i, j) in sol.B:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = c[i][j] - u[i]
                changed = True
            elif not np.isnan(v[j]) and np.isnan(u[i]):
                u[i] = c[i][j] - v[j]
                changed = True

    return u, v


def find_cycle_elimination(B: list[tuple[int, int]], enter: tuple[int, int]):
    positions = set(B) | {enter}

    changed = True
    while changed:
        changed = False
        for pos in list(positions):
            row_count = sum(1 for p in positions if p[0] == pos[0])
            col_count = sum(1 for p in positions if p[1] == pos[1])
            if row_count == 1 or col_count == 1:
                positions.discard(pos)
                changed = True

    corners = list(positions)
    cycle = [enter]
    used = {enter}
    direction = 'col'

    while len(cycle) < len(corners):
        cur = cycle[-1]
        if direction == 'col':
            nxt = next(p for p in corners if p[1] == cur[1] and p not in used)
        else:
            nxt = next(p for p in corners if p[0] == cur[0] and p not in used)
        cycle.append(nxt)
        used.add(nxt)
        direction = 'row' if direction == 'col' else 'col'

    return cycle


def solve_transport(c: np.ndarray, a: list, b: list) -> BasicTransportSolution | None:
    a, b = np.array(a, float), np.array(b, float)
    if np.sum(a) != np.sum(b):
        return None

    sol = northwest_corner(a, b, c)

    m, n = c.shape

    while True:
        u, v = compute_potentials(c, sol)
        B_set = set(sol.B)
        non_basic = [(i, j) for i in range(m) for j in range(n)
                     if (i, j) not in B_set]
        deltas = {(i, j): c[i, j] - u[i] - v[j] for (i, j) in non_basic}

        negative = {point: delta for point, delta in deltas.items() if delta < 0}
        if not negative:
            return sol

        enter = min(negative.keys())
        cycle = find_cycle_elimination(sol.B, enter)

        signs = [1 if k % 2 == 0 else -1 for k in range(len(cycle))]
        minus_pos = [cycle[k] for k in range(len(cycle)) if signs[k] == -1]

        theta = min(sol.x[i][j] for (i, j) in minus_pos)

        for k, (i, j) in enumerate(cycle):
            sol.x[i][j] += theta * signs[k]

        leave_cands = [(i, j) for (i, j) in minus_pos if sol.x[i][j] == 0]
        leave = min(leave_cands)
        B_set.discard(leave)
        B_set.add(enter)
        sol.B = sorted(B_set)


def main():
    c = np.array([
        [8, 4, 1],
        [8, 4, 3],
        [9, 7, 5],
    ])
    a = [100, 300, 300]
    b = [300, 200, 200]
    sol = solve_transport(c, a, b)
    print(sol)
    print(float(np.sum(c * sol.x)))


if __name__ == "__main__":
    main()
