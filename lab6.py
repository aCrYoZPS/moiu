import numpy as np


class BasicTransportSolution:
    def __init__(self, x: np.ndarray, basis: list[tuple[int, int]]):
        self.x = x
        self.B = basis

    def __str__(self):
        return f"x: {self.x}\nB: {self.B}"


def north_west_corner(a: list, b: list, c: np.ndarray) -> BasicTransportSolution:
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


def test():
    B = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
    pass


def main():
    c = np.array([
        [8, 4, 1],
        [8, 4, 3],
        [9, 7, 5],
    ])
    a = [100, 300, 300]
    b = [300, 200, 200]
    print(north_west_corner(a, b, c))


if __name__ == "__main__":
    main()
