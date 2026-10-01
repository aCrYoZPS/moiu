import numpy as np


def get_path(pred: np.ndarray) -> list[int]:
    n = len(pred)
    if n == 1:
        return [0]

    cur = n - 1
    path = []
    while cur != 0:
        prev = pred[cur]
        if prev == -1:
            break

        path.append(int(cur))
        cur = prev
    if len(path) == 0:
        return path

    path.append(0)
    path.reverse()

    return path


def get_longest_distance(dp: np.ndarray) -> float:
    n = dp.shape[0]
    return dp[n - 1]


def print_path(pred: np.ndarray):
    path = get_path(pred)
    res = ""
    for v in path:
        res += f"->{v}"

    print(res[2:])


def longest_path(n: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dp = np.full(n, -np.inf, dtype=float)
    dp[0] = 0
    pred = np.full(n, -1, dtype=int)
    for i in range(n):
        if dp[i] == -np.inf:
            continue

        for j in range(i + 1, n):
            if dp[j] < dp[i] + d[i][j]:
                dp[j] = dp[i] + d[i][j]
                pred[j] = i

    return dp, pred


def test_1():
    # 1) Single node graph (n = 1)
    print("=== Test 1: single node ===")
    n = 1
    d = np.full((n, n), -np.inf, dtype=float)  # no edges
    dp, pred = longest_path(n, d)
    print("dp:", dp)
    print("pred:", pred)
    print("path:", get_path(pred))
    print("longest distance:", get_longest_distance(dp))
    print("printed path:", end=" ")
    print_path(pred)
    print()

    # 2) Simple chain: 0 -> 1 -> 2 -> 3
    print("=== Test 2: simple chain 0->1->2->3 ===")
    n = 4
    d = np.full((n, n), -np.inf, dtype=float)
    d[0, 1] = 2
    d[1, 2] = 3
    d[2, 3] = 4
    # optional extra edge 0->3 with smaller total weight
    d[0, 3] = 5

    dp, pred = longest_path(n, d)
    print("dp:", dp)
    print("pred:", pred)
    print("path:", get_path(pred))
    print("longest distance:", get_longest_distance(dp))
    print("printed path:", end=" ")
    print_path(pred)
    print()

    # 3) Branching where longer path uses more edges:
    # 0->1 (1), 1->2 (1), 2->3 (1), 0->3 (2)
    # Best: 0->1->2->3 = 3 > 2
    print("=== Test 3: prefer longer route ===")
    n = 4
    d = np.full((n, n), -np.inf, dtype=float)
    d[0, 1] = 1
    d[1, 2] = 1
    d[2, 3] = 1
    d[0, 3] = 2

    dp, pred = longest_path(n, d)
    print("dp:", dp)
    print("pred:", pred)
    print("path:", get_path(pred))
    print("longest distance:", get_longest_distance(dp))
    print("printed path:", end=" ")
    print_path(pred)
    print()

    print("=== Test 4: unreachable target ===")
    n = 3
    d = np.full((n, n), -np.inf, dtype=float)
    d[0, 1] = 5

    dp, pred = longest_path(n, d)
    print("dp:", dp)
    print("pred:", pred)
    print("path:", get_path(pred))
    print("longest distance:", get_longest_distance(dp))
    print("printed path:", end=" ")
    print_path(pred)
    print()

    print("=== Test 5: negative weights ===")
    n = 4
    d = np.full((n, n), -np.inf, dtype=float)
    d[0, 1] = -1
    d[1, 3] = -1
    d[0, 2] = -5
    d[2, 3] = -5
    d[0, 3] = -3

    dp, pred = longest_path(n, d)
    print("dp:", dp)
    print("pred:", pred)
    print("path:", get_path(pred))
    print("longest distance:", get_longest_distance(dp))
    print("printed path:", end=" ")
    print_path(pred)
    print()


def main():
    test_1()


if __name__ == "__main__":
    main()
