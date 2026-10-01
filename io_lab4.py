import numpy as np


def get_path(pred: np.ndarray) -> list[int]:
    n = pred.shape[0]
    cur = n - 1
    path = [cur]
    while cur != 0:
        prev = pred[0][cur]
        if prev == -1:
            break

        path.append(int(prev))
        cur = prev

    return path


def get_longest_distance(dp: np.ndarray) -> float:
    n = dp.shape[0]
    return max(dp[:, n - 1])


def print_path(pred: np.ndarray):
    path = get_path(pred)
    res = ""
    for v in path:
        res += f"->{v}"

    print(res[2:])


def longest_path(n: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dp = np.full((n, n), -np.inf, dtype=float)
    pred = np.full((n, n), -1, dtype=int)
    for k in range(n):
        for l in range(n):
            if k == l:
                dp[k][l] = 0
            elif k < l:
                max_k_l = -np.inf
                max_idx = -1
                for j in range(k, l):
                    candidate_max: int = dp[k][j] + d[j][l]
                    if candidate_max > max_k_l:
                        max_k_l = candidate_max
                        max_idx = j

                if max_idx != -1:
                    dp[k][l] = max_k_l
                    pred[k][l] = max_idx

    return dp, pred


def test_1():
    n = 1
    d = np.array([[-np.inf]])
    dp, x = longest_path(n, d)

    print(dp)
    print_path(x)


def main():
    test_1()


if __name__ == "__main__":
    main()
