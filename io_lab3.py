import numpy as np

def backpack(costs:list[int], weights:list[int], capacity: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(weights)
    dp = np.full((n, capacity + 1), fill_value=-1, dtype=int)
    x = np.zeros((n,capacity + 1), dtype=int)
    for i in range(n):
        for w in range(capacity + 1):
            if i == 0:
                if weights[i] <= w:
                    dp[i][w] = costs[i]
                    x[i][w] = 1
                else:
                    dp[i][w] = 0
                    
                continue
                
            if weights[i] <= w:
                alpha = costs[i] + dp[i-1][w - weights[i]]
                beta = dp[i-1][w]
                if alpha >= beta:
                    dp[i][w] = alpha
                    x[i][w] = 1
                else:
                    dp[i][w] = beta
            else:
                dp[i][w] = dp[i-1][w]
                x[i][w] = 0
    return dp, x

def restore_path(x:np.ndarray, weights: list[int], capacity: int) -> list[int]:
    res: list[int] = []
    w = capacity
    for i in reversed(range(len(weights))):
        if x[i][w] == 1:
            res.append(i)
            w -= weights[i]

    res.reverse()
    
    return res

def test_1():
    costs = [6, 10, 12]
    weights = [1, 2, 3]
    capacity = 5

    dp, x = backpack(costs, weights, capacity)
    optimal_res = 22
    print(dp)
    print(x)
    assert dp[len(costs) - 1][capacity] == optimal_res

    path = restore_path(x, weights, capacity)
    assert path == [1, 2]

def main():
    test_1()
    
if __name__ == "__main__":
    main()