import numpy as np


def distribute(P: int, Q: int, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if A.shape != (P, Q + 1):
        raise ValueError(
            f"Expected A.shape == {(P, Q + 1)}, got {A.shape}"
        )

    B = np.full((P, Q + 1), -np.inf, dtype=float)
    C = np.zeros((P, Q + 1), dtype=int)

    for p in range(P):
        for q in range(Q + 1):
            if p == 0:
                B[p, q] = A[p, q]
                C[p, q] = q
                continue

            for i in range(q + 1):
                candidate = A[p, i] + B[p - 1, q - i]

                if candidate > B[p, q]:
                    B[p, q] = candidate
                    C[p, q] = i

    return B, C


def restore_plan(P: int, Q: int, C: np.ndarray) -> list[int]:
    allocation = [0] * P

    p = P - 1
    q = Q

    while p >= 0:
        allocation[p] = int(C[p, q])
        q -= allocation[p]
        p -= 1

    return allocation


def main() -> None:
    P = 3
    Q = 3

    A = np.array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 2, 2, 3],
    ], dtype=float)

    B, C = distribute(P, Q, A)
    allocation = restore_plan(P, Q, C)

    print(f"Optimal profit: {B[P - 1, Q]}")

    for agent, amount in enumerate(allocation):
        print(f"Agent {agent} gets {amount} resource(s)")


if __name__ == "__main__":
    main()