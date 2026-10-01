import io
import itertools
import random
from contextlib import redirect_stdout

import numpy as np

from io_lab4 import (  # replace `your_module` with the actual module name
    get_path,
    get_longest_distance,
    print_path,
    longest_path,
)

NEG = -np.inf


def make_d(n: int, edges: dict[tuple[int, int], float]) -> np.ndarray:
    d = np.full((n, n), NEG, dtype=float)
    for (i, j), w in edges.items():
        d[i][j] = w
    return d


def make_pred(n: int, path: list[int]) -> np.ndarray:
    pred = np.full((n, n), -1, dtype=int)
    for a, b in zip(path, path[1:]):
        pred[0][b] = a
    return pred


def path_from_pred(pred: np.ndarray, k: int, l: int) -> list[int]:
    path = [l]
    cur = l
    steps = 0
    while cur != k:
        cur = int(pred[k][cur])
        assert cur != -1, "broken predecessor chain"
        path.append(cur)
        steps += 1
        assert steps <= pred.shape[0], "cycle in predecessor chain"
    path.reverse()
    return path


def brute_force_longest(n: int, d: np.ndarray):
    best = NEG
    best_paths = []
    inner = list(range(1, n - 1))
    for r in range(len(inner) + 1):
        for mid in itertools.combinations(inner, r):
            nodes = [0, *mid, n - 1] if n > 1 else [0]
            total = 0.0
            for a, b in zip(nodes, nodes[1:]):
                total += d[a][b]
            if total > best:
                best = total
                best_paths = [nodes]
            elif total == best:
                best_paths.append(nodes)
    return best, best_paths


def path_weight(path: list[int], d: np.ndarray) -> float:
    return sum(d[a][b] for a, b in zip(path, path[1:]))


def test_get_path_simple_chain():
    pred = make_pred(4, [0, 1, 2, 3])
    assert get_path(pred) == [0, 1, 2, 3]


def test_get_path_direct_jump():
    pred = make_pred(4, [0, 3])
    assert get_path(pred) == [0, 3]


def test_get_path_two_nodes():
    pred = make_pred(2, [0, 1])
    assert get_path(pred) == [0, 1]


def test_get_path_single_node():
    pred = np.full((1, 1), -1, dtype=int)
    assert get_path(pred) == [0]


def test_get_path_with_skips():
    pred = make_pred(6, [0, 2, 3, 5])
    assert get_path(pred) == [0, 2, 3, 5]


def test_get_path_no_path_returns_empty():
    pred = np.full((3, 3), -1, dtype=int)
    assert get_path(pred) == []


def test_get_path_uses_source_zero_row_only():
    pred = make_pred(4, [0, 1, 3])
    pred[1][3] = 2
    pred[2][3] = 2
    assert get_path(pred) == [0, 1, 3]


def test_print_path_output():
    pred = make_pred(4, [0, 1, 2, 3])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->1->2->3"


def test_print_path_two_nodes():
    pred = make_pred(2, [0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->1"


def test_print_path_direct_jump():
    pred = make_pred(5, [0, 4])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->4"


def test_get_longest_distance_reads_dp_corner():
    dp = np.array([[0, 2, 6, 8],
                   [NEG, 0, 4, 6],
                   [NEG, NEG, 0, 2],
                   [NEG, NEG, NEG, 0]], dtype=float)
    assert get_longest_distance(dp) == 8


def test_get_longest_distance_single_node():
    dp = np.array([[0.0]])
    assert get_longest_distance(dp) == 0


def test_longest_path_single_node():
    dp, pred = longest_path(1, np.array([[NEG]]))
    assert dp.shape == (1, 1)
    assert pred.shape == (1, 1)
    assert dp[0][0] == 0
    assert pred[0][0] == -1
    assert get_path(pred) == [0]
    assert get_longest_distance(dp) == 0


def test_longest_path_two_nodes():
    d = make_d(2, {(0, 1): 5})
    dp, pred = longest_path(2, d)
    assert dp[0][0] == 0 and dp[1][1] == 0
    assert dp[0][1] == 5
    assert dp[1][0] == NEG
    assert pred[0][1] == 0
    assert pred[0][0] == -1 and pred[1][1] == -1 and pred[1][0] == -1
    assert get_path(pred) == [0, 1]
    assert get_longest_distance(dp) == 5


def test_longest_path_basic_dag():
    d = make_d(4, {(0, 1): 2, (0, 2): 3, (1, 2): 4, (1, 3): 1, (2, 3): 2})
    dp, pred = longest_path(4, d)
    assert dp[0][1] == 2
    assert dp[0][2] == 6
    assert dp[0][3] == 8
    assert dp[1][2] == 4
    assert dp[1][3] == 6
    assert dp[2][3] == 2
    assert pred[0][1] == 0
    assert pred[0][2] == 1
    assert pred[0][3] == 2
    assert pred[1][2] == 1
    assert pred[1][3] == 2
    assert pred[2][3] == 2
    assert path_from_pred(pred, 0, 3) == [0, 1, 2, 3]
    assert get_path(pred) == [0, 1, 2, 3]
    assert get_longest_distance(dp) == 8


def test_longest_path_prefers_direct_edge():
    d = make_d(3, {(0, 1): 1, (1, 2): 2, (0, 2): 10})
    dp, pred = longest_path(3, d)
    assert dp[0][2] == 10
    assert pred[0][2] == 0
    assert pred[0][1] == 0
    assert get_path(pred) == [0, 2]
    assert get_longest_distance(dp) == 10


def test_longest_path_prefers_long_route():
    d = make_d(3, {(0, 1): 6, (1, 2): 7, (0, 2): 10})
    dp, pred = longest_path(3, d)
    assert dp[0][2] == 13
    assert pred[0][2] == 1
    assert get_path(pred) == [0, 1, 2]
    assert get_longest_distance(dp) == 13


def test_longest_path_missing_edges():
    d = make_d(4, {(0, 1): 3, (1, 3): 4, (0, 2): 100})
    dp, pred = longest_path(4, d)
    assert dp[0][2] == 100
    assert dp[0][3] == 7
    assert dp[2][3] == NEG
    assert pred[0][2] == 0
    assert pred[0][3] == 1
    assert pred[2][3] == -1
    assert get_path(pred) == [0, 1, 3]
    assert get_longest_distance(dp) == 7


def test_longest_path_unreachable_target():
    d = make_d(3, {(0, 1): 4})
    dp, pred = longest_path(3, d)
    assert dp[0][2] == NEG
    assert pred[0][2] == -1
    assert pred[1][2] == -1
    assert get_path(pred) == []


def test_longest_path_diagonal_and_lower_triangle():
    n = 5
    d = make_d(n, {(i, j): float(i + j) for i in range(n) for j in range(i + 1, n)})
    dp, pred = longest_path(n, d)
    for i in range(n):
        assert dp[i][i] == 0
        assert pred[i][i] == -1
        for j in range(i):
            assert dp[i][j] == NEG
            assert pred[i][j] == -1


def test_longest_path_pred_defined_iff_dp_finite():
    d = make_d(5, {(0, 1): 1, (1, 2): 2, (0, 3): 4, (3, 4): 1})
    dp, pred = longest_path(5, d)
    for k in range(5):
        for l in range(k + 1, 5):
            if dp[k][l] == NEG:
                assert pred[k][l] == -1
            else:
                assert k <= pred[k][l] < l


def test_longest_path_pred_consistent_with_dp():
    d = make_d(5, {(i, j): float((i * 3 + j * 5) % 7 - 2) for i in range(5) for j in range(i + 1, 5)})
    dp, pred = longest_path(5, d)
    for k in range(5):
        for l in range(k + 1, 5):
            if dp[k][l] != NEG:
                p = int(pred[k][l])
                assert dp[k][l] == dp[k][p] + d[p][l]


def test_longest_path_negative_weights():
    d = make_d(4, {(0, 1): -1, (1, 3): -1, (0, 2): -5, (2, 3): -5, (0, 3): -3})
    dp, pred = longest_path(4, d)
    assert dp[0][3] == -2
    assert pred[0][3] == 1
    assert get_path(pred) == [0, 1, 3]
    assert get_longest_distance(dp) == -2


def test_longest_path_zero_weights():
    d = make_d(3, {(0, 1): 0, (1, 2): 0, (0, 2): 0})
    dp, pred = longest_path(3, d)
    assert dp[0][2] == 0
    path = get_path(pred)
    assert path[0] == 0 and path[-1] == 2


def test_longest_path_float_weights():
    d = make_d(3, {(0, 1): 0.5, (1, 2): 0.25, (0, 2): 0.7})
    dp, pred = longest_path(3, d)
    assert abs(dp[0][2] - 0.75) < 1e-12
    assert get_path(pred) == [0, 1, 2]


def test_longest_path_complete_graph_all_ones():
    n = 6
    d = make_d(n, {(i, j): 1 for i in range(n) for j in range(i + 1, n)})
    dp, pred = longest_path(n, d)
    assert dp[0][n - 1] == n - 1
    assert get_path(pred) == list(range(n))
    for l in range(1, n):
        assert pred[0][l] == l - 1
    assert get_longest_distance(dp) == n - 1


def test_longest_path_subpath_values():
    d = make_d(5, {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (0, 4): 3})
    dp, pred = longest_path(5, d)
    assert dp[0][1] == 1
    assert dp[0][2] == 2
    assert dp[0][3] == 3
    assert dp[0][4] == 4
    assert dp[1][4] == 3
    assert path_from_pred(pred, 0, 4) == [0, 1, 2, 3, 4]
    assert path_from_pred(pred, 1, 4) == [1, 2, 3, 4]
    assert path_from_pred(pred, 2, 4) == [2, 3, 4]


def test_longest_path_rows_are_independent():
    d = make_d(4, {(0, 1): 1, (1, 3): 1, (0, 2): 1, (2, 3): 1, (1, 2): 50})
    dp, pred = longest_path(4, d)
    assert dp[0][3] == 52
    assert path_from_pred(pred, 0, 3) == [0, 1, 2, 3]
    assert path_from_pred(pred, 1, 3) == [1, 2, 3]
    assert path_from_pred(pred, 2, 3) == [2, 3]


def test_longest_path_pred_is_integer_matrix():
    d = make_d(4, {(0, 1): 1, (1, 2): 1, (2, 3): 1})
    dp, pred = longest_path(4, d)
    assert dp.shape == (4, 4)
    assert pred.shape == (4, 4)
    assert np.issubdtype(pred.dtype, np.integer)


def test_longest_path_does_not_mutate_input():
    d = make_d(4, {(0, 1): 2, (0, 2): 3, (1, 2): 4, (1, 3): 1, (2, 3): 2})
    d_copy = d.copy()
    longest_path(4, d)
    assert np.array_equal(d, d_copy)


def test_longest_path_every_pair_reconstruction_matches_dp():
    n = 6
    rng = random.Random(7)
    d = make_d(n, {(i, j): float(rng.randint(1, 9)) for i in range(n) for j in range(i + 1, n)})
    dp, pred = longest_path(n, d)
    for k in range(n):
        for l in range(k + 1, n):
            path = path_from_pred(pred, k, l)
            assert path[0] == k and path[-1] == l
            assert path == sorted(set(path))
            assert path_weight(path, d) == dp[k][l]


def test_longest_path_random_against_brute_force():
    rng = random.Random(12345)
    for _ in range(200):
        n = rng.randint(2, 8)
        edges = {}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.7 or (i == 0 and j == n - 1):
                    edges[(i, j)] = float(rng.randint(-5, 20))
        d = make_d(n, edges)
        dp, pred = longest_path(n, d)
        expected, expected_paths = brute_force_longest(n, d)
        assert dp[0][n - 1] == expected, (n, d)
        assert get_longest_distance(dp) == expected
        path = get_path(pred)
        assert path == path_from_pred(pred, 0, n - 1)
        assert path[0] == 0 and path[-1] == n - 1
        assert path == sorted(set(path))
        assert path_weight(path, d) == expected
        assert path in expected_paths


def main():
    tests = [
        test_get_path_simple_chain,
        test_get_path_direct_jump,
        test_get_path_two_nodes,
        test_get_path_single_node,
        test_get_path_with_skips,
        test_get_path_no_path_returns_empty,
        test_get_path_uses_source_zero_row_only,
        test_print_path_output,
        test_print_path_two_nodes,
        test_print_path_direct_jump,
        test_get_longest_distance_reads_dp_corner,
        test_get_longest_distance_single_node,
        test_longest_path_single_node,
        test_longest_path_two_nodes,
        test_longest_path_basic_dag,
        test_longest_path_prefers_direct_edge,
        test_longest_path_prefers_long_route,
        test_longest_path_missing_edges,
        test_longest_path_unreachable_target,
        test_longest_path_diagonal_and_lower_triangle,
        test_longest_path_pred_defined_iff_dp_finite,
        test_longest_path_pred_consistent_with_dp,
        test_longest_path_negative_weights,
        test_longest_path_zero_weights,
        test_longest_path_float_weights,
        test_longest_path_complete_graph_all_ones,
        test_longest_path_subpath_values,
        test_longest_path_rows_are_independent,
        test_longest_path_pred_is_integer_matrix,
        test_longest_path_does_not_mutate_input,
        test_longest_path_every_pair_reconstruction_matches_dp,
        test_longest_path_random_against_brute_force,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()