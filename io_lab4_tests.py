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
        d[i, j] = w
    return d


def path_weight(path: list[int], d: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return sum(d[a, b] for a, b in zip(path, path[1:]))


def brute_force_longest_from_0(n: int, d: np.ndarray):
    if n == 1:
        return 0.0, [[0]]

    best = NEG
    best_paths = []
    inner = list(range(1, n - 1))

    for r in range(len(inner) + 1):
        for mid in itertools.combinations(inner, r):
            nodes = [0, *mid, n - 1]
            total = 0.0
            ok = True
            for a, b in zip(nodes, nodes[1:]):
                w = d[a, b]
                if w == NEG:
                    ok = False
                    break
                total += w
            if not ok:
                continue
            if total > best + 1e-12:
                best = total
                best_paths = [nodes]
            elif abs(total - best) <= 1e-12:
                best_paths.append(nodes)

    return best, best_paths


def test_get_path_simple_chain():
    pred = np.array([-1, 0, 1, 2], dtype=int)
    assert get_path(pred) == [0, 1, 2, 3]


def test_get_path_direct_jump():
    pred = np.array([-1, -1, -1, 0], dtype=int)
    assert get_path(pred) == [0, 3]


def test_get_path_two_nodes():
    pred = np.array([-1, 0], dtype=int)
    assert get_path(pred) == [0, 1]


def test_get_path_single_node():
    pred = np.array([-1], dtype=int)
    assert get_path(pred) == [0]


def test_get_path_with_skips():
    # 0 -> 2 -> 3 -> 5 in n=6
    pred = np.array([-1, -1, 0, 2, -1, 3], dtype=int)
    assert get_path(pred) == [0, 2, 3, 5]


def test_get_path_no_path_returns_empty():
    # target unreachable: pred[n-1] == -1 and n > 1
    pred = np.array([-1, -1, -1], dtype=int)  # n=3, 0 cannot reach 2
    assert get_path(pred) == []


def test_get_path_ignores_other_predecessors():
    # Suppose some other row-like info existed; here we just ensure get_path
    pred = np.array([-1, 0, 0, 1], dtype=int)
    # Path 0 -> 1 -> 3
    assert get_path(pred) == [0, 1, 3]


# ---------------- print_path tests ----------------


def test_print_path_output():
    pred = np.array([-1, 0, 1, 2], dtype=int)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->1->2->3"


def test_print_path_two_nodes():
    pred = np.array([-1, 0], dtype=int)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->1"


def test_print_path_direct_jump():
    pred = np.array([-1, -1, -1, -1, 0], dtype=int)  # 0 -> 4
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0->4"


def test_print_path_single_node():
    pred = np.array([-1], dtype=int)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_path(pred)
    assert buf.getvalue().strip() == "0"


# ---------------- get_longest_distance tests ----------------


def test_get_longest_distance_reads_dp_corner():
    dp = np.array([0.0, 2.0, 6.0, 8.0], dtype=float)
    assert get_longest_distance(dp) == 8.0


def test_get_longest_distance_single_node():
    dp = np.array([0.0], dtype=float)
    assert get_longest_distance(dp) == 0.0


def test_get_longest_distance_unreachable():
    dp = np.array([0.0, NEG, NEG], dtype=float)
    # Longest distance to last node is -inf
    assert get_longest_distance(dp) == NEG


# ---------------- longest_path tests ----------------


def test_longest_path_single_node():
    d = make_d(1, {})
    dp, pred = longest_path(1, d)
    assert dp.shape == (1,)
    assert pred.shape == (1,)
    assert dp[0] == 0.0
    assert pred[0] == -1
    assert get_path(pred) == [0]
    assert get_longest_distance(dp) == 0.0


def test_longest_path_two_nodes():
    d = make_d(2, {(0, 1): 5.0})
    dp, pred = longest_path(2, d)
    assert dp.shape == (2,)
    assert pred.shape == (2,)
    assert dp[0] == 0.0
    assert dp[1] == 5.0
    assert pred[0] == -1
    assert pred[1] == 0
    assert get_path(pred) == [0, 1]
    assert get_longest_distance(dp) == 5.0


def test_longest_path_basic_dag():
    # 0->1 (2), 0->2 (3), 1->2 (4), 1->3 (1), 2->3 (2)
    d = make_d(4, {
        (0, 1): 2.0,
        (0, 2): 3.0,
        (1, 2): 4.0,
        (1, 3): 1.0,
        (2, 3): 2.0,
    })
    dp, pred = longest_path(4, d)

    # From 0:
    # dp[0] = 0
    # dp[1] = 2 (0->1)
    # dp[2] = 6 (0->1->2)
    # dp[3] = 8 (0->1->2->3)
    assert dp[0] == 0.0
    assert dp[1] == 2.0
    assert dp[2] == 6.0
    assert dp[3] == 8.0

    assert pred[0] == -1
    assert pred[1] == 0
    assert pred[2] == 1
    assert pred[3] == 2

    assert get_path(pred) == [0, 1, 2, 3]
    assert get_longest_distance(dp) == 8.0


def test_longest_path_prefers_direct_edge():
    # 0->1 (1), 1->2 (2), 0->2 (10)
    d = make_d(3, {
        (0, 1): 1.0,
        (1, 2): 2.0,
        (0, 2): 10.0,
    })
    dp, pred = longest_path(3, d)
    assert dp[0] == 0.0
    assert dp[1] == 1.0
    assert dp[2] == 10.0
    assert pred[1] == 0
    assert pred[2] == 0
    assert get_path(pred) == [0, 2]
    assert get_longest_distance(dp) == 10.0


def test_longest_path_prefers_long_route():
    # 0->1 (6), 1->2 (7), 0->2 (10)
    d = make_d(3, {
        (0, 1): 6.0,
        (1, 2): 7.0,
        (0, 2): 10.0,
    })
    dp, pred = longest_path(3, d)
    assert dp[0] == 0.0
    assert dp[1] == 6.0
    assert dp[2] == 13.0
    assert pred[1] == 0
    assert pred[2] == 1
    assert get_path(pred) == [0, 1, 2]
    assert get_longest_distance(dp) == 13.0


def test_longest_path_missing_edges():
    # 0->1 (3), 1->3 (4), 0->2 (100), no edge 2->3
    d = make_d(4, {
        (0, 1): 3.0,
        (1, 3): 4.0,
        (0, 2): 100.0,
    })
    dp, pred = longest_path(4, d)
    assert dp[0] == 0.0
    assert dp[1] == 3.0
    assert dp[2] == 100.0
    assert dp[3] == 7.0  # 0->1->3

    assert pred[0] == -1
    assert pred[1] == 0
    assert pred[2] == 0
    assert pred[3] == 1

    assert get_path(pred) == [0, 1, 3]
    assert get_longest_distance(dp) == 7.0


def test_longest_path_unreachable_target():
    # 0->1 (4), node 2 unreachable from 0
    d = make_d(3, {
        (0, 1): 4.0,
    })
    dp, pred = longest_path(3, d)
    assert dp[0] == 0.0
    assert dp[1] == 4.0
    assert dp[2] == NEG

    assert pred[0] == -1
    assert pred[1] == 0
    assert pred[2] == -1

    assert get_path(pred) == []
    assert get_longest_distance(dp) == NEG


def test_longest_path_diagonal_and_lower_triangle():
    n = 5
    d = make_d(n, {(i, j): float(i + j) for i in range(n) for j in range(i + 1, n)})
    dp, pred = longest_path(n, d)

    assert dp[0] == 0.0
    assert pred[0] == -1

    # All nodes reachable; just check shapes and that dp is finite
    assert dp.shape == (n,)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(dp))
    assert pred[0] == -1


def test_longest_path_pred_defined_iff_dp_finite():
    d = make_d(5, {
        (0, 1): 1.0,
        (1, 2): 2.0,
        (0, 3): 4.0,
        (3, 4): 1.0,
    })
    dp, pred = longest_path(5, d)
    for i in range(5):
        if dp[i] == NEG:
            assert pred[i] == -1
        else:
            if i != 0:
                assert 0 <= pred[i] < i


def test_longest_path_pred_consistent_with_dp():
    d = make_d(
        5,
        {
            (i, j): float((i * 3 + j * 5) % 7 - 2)
            for i in range(5)
            for j in range(i + 1, 5)
        },
    )
    dp, pred = longest_path(5, d)
    for i in range(1, 5):
        if dp[i] != NEG:
            p = int(pred[i])
            assert 0 <= p < i
            assert dp[i] == dp[p] + d[p, i]


def test_longest_path_negative_weights():
    # 0->1 (-1), 1->3 (-1), 0->2 (-5), 2->3 (-5), 0->3 (-3)
    d = make_d(4, {
        (0, 1): -1.0,
        (1, 3): -1.0,
        (0, 2): -5.0,
        (2, 3): -5.0,
        (0, 3): -3.0,
    })
    dp, pred = longest_path(4, d)
    # Best: 0->1->3 = -2
    assert dp[0] == 0.0
    assert dp[1] == -1.0
    assert dp[2] == -5.0
    assert dp[3] == -2.0
    assert pred[1] == 0
    assert pred[3] == 1
    assert get_path(pred) == [0, 1, 3]
    assert get_longest_distance(dp) == -2.0


def test_longest_path_zero_weights():
    d = make_d(3, {
        (0, 1): 0.0,
        (1, 2): 0.0,
        (0, 2): 0.0,
    })
    dp, pred = longest_path(3, d)
    assert dp[0] == 0.0
    assert dp[2] == 0.0
    path = get_path(pred)
    assert path[0] == 0 and path[-1] == 2


def test_longest_path_float_weights():
    d = make_d(3, {
        (0, 1): 0.5,
        (1, 2): 0.25,
        (0, 2): 0.7,
    })
    dp, pred = longest_path(3, d)
    assert abs(dp[2] - 0.75) < 1e-12
    assert get_path(pred) == [0, 1, 2]


def test_longest_path_complete_graph_all_ones():
    n = 6
    d = make_d(n, {(i, j): 1.0 for i in range(n) for j in range(i + 1, n)})
    dp, pred = longest_path(n, d)

    # Longest path from 0 to n-1 visits all vertices: length n-1
    assert dp[n - 1] == n - 1
    assert get_path(pred) == list(range(n))

    for i in range(1, n):
        assert pred[i] == i - 1

    assert get_longest_distance(dp) == n - 1


def test_longest_path_subpath_values():
    # 0-1-2-3-4 each weight 1, plus direct 0->4 weight 3
    d = make_d(5, {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (3, 4): 1.0,
        (0, 4): 3.0,
    })
    dp, pred = longest_path(5, d)

    assert dp[0] == 0.0
    assert dp[1] == 1.0
    assert dp[2] == 2.0
    assert dp[3] == 3.0
    assert dp[4] == 4.0  # via 0-1-2-3-4

    assert pred[1] == 0
    assert pred[2] == 1
    assert pred[3] == 2
    assert pred[4] == 3

    path = get_path(pred)
    assert path == [0, 1, 2, 3, 4]
    assert path_weight(path, d) == 4.0


def test_longest_path_pred_is_integer_vector():
    d = make_d(4, {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
    })
    dp, pred = longest_path(4, d)
    assert dp.shape == (4,)
    assert pred.shape == (4,)
    assert np.issubdtype(pred.dtype, np.integer)


def test_longest_path_does_not_mutate_input():
    d = make_d(4, {
        (0, 1): 2.0,
        (0, 2): 3.0,
        (1, 2): 4.0,
        (1, 3): 1.0,
        (2, 3): 2.0,
    })
    d_copy = d.copy()
    longest_path(4, d)
    assert np.array_equal(d, d_copy)


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

        expected, expected_paths = brute_force_longest_from_0(n, d)

        assert abs(dp[n - 1] - expected) < 1e-9, (n, expected, dp[n - 1])
        assert abs(get_longest_distance(dp) - expected) < 1e-9

        path = get_path(pred)
        if expected == NEG:
            assert path == []
        else:
            assert path[0] == 0 and path[-1] == n - 1
            assert path == sorted(set(path))
            assert abs(path_weight(path, d) - expected) < 1e-9
            assert path in expected_paths


def main():
    tests = [
        test_get_path_simple_chain,
        test_get_path_direct_jump,
        test_get_path_two_nodes,
        test_get_path_single_node,
        test_get_path_with_skips,
        test_get_path_no_path_returns_empty,
        test_get_path_ignores_other_predecessors,
        test_print_path_output,
        test_print_path_two_nodes,
        test_print_path_direct_jump,
        test_print_path_single_node,
        test_get_longest_distance_reads_dp_corner,
        test_get_longest_distance_single_node,
        test_get_longest_distance_unreachable,
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
        test_longest_path_pred_is_integer_vector,
        test_longest_path_does_not_mutate_input,
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
