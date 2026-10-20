from dataclasses import dataclass


@dataclass
class BipartileGraph:
    v1: list[int]
    v2: list[int]
    neighbours: dict[int, list[int]]

    def vertice_count(self) -> int:
        return len(self.v1) + len(self.v2)


def find_path(g: BipartileGraph, s: int, t: int) -> list[int]:
    parent: dict[int, int | None] = {s: None}
    stack = [s]
    while stack:
        u = stack.pop()
        if u == t:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in g.neighbours[u]:
            if v not in parent:
                parent[v] = u
                stack.append(v)
    return []


def extend_with_s_t(g: BipartileGraph) -> tuple[BipartileGraph, int, int]:
    n = g.vertice_count()
    s = 0
    t = n + 1

    n1 = len(g.v1)
    left_id = {u: i + 1 for i, u in enumerate(g.v1)}
    right_id = {v: n1 + i + 1 for i, v in enumerate(g.v2)}

    new_neighbours: dict[int, list[int]] = {i: [] for i in range(n + 2)}

    for u in g.v1:
        new_neighbours[s].append(left_id[u])

    for v in g.v2:
        new_neighbours[right_id[v]].append(t)

    for u in g.v1:
        lu = left_id[u]
        for v in g.neighbours.get(u, []):
            new_neighbours[lu].append(right_id[v])

    ext_g = BipartileGraph(
        v1=[left_id[u] for u in g.v1],
        v2=[right_id[v] for v in g.v2],
        neighbours=new_neighbours,
    )
    return ext_g, s, t


def reverse_arcs(g: BipartileGraph, path: list[int]) -> None:
    for a, b in zip(path, path[1:]):
        g.neighbours[a].remove(b)
        g.neighbours[b].append(a)


def find_max_matching(
    g: BipartileGraph,
) -> tuple[list[tuple[int, int]], BipartileGraph]:
    ext_g, s, t = extend_with_s_t(g)

    while True:
        path = find_path(ext_g, s, t)
        if not path:
            break
        reverse_arcs(ext_g, path)

    left = len(g.v1)
    result: list[tuple[int, int]] = []
    for i, v in enumerate(g.v2):
        for b in ext_g.neighbours[left + i + 1]:
            if 1 <= b <= left:
                result.append((g.v1[b - 1], v))
    return result, ext_g


def test1():
    g = BipartileGraph(
        v1=[0],
        v2=[1, 2, 3],
        neighbours={
            0: [1, 2, 3],
        },
    )

    matching, _ = find_max_matching(g)
    print("Matching:", matching)
    print("Size:", len(matching))


def test2():
    g = BipartileGraph(
        v1=[0, 1, 2],
        v2=[3, 4, 5],
        neighbours={
            0: [3],
            1: [3, 4],
            2: [3, 4, 5],
        },
    )

    matching = find_max_matching(g)
    print("Matching:", matching)
    print("Size:", len(matching))


def main():
    test1()
    test2()


if __name__ == "__main__":
    main()
