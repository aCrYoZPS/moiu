from dataclasses import dataclass

from numpy.random.mtrand import f

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


@dataclass
class BipartileGraph:
    v1: list[int]
    v2: list[int]
    neighbours: dict[int, list[int]]

    def vertice_count(self) -> int:
        return len(self.v1) + len(self.v2)


def is_reachable(g: BipartileGraph, s: int, t: int) -> bool:
    stack: list[int] = [s]
    discovered: list[bool] = [False] * g.vertice_count()
    while len(stack) > 0:
        v = stack.pop()
        if not discovered[v]:
            discovered[v] = True
            for u in g.neighbours[v]:
                stack.append(u)

    return discovered[t]


def extend_with_s_t(g: BipartileGraph) -> BipartileGraph:
    n = g.vertice_count()
    new_neighbours = {0: [v + 1 for v in g.v1]}

    v1 = [v + 1 for v in g.v1] + [n]
    v2 = [0] + [v + 1 for v in g.v2]

    for v in range(1, n + 1):
        new_neighbours[v] = [n + 1 for n in g.neighbours.get(v - 1, [])]
        if v in g.v2:
            new_neighbours[v].append(n)

    return BipartileGraph(v1, v2, new_neighbours)

def find_max_matching(g:BipartileGraph):
    ext_g = extend_with_s_t(g)

def draw_bipartite_graph(
    graph: BipartileGraph,
    *,
    title: str | None = "Bipartite graph",
    edge_color: str = "tab:gray",
    figsize: tuple[float, float] = (7, 6),
    node_size: float = 700,
    show: bool = True,
):
    """Draw a bipartite graph: v1 (blue) on the left, v2 (green) on the right.
    Edges are arrows in both directions between parts (v1 ↔ v2), never within a part.

    Indexing:
      - v1 vertices: 0 .. len(v1)-1
      - v2 vertices: len(v1) .. len(v1)+len(v2)-1
      - neighbours keys/values use these global indices.
    """

    def column_positions(vertices: list[int], x: float) -> dict[int, tuple[float, float]]:
        n = len(vertices)
        return {v: (x, (n - 1) / 2 - i) for i, v in enumerate(vertices)}

    pos1 = column_positions(graph.v1, 0.0)
    pos2 = column_positions(graph.v2, 1.0)

    fig, ax = plt.subplots(figsize=figsize)

    radius_pts = (node_size ** 0.5) / 2

    # Edges: draw arrows for each directed edge (u -> w) where u and w are in different parts
    for u, ws in graph.neighbours.items():
        if u in pos1:
            from_pos, to_pos_dict = pos1, pos2
        elif u in pos2:
            from_pos, to_pos_dict = pos2, pos1
        else:
            raise ValueError(f"Vertex {u} in neighbours is not in v1 or v2")

        for w in ws:
            if w not in to_pos_dict:
                raise ValueError(
                    f"Neighbour {w} of vertex {u} is not in the opposite part. "
                    f"v1 = {graph.v1}, v2 = {graph.v2}"
                )
            arrow = FancyArrowPatch(
                from_pos[u], to_pos_dict[w],
                arrowstyle="-|>",
                mutation_scale=14,
                color=edge_color,
                linewidth=1.4,
                alpha=0.85,
                shrinkA=radius_pts,
                shrinkB=radius_pts,
                zorder=1,
            )
            ax.add_patch(arrow)

    # Nodes
    for positions, color in ((pos1, "tab:blue"), (pos2, "tab:green")):
        if not positions:
            continue
        xs, ys = zip(*positions.values())
        ax.scatter(xs, ys, s=node_size, c=color, edgecolors="black", zorder=2)
        for v, (x, y) in positions.items():
            ax.text(x, y, str(v), ha="center", va="center",
                    color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-0.3, 1.3)
    max_n = max(len(graph.v1), len(graph.v2), 1)
    ax.set_ylim(-(max_n - 1) / 2 - 0.7, (max_n - 1) / 2 + 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax

def main():
    g = BipartileGraph(
            v1=[0, 1, 2, 3],
            v2=[4, 5, 6],
            neighbours={0: [4, 5], 1: [5], 2: [4, 6], 3: [6]},
        )
    _ = draw_bipartite_graph(g)
    ext_g = extend_with_s_t(g)
    _ = draw_bipartite_graph(ext_g)


if __name__ == "__main__":
    main()
