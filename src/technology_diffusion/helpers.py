"""Graph generation and diffusion helpers."""

import os
import sys
import time
import random
from typing import Iterable, Mapping, Optional

import networkx as nx
import numpy as np


def suppress_print() -> None:
    """Redirect standard output to the null device.

    The original stream is saved so that :func:`resume_print` can restore it.
    Calls should therefore be paired with ``resume_print()``.
    """
    sys._stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")


def resume_print() -> None:
    """Restore the standard-output stream saved by :func:`suppress_print`."""
    if hasattr(sys, "_stdout"):
        sys.stdout.close()
        sys.stdout = sys._stdout


def time_exceeded(start_time: float, max_time: Optional[float]) -> bool:
    """Return whether a wall-clock budget measured from ``start_time`` expired.

    A ``None`` budget disables the time limit.
    """
    return max_time is not None and (time.perf_counter() - start_time) > max_time


def make_vector(init_seeds: Iterable[int], n_nodes: int) -> np.ndarray:
    """Encode a seed set as a length-``n_nodes`` binary vector.

    Parameters
    ----------
    init_seeds:
        Node indices that should be marked active.
    n_nodes:
        Total number of nodes in the graph.
    """
    x0 = np.zeros(n_nodes, dtype=float)
    x0[list(init_seeds)] = 1.0
    return x0


def create_pa_graph(
    n_nodes: int,
    c: int,
    seed: Optional[int] = None,
    init_nodes: Optional[int] = None,
    init_mode: str = "complete",
) -> tuple[nx.Graph, dict[int, int]]:
    """Generate a preferential-attachment graph and node thresholds.

    New nodes attach to existing nodes with probability proportional to their
    degree. Each node receives a ``theta`` attribute sampled from multiples of
    ``c`` within the graph-size range.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the generated graph.
    c:
        Spacing between possible threshold values.
    seed:
        Seed used for both graph and threshold random-number generators.
    init_nodes:
        Size of the initial graph. Defaults to ``max(2, c)``.
    init_mode:
        Structure of the initial graph: ``"complete"`` or ``"tree"``.

    Returns
    -------
    tuple
        The generated graph and a mapping from node index to threshold.

    Raises
    ------
    ValueError
        If ``c``, ``init_nodes``, or ``init_mode`` is invalid.
    """
    if c <= 0:
        raise ValueError("c must be a positive integer.")
    if init_nodes is None:
        init_nodes = max(2, c)
    if not (2 <= init_nodes <= n_nodes):
        raise ValueError("init_nodes must satisfy 2 <= init_nodes <= n_nodes")

    rng_graph = np.random.default_rng(seed)
    rng_theta = np.random.default_rng(seed)

    if init_mode == "complete":
        g = nx.complete_graph(init_nodes)
    elif init_mode == "tree":
        g = nx.random_labeled_tree(init_nodes, seed=seed)
    else:
        raise ValueError("init_mode must be 'complete' or 'tree'")

    m_choices = (1, 2, 3, 4)

    for u in range(init_nodes, n_nodes):
        g.add_node(u)
        m = int(rng_graph.choice(m_choices))
        existing = np.array([v for v in g.nodes if v != u], dtype=int)
        m = min(m, len(existing))
        degrees = np.array([g.degree(v) for v in existing], dtype=float)
        probs = degrees / degrees.sum()
        targets = rng_graph.choice(existing, size=m, replace=False, p=probs)
        for v in targets:
            g.add_edge(u, int(v))

    k_max = int(np.ceil(n_nodes / c))
    values = list(range(max(2, c), min(k_max * c, n_nodes) + 1, c))
    theta = {v: int(rng_theta.choice(values)) for v in g.nodes()}
    nx.set_node_attributes(g, theta, name="theta")
    return g, theta

def make_subset_connected(
    g: nx.Graph,
    x: Iterable[int],
    use_thetas: int = 0,
    thetas: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Connect selected nodes while preserving the selection's cardinality.

    The selected terminals are first joined through a minimum spanning tree of
    their shortest-path metric closure. Non-articulation nodes are then removed
    until the original number of selected nodes is restored.

    Parameters
    ----------
    g:
        Graph containing the selected nodes.
    x:
        Selected node indices.
    use_thetas:
        If truthy, prefer removing nodes with lower thresholds.
    thetas:
        Thresholds used to rank removal candidates when ``use_thetas`` is
        enabled.

    Returns
    -------
    tuple
        Sorted node indices of the resulting subset.
    """
    s = set(x)
    target_size = len(s)

    if not s:
        return tuple()

    if len(s) > 1:
        terminals = sorted(s)
        closure = nx.Graph()
        closure.add_nodes_from(terminals)

        for i, u in enumerate(terminals[:-1]):
            lengths, paths = nx.single_source_dijkstra(g, source=u)
            for v in terminals[i + 1 :]:
                if v in lengths:
                    closure.add_edge(u, v, weight=float(lengths[v]), path=paths[v])

        if closure.number_of_edges() > 0:
            # The terminal metric closure identifies a compact collection of
            # original-graph paths that connects the selected nodes.
            mst = nx.minimum_spanning_tree(closure, weight="weight")
            connected_nodes = set(s)
            for u, v in mst.edges():
                connected_nodes.update(closure[u][v]["path"])
            s = connected_nodes

    while len(s) > target_size:
        induced = g.subgraph(s)
        articulation = set(nx.articulation_points(induced)) if len(s) > 2 else set()
        candidates = [node for node in s if node not in articulation]

        if not candidates:
            break

        # Removing a non-articulation node preserves connectivity of the
        # current induced subgraph.
        if use_thetas and thetas is not None:
            candidates.sort(key=lambda node: (thetas[node], node))
        else:
            candidates.sort()

        s.remove(candidates[0])

    return tuple(sorted(s))


def connected_component_update(
    g: nx.Graph,
    active_mask: np.ndarray,
    thetas: Mapping[int, int] | np.ndarray,
) -> np.ndarray:
    """Compute the nodes activated by one synchronous diffusion round.

    An inactive node observes each distinct neighboring active component once.
    It activates when the total size of those components is at least
    ``theta[v] - 1``; the extra node itself then brings the merged component to
    its threshold.

    Parameters
    ----------
    g:
        Diffusion graph with nodes indexed from ``0`` to ``n - 1``.
    active_mask:
        Boolean-like vector identifying nodes active before this round.
    thetas:
        Activation threshold for each node.

    Returns
    -------
    numpy.ndarray
        Boolean mask containing only the nodes newly activated this round.
    """
    n_nodes = g.number_of_nodes()
    actives = np.zeros(n_nodes, dtype=bool)

    if not np.any(active_mask):
        return actives

    active_nodes = np.flatnonzero(active_mask)
    comp_id_of = {}
    comp_sizes = []
    active_set = set(active_nodes)
    visited = set()
    cid = 0

    for u in active_nodes:
        if u in visited:
            continue
        stack = [u]
        visited.add(u)
        comp_nodes = [u]

        while stack:
            x = stack.pop()
            for nbr in g.neighbors(x):
                if nbr in active_set and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
                    comp_nodes.append(nbr)

        comp_sizes.append(len(comp_nodes))
        for w in comp_nodes:
            comp_id_of[w] = cid
        cid += 1

    comp_sizes = np.asarray(comp_sizes, dtype=int)
    inactive_nodes = np.flatnonzero(~active_mask)

    for v in inactive_nodes:
        # A component may touch v through several neighbors, but its size must
        # contribute only once to the reachable active population.
        touched = set()
        for u in g.neighbors(v):
            if active_mask[u]:
                c = comp_id_of.get(u)
                if c is not None:
                    touched.add(c)

        if touched:
            reachable = int(comp_sizes[list(touched)].sum())
            if reachable >= (int(thetas[v])-1):
                actives[v] = True

    return actives


def connected_component_spread(
    g: nx.Graph,
    x0: np.ndarray,
    thetas: Mapping[int, int] | np.ndarray,
    max_t: int = 1000,
) -> tuple[int, list[int], np.ndarray]:
    """Run component-based diffusion until convergence or ``max_t`` rounds.

    Parameters
    ----------
    g:
        Diffusion graph with nodes indexed from ``0`` to ``n - 1``.
    x0:
        Binary-like vector identifying the initially active seed nodes.
    thetas:
        Activation threshold for each node.
    max_t:
        Maximum number of synchronous update rounds.

    Returns
    -------
    tuple
        Final number of active nodes, active-count history, and the final
        binary activation vector.
    """
    n_nodes = g.number_of_nodes()
    active = (x0 > 0).astype(bool)
    spread_hist = [int(np.sum(active))]

    for _ in range(max_t):
        prev = active.copy()
        new = connected_component_update(g, prev, thetas)
        active = prev | new
        if np.array_equal(active, prev):
            break
        spread_hist.append(int(np.sum(active)))

    final_spread = int(np.sum(active))
    final_x = np.zeros(n_nodes, dtype=float)
    final_x[active] = 1.0
    return final_spread, spread_hist, final_x
