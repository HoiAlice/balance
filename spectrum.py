"""Algorithms for processing strongly separated families of [T] subsets."""
import networkx as nx
import numpy as np
import cdd
from itertools import combinations

def strongly_le(left_spec: tuple[int], right_spec: tuple[int]) -> bool:
    """Defines << order over spectrum."""

    size = len(left_spec)
    if len(right_spec) != size:
        raise ValueError("Both spectrum should be over same set.")
    left_indexes = {index for index in range(size) if left_spec[index] == 1 and right_spec[index] == 0}
    right_indexes = {index for index in range(size) if left_spec[index] == 0 and right_spec[index] == 1}
    if len(left_indexes) == 0 or len(right_indexes) == 0:
        return True
    return max(left_indexes) < min(right_indexes)


def get_all_subsets(size: int) -> list[tuple]:
    """Returns list of all subsets of [T]."""

    if size <= 0:
        raise ValueError("Size should be positive.")
    subsets = []
    for i in range(1 << size):
        subset = tuple((i >> j) & 1 for j in range(size))
        subsets.append(subset)
    return subsets


def tiling_spectrum(size: int) -> list[tuple[tuple[int]]]:
    """Returns list of all (id, sigma)-tiling's spectrum."""
    # NOTE: 7 is a max size my computer able to process
    subsets = get_all_subsets(size + 1)
    graph = nx.Graph()
    for left_index in range(len(subsets)):
        for right_index in range(left_index + 1, len(subsets)):
            left_spec = subsets[left_index]
            right_spec = subsets[right_index]
            if (strongly_le(left_spec, right_spec)
                or strongly_le(right_spec, left_spec)):
                graph.add_edge(left_index, right_index)

    mis_list = list(nx.find_cliques(graph))
    spec_list = {tuple(sorted(subsets[i][:-1] for i in mis if subsets[i][-1] == 0)) for mis in mis_list}
    return list(spec_list)


def get_inequalities(spec: list[tuple]):
    """Returns all normals to spectrum cone."""
    A = -1 * np.array(spec)
    b = np.zeros((A.shape[0], 1))
    H = np.hstack((b, A))

    mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)
    
    return [tuple(-int(x) for x in row[1:]) for row in ext.array if row[0] == 0]


class SpectrumParityGraph:
    """Parity graph associated with a tiling spectrum."""

    def __init__(self, spectrum: list[tuple[int]]):
        if not spectrum:
            raise ValueError("Spectrum should be non-empty.")
        self._spectrum = spectrum
        self._spectrum_set = set(spectrum)
        self._size = len(spectrum[0])

    @staticmethod
    def _flip_bits(spec: tuple[int], indexes: tuple[int]) -> tuple[int]:
        """Flip bits in given positions."""
        spec = list(spec)
        for i in indexes:
            spec[i] ^= 1
        return tuple(spec)

    def _build_graph(self) -> nx.Graph:
        """
        Build graph on components [0..n-1].

        Vertices i,j are connected if there exists an even X such that:
          X, X ∪ {i}, X ∪ {i, j} ∈ spectrum
          with i ≠ j and i,j ∉ X.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self._size))

        for X in self._spectrum:
            # X must be even
            if sum(X) % 2 != 0:
                continue

            zero_indexes = [i for i in range(self._size) if X[i] == 0]

            # IMPORTANT: ordered pairs (i, j)
            for i in zero_indexes:
                Xi = self._flip_bits(X, (i,))
                if Xi not in self._spectrum_set:
                    continue

                for j in zero_indexes:
                    if j == i:
                        continue

                    Xij = self._flip_bits(X, (i, j))
                    if Xij in self._spectrum_set:
                        graph.add_edge(i, j)

        return graph

    def is_bipartite(self) -> bool:
        """Check whether the parity graph is bipartite."""
        graph = self._build_graph()
        return nx.is_bipartite(graph)
    

class TemplateFinder:
    """Check if a template spectrum is contained in a tiling spectrum."""

    def __init__(self, template: tuple[tuple[int]]):
        if not template:
            raise ValueError("Template should be non-empty.")
        self._template = template
        self._template_size = len(template[0])
        self._template_set = set(template)

    def exists_in(self, tiling_spectrum: list[tuple[int]]) -> bool:
        """
        Check whether the template exists in the given tiling spectrum.

        Elements of the tiling spectrum can have higher dimension than the template.
        Method tries all combinations of indices to project tiling elements to template dimension.
        """
        if not tiling_spectrum:
            return False

        n = len(tiling_spectrum[0])
        k = self._template_size

        if k > n:
            return False

        # Try all combinations of k coordinates from n
        for idx in combinations(range(n), k):
            projected_set = {tuple(spec[i] for i in idx) for spec in tiling_spectrum}
            if self._template_set.issubset(projected_set):
                return True
        return False
