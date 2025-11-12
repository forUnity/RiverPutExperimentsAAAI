"""
    A multiprocessing-safe shortend version of MarginGraph.
"""

import networkx as nx
from pref_voting.weighted_majority_graphs import MajorityGraph

class MarginGraphNoLambdas(MajorityGraph):
    def strength_matrix(self, curr_cands=None, strength_function=None):
        """
        Return the strength matrix of the profile. The strength matrix is a matrix where the entry in row i and column j is the number of voters that rank the candidate with index i over the candidate with index j. If curr_cands is provided, then the strength matrix is restricted to the candidates in curr_cands. If strength_function is provided, then the strength matrix is computed using the strength function.
        """
        import numpy as np
        if curr_cands is not None:
            cindices = list(range(len(curr_cands)))
            def cindex_to_cand(cidx):
                return curr_cands[cidx]
            def cand_to_cindex(c):
                return curr_cands.index(c)
            strength_function = self.margin if strength_function is None else strength_function
            strength_matrix = np.array([
                [strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices]
                for a_idx in cindices
            ])
        else:
            cindices = self.cindices
            cindex_to_cand = self.cindex_to_cand
            cand_to_cindex = self.cand_to_cindex
            strength_matrix = np.array(self.margin_matrix) if strength_function is None else np.array([
                [strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices]
                for a_idx in cindices
            ])
        return strength_matrix, cand_to_cindex
    """A margin graph is a weighted asymmetric directed graph. This version avoids lambdas for multiprocessing safety."""
    def __init__(self, candidates, w_edges, cmap=None):
        mg = nx.DiGraph()
        mg.add_nodes_from(candidates)
        for c1, c2, w in w_edges:
            mg.add_edge(c1, c2, weight=w)
        self.mg = mg
        self.cmap = cmap if cmap is not None else {c: str(c) for c in candidates}
        self.candidates = list(candidates)
        self.num_cands = len(self.candidates)
        self.cindices = list(range(self.num_cands))
        self._cand_to_cindex = {c: i for i, c in enumerate(self.candidates)}
        self._cindex_to_cand = {i: c for i, c in enumerate(self.candidates)}
        # No lambdas, use methods

    def cand_to_cindex(self, c):
        return self._cand_to_cindex[c]

    def cindex_to_cand(self, i):
        return self._cindex_to_cand[i]

    def margin(self, c1, c2):
        # Return the margin (weight) from c1 to c2, or 0 if no edge
        return self.mg[c1][c2]["weight"] if self.mg.has_edge(c1, c2) else 0

    def majority_prefers(self, c1, c2):
        return self.margin(c1, c2) > 0

    def is_tied(self, c1, c2):
        return self.margin(c1, c2) == 0 and self.margin(c2, c1) == 0

    @property
    def margin_matrix(self):
        return [
            [self.margin(self.cindex_to_cand(i), self.cindex_to_cand(j)) for j in self.cindices]
            for i in self.cindices
        ]