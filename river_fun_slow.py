from pref_voting.voting_method import  *
import networkx as nx

from enum import Enum
class EdgeState(Enum):
    NONE = 0,
    REJECT = 1,
    FIX = 2,
    BrCh = 3,
    CyCh = 4,
    CyBrCh = 5

flatten = lambda l: [item for sublist in l for item in sublist]


# Check if there is a path from x to y in the fun diagram
def has_path_k(fun_diagram, x, y, at_least_k):
    fun_diagram_at_least_k = fun_diagram.copy()
    # Remove all edges with weight less than larger_k
    fun_diagram_at_least_k.remove_edges_from([(edge[0], edge[1]) for edge in fun_diagram_at_least_k.edges(data=True) if edge[2]['weight'] < at_least_k])

    return nx.has_path(fun_diagram_at_least_k, x, y)

#TODO: rewrite in pref_voting style, optimize, and remove networkx
def river_fun(edata, curr_cands=None, strength_function=None):   
    """
    Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle *and edges in which there is already an edge pointing to the target*.  Break ties using a tie-breaking  linear ordering over the edges.  A candidate is a River winner if it wins according to some tie-breaking rule. See https://electowiki.org/wiki/River.
    Using the River PUT algorithm to determine the winners of the election in Polynomial time.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        strength_function (function, optional): The strength function to be used to calculate the strength of a path.   The default is the margin method of ``edata``.   This only matters when the ballots are not linear orders. 

    Returns: 
        A sorted list of candidates. 

    """
    candidates = edata.candidates if curr_cands is None else curr_cands    

    strength_function = edata.margin if strength_function is None else strength_function    

    cw = edata.condorcet_winner(curr_cands=curr_cands)
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if c1 != c2 and (edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2))]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        sorted_edges = [[e for e in w_edges if e[2] == s] for s in strengths]
        sorted_edges = flatten(sorted_edges)



        #create an empty graph with the same vertices as the margin graph
        fun_diagram = nx.DiGraph()
        fun_diagram.add_nodes_from(candidates)

        #initialize the edge state of each edge to none
        edge_states = {edge: EdgeState.NONE for edge in sorted_edges}


        for e in sorted_edges:
            x,y,k = e
            # Create a directed graph for the fun diagram

            #--branching reject--
            if any(edge[2]['weight'] > k and (edge_states[(edge[0], edge[1], edge[2]['weight'])] == EdgeState.FIX or edge_states[(edge[0], edge[1], edge[2]['weight'])] == EdgeState.BrCh or edge_states[(edge[0], edge[1], edge[2]['weight'])] == EdgeState.CyBrCh) for edge in fun_diagram.in_edges(y, data=True)):
                edge_states[(x,y,k)] = EdgeState.REJECT
                continue
            
            #(compute fun diagram with edges >k)
            fun_diagram_larger_k = fun_diagram.copy() #TODO Optimize this by running update
            #remove all edges with weight less than k
            fun_diagram_larger_k.remove_edges_from([(edge[0], edge[1]) for edge in fun_diagram_larger_k.edges(data=True) if edge[2]['weight'] <= k])

            fun_diagram_larger_k_no_y = fun_diagram_larger_k.copy()
            fun_diagram_larger_k_no_y.remove_edges_from(fun_diagram_larger_k.in_edges(y)) 

            reaching_nodes = nx.bfs_tree(fun_diagram_larger_k_no_y, x, reverse=True).nodes
            fun_diagram_induced = fun_diagram.subgraph(reaching_nodes).copy() #TODO Maybe optimize this 
            #remove all edges incoming to y
            fun_diagram_induced.remove_edges_from(fun_diagram.in_edges(y))

            subgraph_CyCh_edges = []
            for edge in fun_diagram_induced.edges(data=True):
                if edge_states[(edge[0], edge[1], edge[2]['weight'])] == EdgeState.CyCh and has_path_k(fun_diagram_induced, edge[1], edge[0], edge[2]['weight']): #done: FIX ERROR!!! has_path needs to be larger than (0,1)
                    subgraph_CyCh_edges.append((edge[0], edge[1]))
            fun_diagram_induced.remove_edges_from(subgraph_CyCh_edges)

            #check if y is the only node without incoming edges
            nodes_that_could_win = [node for node in fun_diagram_induced.nodes if len(fun_diagram_induced.in_edges(node)) == 0]
            is_y_only_with_no_in_edge = [y] == nodes_that_could_win

            try:
                is_y_x_path = nx.has_path(fun_diagram_induced, y, x)
            except nx.NodeNotFound:
                is_y_x_path = False

            if is_y_only_with_no_in_edge and is_y_x_path:
                edge_states[(x,y,k)] = EdgeState.REJECT
                continue

            #--accepted--
            fun_diagram.add_edge(x, y, weight=k)
            #tentative state
            if len(fun_diagram.in_edges(y)) == 1:
                edge_states[(x,y,k)] = EdgeState.FIX
            elif any(edge[2]['weight'] > k for edge in fun_diagram.in_edges(y, data=True)):
                edge_states[(x,y,k)] = EdgeState.CyBrCh
            else:
                for edge in fun_diagram.in_edges(y, data=True):
                    edge_states[(edge[0], edge[1], edge[2]['weight'])] = EdgeState.BrCh


            #update cycle choice edges
            fun_diagram_larger_equal_k = fun_diagram.copy() #TODO Optimize this by running update
            #remove all edges with weight less than k
            fun_diagram_larger_equal_k.remove_edges_from([(edge[0], edge[1]) for edge in fun_diagram_larger_equal_k.edges(data=True) if edge[2]['weight'] < k])

            fun_diagram_larger_equal_k_no_x = fun_diagram_larger_equal_k.copy()
            fun_diagram_larger_equal_k_no_x.remove_edges_from(fun_diagram_larger_equal_k.out_edges(x))

            fun_diagram_larger_equal_k_no_y = fun_diagram_larger_equal_k.copy()
            fun_diagram_larger_equal_k_no_y.remove_edges_from(fun_diagram_larger_equal_k.in_edges(y))

            Sy_k = nx.bfs_tree(fun_diagram_larger_equal_k_no_x, y, reverse=False).nodes
            S_t_x_k = nx.bfs_tree(fun_diagram_larger_equal_k_no_y, x, reverse=True).nodes
            S = set(Sy_k).intersection(S_t_x_k)
            for edge in fun_diagram.edges(data=True):
                if edge[0] in S and edge[1] in S and edge[2]['weight'] == k:
                    edge_states[(edge[0], edge[1], edge[2]['weight'])] = EdgeState.CyCh

        winners = set()
        for node in fun_diagram.nodes:
            if all(edge_states[edge[0], edge[1], edge[2]['weight']] == EdgeState.CyCh or edge_states[edge[0], edge[1], edge[2]['weight']] == EdgeState.REJECT for edge in fun_diagram.in_edges(node, data=True)):
                winners.add(node)
    return sorted(list(winners))