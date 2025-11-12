import os
from b_run_voting_methods import load_data, Result, get_relevant_paths, load_single_election, get_m_from_filename
from d_analysis import load_results

from collections import namedtuple

from pref_voting.weighted_majority_graphs import MarginGraph

def sets_of_tied_edges(profile):
    #convert profile to margin graph
    mg = MarginGraph.from_profile(profile)
    #get all edges with the same margin
    tied_edges = {}
    for edge in mg.edges:
        margin = edge[2]
        if margin not in tied_edges:
            tied_edges[margin] = []
        tied_edges[margin].append(edge)
    #filter out margins with only one edge
    tied_edges = {margin: edges for margin, edges in tied_edges.items() if len(edges) > 1}
    return tied_edges


if __name__ == "__main__":
    folder_path = "data/seed=574_varyn_model=mallowsnorm_phi=0.35_CW=no/"
    M = list(range(5,51,1)) #[5,6,7,8,9,10]#, 20, 50, 100, 200]
    N = [10, 50, 100, 200]

    ###-----Load Profiles
    #Check that no profiles are identical
    paths = get_relevant_paths(folder_path, M, N)
    for m in M:
        paths_for_m = [path for path in paths if get_m_from_filename(os.path.basename(path)) == m]
        all_profiles_for_m = []
        for path in paths_for_m:
            p = load_single_election(path)
            all_profiles_for_m.append(p)
        #pairwise check all profiles for identity. Find identical profiles.
        print(f"Loaded {len(all_profiles_for_m)} profiles for m={m}")
        identical_profiles = []
        for i, profile in enumerate(all_profiles_for_m):
            for j in range(i+1, len(all_profiles_for_m)):
                if profile == all_profiles_for_m[j]:
                    identical_profiles.append((i, j))
        print("Identical profiles found?:")
        print(f"m={m}, identical profiles: {identical_profiles}")

    ###-----Load Results
    results = load_results(folder_path, M)
    print(f"Loaded {len(results)} results from {folder_path}")
    #--------Check that for the same m and profile, the river and river_fun results are the same
    results_with_same_m_and_seed = [set([r1,r2]) for r1 in results for r2 in results if r1.m == r2.m and r1.profile_reference == r2.profile_reference and r1.method != r2.method]
    all_results_match = True
    for result in results_with_same_m_and_seed:
        river_result = [r for r in result if r.method == "River"][0]
        river_fun_result = [r for r in result if r.method == "RiverFun"][0]
        if river_result is None or river_fun_result is None:
            print(f"Skipping result {result} because one of the methods is missing.")
            continue
        if river_result.winners != river_fun_result.winners:
            print(f"Discrepancy found for m={river_result.m}, profile={river_result.profile_reference}: River winners: {river_result.winners}, RiverFun winners: {river_fun_result.winners}")
            all_results_match = False
    print(f"All results match: {all_results_match}")

        

    
    


