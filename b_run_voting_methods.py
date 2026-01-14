from pref_voting.io.readers import preflib_to_profile
import os
from typing import Callable, Dict, List, Tuple

import random

from pref_voting.profiles import Profile
from margin_graph_no_lambdas import MarginGraphNoLambdas
from pref_voting.margin_based_methods import river, _ranked_pairs_basic, _ranked_pairs_from_stacks, _split_cycle_floyd_warshall, _stable_voting_with_condorcet_check, _beat_path_floyd_warshall, _beat_path_basic, _split_cycle_basic, _stable_voting_basic
from river_fun_slow import river_fun

import csv

import time

from collections import namedtuple

Result = namedtuple("Result", ["m", "n", "seed", "method", "winners", "process_time_ns", "perf_time_ns"])

def target(args):
    edata, method_name = args
    try:
        import sys
        import importlib
        method = globals().get(method_name)
        try:
            result = run_voting_method(edata, method)
        except MemoryError:
            print(f"Out of memory in method {method_name}")
            return ([], -1, -1)
        return result
    except Exception as e:
        print(f"Error in method {method_name}: {e}")
        return ([], 0, 0)

def async_wrapper(edata, method, timeout_in_s=20):
    import multiprocessing
    with multiprocessing.Pool(processes=1) as pool:
        async_result = pool.apply_async(target, ((edata, method.__name__),))
        try:
            result = async_result.get(timeout=timeout_in_s)
            if result == ([], -1, -1):
                print("Out of memory detected in child process.")
                return [], -1, -1
            return result
        except multiprocessing.TimeoutError:
            print("Timeout")
            pool.terminate()
            pool.join()
            return [], 0, timeout_in_s * 1_000_000_000


def run_voting_method(edata, method):
    tstart = time.perf_counter_ns()
    tstart_process = time.process_time_ns()
    winners = method(edata)
    process_time_ns = time.process_time_ns() - tstart_process
    perf_time_ns = time.perf_counter_ns() - tstart
    return winners, process_time_ns, perf_time_ns


def get_all_M_and_N_in_folder(folder_path : str):
    filenames = os.listdir(folder_path)
    M = set()
    N = set()
    for name in filenames:
        if not name.endswith('.soc'):
            continue
        for part in name.split('_'):
            if part.startswith('m='):
                M.add(int(part.split('=')[1]))
            elif part.startswith('n='):
                N.add(int(part.split('=')[1]))
    return sorted(list(M)), sorted(list(N))

def get_all_N_for_M(folder_path: str, m: int) -> List[int]:
    N = set()
    for dir in os.listdir(folder_path):
        if f"m={m}" not in dir:
            continue
        path_for_m = os.path.join(folder_path, dir) + "/" 
        names = os.listdir(path_for_m)
        for name in names:
            if not name.endswith('.soc'):
                continue
            if f"m={m}" not in name:
                continue
            for part in name.split('_'):
                if part.startswith('n='):
                    N.add(int(part.split('=')[1]))
    return sorted(list(N))

def get_all_m_in_foldernames(folder_path: str) -> List[int]:
    M = set()
    for name in os.listdir(folder_path):
        #name must belong to a folder
        if not name.startswith('m='):
            continue
        M.add(int(name.split('=')[1]))
    return sorted(list(M))

#experiment single load. This is faster but may have sequence effects
LoadedProfile = namedtuple("LoadedProfile", ["filename", "profile"])
def load_data(folder_path: str, m: int, N: list[int]) -> Dict[int, List[LoadedProfile]]:
    profiles_for_m_for_N = {n: [] for n in N}
    folders = os.listdir(folder_path)
    for foldername in folders:
        if not f"m={m}" == foldername:
            continue
        path_for_m = os.path.join(folder_path, foldername) + "/"
        #load selected files in the folder
        print(f"Loading profiles for m={m} in folder {path_for_m}")
        files = os.listdir(path_for_m)
        for filename in files:
            n = int(filename.split('_')[1].split('=')[1])
            if n not in N:
                continue
            print("Loading file:", filename)
            # if f"n={n}" in filename:
            file_path = os.path.join(path_for_m, filename)
            profile = preflib_to_profile(file_path)
            # If the loaded profile is a ProfileWithTies, convert to MarginGraphNoLambdas using its margin matrix
            if hasattr(profile, '__class__') and profile.__class__.__name__ == 'ProfileWithTies':
                margin_matrix = profile.margin_matrix
                candidates = profile.candidates
                w_edges = []
                for i, c1 in enumerate(candidates):
                    for j, c2 in enumerate(candidates):
                        w = margin_matrix[i][j]
                        if w != 0:
                            w_edges.append((c1, c2, w))
                profile = MarginGraphNoLambdas(candidates, w_edges)
            profiles_for_m_for_N[n].append(LoadedProfile(filename,profile))

    return profiles_for_m_for_N

def do_experiment(folder_path: str, m : int, N : list[int], setting_async: bool, timeout_in_s : int):
    OVERWRITE_RESULTS = False
    voting_methods = {
        "RiverFun": river_fun,
        "River": river,
        "RankedPairsBasic": _ranked_pairs_basic,
        #"RankedPairsStacks": _ranked_pairs_from_stacks,
        "SplitCycleFW" :_split_cycle_floyd_warshall,
        # "SplitCycleBasic" : _split_cycle_basic,
        "StableVotingCW" : _stable_voting_with_condorcet_check,
        # "StableVotingBasic" : _stable_voting_basic,
        "BeatPathFW": _beat_path_floyd_warshall,
        # "BeatPathBasic": _beat_path_basic, # <- A lot Slower than others for growing m
    }

    if N is None or len(N) == 0:
        #do all N
        N = get_all_N_for_M(folder_path, m)

    print("Running experiment for m=", m, "N=", N, "async=", setting_async, "timeout_in_s=", timeout_in_s)
    profiles_for_N = load_data(folder_path, m, N)
    print(f"Loaded for n={N} with {[len(profiles_for_N[i]) for i in N]} entries")
    results : list[Result] = [] # m,n, seed, method, winners, process_time_ns

    for n in N:
        for i, profile in enumerate(profiles_for_N[n]):
            for method_name, voting_method in voting_methods.items():
                if not setting_async:
                    winners, process_time_ns, perf_time_ns = run_voting_method(profile.profile, voting_method)
                else:
                    winners, process_time_ns, perf_time_ns = async_wrapper(profile.profile, voting_method, timeout_in_s=timeout_in_s)
                results.append(Result(m, n, i, method_name, winners, process_time_ns, perf_time_ns))
                print(f"Result m={m}, n={n}, profile={i}, method={method_name}, winners={winners}, time={process_time_ns} ns (in s = {process_time_ns / 1_000_000_000}), perf_time={perf_time_ns} ns (in s = {perf_time_ns / 1_000_000_000})")
    

    if setting_async:
        output_file = os.path.join(folder_path, f"results_m={list(M)}_async.csv")
    else:
        output_file = os.path.join(folder_path, f"results_m={list(M)}.csv")
    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    if OVERWRITE_RESULTS:
        write_header = True
    with open(output_file, mode=('w' if OVERWRITE_RESULTS else 'a'), newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["m", "n", "seed", "method", "winners", "process_time_ns, perf_time_ns"])
        for result in results:
            data_name = profiles_for_N[result.n][result.seed].filename
            # Extract seed from the filename
            parts = data_name.split('_')
            seed = int(parts[2].split('=')[1].split('.')[0])
            writer.writerow([result.m, result.n, seed,
                             result.method, 
                             str(result.winners), result.process_time_ns, result.perf_time_ns])
    print(f"Results written to {output_file}")
  

#experiment with single instance. This is slower but cleaner.
def load_single_election(election_path: str) -> Tuple[str, LoadedProfile, int, int,int]:
    if not os.path.exists(election_path):
        raise FileNotFoundError(f"Election file {election_path} does not exist.")
    try:
        profile = preflib_to_profile(election_path)
    except Exception as e:
        print(f"Error loading election data from {election_path}: {e}", file=sys.stderr)
        return "", None, 0, 0, 0
    filename = os.path.basename(election_path)
    m = len(profile.candidates)
    n = profile.num_voters
    seed = int(filename.split('_')[2].split('=')[1].split('.')[0])
    #convert to MarginGraphNoLambdas 
    if hasattr(profile, '__class__') and profile.__class__.__name__ == 'ProfileWithTies':
        margin_matrix = profile.margin_matrix
        candidates = profile.candidates
        w_edges = []
        for i, c1 in enumerate(candidates):
            for j, c2 in enumerate(candidates):
                w = margin_matrix[i][j]
                if w != 0:
                    w_edges.append((c1, c2, w))
        profile = MarginGraphNoLambdas(candidates, w_edges)
        return filename, LoadedProfile(filename, profile), m, n, seed
    else:
        print(f"ERROR : Loaded profile is not a ProfileWithTies.")
        return "", None, m, n, seed

# Shorten filename 
def m_list_name(m_list):
    m_list = sorted(m_list)
    ranges = []
    start = m_list[0]
    end = m_list[0]
    for m in m_list[1:]:
        if m == end + 1:
            end = m
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = m
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return "[" +  ",".join(ranges) + "]"

def do_experiment_atomized(election_path : str, atom_method : tuple[str, Callable], setting_async : bool, timeout_in_s : int):
    print(f"Running experiment for {atom_method[0]} on {election_path} with async={setting_async} and timeout={timeout_in_s}s at {time.strftime('%d %H:%M:%S')}")
    filename, profile,m,n,seed = load_single_election(election_path)
    if profile is None:
        print(f"Skipping experiment for {election_path} due to loading error.")
        return None
    method_name, voting_method = atom_method
    if not setting_async:
        winners, process_time_ns, perf_time_ns = run_voting_method(profile.profile, voting_method)
    else:
        winners, process_time_ns, perf_time_ns = async_wrapper(profile.profile, voting_method, timeout_in_s=timeout_in_s)
    result = Result(m, n, seed, method_name, winners, process_time_ns, perf_time_ns)
    print(f"Result m={m}, n={n}, seed={seed}, method={method_name},    winners={winners}, time={process_time_ns} ns (in s = {process_time_ns / 1_000_000_000}), perf_time={perf_time_ns} ns (in s = {perf_time_ns / 1_000_000_000})")


    m_str = m_list_name(M)
    output_file = os.path.join(folder_path, f"new_results_m={m_str}.csv")
    if setting_async:
        output_file = output_file.replace(".csv", "_async.csv")

    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["m", "n", "seed", "method", "winners", "process_time_ns, perf_time_ns"])
        data_name = filename

        parts = data_name.split('_')
        seed = int(parts[2].split('=')[1].split('.')[0])
        writer.writerow([result.m, result.n, seed,
                        result.method, 
                        str(result.winners), result.process_time_ns, result.perf_time_ns])
    print(f"Result in {output_file}")
    return result
  
def get_m_from_filename(filename: str) -> int:
    return int(filename.split('_')[0].split('=')[1])
def get_n_from_filename(filename: str) -> int:
    return int(filename.split('_')[1].split('=')[1])
def get_relevant_paths(folder_path: str, M: list[int], N: list[int]) -> List[str]:
    paths = []
    folders = os.listdir(folder_path)
    for foldername in folders:
        relevant_M = False
        for m in M:
            if f"m={m}" == foldername:
                relevant_M = True
                break
        if not relevant_M:
            continue
        path_for_m = os.path.join(folder_path, foldername) + "/"
        print(f"Finding paths for m={m} in folder {path_for_m}")
        files = os.listdir(path_for_m)
        for filename in files:
            n = int(filename.split('_')[1].split('=')[1])
            if not N == [] and n not in N:
                continue
            file_path = os.path.join(path_for_m, filename)
            if not file_path.endswith('.soc'):
                continue
            paths.append(file_path)
    return paths

if __name__ == "__main__":
    import sys
    sys.stderr = open("error", "w")
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    folder_path = "data/seed=785_varyn_model=mallowsnorm_phi=0.7_CW=no/"
    setting_async = True
    # timeout_in_s = 30 * 60 #for EXP
    # timeout_in_s = 5 * 60 #for P
    
    allow_timeouts_per_method_and_m = 3

    do_atomized_shuffled = True
    run_same_conditionion_times = 1
    do_all_M_and_N_in_folder = True


    M = [100, 200] # URN m=6 profile 3 River -> Memory Error. River large m -> system crashes
    N = [100]

    if do_all_M_and_N_in_folder:
        M = get_all_m_in_foldernames(folder_path)
        # exclude_m = list(range(13, 51, 1)) + [100, 200, 500, 1000] #for EXP
        exclude_m = [100, 200, 500, 1000]
        # exclude_m =[] #for P
        M = [m for m in M if m not in exclude_m]
        N = []
        print(f"Found M={M} in folder {folder_path}")
    
    if do_atomized_shuffled:
        paths = get_relevant_paths(folder_path, M, N)
        N = sorted(list(set(get_n_from_filename(os.path.basename(path)) for path in paths)))

        voting_methods = {
        "RiverFun": river_fun,
        "River": river, #EXP
        "RankedPairsBasic": _ranked_pairs_basic, #EXP
        # #"RankedPairsStacks": _ranked_pairs_from_stacks, #slower
        "SplitCycleFW" :_split_cycle_floyd_warshall, #P
        # # "SplitCycleBasic" : _split_cycle_basic, #slower
        "StableVotingCW" : _stable_voting_with_condorcet_check, #P
        # # "StableVotingBasic" : _stable_voting_basic, #slower
        "BeatPathFW": _beat_path_floyd_warshall, #P
        # # "BeatPathBasic": _beat_path_basic, # <- A lot Slower than others for growing m
        }

        time_out_in_s_per_method = { "RiverFun": 5*60, "River": 30*60, "RankedPairsBasic": 30*60, "SplitCycleFW": 5*60, "StableVotingCW": 5*60, "BeatPathFW": 5*60}
        do_timeout_n_wise_per_method = { "RiverFun": True, "River": False, "RankedPairsBasic": False, "SplitCycleFW": True, "StableVotingCW": True, "BeatPathFW": True}
        limit_method_max_m = { "River": 12, "RankedPairsBasic": 12 }

        exclude_where_less_than_20_profiles_found = True
        if exclude_where_less_than_20_profiles_found:
            num_profiles_per_m_and_n = {}
            for path in paths:
                m = get_m_from_filename(os.path.basename(path))
                n = get_n_from_filename(os.path.basename(path))
                if (m,n) not in num_profiles_per_m_and_n:
                    num_profiles_per_m_and_n[(m, n)] = 0
                num_profiles_per_m_and_n[(m, n)] += 1
            new_paths = [path for path in paths if num_profiles_per_m_and_n[(get_m_from_filename(os.path.basename(path)), get_n_from_filename(os.path.basename(path)))] >= 20]
            removed_paths = [path for path in paths if path not in new_paths]
            paths = new_paths
            print(f"Removed {len(removed_paths)} paths with less than 20 profiles found.")
        

        gen = ((path, (method_name, method)) for path in paths for method_name, method in voting_methods.items())
        gen = list(gen)
        gen = [g for g in gen if not (g[1][0] in limit_method_max_m and limit_method_max_m[g[1][0]] < get_m_from_filename(os.path.basename(g[0])))]
        if run_same_conditionion_times > 1:
            #dublicate every item in the list run_same_conditionion_times times
            gen = gen * run_same_conditionion_times
        #shuffle generator
        random.shuffle(gen)
        timeouts_per_method_and_m_and_n = {(method_name, m, n): 0 for method_name in voting_methods.keys() for m in M for n in N}
        timeouts_per_method_and_m = {(method_name, m): 0 for method_name in voting_methods.keys() for m in M}

        info_done_counter = 0
        info_vals_to_avg = [1_000_000_000 * 10] * 30
        for path, atom_method in gen:
            test_time = time.perf_counter_ns()
            # Check if timeouts for this method and any smaller m exceed the allowed limit
            method_name = atom_method[0]
            m = get_m_from_filename(os.path.basename(path))
            n = get_n_from_filename(os.path.basename(path))
 
            if  timeouts_per_method_and_m_and_n[(method_name, m,n)] > allow_timeouts_per_method_and_m if do_timeout_n_wise_per_method[method_name] else timeouts_per_method_and_m[(method_name, m)] > allow_timeouts_per_method_and_m:
                print(f"Skipping {method_name} for m={m} due to exceeded timeouts.")
                info_done_counter += 1
                info_vals_to_avg.append(0)
                info_vals_to_avg.pop(0)
                continue
            result = do_experiment_atomized(path, atom_method, setting_async, time_out_in_s_per_method[method_name])
            if result.winners == []:
                #timeout
                print(f"Timeout for method {method_name} and m={m}.")
                if do_timeout_n_wise_per_method[method_name]:
                    timeouts_per_method_and_m_and_n[(method_name, m, n)] += 1
                else:
                    timeouts_per_method_and_m[(atom_method[0], result.m)] += 1
            info_test_duration = time.perf_counter_ns() - test_time
            info_done_counter += 1
            info_vals_to_avg.append(info_test_duration)
            info_vals_to_avg.pop(0)
            info_rolling_average_time_needed = sum(info_vals_to_avg) / len(info_vals_to_avg)
            percent_done = (info_done_counter / len(gen)) * 100 if len(gen) > 0 else 0
            avg_time_s = info_rolling_average_time_needed / 1_000_000_000
            time_left_h = (info_rolling_average_time_needed * (len(gen) - info_done_counter) / 1_000_000_000) / (60*60)
            print(f"Atom duration: {info_test_duration / 1_000_000_000} s")

            print(f"Done {info_done_counter}/{len(gen)} ({percent_done:.2f}%) avg time per {avg_time_s:.2f} s so {time_left_h:.2f}h left")

    else:
        skip_M = []
        for m in M:
            if m in skip_M:
                continue
            do_experiment(folder_path, m, N, setting_async, 5 * 60)

    print("------------Done")

#Notes: Stable voting and Beat Path slow for m>=50