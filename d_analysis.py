from b_run_voting_methods import Result, m_list_name
from collections import namedtuple

def load_results_from_file(file_path : str) -> list[Result]:
    import os
    import csv
    results = []
    print(f"Loading results from {file_path}")
    if not os.path.exists(file_path):
        print(f"No results file found at {file_path}.")
        return results
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        #contains header that should match Result namedtuple
        header = next(reader)
        for row in reader:
            m = int(row[0])
            n = int(row[1])
            seed = int(row[2])
            method = row[3]
            winners = row[4]
            process_time_ns = int(row[5])
            perf_time_ns = int(row[6])
            results.append(Result(m, n, seed, method, winners, process_time_ns, perf_time_ns))
    return results

def load_results(folder_path: str, M : list[int], load_all : bool = False, load_async_results : bool = True) -> list[Result]:
    import os
    import csv
    results = []
    # output_file = os.path.join(folder_path, f"results_m={list(M)}.csv")
    output_file = os.path.join(folder_path, f"new_results_m={m_list_name(M)}.csv")
    if load_all:
        output_file = os.path.join(folder_path, f"results_m=ALL.csv")
    if load_async_results:
        output_file = output_file.replace(".csv", "_async.csv")
    print(f"Loading results from {output_file}")
    if not os.path.exists(output_file):
        print(f"No results file found at {output_file}.")
        return results
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        #contains header that should match Result namedtuple
        header = next(reader)
        for row in reader:
            m = int(row[0])
            n = int(row[1])
            seed = int(row[2])
            method = row[3]
            winners = row[4]
            process_time_ns = int(row[5])
            perf_time_ns = int(row[6])
            results.append(Result(m, n, seed, method, winners, process_time_ns, perf_time_ns))
    return results

def rename_method(results: list[Result], old_name : str, new_name :str) -> list[Result]:
    return [
        Result(
        r.m, r.n, r.seed, new_name if r.method == old_name else r.method,
        r.winners, r.process_time_ns, r.perf_time_ns
        )
        for r in results
    ]

def bar_plot_number_of_timeouts(results : list[Result]) -> None:
    import matplotlib.pyplot as plt
    from collections import defaultdict
    # Count timeouts per method and m
    timeouts_count = defaultdict(lambda: defaultdict(int))
    for result in results:
        if result.winners == '[]':
            timeouts_count[result.method][result.m] += 1

    # Prepare data for plotting
    methods = list(timeouts_count.keys())
    m_values = sorted(set(m for method in timeouts_count.values() for m in method.keys()))
    counts = {method: [timeouts_count[method].get(m, 0) for m in m_values] for method in methods}

    # Plotting
    plt.figure(figsize=(12, 8))
    for method, count in counts.items():
        plt.plot(m_values, count, label=method)
    plt.xlabel("m")
    plt.ylabel("Number of Timeouts")
    plt.title("Number of Timeouts per Method and m")
    plt.legend()
    plt.grid()
    plt.show()


def analyze_results(results: list[Result]) -> None:
    import matplotlib.pyplot as plt
    from collections import defaultdict
    # Group results by method and m
    grouped_results = defaultdict(lambda: defaultdict(list))

    filter_out_timeouts = False
    filter_max_timeouts = False
    cross_out_timeout_datapoints = False
    print_timeouts = True
    timeout_line = True

    # Option to compute average performance time per method per m
    average_per_m = True 
    # Option to plot time on a log scale
    log_scale = True 
    # Option to differentiate different n
    differentiate_n = True 
    print_only_max_timeouts = True



    timeout_limit_to_remove_other_results = 0
    timeout_results = [r for r in results if r.winners == '[]']
    timeout_marker = 'x'
    timeout_per_method_and_m_and_n = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in timeout_results:
        timeout_per_method_and_m_and_n[result.method][result.m][result.n].append(result)

    deemph_results = []
    for result in results:
        if filter_out_timeouts and result.winners == '[]':
            continue
        if filter_max_timeouts and len(timeout_per_method_and_m_and_n[result.method][result.m][result.n]) > timeout_limit_to_remove_other_results:
            continue
        if not filter_max_timeouts and len(timeout_per_method_and_m_and_n[result.method][result.m][result.n]) > timeout_limit_to_remove_other_results:
            deemph_results.append(result)

        grouped_results[result.method][result.m].append(result)

    deemph_method_and_m_and_n = [ (r.method, r.m, r.n) for r in deemph_results]

    if average_per_m:
        if differentiate_n:
            # Compute average per method, m, n
            for method in grouped_results:
                for m in grouped_results[method]:
                    # Group by n within each (method, m)
                    n_groups = {}
                    for r in grouped_results[method][m]:
                        r_n = r.n[0] if isinstance(r.n, tuple) else r.n
                        n_groups.setdefault(r_n, []).append(r)
                    grouped_results[method][m] = [
                        Result(
                            m,
                            n,
                            0,  # seed
                            method,
                            res[0].winners,
                            0,  # process_time_ns
                            sum(r.perf_time_ns for r in res) / len(res)
                        )
                        for n, res in n_groups.items()
                    ]
        else:
            for method in grouped_results:
                for m in grouped_results[method]:
                    res = grouped_results[method][m]
                    avg_perf_time = sum(r.perf_time_ns for r in res) / len(res)
                    template = res[0]
                    grouped_results[method][m] = [Result(
                        template.m,
                        0, 0,
                        template.method,
                        template.winners,
                        0,
                        avg_perf_time
                    )]

    plt.figure(figsize=(12, 8))
    # colors = plt.cm.get_cmap('tab10', len(grouped_results))
    colors = plt.cm.get_cmap('tab10', 10)
    color_per_method = { "River FUN": colors(2),
                        "River": colors(1),
                        "Ranked Pairs": colors(0),
                        "BeatPathFW": colors(4),
                        "StableVotingCW": colors(5),
                        "SplitCycleFW": colors(6)
                        }

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>']

    if differentiate_n:
        n_values = set(r.n for method in grouped_results.values() for m_results in method.values() for r in m_results)
        n_values = sorted(n_values)
        n_marker_map = {n: markers[i % len(markers)] for i, n in enumerate(n_values)}

        for idx, (method, m_results) in enumerate(grouped_results.items()):
            timeout_crossout_points_x = []
            timeout_crossout_points_y = []
            for n in n_values:
                ms = []
                perf_times = []
                alpha = []
                for m, res in m_results.items():
                    for r in res:
                        if r.n == n:
                            ms.append(m)
                            perf_times.append(r.perf_time_ns)
                            alpha.append(0.2 if (r.method, r.m, r.n) in deemph_method_and_m_and_n else 0.7)  # Deemphasize timeouts
                            if (r.method, r.m, r.n) in deemph_method_and_m_and_n:
                                timeout_crossout_points_x.append(m)
                                timeout_crossout_points_y.append(r.perf_time_ns)
                                
                if ms:
                    label = f"{method} on n={n}"
                    marker = n_marker_map[n]
                    
                    plt.scatter(ms, perf_times, label=label, color=color_per_method[method], alpha=alpha, marker=marker)
            if print_timeouts and timeout_crossout_points_x:
                if cross_out_timeout_datapoints:
                    plt.scatter(
                        timeout_crossout_points_x,
                        timeout_crossout_points_y,
                        label=f"{method} exceeded a timeout on that n",
                        color=color_per_method[method],
                        alpha=.7,
                        marker=timeout_marker,
                        s=18
                    ) 
                else:
                    timeout_ms = []
                    timeout_perf_times = []
                    for n in n_values:
                        timeout_ms += [timeout.m for timeout in timeout_results if timeout.n == n and timeout.method == method]
                        if print_only_max_timeouts:
                            timeout_ms = [m for m in timeout_ms if len(timeout_per_method_and_m_and_n[method][m]) > timeout_limit_to_remove_other_results]
                            # timeout_ms = list(set(timeout_ms))
                        if timeout_ms:
                            timeout_time = [r.perf_time_ns for r in timeout_results if r.method==method][0]
                            timeout_perf_times = [timeout_time] * len(timeout_ms)
                    if timeout_ms:
                        plt.scatter(
                            timeout_ms,
                            timeout_perf_times,
                            label=f"{method} exceeded a timeout",
                                # color='red',
                                # label=f"{method} exceeded timeouts",
                                color=color_per_method[method],
                                alpha=1,
                                marker=timeout_marker,
                                s = 18
                        )
    else:
        for idx, (method, m_results) in enumerate(grouped_results.items()):
            ms = []
            perf_times = []
            for m, res in m_results.items():
                for r in res:
                    ms.append(m)
                    perf_times.append(r.perf_time_ns)
            if ms:
                plt.scatter(ms, perf_times, label=method, color=color_per_method[method], alpha=0.7, marker=markers[idx % len(markers)])
            if print_timeouts:
                timeout_ms = [r.m for r in timeout_results]
                timeout_perf_times = [r.perf_time_ns for r in timeout_results]
                plt.scatter(timeout_ms, timeout_perf_times, label="Timeouts", color='red', alpha=0.5, marker=timeout_marker)

    if log_scale:
        plt.yscale('log')
    
    if timeout_line:
        # Add a red line for the timeout value at y = 5 minutes
        timeout_ns = 5 * 60 * 1_000_000_000  # 5 minutes in nanoseconds
        plt.axhline(timeout_ns, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Timeout (5 min)')

    plt.xlabel('Number of Candidates m', fontsize=14)
    plt.ylabel('Running Time (in nanoseconds)', fontsize=14)
    # plt.title('River vs RiverFun on profiles without Condorcet winners')
    plt.title('River FUN vs Other Polynomial Methods on Mallows Profiles Without Condorcet Winners', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.legend(fontsize=10)

    plt.grid(True)
    plt.tight_layout()

    # Show only integer m on x axis
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    #save plot to file at folder_path
    import os
    plot_file = os.path.join(export_results_folder, "plot.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    plt.show()




def scatter_data_exp(results: list[Result]) -> None:
    import matplotlib.pyplot as plt
    from collections import defaultdict
    # Group results by method and m
    grouped_results = defaultdict(lambda: defaultdict(list))

    # Option to plot time on a log scale
    log_scale = True 

    timeout_limit_to_remove_other_results = 3
    timeout_results = [r for r in results if r.winners == '[]']
    timeout_marker = 'x'
    timeout_per_method_and_m = defaultdict(lambda: defaultdict(list[Result]))
    for result in timeout_results:
        timeout_per_method_and_m[result.method][result.m].append(result)

    timeout_result = []
    for result in results:
        if len(timeout_per_method_and_m[result.method][result.m]) > timeout_limit_to_remove_other_results:
            timeout_result.append(result)
        grouped_results[result.method][result.m].append(result)

    deemph_method_and_m = [ (r.method, r.m) for r in timeout_result]

    plt.figure(figsize=(12, 8))
    # colors = plt.cm.get_cmap('tab10', len(grouped_results))
    colors = plt.cm.get_cmap('tab10', 10)
    color_per_method = { "River FUN": colors(2),
                        "River (RV-PUT)": colors(1),
                        "Ranked Pairs (RP-PUT)": colors(0),
                        "BeatPathFW": colors(4),
                        "StableVotingCW": colors(5),
                        "SplitCycleFW": colors(6)
                        }
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>']

   
    for idx, (method, m_results) in enumerate(grouped_results.items()):
        ms = []
        perf_times = []
        for m, res in m_results.items():
            for r in res:
                ms.append(m)
                perf_times.append(r.perf_time_ns)

        plt.scatter(ms, perf_times, alpha=0.7, label=method, color=color_per_method[method], marker=markers[idx % len(markers)])

        timeout_for_method = timeout_per_method_and_m[method]
        if timeout_for_method:
            timeout_for_method = {m: res for m, res in timeout_for_method.items() if len(res) > timeout_limit_to_remove_other_results}
            if not timeout_for_method:
                continue
            timeout_ms = []
            timeout_perf_times = []
            for m, res in timeout_for_method.items():
                readability_offset = 0.02
                timeout_ms += [r.m + readability_offset if r.method == "Ranked Pairs (RP-PUT)" else r.m - readability_offset for r in res]
                timeout_perf_times += [r.perf_time_ns for r in res]
            if timeout_ms:
                plt.scatter(
                    timeout_ms,
                    timeout_perf_times,
                    label=f"{method} exceeded 3 timeouts",
                    color=color_per_method[method],
                    alpha=1,
                    marker=timeout_marker,
                    s=120
                )

    if log_scale:
        plt.yscale('log')
    
    timeout_ns = 30 * 60 * 1_000_000_000  # 30 minutes in nanoseconds
    plt.axhline(timeout_ns, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Timeout (30 min)')
    plt.axhline(1_000_000_000, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    plt.xlabel('Number of Candidates m', fontsize=14)
    plt.ylabel('Running Time (in nanoseconds)', fontsize=14)
    # plt.title('River vs RiverFun on profiles without Condorcet winners')
    plt.title('River FUN vs River and Ranked Pairs on Mallows Profiles Without Condorcet Winners', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Show only integer m on x axis
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    #save plot to file at folder_path
    import os
    plot_file = os.path.join(export_results_folder, "plot_exp.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    plt.show()

def analyze_p_methods() -> None: 
    #ANALYZE P METHODS
    file_path = "data/seed=785_varyn_model=mallowsnorm_phi=0.7_CW=no/new_results_m=[5-50]_async.csv"
    results = load_results_from_file(file_path)

    filter_for_methods = ["RiverFun","BeatPathFW", "StableVotingCW", "SplitCycleFW"]
    results = [r for r in results if r.method in filter_for_methods]

    #incomplete data for this n because of aborted generation
    filter_out_N = [500]
    results = [r for r in results if r.n not in filter_out_N]

    #removed pairs with less generated instances according to the cw_ratios
    # remove_m_and_n_pair = [(5,100), (5,200), (6,200), (7,200), (8,200), (9,200)]
    # results = [r for r in results if (r.m, r.n) not in remove_m_and_n_pair]

    results = rename_method(results, "RankedPairsBasic", "Ranked Pairs")
    results = rename_method(results, "RiverFun", "River FUN")


    print(f"Loaded {len(results)} results from {file_path}")
    analyze_results(results)

def analyze_exp_methods() -> None:
    #ANALYZE EXPERIMENT METHODS
    file_path = "data/seed=785_varyn_model=mallowsnorm_phi=0.7_CW=no/new_results_m=[5-50]_async.csv"
    results = load_results_from_file(file_path)


    filter_for_methods = ["RiverFun","RankedPairsBasic","River"]
    results = [r for r in results if r.method in filter_for_methods]

    filter_for_M = list(range(5, 13))
    results = [r for r in results if r.m in filter_for_M]

    #incomplete data for this n because of aborted generation
    # filter_out_N = [500]
    # results = [r for r in results if r.n not in filter_out_N]

    #removed pairs with less generated instances according to the cw_ratios
    # remove_m_and_n_pair = [(5,100), (5,200), (6,200), (7,200), (8,200), (9,200)]
    # results = [r for r in results if (r.m, r.n) not in remove_m_and_n_pair]

    results = rename_method(results, "RankedPairsBasic", "Ranked Pairs (RP-PUT)")
    results = rename_method(results, "RiverFun", "River FUN")
    results = rename_method(results, "River", "River (RV-PUT)")
    print(f"Loaded {len(results)} results from {file_path}")
    scatter_data_exp(results)

export_results_folder = "export/"
import os
if not os.path.exists(export_results_folder):
    os.makedirs(export_results_folder, exist_ok=True)

if __name__ == "__main__":
    analyze_p_methods()
    analyze_exp_methods()

