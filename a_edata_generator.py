import pref_voting
# from pref_voting.generate_profiles import * 
from pref_voting.generate_weighted_majority_graphs import *
from pref_voting.generate_profiles import generate_profile

from pref_voting.io.writers import write_preflib

from b_run_voting_methods import m_list_name
import random

import os

root_dir = "data/"
skip_CW_winners = True
give_up_after_tries_for_m = 50_000
def generate_election(seed, num_candidates, num_voters):
    # Mallows 
    # beware: mallows with variying n may mislead. Boehmer et al. 2023. Use 0.35 as closest to some real elections
    profiles = generate_profile(num_candidates, num_voters,
                        probmodel="mallows",
                        phi=0.35,#0.5
                        normalise_phi=True,
                        seed=seed)
    #Urn
    # profiles = generate_profile(num_candidates, num_voters,
    #                     probmodel="URN-R",
    #                     seed=seed)
    return profiles

def generate_data():
    num_profiles_each = 20 
    N = [10, 50, 100, 200] #, 500, 1000]
    M = list(range(5,51, 1))# + [100, 200]


    #this range was chosen to not lengthen the filename too much
    seed_seed = random.randint(0, 1_000) # we got 574 here on our first run.
    random.seed(seed_seed)
    output_dir = root_dir + f"seed={seed_seed}_varyn_model=mallowsnorm_phi=0.35_CW=no/"
    os.makedirs(output_dir, exist_ok=True)

    for n in N:
        print(f"Generating data for n={n}....")
        number_generated_profiles_per_m = {m: 0 for m in M}
        numer_CW_generated_profiles_per_m = {m: 0 for m in M}
        for m in M:
            m_dir = output_dir + f"m={m}/"
            os.makedirs(m_dir, exist_ok=True)
            tries = 0
            for i in range(num_profiles_each):
                while tries < give_up_after_tries_for_m:
                    seed = random.randint(0, 1_000_000_000)
                    profile = generate_election(seed, m, n)
                    number_generated_profiles_per_m[m] += 1
                    tries += 1
                    if profile.condorcet_winner() is None or not skip_CW_winners:
                        filename = f"m={m}_n={n}_seed={seed}"
                        write_preflib(profile, m_dir + filename)
                        print(f"Saved election with seed={seed}, num_candidates={m}, num_voters={n}")
                        numer_CW_generated_profiles_per_m[m] += 1
                        break
        
            if skip_CW_winners:
                cw_ratio_per_m = numer_CW_generated_profiles_per_m[m] / number_generated_profiles_per_m[m]
                #save the ratio of skipped profiles to txt
                with open(output_dir + f"cw_ratio_n={n}_m={m_list_name(M)}.txt", "a") as f:
                    f.write(f"n={n} m={m}: {cw_ratio_per_m} from {numer_CW_generated_profiles_per_m[m]} / {number_generated_profiles_per_m[m]}\n")



def sort_data_into_folders_by_m(folder_path):
    """
    Sorts the data in the folder by m and creates subfolders for each m.
    """
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".soc"):
            parts = file.split('_')
            m = int(parts[0].split('=')[1])
            m_dir = os.path.join(folder_path, f"m={m}")
            os.makedirs(m_dir, exist_ok=True)
            os.rename(os.path.join(folder_path, file), os.path.join(m_dir, file))
            print(f"Moved {file} to {m_dir}")

if __name__ == "__main__":
    generate_data()
    print("Data generation complete.")
