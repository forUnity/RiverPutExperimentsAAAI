# RiverPutExperimentsAAAI
This repo archives the code for the experiments on the River FUN algorithm. River FUN computes River PUT in polynomial time. A section on these experiments can be found in the long version of the Paper "Cost-Free Neutrality for the River Voting Method".

## Setup
1. Generate election data by running a_edata_generator.py . Parameters like, n, m and \phi can be adjusted in code. See the cw_ratios.txt to see if the given compute sufficed to find enough CW-free profiles for the given parameters. Adjust the script in 3. to filter those out. 
2. Run voting methods on the generated elections with b_run_voting_methods.py . Which voting methods are run with what timeout, as well as if multiprocessing should be used, can be adjusted in code.
3. Run d_analysis.py to get plots of the resulting data. Adjust the path to point to the data.

## To reproduce our results
Our run of a_edata_generator.py used a seed of 574. Everything else can be run as is.