# RiverPutExperimentsAAAI
This repo archives the code for the experiments on the River FUN algorithm. River FUN computes River PUT in polynomial time. A section on these experiments can be found in the long version of the Paper "Cost-Free Neutrality for the River Voting Method".

## Setup
1. Generate election data by running a_edata_generator.py. Parameters like, n, m and \phi can be adjusted in code. See the cw_ratios.txt to see if the given compute sufficed to find enough CW-free profiles for the given parameters. Adjust the script in 2. or 3. to filter cases where to few elections where generated. 
2. Run voting methods on the generated elections with b_run_voting_methods.py. Which voting methods are run with what timeout can be adjusted in code.
3. Run d_analysis.py to get plots of the resulting data. Adjust the path to point to the data.


## To reproduce our results
Our run of a_edata_generator.py used a seed of 574. Everything else can be run as is. Adjust the path variable in each script to point to the result of the previous script (as the seed is in the file name).
