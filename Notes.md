## Defs

- Branch Heavy workload - Compiled code contains lots of branch heavy instructions
    - has a very high control flow complexity 
    - performance strongly dependent on branch preduvtion schemes
- Compute Heavy workload
    - High alu sort of computes
    - percentage alu sort / total is greater than branch sort / total 


## How to build

```
/home/arpit/Desktop/iitd/sem_7/COL718/projects/gem5/build/X86/gem5.opt \
        --outdir="out100_/${cpu}_${freq}_${mem}_out" \
        ./config_script.py \
        --cpu_type="$cpu" \
        --cpu_freq="$freq" \
        --mem_type="$mem" \
        --cache_type=MESITwoLevelCacheHierarchy \
        --mem_size=2GiB \
        --mode=$mode
```
## Goal 

- implement serveral brach preduction schemes on gem5
- run them against out of order workloads
- analyse the effect of predictor design on IPC, misprediction rates, overall processor performance 

## File Structure

Based out of: https://www.gem5.org/documentation/gem5art/tutorials/microbench-tutorial
 
- microbench
    - run_micro.py
    - system_configs.py (may have to be changed wrt launch_micro_tests.py)

- launch_micro_test.py

Run the following
```
python3 Branch-Predictor-Test/launch_micro_tests.py
```

We may have to change the directory of launch_micro later

## Step 1

- Build gem5 with out of order cpu model (O3)

- chose a workload 
    - Branch heavy workload:
        - MiBench 
        - parsec
    - Compute heavy 
        - MM
        - fft

- the tutorial runs on benchmarks for SE
- in launch you can edit the bm list for different functionlau

- capture baseline stats

## General Pointer

- deactivate from env using deactivate
- 