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

## Predictors available in gem5

```
../gem5/src/cpu/pred/tage.hh:class TAGE: public BPredUnit
../gem5/src/cpu/pred/tournament.hh:class TournamentBP : public BPredUnit
../gem5/src/cpu/pred/BranchPredictor.py:    cxx_class = "gem5::branch_prediction::BPredUnit"
../gem5/src/cpu/pred/BranchPredictor.py:class LocalBP(BranchPredictor):
../gem5/src/cpu/pred/BranchPredictor.py:    cxx_class = "gem5::branch_prediction::LocalBP"
../gem5/src/cpu/pred/BranchPredictor.py:class TournamentBP(BranchPredictor):
../gem5/src/cpu/pred/BranchPredictor.py:    cxx_class = "gem5::branch_prediction::TournamentBP"
../gem5/src/cpu/pred/BranchPredictor.py:class BiModeBP(BranchPredictor):
../gem5/src/cpu/pred/BranchPredictor.py:    cxx_class = "gem5::branch_prediction::BiModeBP"
../gem5/src/cpu/pred/bpred_unit.hh:class BPredUnit : public SimObject
../gem5/src/cpu/pred/2bit_local.hh:class LocalBP : public BPredUnit
../gem5/src/cpu/pred/multiperspective_perceptron.hh:class MultiperspectivePerceptron : public BPredUnit
../gem5/src/cpu/pred/bi_mode.hh:class BiModeBP : public BPredUnit
```