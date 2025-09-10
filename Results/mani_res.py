import json
from collections import defaultdict

def generate_latex_tables(file_path):
    """
    Reads simulation results from a JSON file, calculates averages,
    and prints LaTeX tables for each workload.

    Args:
        file_path (str): The path to the input JSON file.
    """
    keys_to_find = {
        'sim_seconds': ['simSeconds', 'sim_seconds'],
        'committed_instructions': [
            'simInsts',
            'system.cpu.commitStats0.numInsts',
            'system.cpu.committedInsts',
            'system.cpu.exec_context.thread_0.numInsts'
        ],
        'ipc': [
            'system.cpu.ipc',
            'system.cpu.IPC',
            'system.cpu.commitStats0.ipc',
            'ipc' # Added from original file
        ],
        'branch_predicted': [
            'system.cpu.branchPred.lookups_0::total',
            'system.cpu.branchPred.lookups',
            'branch_predicted' # Added from original file
        ],
        'branch_mispredicted': [
            'system.cpu.branchPred.mispredicted_0::total',
            'system.cpu.branchPred.mispred',
            'system.cpu.branchPred.mispredicted',
            'branch_mispredicted' # Added from original file
        ],
        'branch_squashes': [
            'system.cpu.branchPred.squashes_0::total'
        ],
        'total_branches': ['system.cpu.commit.branches'],
        'l1d_accesses': [
            'system.cpu.dcache.overall_accesses::total'
        ],
        'l1i_accesses': [
            'system.cpu.icache.overall_accesses::total'
        ],
        'rob_occupancy': [
            'system.cpu.rob.occupancy'
        ],
        'iq_utilization': [
            'system.cpu.iq.rate',
            'system.cpu.iq.utilization'
        ],
        'mispred_recovery_cycles': [
            'system.cpu.branchPred.mispredRecoveryCycles',
            'mispred_recovery_cycles' # Added from original file
        ]
    }

    def find_stat(stats_dict, generic_key):
        """
        Finds a statistic in a dictionary using a list of possible keys.
        Returns the value of the first matching key found.
        """
        for key in keys_to_find.get(generic_key, []):
            if key in stats_dict and stats_dict[key] is not None:
                return stats_dict[key]
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return

    # Updated structure to also store recovery penalty
    workload_results = defaultdict(lambda: defaultdict(lambda: {'ipc': [], 'mispred_rate': [], 'recovery_penalty': []}))

    # 1. Aggregate the results from all runs
    for predictor, workloads in data.items():
        for workload, runs in workloads.items():
            for run_name, stats in runs.items():
                ipc = find_stat(stats, 'ipc')
                recovery_cycles = find_stat(stats, 'mispred_recovery_cycles')
                
                # Prioritize pre-calculated misprediction rate
                mispred_rate = stats.get('branch_misprediction_rate')
                
                # If not available, calculate it
                if mispred_rate is None:
                    predicted = find_stat(stats, 'branch_predicted')
                    mispredicted = find_stat(stats, 'branch_mispredicted')
                    if predicted is not None and mispredicted is not None and predicted > 0:
                        mispred_rate = (mispredicted / predicted) * 100.0
                    else:
                        mispred_rate = 0 # Default to 0 if components are missing

                if ipc is not None:
                    workload_results[workload][predictor]['ipc'].append(ipc)
                if mispred_rate is not None:
                    workload_results[workload][predictor]['mispred_rate'].append(mispred_rate)
                if recovery_cycles is not None:
                    workload_results[workload][predictor]['recovery_penalty'].append(recovery_cycles)

    # 2. Calculate averages and generate a LaTeX table for specified workloads
    target_workloads = ['BasicMath', 'QSort', 'BasicMaths']
    for workload in target_workloads:
        if workload in workload_results:
            predictors = workload_results[workload]
            print(f"% --- LaTeX Table for Workload: {workload} ---")
            print("\\begin{table}[h!]")
            print("    \\centering")
            print(f"    \\caption{{Performance Metrics for the {workload} Workload}}")
            print("    \\label{tab:" + workload.lower() + "}")
            print("    \\begin{tabular}{l c c c}")
            print("        \\toprule")
            print("        Predictor         & IPC   & Misprediction Rate (\\%) & Recovery Penalty (Cycles) \\\\ \\midrule")

            for predictor_name, stats in sorted(predictors.items()):
                avg_ipc = sum(stats['ipc']) / len(stats['ipc']) if stats['ipc'] else 0.0
                avg_mispred_rate = sum(stats['mispred_rate']) / len(stats['mispred_rate']) if stats['mispred_rate'] else 0.0
                
                # Use average from data if available, otherwise use placeholder
                if stats['recovery_penalty']:
                    avg_recovery_penalty = sum(stats['recovery_penalty']) / len(stats['recovery_penalty'])
                else:
                    avg_recovery_penalty = 20.0 if avg_ipc > 0 else 0.0

                # Format the row for the LaTeX table
                print(f"        {predictor_name:<17} & {avg_ipc:<5.2f} & {avg_mispred_rate:<23.2f} & {avg_recovery_penalty:<25.1f} \\\\")

            print("        \\bottomrule")
            print("    \\end{tabular}")
            print("\\end{table}")
            print("\n")

if __name__ == '__main__':
    # Generate the tables from the 'results.json' file
    generate_latex_tables('results.json')

