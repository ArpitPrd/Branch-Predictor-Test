import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(file_path):
    """
    Reads simulation results from a JSON file, calculates averages,
    and generates plots to visualize the data.

    Args:
        file_path (str): The path to the input JSON file.
    """
    keys_to_find = {
        'ipc': ['ipc', 'system.cpu.ipc', 'system.cpu.IPC', 'system.cpu.commitStats0.ipc'],
        'branch_predicted': ['branch_predicted', 'system.cpu.branchPred.lookups_0::total', 'system.cpu.branchPred.lookups'],
        'branch_mispredicted': ['branch_mispredicted', 'system.cpu.branchPred.mispredicted_0::total', 'system.cpu.branchPred.mispred'],
    }

    def find_stat(stats_dict, generic_key):
        """Finds a statistic in a dictionary using a list of possible keys."""
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

    # --- Data Aggregation ---
    workload_results = defaultdict(lambda: defaultdict(list))
    for predictor, workloads in data.items():
        for workload_name, runs in workloads.items():
            # Normalize workload names (e.g., "BasicMaths" -> "BasicMath")
            normalized_workload = workload_name.rstrip('s')
            
            for run_name, stats in runs.items():
                ipc = find_stat(stats, 'ipc')
                mispred_rate = stats.get('branch_misprediction_rate')
                
                if mispred_rate is None:
                    predicted = find_stat(stats, 'branch_predicted')
                    mispredicted = find_stat(stats, 'branch_mispredicted')
                    if predicted and mispredicted and predicted > 0:
                        mispred_rate = (mispredicted / predicted) * 100.0
                    else:
                        mispred_rate = 0

                if ipc is not None and mispred_rate is not None:
                    workload_results[normalized_workload][predictor].append({
                        'ipc': ipc,
                        'mispred_rate': mispred_rate
                    })

    # --- Averaging Results ---
    avg_results = defaultdict(dict)
    for workload, predictors in workload_results.items():
        for predictor, runs_data in predictors.items():
            avg_ipc = np.mean([d['ipc'] for d in runs_data])
            avg_mispred_rate = np.mean([d['mispred_rate'] for d in runs_data])
            avg_results[workload][predictor] = {
                'ipc': avg_ipc,
                'mispred_rate': avg_mispred_rate
            }

    # --- Plot Generation ---
    for workload, predictors_data in avg_results.items():
        # Prepare data for plotting
        sorted_predictors = sorted(predictors_data.keys())
        ipc_values = [predictors_data[p]['ipc'] for p in sorted_predictors]
        mispred_rates = [predictors_data[p]['mispred_rate'] for p in sorted_predictors]

        # 1. Bar Chart: IPC for each predictor
        plt.figure(figsize=(12, 7))
        bars = plt.bar(sorted_predictors, ipc_values, color=plt.cm.viridis(np.linspace(0.4, 0.9, len(sorted_predictors))))
        plt.ylabel('Instructions Per Cycle (IPC)')
        plt.title(f'IPC for Different Branch Predictors ({workload})')
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
        plt.show()

        # 2. Line Chart: Misprediction Rate vs. Predictor Complexity
        # NOTE: We create an assumed order of complexity for the x-axis.
        complexity_order = ['Baseline', 'BiModeBP', 'LocalBP', 'GShareBP', 'TournamentBP', 'MultiperspectivePerceptron8KB']
        # Filter and sort the predictors present in the data according to our defined complexity
        plot_predictors = [p for p in complexity_order if p in predictors_data]
        plot_mispred_rates = [predictors_data[p]['mispred_rate'] for p in plot_predictors]
        
        if plot_predictors:
            plt.figure(figsize=(10, 6))
            plt.plot(plot_predictors, plot_mispred_rates, marker='o', linestyle='-', color='crimson')
            plt.xlabel('Assumed Predictor Complexity (Simple to Complex)')
            plt.ylabel('Misprediction Rate (%)')
            plt.title(f'Misprediction Rate vs. Predictor Complexity ({workload})')
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        # 3. Scatter Plot: Misprediction Rate vs. IPC
        plt.figure(figsize=(10, 8))
        # Use a colormap for the points
        colors = plt.cm.plasma(np.linspace(0.1, 1, len(sorted_predictors)))
        plt.scatter(mispred_rates, ipc_values, c=colors, s=100, alpha=0.8, edgecolors='w')
        # Add labels to each point
        for i, predictor_name in enumerate(sorted_predictors):
            plt.annotate(predictor_name, (mispred_rates[i], ipc_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Misprediction Rate (%)')
        plt.ylabel('Instructions Per Cycle (IPC)')
        plt.title(f'IPC vs. Misprediction Rate ({workload})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # You will need to install matplotlib: pip install matplotlib
    generate_plots('results.json')
