import os
import re
import json

def parse_stat_file(filepath):
    """
    Parses a gem5 stats.txt file and extracts key metrics. This version is
    more robust to variations in stat names based on the provided sample file.

    Args:
        filepath (str): The path to the stats.txt file.

    Returns:
        dict: A dictionary containing the extracted statistics.
    """
    stats = {}
    
    # Dictionary mapping the desired JSON key to a list of possible
    # stat names found in gem5's stats.txt files. Updated with keys
    # from your stats1.txt file.
    keys_to_find = {
        'sim_seconds': ['simSeconds', 'sim_seconds'],
        'committed_instructions': [
            'simInsts',  # Fallback
            'system.cpu.commitStats0.numInsts', # From your stats file
            'system.cpu.committedInsts', 
            'system.cpu.exec_context.thread_0.numInsts'
        ],
        'ipc': [
            'system.cpu.ipc', 
            'system.cpu.IPC',
            'system.cpu.commitStats0.ipc' # From your stats file
        ],
        'branch_predicted': [
            'system.cpu.branchPred.lookups_0::total', # From your stats file
            'system.cpu.branchPred.lookups'
        ],
        'branch_mispredicted': [
            'system.cpu.branchPred.mispredicted_0::total', # From your stats file
            'system.cpu.branchPred.mispred', 
            'system.cpu.branchPred.mispredicted'
        ],
        'branch_squashes': [ # For squash count
            'system.cpu.branchPred.squashes_0::total' # From your stats file
        ],
        'total_branches': ['system.cpu.commit.branches'],
        'l1d_accesses': [
            'system.cpu.dcache.overall_accesses::total'
            # Note: Not found in your sample, will be null if not in other files
        ],
        'l1i_accesses': [
            'system.cpu.icache.overall_accesses::total'
             # Note: Not found in your sample, will be null if not in other files
        ],
        'rob_occupancy': [
            'system.cpu.rob.occupancy'
             # Note: Not found in your sample, will be null if not in other files
        ],
        'iq_utilization': [
            'system.cpu.iq.rate', 
            'system.cpu.iq.utilization'
             # Note: Not found in your sample, will be null if not in other files
        ],
        'mispred_recovery_cycles': [
            'system.cpu.branchPred.mispredRecoveryCycles'
             # Note: Not found in your sample, will be null if not in other files
        ]
    }

    found_keys = set()

    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0].startswith('-') or len(parts) < 2:
                    continue
                
                stat_from_file = parts[0]

                # Iterate through our desired metrics
                for json_key, possible_names in keys_to_find.items():
                    # Check if this line matches any of the possible stat names
                    for name in possible_names:
                        if stat_from_file == name:
                            try:
                                value = float(parts[1])
                                stats[json_key] = value
                                found_keys.add(json_key)
                                break # Move to the next metric once found
                            except (ValueError, IndexError):
                                stats[json_key] = None
                    if json_key in found_keys:
                        continue
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        return None

    # --- Fill in any missing keys with None ---
    for key in keys_to_find:
        if key not in stats:
            stats[key] = None

    # --- Calculated Metrics ---
    predicted = stats.get('branch_predicted')
    mispredicted = stats.get('branch_mispredicted')

    if predicted and mispredicted is not None and predicted > 0:
        stats['branch_misprediction_rate'] = (mispredicted / predicted) * 100
    else:
        stats['branch_misprediction_rate'] = 0

    return stats

def main(root_dir='.'):
    """
    Walks through the directory structure, parses stat files,
    and generates a JSON summary.

    Args:
        root_dir (str): The root directory containing the simulation results.
    """
    results = {}
    
    # Get all subdirectories in the root, assuming they are branch predictors
    try:
        # Exclude files like '.DS_Store' and the script itself
        predictors = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except FileNotFoundError:
        print(f"Error: Root directory '{root_dir}' not found.")
        return

    for bp in predictors:
        bp_path = os.path.join(root_dir, bp)
        results[bp] = {}
        
        workloads = [d for d in os.listdir(bp_path) if os.path.isdir(os.path.join(bp_path, d))]
        
        for wl in workloads:
            wl_path = os.path.join(bp_path, wl)
            results[bp][wl] = {}
            
            # Find all files matching 'stats*.txt'
            stat_files = [f for f in os.listdir(wl_path) if f.startswith('stats') and f.endswith('.txt')]
            
            for sf in sorted(stat_files):
                run_number = re.findall(r'\d+', sf)
                if run_number:
                    run_id = f"run_{run_number[0]}"
                    filepath = os.path.join(wl_path, sf)
                    
                    print(f"Parsing: {filepath}")
                    
                    parsed_data = parse_stat_file(filepath)
                    if parsed_data:
                        results[bp][wl][run_id] = parsed_data

    # Write the results to a JSON file
    output_filename = 'results.json'
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSuccessfully parsed all files. Results saved to '{output_filename}'")

if __name__ == '__main__':
    # Assumption: You run this script from the directory containing the
    # 'BiModeBP', 'GShareBP', etc. folders.
    main()

