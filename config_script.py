from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory.single_channel import (
    SingleChannelDDR3_1600,
    SingleChannelDDR4_2400,
    SingleChannelHBM
)
from gem5.components.memory.multi_channel import DualChannelDDR3_1600
import argparse
from gem5.components.processors.simple_processor import SimpleProcessor
from m5.objects import DerivO3CPU
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import (
    PrivateL1CacheHierarchy,
)
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import (
    MESITwoLevelCacheHierarchy,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from m5.objects import BiModeBP, LocalBP, TournamentBP
import sys
import csv
import time
import json

def get_branch_predictor(bp_type: str):
    """
    Return the branch predictor object based on name.
    Current support:
        - bimode
        - local
        - tournament
    """
    if bp_type.lower() == "bimode":
        return BiModeBP()
    elif bp_type.lower() == "local":
        return LocalBP()
    elif bp_type.lower() == "tournament":
        return TournamentBP()
    else:
        print(f"Branch predictor {bp_type} not supported")
        sys.exit(1)


def get_simulator(board:any) -> any:
    """
    return the simulator
    """
    return Simulator(board)

def get_binary_file(filename:str) -> any:
    """
    returns the binary file modidy any change of obtaining resources only inside this
    currently supports:\n
        1. BinaryResource
    outsourcing not supported
    """
    return BinaryResource(filename)

def get_cache_hierarchy(cache_type:str):
    """
    keep adding support for additional caches
    current support for:\n
        1. NoCache NoCache()
        2. PrivateL1CacheHierarchy
        3. MESITwoLevelCacheHierarchy (built from Ruby and works the best)
    """

    if cache_type=="NoCache":
        return NoCache()
    
    if cache_type=="PrivateL1CacheHierarchy":
        return PrivateL1CacheHierarchy(
            l1d_size="32kB",
            l1i_size="32kB",
        )

    if cache_type=="MESITwoLevelCacheHierarchy":
        return MESITwoLevelCacheHierarchy(
            l1d_size="16KiB",
            l1d_assoc=8,
            l1i_size="16KiB",
            l1i_assoc=8,
            l2_size="256KiB",
            l2_assoc=16,
            num_l2_banks=1,
        )


    else:
        print(f"cache {cache_type} Not Supported")
        sys.exit()
    
def get_memory(mem_type:str, size:str):
    """
    keep adding support for additional memory types

    current support for:\n
        1. DDR3 SingleChannelDDR3_1600(size)
        2. DDR4 SingleChannelDDR4_2400(size)
        3. HBM SingleChannelHBM_1000(size) (have not tested, shall be tested)
        4. DualDDR3 DualChannelDDR3_1600(size)
    """
    if mem_type=="DDR3":
        return SingleChannelDDR3_1600(size)
    if mem_type=="DDR4":
        return SingleChannelDDR4_2400(size)
    if mem_type=="HBM":
        return SingleChannelHBM(size)
    if mem_type=="DualDDR3":
        return DualChannelDDR3_1600(size)
    else:
        print(f"mem type {mem_type} not supported")
        sys.exit()
    
def prepare_board(freq:str, processor:any, memory:any, cache_hierarchy:any) -> any:
    """
    prepares a board with above listed features
    """
    return SimpleBoard(
        clk_freq=freq,
        processor=processor,
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )

def set_workload(board:any, type_of_workload:str, binary:any) -> None:
    """
    sets your binary workload
    current support for 
        1. SE board.set_se_binary_workload(binary)
    we do not load the entire OS
    """
    if type_of_workload=="SE":
        board.set_se_binary_workload(binary)
    
    else:
        print(f"{type_of_workload} not supported")
        sys.exit()

def get_cpu_type(cpu_type:str) -> any:
    """
    keep adding more cpu types
    current support for
        1. ATMOIC cpu_types.ATOMIC
    """
    if cpu_type=="ATOMIC":
        return CPUTypes.ATOMIC
    if cpu_type=="O3":
        return CPUTypes.O3
    if cpu_type=="TIMING":
        return CPUTypes.TIMING
    else:
        print(f"cpu type {cpu_type} not supported")
    
def get_isa(isa_type:str) -> any:
    """
    keep adding more isa_types
    current support for 
        1. X86
    no cross compiler support, cross compiling should be done at the users end and provide us with the binary
    """
    if isa_type=="X86":
        return ISA.X86
    else:
        print(f"isa type {isa_type} not supported")
        sys.exit()

def get_processor(
    processor_type: str,
    cpu_type: str,
    num_cores: int,
    isa_type: str,
    bp_type: str = None,
) -> SimpleProcessor:
    _cpu_type = get_cpu_type(cpu_type)
    _isa = get_isa(isa_type)

    if processor_type.upper() == "SIMPLE":
        processor = SimpleProcessor(
            cpu_type=_cpu_type, num_cores=num_cores, isa=_isa
        )

        if _cpu_type == CPUTypes.O3 and bp_type:
            for core in processor.cores:
                # Create a new branch predictor for the core.
                branch_predictor = get_branch_predictor(bp_type)
                
                # **THE FIX**: Manually set numThreads to resolve the proxy.
                # In SE mode, each O3 core has one thread.
                branch_predictor.numThreads = 1
                
                # Assign the fully configured predictor to the core.
                core.branch_predictor = branch_predictor

        return processor

    else:
        print(f"processor type {processor_type} not supported")
        sys.exit(1)


def prepare_simulator(
        processor_type:str,
        cpu_type:str, 
        num_cores:int, 
        isa_type:str, 
        mem_type:str,
        mem_size:str,
        cache_type:str,
        cpu_freq:str,
        binary_filename:str,
        type_of_workload:str,
        bp_type:str = None
    ) -> any:
    """
    Breif:
        sets up the entire system for run, provides one with the following\n
        - motherboard
        - processor
        #assumes no cache hierarchy (new version does) deprecated
    Args:
        processor_type: [SIMPLE]
        cpu_type: [ATOMIC, TIMING, O3]
        num_cores >= 1
        isa_type: [X86]
        mem_type: [DDR3, DDR4, HBM, DualDDR3]
        mem_size: in GiB
        cache_type: [NoCache, PrivateL1CacheHierarchy, MESITwoLevelCacheHierarchy]
        cpu_freq: in MHz
        binary_filename: must be X86 compatible as of now
        type_of_workload: only SE no FS (takes a lot of time)
    
    Reuturns:
        A simulator

    Waarnings:
        make sure to use within the boundaries defined above
    """
    processor = get_processor(processor_type, cpu_type=cpu_type, num_cores=num_cores, isa_type=isa_type, bp_type=bp_type)

    cache_hierarchy = get_cache_hierarchy(cache_type)
    memory = get_memory(mem_type, mem_size)
    
    board = prepare_board(cpu_freq, processor, memory, cache_hierarchy)

    binary = get_binary_file(binary_filename)
    
    set_workload(board, type_of_workload, binary)

    simulator = get_simulator(board)
    
    return simulator

def _run_simulator(simulator) -> None:
    """
    general purpose doc for running a simulator
    """
    print("running the simulation")
    simulator.run()

def run_simulation(
        processor_type:str,
        cpu_type:str, 
        num_cores:int, 
        isa_type:str, 
        mem_type:str,
        mem_size:str,
        cache_type:str,
        cpu_freq:str,
        binary_filename:str,
        type_of_workload:str,
        bp_type:str = None
    ) -> list:
    """
    simulator runner
    support for:\n
    Args:
        processor_type: [SIMPLE]
        cpu_type: [ATOMIC, TIMING, O3]
        num_cores >= 1
        isa_type: [X86]
        mem_type: [DDR3, DDR4, HBM, DualDDR3]
        mem_size: in GiB
        cache_type: [NoCache, PrivateL1CacheHierarchy, MESITwoLevelCacheHierarchy]
        cpu_freq: in MHz
        binary_filename: must be X86 compatible as of now
        type_of_workload: only SE no FS (takes a lot of time)
    """
    simulator = prepare_simulator(processor_type,
        cpu_type,
        num_cores,
        isa_type,
        mem_type,
        mem_size,
        cache_type,
        cpu_freq,
        binary_filename,
        type_of_workload,
        bp_type
    )
    
    _run_simulator(simulator)

def _get_stats(filename: str):
    """
    Reads gem5 stats.txt and returns {stat_name: float_value}.
    Skips histogram table rows that don't follow the key-value format.
    """
    # filename="m5out/hello-world.cpp"
    stats = {}
    f = open(filename, 'r')
    lines = f.readlines()
    print(f"lines length = {lines}")
    for line in lines:
        print(line)
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        parts = line.split()
        # normal stat lines have: key, value, "#", comment...
        if len(parts) >= 2 and parts[1].replace('.', '', 1).replace(',', '').isdigit():
            key = parts[0]
            val = float(parts[1].replace(",", ""))
            stats[key] = val
    f.close()
    return stats

def compile_stats(stats:dict) -> tuple:
    """
    extracts ueful information
    as of now extracts only ipc, sim_seconds
    """
    ipc = stats.get("board.processor.cores.core.ipc", 0)
    sim_seconds = stats.get("simSeconds", 0)

    return (ipc, sim_seconds)

def get_stats(filename:str) -> tuple:
    """
    wrapper to get and compile
    """
    stats = _get_stats(filename)
    return compile_stats(stats)

def save(lists_to_save:list, field_names:list, save_loc:str, mode:str):
    """
    general purpose saver
    """
    with open(save_loc, mode=mode, newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        # writer.writeheader()
        for list_to_save in lists_to_save:
            dict_to_save = {}
            for i, field_name in enumerate(field_names):
                dict_to_save[field_name] = list_to_save[i]
            writer.writerow(dict_to_save)

        

    print(f"file saved @ {save_loc}")

def run_multiple_simulation(
        processor_type,
        cpu_types:list[str],
        num_cores:int,
        isa_type:str,
        mem_types:list[str],
        mem_sizes:list[str],
        cache_type:str,
        cpu_freqs:list[str],
        binary_filename:str,
        type_of_workload:str,
        bp_type:str
    ) -> list:
    """
    Args:
        processor_type: [SIMPLE]
        cpu_type: [ATOMIC, TIMING, O3], list
        num_cores >= 1
        isa_type: [X86]
        mem_type: [DDR3, DDR4, HBM, DualDDR3] list
        mem_size: in GiB list
        cache_type: [NoCache, PrivateL1CacheHierarchy, MESITwoLevelCacheHierarchy]
        cpu_freq: in MHz list
        binary_filename: must be X86 compatible as of now
        type_of_workload: only SE no FS (takes a lot of time)

    Returns:
        list of result, where each result contains the following
            (cpu_type, cpu_freq, mem_type, mem_size, ipc, sim_seconds)

    Warnings:
        No warnings as of now
    """
    results = []
    for cpu_type in cpu_types:
        for cpu_freq in cpu_freqs:
            for mem_type, mem_size in zip(mem_types, mem_sizes):
                print(f"working on {cpu_type} @ freq={cpu_freq} with mem: {mem_type} of size={mem_size}")
                run_simulation(
                    processor_type,
                    cpu_type,
                    num_cores,
                    isa_type,
                    mem_type,
                    mem_size,
                    cache_type,
                    cpu_freq,
                    binary_filename,
                    type_of_workload,
                    bp_type
                )

                ipc, sim_seconds = get_stats("m5out/stats.txt")
                results.append(
                    [cpu_type, cpu_freq, mem_type, ipc, sim_seconds]
                )

    return results

def args_to_dict(args):

    info = {
        "processor_type":"SIMPLE",
        "cpu_type":args.cpu_type,
        "num_cores":1,
        "isa_type":"X86",
        "mem_type":args.mem_type,
        "mem_size":args.mem_size,
        "cache_type":args.cache_type,
        "cpu_freq":args.cpu_freq,
        "binary_filename":"mm",
        "type_of_workload":"SE",
        "bp_type":args.bp_type
    }

    return info
    

parser = argparse.ArgumentParser(description="pass the required information to set the memory up")
parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file")
args = parser.parse_args()

# load JSON config
with open(args.config, "r") as f:
    args_dict = json.load(f)

run_simulation(
    args_dict["processor_type"],
    args_dict["cpu_type"],
    args_dict["num_cores"],
    args_dict["isa_type"],
    args_dict["mem_type"],
    args_dict["mem_size"],
    args_dict["cache_type"],
    args_dict["cpu_freq"],
    args_dict["binary_filename"],
    args_dict["type_of_workload"],
    args_dict["bp_type"]
)
