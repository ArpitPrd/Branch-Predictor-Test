from m5.objects.BranchPredictor import BranchPredictor
from m5.params import *


class GShareBP(BranchPredictor):
    type = "GShareBP"
    cxx_class = "gem5::branch_prediction::GShareBP"
    cxx_header = "cpu/pred/gshare.hh"

    # Predictor configuration parameters
    tableEntries = Param.Unsigned(8192, "Number of PHT entries")
    historyBits = Param.Unsigned(12, "Length of global history")
    counterBits = Param.Unsigned(2, "Saturating counter bits")
    initCounter = Param.Unsigned(1, "Initial counter value (0..2^bits-1)")
