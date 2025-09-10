#ifndef __CPU_PRED_GSHARE_HH__
#define __CPU_PRED_GSHARE_HH__

#include "cpu/pred/bpred_unit.hh"
#include "params/GShareBP.hh"
#include "base/types.hh"
#include "cpu/static_inst.hh"
#include "arch/generic/pcstate.hh"
#include <vector>
#include <cstdint>

namespace gem5 {
namespace branch_prediction {

/**
 * GShare Branch Predictor
 *
 * Uses global history XORed with PC bits to index into a table of saturating counters.
 */
class GShareBP : public BPredUnit
{
  public:
    // Constructor required for SimObject params system
    GShareBP(const GShareBPParams &params);

    ~GShareBP() = default;

    // Override the pure virtual functions from BPredUnit
    bool lookup(ThreadID tid, Addr pc, void * &bp_history) override;

    void updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                        Addr target, const StaticInstPtr &inst, void * &bp_history) override;

    void update(ThreadID tid, Addr pc, bool taken, void * &bp_history,
                bool squashed, const StaticInstPtr &inst, Addr target) override;

    void squash(ThreadID tid, void * &bp_history) override;

    // Note: reset() is not a virtual function in BPredUnit, so don't mark it override
    void reset();

  private:
    struct HistSnapshot {
        uint64_t gh_snapshot;
        uint32_t idx;
        bool pred;
    };

    const unsigned historyBits;
    const unsigned counterBits;
    const unsigned numEntries;
    const uint8_t  initCounter;
    const uint32_t indexMask;
    const uint8_t  ctrMax;
    const uint8_t  ctrThreshold;

    std::vector<uint8_t> pht;    // Pattern history table
    std::vector<uint64_t> ghr;   // Per-thread global history

    inline uint32_t index(Addr pc, ThreadID tid) const {
        uint32_t pc_idx = static_cast<uint32_t>((pc >> 2) & indexMask);
        uint32_t gh_idx = static_cast<uint32_t>(ghr[tid] & indexMask);
        return (pc_idx ^ gh_idx) & indexMask;
    }

    inline bool counterTaken(uint8_t c) const { return c >= ctrThreshold; }

    inline void specUpdateGH(ThreadID tid, bool taken) {
        if (historyBits >= 64) {
            ghr[tid] = (ghr[tid] << 1) | (taken ? 1ULL : 0ULL);
        } else {
            ghr[tid] = ((ghr[tid] << 1) | (taken ? 1ULL : 0ULL)) &
                       ((1ULL << historyBits) - 1ULL);
        }
    }

    inline void commitCounter(uint32_t idx, bool taken) {
        uint8_t &c = pht[idx];
        if (taken) {
            if (c < ctrMax) ++c;
        } else {
            if (c > 0) --c;
        }
    }
};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_GSHARE_HH__
