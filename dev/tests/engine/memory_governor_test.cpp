#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

// The governor reads nothing from the backend but its memory statistics, so
// this stand-in lets the tests set them without a GPU.
namespace splash::metal {
namespace {
MetalMemoryStats statistics;
} // namespace

struct MetalBackend::Impl {};
MetalBackend::MetalBackend(std::string, double)
    : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;
MetalMemoryStats MetalBackend::memoryStats() const noexcept {
  return statistics;
}
MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
  return statistics;
}
} // namespace splash::metal

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

// A Mac whose recommended working set is numerator/denominator of its memory,
// as Metal reports it (2/3 at 32 GB, 3/4 at 36 GB).
DeviceCapabilities mac(uint64_t physicalGiB, uint64_t numerator,
                       uint64_t denominator) {
  DeviceCapabilities device;
  device.deviceName = "governor-test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = physicalGiB * kGiB;
  device.recommendedMaxWorkingSetBytes =
      device.physicalMemoryBytes / denominator * numerator;
  device.maxBufferLengthBytes = device.recommendedMaxWorkingSetBytes;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  return device;
}

// Grows one request's KV extent by extent, as PageStorage maps it, and
// returns the pages the governor granted.
uint32_t grantKvPages(MemoryGovernor &governor,
                      const EngineMemoryBreakdown &budget) {
  uint32_t granted = 0;
  while (granted < budget.kvVirtualPages) {
    const uint32_t pages =
        std::min(budget.kvExtentPages, budget.kvVirtualPages - granted);
    const uint64_t bytes = uint64_t{pages} * budget.kvPageBytes;
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      break;
    metal::statistics.sparseResidentBytes += bytes;
    metal::statistics.deviceCurrentAllocatedBytes += bytes;
    reservation->commit();
    granted += pages;
  }
  return granted;
}

// The plan budgets the memory Metal holds outside the backend's buffers
// (pipelines, driver allocations) inside its reserves and advertises the
// context the rest of the budget holds. One request must reach that context
// whatever part of the reserves this memory takes; memory beyond them is
// still charged, and the device never exceeds the hard budget.
void testAdvertisedContextIsGrantable() {
  struct Machine {
    const char *name;
    uint64_t physicalGiB, numerator, denominator;
    kv::Format format;
    uint32_t advertisedTokens;
  };
  for (const Machine &machine :
       {Machine{"32 GB INT8", 32, 2, 3, kv::Format::Int8, 69'625},
        Machine{"36 GB INT8", 36, 3, 4, kv::Format::Int8, 253'945},
        Machine{"36 GB BF16", 36, 3, 4, kv::Format::BFloat16, 129'241}}) {
    // The 27B with its draft and vision tower: 16.2 GiB of weights.
    ModelMemoryProfile model =
        test::modelMemoryProfile(15 * kGiB, kGiB / 2, 7 * kGiB / 10);
    model.targetKvLayout.format = machine.format;
    const EngineMemoryPlan plan = requireEngineMemoryPlan(
        mac(machine.physicalGiB, machine.numerator, machine.denominator),
        model);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    require(plan.maximumContextTokens() == machine.advertisedTokens,
            std::string(machine.name) + ": unexpected advertised context");
    const uint64_t reserves =
        budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
    for (const uint64_t untracked :
         {uint64_t{0}, 64 * kMiB, 150 * kMiB, 300 * kMiB, reserves,
          reserves + 256 * kMiB}) {
      // After warmup the weights, the arenas and the request's state cell
      // are the backend's buffers; Metal holds the untracked bytes besides.
      metal::statistics = {};
      metal::statistics.allocatedBytes =
          budget.targetWeightsBytes + budget.draftWeightsBytes +
          budget.visionWeightsBytes + budget.sharedPrefillBytes +
          budget.sharedDecodeBytes + budget.activeStateCellBytes;
      metal::statistics.deviceCurrentAllocatedBytes =
          metal::statistics.allocatedBytes + untracked;
      metal::MetalBackend backend("unused");
      // Configured as RuntimeResources configures it.
      MemoryGovernor governor(
          backend, budget.hardBudgetBytes, 2 * kGiB,
          [] { return std::optional<uint64_t>(200 * kGiB); }, reserves);
      const uint32_t granted = grantKvPages(governor, budget);
      const std::string context = std::string(machine.name) + ", " +
                                  std::to_string(untracked / kMiB) +
                                  " MiB untracked: granted " +
                                  std::to_string(granted) + " of " +
                                  std::to_string(budget.kvVirtualPages) +
                                  " KV pages";
      require(metal::statistics.deviceCurrentAllocatedBytes <=
                  budget.hardBudgetBytes,
              context + ", beyond the hard budget");
      if (untracked <= reserves)
        require(granted == budget.kvVirtualPages,
                context + ", short of the advertised context");
      else
        require(granted < budget.kvVirtualPages,
                context + ", memory beyond the reserves was not charged");
    }
  }
}

} // namespace

int main() {
  try {
    testAdvertisedContextIsGrantable();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
