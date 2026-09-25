#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
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

// Available host memory is what macOS can hand out without compressing or
// swapping: free pages and pageable file-backed and purgeable pages.
void testHostAvailabilityCountsReclaimablePages() {
  constexpr uint64_t pageSize = 16384;
  auto availablePages = [](const HostMemoryPages &pages) {
    return estimateHostAvailableMemory(pages, pageSize) / pageSize;
  };
  // free_count includes the speculative pages; they are file-backed too.
  HostMemoryPages pages{
      .free = 15, .speculative = 5, .fileBacked = 35, .purgeable = 5};
  require(availablePages(pages) == 50,
          "host availability does not match macOS reclaimable accounting");
  pages.free += 5;
  pages.speculative += 5;
  require(availablePages(pages) == 50,
          "speculative file pages were counted twice");
  pages.free -= 10;
  require(availablePages(pages) == 40,
          "anonymous or compressed pages did not consume capacity");
  pages.purgeable = 0;
  require(availablePages(pages) == 35,
          "non-purgeable backing received reclaimable credit");
  // Wired file pages leave external_page_count: GPU pinning must reduce
  // available memory, rather than crediting hot weights for KV growth.
  pages.fileBacked -= 10;
  require(availablePages(pages) == 25,
          "wired weights remained available for new allocations");

  // Reading a file into clean cache does not require a second full copy
  // when that same immutable file is mapped again on the next startup.
  require(availablePages({.free = 75}) == 75 &&
              availablePages({.free = 35, .fileBacked = 40}) == 75,
          "cached weights reduced model reload capacity");
  // A 64 GB M5 Pro, whose hw.memsize less its VM queues left 1.2 GiB more
  // (the firmware carve-out, tag storage) that no allocation can have.
  require(estimateHostAvailableMemory(
              {.free = 2'349'632, .speculative = 90'428,
               .fileBacked = 417'802, .purgeable = 23'495},
              pageSize) == 44'245'008'384ULL,
          "unexpected available memory for a 64 GB snapshot");
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(estimateHostAvailableMemory({.free = maximum, .fileBacked = 1}, 1) ==
                  0 &&
              estimateHostAvailableMemory(
                  {.fileBacked = maximum, .purgeable = 1}, 1) == 0 &&
              estimateHostAvailableMemory({.free = maximum}, pageSize) == 0 &&
              estimateHostAvailableMemory({.free = 1, .speculative = 2}, 1) ==
                  0 &&
              estimateHostAvailableMemory({.free = maximum}, 1) == maximum &&
              estimateHostAvailableMemory(pages, 0) == 0,
          "invalid host counters or arithmetic overflow did not fail closed");
  require(EngineMemoryPolicy::hostAvailableReserveBytes(16 * kGiB) ==
                  16 * kGiB / 10 &&
              EngineMemoryPolicy::hostAvailableReserveBytes(48 * kGiB) ==
                  2 * kGiB &&
              EngineMemoryPolicy::hostAvailableReserveBytes(128 * kGiB) ==
                  2 * kGiB,
          "the macOS reserve is a tenth of a small machine, 2 GiB above 20 GiB");
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
    testHostAvailabilityCountsReclaimablePages();
    testAdvertisedContextIsGrantable();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
