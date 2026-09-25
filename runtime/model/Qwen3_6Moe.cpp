#include "Qwen3_6Moe.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFM0002";

void requireLayout(const Qwen3_6MoeLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize) {
    throw WeightStoreError("Qwen3.6 MoE layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          2 * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.expertsPerToken > layout.experts ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("Qwen3.6 MoE layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      Qwen3_6MoeLayout layout) {
  requireLayout(layout);
  return loadQwenTargetWeights<Qwen3_6MoeWeights>(
      backend, directory, layout, kHeadMagic,
      [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
        const uint32_t hidden = layout.hiddenSize, width = layout.expertIntermediateSize;
        layer.ffn = ops::AffineMoeWeights{
            .router = readAffineQ8Projection(file, layout.experts, hidden, "router"),
            .expertGate = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-gate"),
            .expertUp = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-up"),
            .expertDown = readAffineExpertProjection(file, layout.experts, hidden, width, "experts-down"),
            .sharedGate = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-gate"),
            .sharedUp = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-up"),
            .sharedDown = readAffineExpertProjection(file, 1, hidden, width, "shared-expert-down"),
            .sharedScalarGate =
                readAffineQ8Projection(file, kQ4StorageN, hidden, "shared-expert-scalar-gate"),
        };
      });
}

} // namespace splash::model
