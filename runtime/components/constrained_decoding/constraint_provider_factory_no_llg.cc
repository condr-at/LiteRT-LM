#include "runtime/components/constrained_decoding/constraint_provider_factory.h"

#include <memory>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/external_constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/tokenizer.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateConstraintProvider(
    const ConstraintProviderConfig& constraint_provider_config,
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids) {
  if (std::holds_alternative<ExternalConstraintConfig>(
          constraint_provider_config)) {
    return std::make_unique<ExternalConstraintProvider>();
  } else if (std::holds_alternative<LlGuidanceConfig>(
                 constraint_provider_config)) {
    return absl::UnimplementedError(
        "LlGuidance constraint provider is disabled in this Android build.");
  }

  return absl::UnimplementedError("Unknown ConstraintProviderConfig type.");
}

}  // namespace litert::lm
