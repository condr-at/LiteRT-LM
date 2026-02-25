// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/framework/resource_management/utils/movable_mutex_lock.h"
#include "runtime/framework/resource_management/utils/resource_manager_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Saves the current processed context within the llm_executor to the previous
// handler's shared_processed_context, and link the current handler to an
// empty processed_context, which represents the current handler's processed
// context is already loaded in the llm_executor. Notice that the
// context_handler is assumed to be loaded in the llm_executor via the
// AcquireWithContext method, meaning the context_handler should not own any
// RuntimeState, RuntimeConfig and actual ProcessedContext within it, as they
// are all owned by the llm_executor at this point.
absl::Status SaveProcessedContextAndSeparateLoadedHandler(
    std::shared_ptr<ContextHandler> context_handler,
    std::shared_ptr<LlmExecutor> llm_executor) {
  const bool has_runtime_config = context_handler->HasRuntimeConfig();
  const bool has_runtime_state = context_handler->HasRuntimeState();
  const bool has_processed_context =
      context_handler->shared_processed_context()->HasProcessedContext();
  if (has_runtime_config || has_runtime_state || has_processed_context) {
    ABSL_LOG(ERROR)
        << "SaveProcessedContextAndSeparateLoadedHandler: context handler "
           "owns context artifacts unexpectedly; refusing unsafe normalization "
           "and failing fast."
        << " has_runtime_config=" << has_runtime_config
        << " has_runtime_state=" << has_runtime_state
        << " has_processed_context=" << has_processed_context;
    return absl::InternalError(
        "OWNERSHIP_INVARIANT_VIOLATION: context_handler should not own any "
        "RuntimeState, RuntimeConfig and actual ProcessedContext within it "
        "when calling SaveProcessedContextAndSeparateLoadedHandler.");
  }
  ASSIGN_OR_RETURN(auto llm_context, llm_executor->CloneContext());
  ASSIGN_OR_RETURN(auto current_processed_context,
                   llm_context->RetrieveProcessedContext());
  RETURN_IF_ERROR(
      context_handler->shared_processed_context()->SetProcessedContext(
          std::move(current_processed_context)));

  auto new_shared_processed_context =
      std::make_shared<ContextHandler::SharedProcessedContext>(nullptr);
  RETURN_IF_ERROR(context_handler->UpdateSharedProcessedContext(
      new_shared_processed_context));
  return absl::OkStatus();
}

}  // namespace

class LockedVisionExecutor : public VisionExecutor {
 public:
  LockedVisionExecutor(std::shared_ptr<VisionExecutor> vision_executor,
                       MovableMutexLock lock)
      : vision_executor_(std::move(vision_executor)), lock_(std::move(lock)) {}

  absl::StatusOr<ExecutorVisionData> Encode(
      const TensorBuffer& input_image_tensor) override {
    return vision_executor_->Encode(input_image_tensor);
  }

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override {
    return vision_executor_->GetExpectedInputDimension();
  }

 private:
  std::shared_ptr<VisionExecutor> vision_executor_;
  // The mutex lock.
  MovableMutexLock lock_;
};

class LockedAudioExecutor : public AudioExecutor {
 public:
  LockedAudioExecutor(std::shared_ptr<AudioExecutor> audio_executor,
                      MovableMutexLock lock)
      : audio_executor_(std::move(audio_executor)), lock_(std::move(lock)) {}

  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& input_spectrogram_tensor) override {
    return audio_executor_->Encode(input_spectrogram_tensor);
  }

  absl::Status Reset() override { return audio_executor_->Reset(); }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return audio_executor_->GetAudioExecutorProperties();
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CreateNewContext() override {
    return audio_executor_->CreateNewContext();
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext() override {
    return audio_executor_->CloneContext();
  }

  absl::Status RestoreContext(
      std::unique_ptr<AudioContext> audio_context) override {
    return audio_executor_->RestoreContext(std::move(audio_context));
  }

 private:
  std::shared_ptr<AudioExecutor> audio_executor_;
  // The mutex lock.
  MovableMutexLock lock_;
};

// LockedLlmExecutor's behavior should be the same as LlmExecutor, but instead
// wraps the llm executor, the corresponding mutex lock, and some additional
// optimization logic before forwarding the request to the llm executor.
// The Optimization includes:
// 1(remove matching tokens): Update the input_ids and current_step by
// removing
//    the matching tokens from the processed tokens.
// 2(Copy on write): If the current handler is not the longest handler,
// retrieve
//    the processed_context for the previous handler, and update the current
//    handler's shared_processed_context.
// For more details, please refer to go/llm_resource_manager.
class LockedLlmExecutor : public LlmExecutor {
 public:
  // LockedLlmExecutor takes ownership of the mutex lock and holds the
  // shared_ptr to the executor.
  LockedLlmExecutor(std::shared_ptr<LlmExecutor> executor,
                    MovableMutexLock lock,
                    std::shared_ptr<ContextHandler> current_handler = nullptr)
      : current_handler_(current_handler),
        llm_executor_(std::move(executor)),
        lock_(std::move(lock)) {}

  absl::string_view ExecutorBackendName() const override {
    return llm_executor_->ExecutorBackendName();
  }

  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return Prefill(inputs, ExecutorPrefillParams());
  }

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    // If the executor is not acquired by any handler, forward the prefill
    // request to the executor directly.
    if (current_handler_ == nullptr) {
      return llm_executor_->Prefill(inputs, prefill_params);
    }
    // Check if the input token is 1 batch. Currently only support 1 batch per
    // prefill.
    ASSIGN_OR_RETURN(auto token_ids, inputs.GetTextTokenIdsPtr());
    LITERT_ASSIGN_OR_RETURN(auto token_ids_tensor_type,
                            token_ids->TensorType());
    RET_CHECK_EQ(token_ids_tensor_type.Layout().Dimensions()[0], 1);
    if (token_ids_tensor_type.Layout().Dimensions()[1] == 0) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    if (prefill_params.GetCurrentStep() != -1) {
      current_step = prefill_params.GetCurrentStep();
    }
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    // clone35: Clamp current_step to token_count to prevent
    // "New step cannot be greater than the max step" when a context switch
    // has restored a context with fewer tokens than the previous session's
    // current_step. This is a safe no-op when current_step <= token_count.
    const int token_count = processed_tokens->TokenCount();
    if (current_step > token_count) {
      ABSL_LOG(WARNING)
          << "prefill_current_step_clamped"
          << " original_current_step=" << current_step
          << " token_count=" << token_count;
      current_step = token_count;
    }
    // If the current_step is pointing at the step right after the last
    // processed token, call executor directly, no optimization for the
    // input can be done.
    if (token_count == current_step) {
      return llm_executor_->Prefill(inputs, prefill_params);
    }

    LITERT_ASSIGN_OR_RETURN(
        auto input_ids_vec,
        CopyFromTensorBuffer<int32_t>(*(*inputs.GetTextTokenIdsPtr())));

    // If the current_step is not pointing at the step right after the last
    // processed token, update the input_ids and current_step by removing
    // the matching tokens, and then call llm_executor_->Prefill with the
    // optimized inputs and time step.

    // If the processed tokens size is larger than the current step, update
    // the input_ids and current_step by removing the matching tokens.
    RETURN_IF_ERROR(RemoveMatchingTokens(processed_tokens->GetCopyOfTokens()[0],
                                         &input_ids_vec, &current_step));
    // clone35: Re-clamp after RemoveMatchingTokens in case it moved
    // current_step beyond token_count.
    if (current_step > token_count) {
      ABSL_LOG(WARNING)
          << "prefill_current_step_clamped_post_remove_matching"
          << " original_current_step=" << current_step
          << " token_count=" << token_count;
      current_step = token_count;
    }
    // If the updated input_ids is empty, meaning all required prefill
    // tokens have been processed previously, just set the current step and
    // return.
    if (input_ids_vec.empty()) {
      RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step));
      return absl::OkStatus();
    }

    // TODO: b/409401231 - Add unit tests for the new_inputs creation.
    LITERT_ASSIGN_OR_RETURN(
        auto new_inputs_token_ids,
        CopyToTensorBuffer(
            absl::MakeConstSpan(input_ids_vec.data(), input_ids_vec.size()),
            {1, static_cast<int>(input_ids_vec.size())}));
    std::optional<ExecutorVisionData> new_vision_data = std::nullopt;
    std::optional<ExecutorAudioData> new_audio_data = std::nullopt;
    if (inputs.GetVisionDataPtr().ok()) {
      new_vision_data = ExecutorVisionData();
      LITERT_ASSIGN_OR_RETURN(
          auto new_vision_embeddings,
          inputs.GetVisionEmbeddingsPtr().value()->Duplicate());
      new_vision_data->SetEmbeddings(std::move(new_vision_embeddings));
      if (inputs.GetVisionDataPtr().value()->GetPerLayerEmbeddingsPtr().ok()) {
        LITERT_ASSIGN_OR_RETURN(auto new_per_layer_embeddings,
                                inputs.GetVisionDataPtr()
                                    .value()
                                    ->GetPerLayerEmbeddingsPtr()
                                    .value()
                                    ->Duplicate());
        new_vision_data->SetPerLayerEmbeddings(
            std::move(new_per_layer_embeddings));
      }
    }
    if (inputs.GetAudioEmbeddingsPtr().ok()) {
      new_audio_data = ExecutorAudioData();
      LITERT_ASSIGN_OR_RETURN(
          auto new_audio_embeddings,
          inputs.GetAudioEmbeddingsPtr().value()->Duplicate());
      new_audio_data->SetEmbeddings(std::move(new_audio_embeddings));
      if (inputs.GetAudioDataPtr().value()->GetPerLayerEmbeddingsPtr().ok()) {
        LITERT_ASSIGN_OR_RETURN(auto new_per_layer_embeddings,
                                inputs.GetAudioDataPtr()
                                    .value()
                                    ->GetPerLayerEmbeddingsPtr()
                                    .value()
                                    ->Duplicate());
        new_audio_data->SetPerLayerEmbeddings(
            std::move(new_per_layer_embeddings));
      }
    }
    auto new_inputs =
        ExecutorInputs(ExecutorTextData(std::move(new_inputs_token_ids)),
                       std::move(new_vision_data), std::move(new_audio_data));

    auto new_prefill_query_params = prefill_params;
    new_prefill_query_params.SetCurrentStep(current_step);
    // If the current_step is pointing at the step following the last
    // processed token after removing the matching tokens, call executor
    // directly.
    if (processed_tokens->TokenCount() == current_step) {
      return llm_executor_->Prefill(new_inputs, new_prefill_query_params);
    }
    // If the updated current_steps_ is still less than the processed tokens
    // size, meaning part of the processed tokens does not match the
    // input_ids.
    // Confirm if the current handler is the longest handler. If not,
    // cloning processed context is required to avoid modifying the
    // processed context of other handlers.
    ASSIGN_OR_RETURN(
        int largest_time_step,
        current_handler_->shared_processed_context()->LongestHandlerTimeStep(
            *llm_executor_));
    if (largest_time_step != current_step) {
      // If the current handler is not the longest handler, retrieve the
      // processed_context for the previous handler, and update the current
      // handler's shared_processed_context.
      RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
          current_handler_, llm_executor_));
    }
    // Update the current step since the new processed context (set above)
    // might not match the executor's current step, and the processed
    // context may need to be truncated.
    // TODO: b/418002952 - Consider setting the current step within Prefill
    // rather than relying on the caller.
    RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step));
    return llm_executor_->Prefill(new_inputs, new_prefill_query_params);
  }

  absl::Status Decode(TensorBuffer& output_tokens) override {
    return Decode(output_tokens, ExecutorDecodeParams());
  }

  absl::Status Decode(TensorBuffer& output_tokens,
                      const ExecutorDecodeParams& decode_params) override {
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->Decode(output_tokens, decode_params);
  }

  absl::Status Decode(const ExecutorInputs& inputs,
                      TensorBuffer& output_logits) override {
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->Decode(output_logits);
  }

  absl::StatusOr<TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override {
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    // If the current step is pointing at right after the pending token, set
    // the current step to the previous step. This ensures that the current
    // step points to the token to be processed, as expected by
    // llm_executor_->DecodeLogits().
    if (current_step == processed_tokens->TokenCount() &&
        !processed_tokens->GetNextUnprocessedToken().token.empty()) {
      RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step - 1));
    }
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->DecodeLogits(inputs);
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    return llm_executor_->CloneContext();
  }

  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    return llm_executor_->RestoreContext(std::move(llm_context));
  }

  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    return llm_executor_->UpdateRuntimeConfig(runtime_config);
  }

  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return llm_executor_->GetRuntimeConfig();
  }

  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    return llm_executor_->UpdateRuntimeState(runtime_state);
  }

  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return llm_executor_->GetRuntimeState();
  }

  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return llm_executor_->GetExecutorSettings();
  }

  absl::StatusOr<int> GetCurrentStep() const override {
    return llm_executor_->GetCurrentStep();
  }

  absl::Status SetCurrentStep(int new_step) override {
    return llm_executor_->SetCurrentStep(new_step);
  }

  absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const override {
    return llm_executor_->GetProcessedTokens();
  }

  absl::Status LoadLoRA(uint32_t lora_id,
                        const ModelAssets& model_assets) override {
    return llm_executor_->LoadLoRA(lora_id, model_assets);
  }

  absl::Status Reset() override { return llm_executor_->Reset(); }

  absl::StatusOr<int> GetVocabSize() override {
    return llm_executor_->GetVocabSize();
  }

 private:
  absl::Status MaybeTruncateProcessedTokens() {
    if (current_handler_ == nullptr) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    if (processed_tokens->TokenCount() == current_step) {
      return absl::OkStatus();
    }

    // Confirm if the current handler is the longest handler. If not,
    // cloning processed context is required to avoid modifying the
    // processed context of other handlers.
    ASSIGN_OR_RETURN(
        int largest_time_step,
        current_handler_->shared_processed_context()->LongestHandlerTimeStep(
            *llm_executor_));
    if (largest_time_step != current_step) {
      // If the current handler is not the longest handler, retrieve the
      // processed_context for the previous handler, and update the current
      // handler's shared_processed_context.
      RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
          current_handler_, llm_executor_));
    }
    // Update the current step since the new processed context (set above)
    // might not match the executor's current step, and the processed
    // context may need to be truncated.
    // TODO: b/418002952 - Consider setting the current step within Decode
    // rather than relying on the caller.
    return llm_executor_->SetCurrentStep(current_step);
  }

  // The current context handler;
  std::shared_ptr<ContextHandler> current_handler_;

  // The executor.
  std::shared_ptr<LlmExecutor> llm_executor_;

  // The mutex lock.
  MovableMutexLock lock_;
};

std::optional<uint32_t> ResourceManager::AssignLoraId(
    std::string lora_path, bool has_scoped_lora_file) {
  if (lora_path.empty() && !has_scoped_lora_file) {
    return std::nullopt;
  }
  std::optional<uint32_t> lora_id;
  // If this session is using a new lora, assign a new unique lora id. Else,
  // assign the corresponding lora id according to the provided lora path in
  // the session config.
  if (!lora_path.empty()) {
    // Lora provided by both path and scoped file will use lora path as the
    // reference key if provided.
    if (lora_hash_to_id_.find(lora_path) == lora_hash_to_id_.end()) {
      // If the lora is new, assign the id.
      lora_hash_to_id_[lora_path] = lora_hash_to_id_.size();
    }
    lora_id = lora_hash_to_id_[lora_path];
  } else if (has_scoped_lora_file) {
    // Lora provided by scoped file but without lora path will be assumed to
    // be used only once. Assign a unique id for this session only.
    // TODO: b/346421150 - Extend support to map from scoped file to hash
    // key, for multiple same scoped file use case.
    lora_id = lora_hash_to_id_.size();
    lora_hash_to_id_["scoped_lora:" + absl::StrCat(lora_hash_to_id_.size())] =
        lora_id.value();
  }
  return lora_id;
}

absl::Status ResourceManager::MaybeCreateLitertEnv() {
  if (litert_env_ != nullptr) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(
      auto new_litert_env,
      litert::Environment::Create(std::vector<litert::Environment::Option>()));
  backup_litert_env_ =
      std::make_unique<litert::Environment>(std::move(new_litert_env));
  litert_env_ = backup_litert_env_.get();
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ContextHandler>>
ResourceManager::CreateContextHandler(const SessionConfig& session_config) {
  // TODO: b/462499294 -
  //   1. Check if lora is loaded or not.
  //   2. Get the lora id.
  //   3. If lora is not loaded, load the lora.

  // Check if the lora is already loaded.
  // TODO: b/462499294 - Use the real lora path.
  bool lora_is_loaded =
      lora_hash_to_id_.find("fake_lora_path") != lora_hash_to_id_.end();

  // Find the lora id. If lora_id is not nullopt, it means the lora is used.
  std::optional<uint32_t> lora_id = AssignLoraId(
      /*lora_path=*/"",
      /*has_scoped_lora_file=*/session_config.GetScopedLoraFile() != nullptr);

  // If lora is used and not loaded, load the lora.
  if (lora_id.has_value() && !lora_is_loaded) {
    RET_CHECK(session_config.GetScopedLoraFile() != nullptr);
    ASSIGN_OR_RETURN(ModelAssets model_assets,
                     ModelAssets::Create(session_config.GetScopedLoraFile(),
                                         /*model_path=*/""));
    MovableMutexLock lock(&executor_mutex_);
    RETURN_IF_ERROR(llm_executor_->LoadLoRA(lora_id.value(), model_assets));
  }
  auto runtime_config = RuntimeConfig{
      .output_heads = session_config.GetNumOutputCandidates(),
      // b/368348506 - Make tokens_per_decode configurable.
      .tokens_per_decode = 1,
  };

  std::unique_ptr<litert::lm::LlmContext> llm_context;
  {
    MovableMutexLock lock(&executor_mutex_);
    ASSIGN_OR_RETURN(llm_context,
                     llm_executor_->CreateNewContext(
                         std::move(lora_id), std::move(runtime_config)));
  }
  std::unique_ptr<AudioContext> audio_context;
  if (session_config.AudioModalityEnabled()) {
    RETURN_IF_ERROR(TryLoadingAudioExecutor());
    ASSIGN_OR_RETURN(auto audio_executor, AcquireAudioExecutor());
    auto audio_executor_properties =
        audio_executor->GetAudioExecutorProperties();
    if (audio_executor_properties.ok()) {
      if (audio_executor_properties->is_streaming_model) {
        ASSIGN_OR_RETURN(audio_context, audio_executor->CreateNewContext());
      }
    } else if (!absl::IsUnimplemented(audio_executor_properties.status())) {
      return audio_executor_properties.status();
    }
  }
  return ContextHandler::Create(std::move(llm_context),
                                std::move(audio_context));
}

absl::StatusOr<std::unique_ptr<ContextHandler>>
ResourceManager::CloneContextHandler(
    std::shared_ptr<const ContextHandler> llm_context_handler) {
  RET_CHECK_NE(llm_context_handler, nullptr)
      << "The provided context handler should not be null.";

  RuntimeConfig runtime_config;
  RuntimeState runtime_state;
  ABSL_LOG(INFO) << "resource_manager_clone_context_handler_begin"
                 << " source_has_runtime_config="
                 << (llm_context_handler->HasRuntimeConfig() ? "1" : "0")
                 << " source_has_runtime_state="
                 << (llm_context_handler->HasRuntimeState() ? "1" : "0")
                 << " source_has_processed_context="
                 << (llm_context_handler->shared_processed_context()
                             ->HasProcessedContext()
                         ? "1"
                         : "0")
                 << " source_is_current_handler="
                 << ((current_handler_ == llm_context_handler) ? "1" : "0");

  // If the context handler has the runtime config and runtime state, use
  // them directly.
  if (llm_context_handler->HasRuntimeConfig() &&
      llm_context_handler->HasRuntimeState()) {
    ASSIGN_OR_RETURN(runtime_config, llm_context_handler->GetRuntimeConfig());
    ASSIGN_OR_RETURN(runtime_state, llm_context_handler->GetRuntimeState());
  } else {
    // Otherwise, assume the context handler is loaded by the manager to the
    // executor, and get the runtime config and runtime state from the
    // executor. This is safe because ExecutionManager runs tasks on a single
    // execution thread, so runtime state reads here observe deterministic
    // sequencing across clone/context-switch operations.
    MovableMutexLock lock(&executor_mutex_);
    if (current_handler_ != llm_context_handler) {
      return absl::InternalError(
          "CLONE_RUNTIME_STATE_SOURCE_INVALID: context handler has no runtime "
          "config/state and is not current_handler_. Refusing to clone with "
          "executor state from a different active handler.");
    }
    ASSIGN_OR_RETURN(runtime_config, llm_executor_->GetRuntimeConfig());
    ASSIGN_OR_RETURN(runtime_state, llm_executor_->GetRuntimeState());
  }
  auto processed_context = llm_context_handler->shared_processed_context();

  std::unique_ptr<AudioContext> audio_context;
  if (llm_context_handler->HasAudioContext()) {
    const auto& audio_context_ref = llm_context_handler->GetAudioContext();
    ASSIGN_OR_RETURN(audio_context, audio_context_ref.Clone());
  }
  return ContextHandler::Bundle(
      processed_context, std::make_unique<RuntimeConfig>(runtime_config),
      std::make_unique<RuntimeState>(runtime_state), std::move(audio_context));
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
ResourceManager::AcquireExecutor() {
  MovableMutexLock lock(&executor_mutex_);

  if (llm_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Llm executor should not be null, please do not delete the shared "
        "executor "
        "in ResourceManager at any time.");
  }

  return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock));
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
ResourceManager::AcquireExecutorWithContextHandler(
    std::shared_ptr<ContextHandler> new_context_handler) {
  RET_CHECK_NE(new_context_handler, nullptr)
      << "The provided context handler should not be null.";

  MovableMutexLock lock(&executor_mutex_);
  RET_CHECK_NE(llm_executor_, nullptr) << "Llm executor should not be null, "
                                          "please do not delete the shared "
                                          "executor in ResourceManager at "
                                          "any time.";
  ABSL_LOG(INFO) << "resource_manager_switch_begin"
                 << " has_current_handler="
                 << (current_handler_ != nullptr ? "1" : "0")
                 << " same_handler="
                 << (new_context_handler == current_handler_ ? "1" : "0")
                 << " same_shared_processed_context="
                 << ((current_handler_ != nullptr &&
                      new_context_handler->shared_processed_context() ==
                          current_handler_->shared_processed_context())
                         ? "1"
                         : "0")
                 << " target_has_runtime_config="
                 << (new_context_handler->HasRuntimeConfig() ? "1" : "0")
                 << " target_has_runtime_state="
                 << (new_context_handler->HasRuntimeState() ? "1" : "0")
                 << " target_has_processed_context="
                 << (new_context_handler->shared_processed_context()
                             ->HasProcessedContext()
                         ? "1"
                         : "0");
  auto take_runtime_config_for_switch =
      [this](std::shared_ptr<ContextHandler> handler)
      -> absl::StatusOr<std::unique_ptr<RuntimeConfig>> {
    if (handler->HasRuntimeConfig()) {
      return handler->RetrieveRuntimeConfig();
    }
    return absl::InternalError(
        "SWITCH_RUNTIME_CONFIG_MISSING: target context handler has no runtime "
        "config while being activated.");
  };
  auto take_runtime_state_for_switch =
      [this](std::shared_ptr<ContextHandler> handler)
      -> absl::StatusOr<std::unique_ptr<RuntimeState>> {
    if (handler->HasRuntimeState()) {
      return handler->RetrieveRuntimeState();
    }
    return absl::InternalError(
        "SWITCH_RUNTIME_STATE_MISSING: target context handler has no runtime "
        "state while being activated.");
  };

  // If the new handler is the same as the current handler, return the
  // executor directly.
  if (new_context_handler == current_handler_) {
    return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock),
                                               current_handler_);
  }

  // If both handler are sharing the same processed context, save the
  // runtime config and runtime state back to the current handler. Then
  // update the executor with the new handler.
  if (current_handler_ != nullptr &&
      new_context_handler->shared_processed_context() ==
          current_handler_->shared_processed_context()) {
    ASSIGN_OR_RETURN(auto current_runtime_config,
                     llm_executor_->GetRuntimeConfig());
    ASSIGN_OR_RETURN(auto current_runtime_state,
                     llm_executor_->GetRuntimeState());
    RETURN_IF_ERROR(current_handler_->SetRuntimeConfig(
        std::make_unique<RuntimeConfig>(current_runtime_config)));
    RETURN_IF_ERROR(current_handler_->SetRuntimeState(
        std::make_unique<RuntimeState>(current_runtime_state)));

    ASSIGN_OR_RETURN(auto new_runtime_config,
                     take_runtime_config_for_switch(new_context_handler));
    ASSIGN_OR_RETURN(auto new_runtime_state,
                     take_runtime_state_for_switch(new_context_handler));
    ASSIGN_OR_RETURN(const ProcessedTokens* active_processed_tokens,
                     llm_executor_->GetProcessedTokens());
    const int active_token_count = active_processed_tokens->TokenCount();
    if (new_runtime_state->current_step > active_token_count) {
      ABSL_LOG(WARNING)
          << "resource_manager_runtime_state_clamped_same_processed_context"
          << " original_current_step=" << new_runtime_state->current_step
          << " token_count=" << active_token_count;
      new_runtime_state->current_step = active_token_count;
    }
    if (new_runtime_state->current_step < 0) {
      ABSL_LOG(WARNING)
          << "resource_manager_runtime_state_clamped_negative_step"
          << " original_current_step=" << new_runtime_state->current_step;
      new_runtime_state->current_step = 0;
    }
    ABSL_LOG(INFO) << "resource_manager_switch_same_processed_context"
                   << " current_session_handler="
                   << (current_handler_ != nullptr ? "1" : "0")
                   << " target_runtime_config_taken=1"
                   << " target_runtime_state_taken=1";
    RETURN_IF_ERROR(llm_executor_->UpdateRuntimeConfig(*new_runtime_config));
    RETURN_IF_ERROR(llm_executor_->UpdateRuntimeState(*new_runtime_state));
  } else {
    // If the new handler is not sharing the same processed context with the
    // current handler, clone the processed context to the new handler. Then
    // restore the executor with the new LlmContext.
    if (current_handler_ != nullptr) {
      ASSIGN_OR_RETURN(auto current_llm_context, llm_executor_->CloneContext());
      ASSIGN_OR_RETURN(auto current_runtime_config,
                       current_llm_context->RetrieveRuntimeConfig());
      ASSIGN_OR_RETURN(auto current_runtime_state,
                       current_llm_context->RetrieveRuntimeState());
      ASSIGN_OR_RETURN(auto current_processed_context,
                       current_llm_context->RetrieveProcessedContext());

      RETURN_IF_ERROR(current_handler_->SetRuntimeConfig(
          std::move(current_runtime_config)));
      RETURN_IF_ERROR(
          current_handler_->SetRuntimeState(std::move(current_runtime_state)));
      RETURN_IF_ERROR(
          current_handler_->shared_processed_context()->SetProcessedContext(
              std::move(current_processed_context)));
    }

    ASSIGN_OR_RETURN(auto new_runtime_config,
                     take_runtime_config_for_switch(new_context_handler));
    ASSIGN_OR_RETURN(auto new_runtime_state,
                     take_runtime_state_for_switch(new_context_handler));
    ASSIGN_OR_RETURN(auto new_processed_context,
                     new_context_handler->shared_processed_context()
                         ->RetrieveProcessedContext());
    const int target_token_count = new_processed_context == nullptr
                                       ? 0
                                       : new_processed_context->processed_tokens()
                                             .TokenCount();
    if (new_runtime_state->current_step > target_token_count) {
      ABSL_LOG(WARNING)
          << "resource_manager_runtime_state_clamped_restored_context"
          << " original_current_step=" << new_runtime_state->current_step
          << " token_count=" << target_token_count;
      new_runtime_state->current_step = target_token_count;
    }
    if (new_runtime_state->current_step < 0) {
      ABSL_LOG(WARNING)
          << "resource_manager_runtime_state_clamped_negative_step"
          << " original_current_step=" << new_runtime_state->current_step;
      new_runtime_state->current_step = 0;
    }
    const int token_count = new_processed_context == nullptr
                                ? 0
                                : new_processed_context->processed_tokens()
                                      .TokenCount();
    const bool is_fresh_context =
        token_count == 0 && new_runtime_state->current_step == 0 &&
        !new_runtime_state->ran_decode;
    ABSL_LOG(INFO) << "resource_manager_restore_context_decision"
                   << " token_count=" << token_count
                   << " current_step=" << new_runtime_state->current_step
                   << " ran_decode=" << new_runtime_state->ran_decode
                   << " has_processed_context="
                   << (new_processed_context != nullptr ? "1" : "0")
                   << " path="
                   << (is_fresh_context ? "fresh_create_new_context"
                                        : "restore_provided_context");
    if (is_fresh_context) {
      std::optional<uint32_t> lora_id = std::nullopt;
      if (new_processed_context != nullptr) {
        lora_id = new_processed_context->lora_id();
      }
      ASSIGN_OR_RETURN(auto llm_context,
                       llm_executor_->CreateNewContext(
                           lora_id, std::move(*new_runtime_config)));
      RETURN_IF_ERROR(llm_executor_->RestoreContext(std::move(llm_context)));
      RETURN_IF_ERROR(llm_executor_->UpdateRuntimeState(*new_runtime_state));
    } else {
      auto llm_context = std::make_unique<LlmContext>(
          std::move(new_processed_context), std::move(new_runtime_config),
          std::move(new_runtime_state));
      RETURN_IF_ERROR(llm_executor_->RestoreContext(std::move(llm_context)));
    }
  }

  // If the current handler has an audio context, update and save the audio
  // context to the current handler.
  if (current_handler_ != nullptr) {
    // If the current handler has an audio context, update it from audio
    // executor and save it back to the current handler.
    if (current_handler_->HasAudioContext()) {
      ASSIGN_OR_RETURN(auto audio_executor, AcquireAudioExecutor());
      ASSIGN_OR_RETURN(auto current_audio_context,
                       audio_executor->CloneContext());
      RETURN_IF_ERROR(
          current_handler_->SetAudioContext(std::move(current_audio_context)));
    }
    // If the new handler has an audio context, audio executor will restore
    // the audio context from the new handler.
    if (new_context_handler->HasAudioContext()) {
      ASSIGN_OR_RETURN(auto audio_executor, AcquireAudioExecutor());
      ASSIGN_OR_RETURN(auto audio_context_cloned,
                       new_context_handler->GetAudioContext().Clone());
      RETURN_IF_ERROR(
          audio_executor->RestoreContext(std::move(audio_context_cloned)));
    }
  }

  current_handler_ = new_context_handler;
  ABSL_LOG(INFO) << "resource_manager_switch_end"
                 << " new_current_handler_set=1"
                 << " current_has_runtime_config="
                 << (current_handler_->HasRuntimeConfig() ? "1" : "0")
                 << " current_has_runtime_state="
                 << (current_handler_->HasRuntimeState() ? "1" : "0")
                 << " current_has_processed_context="
                 << (current_handler_->shared_processed_context()
                             ->HasProcessedContext()
                         ? "1"
                         : "0");

  return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock),
                                             current_handler_);
}

absl::Status ResourceManager::TryLoadingVisionExecutor() {
  return absl::InvalidArgumentError(
      "Vision executor backend is not supported.");
}

absl::StatusOr<std::unique_ptr<VisionExecutor>>
ResourceManager::AcquireVisionExecutor() {
  MovableMutexLock lock(&vision_executor_mutex_);
  if (vision_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Vision executor should not be null, please TryLoadingVisionExecutor() "
        "first.");
  }
  return std::make_unique<LockedVisionExecutor>(vision_executor_,
                                                std::move(lock));
}

absl::Status ResourceManager::TryLoadingAudioExecutor() {
  absl::MutexLock lock(audio_executor_mutex_);
  if (audio_executor_ != nullptr) {
    return absl::OkStatus();
  }
  if (!audio_executor_settings_) {
    return absl::InvalidArgumentError("Audio options should not be null.");
  }
  if (audio_executor_settings_->GetBackend() == litert::lm::Backend::CPU) {
    return absl::InvalidArgumentError(
        "Audio executor backend is not supported.");
  } else if (audio_executor_settings_->GetBackend() ==
             litert::lm::Backend::GPU_ARTISAN) {
    return absl::InvalidArgumentError(
        "Audio executor backend is not supported.");
  } else {
    return absl::InvalidArgumentError(
        "Audio executor backend is not supported.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioExecutor>>
ResourceManager::AcquireAudioExecutor() {
  MovableMutexLock lock(&audio_executor_mutex_);
  if (audio_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Audio executor should not be null, please TryLoadingAudioExecutor() "
        "first.");
  }
  return std::make_unique<LockedAudioExecutor>(audio_executor_,
                                               std::move(lock));
}

absl::StatusOr<std::unique_ptr<ResourceManager>> ResourceManager::Create(
    ModelResources* absl_nullable model_resources,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<litert::lm::AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    ::litert::Environment* absl_nullable litert_env) {
  if (llm_executor == nullptr) {
    return absl::InvalidArgumentError("Llm executor is null.");
  }
  auto llm_resource_manager = std::make_unique<ResourceManager>(
      model_resources, std::move(llm_executor),
      std::move(vision_executor_settings), std::move(audio_executor_settings),
      litert_env);
  return llm_resource_manager;
}

}  // namespace litert::lm
