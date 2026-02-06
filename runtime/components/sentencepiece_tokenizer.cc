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

#include "runtime/components/sentencepiece_tokenizer.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "sentencepiece_model.pb.h"  // from @sentencepiece
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::lm {

namespace {

// Returns the expected total length of a UTF-8 character starting with
// byte_val.
int GetUtf8Length(unsigned char byte_val) {
  if ((byte_val & 0x80) == 0) {
    return 1;  // ASCII
  } else if ((byte_val & 0xE0) == 0xC0) {
    return 2;  // Starts with 110
  } else if ((byte_val & 0xF0) == 0xE0) {
    return 3;  // Starts with 1110
  } else if ((byte_val & 0xF8) == 0xF0) {
    return 4;  // Starts with 11110
  } else if ((byte_val & 0xC0) == 0x80) {
    return 0;  // It's a continuation byte (not a lead)
  }
  return -1;  // Invalid UTF-8
}

int GetUtf8LengthFromByteToken(const std::string& token_piece) {
  unsigned int hex_val;
  if (token_piece.length() == 6 &&
      std::sscanf(token_piece.c_str(), "<0x%x>", &hex_val) == 1) {
    return GetUtf8Length(static_cast<unsigned char>(hex_val));
  }
  return -1;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromFile(absl::string_view model_path) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(model_path);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromBuffer(absl::string_view model_buffer) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->LoadFromSerializedProto(model_buffer);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromProto(
    std::unique_ptr<sentencepiece::ModelProto> model_proto) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(std::move(model_proto));
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> SentencePieceTokenizer::TextToTokenIds(
    absl::string_view text) {
  std::vector<int> ids;
  auto status = processor_->Encode(text, &ids);
  if (!status.ok()) {
    return status;
  }
  return ids;
}

absl::StatusOr<int> SentencePieceTokenizer::TokenToId(absl::string_view token) {
  int id = processor_->PieceToId(token);
  if (id == processor_->unk_id()) {
    return absl::NotFoundError(absl::StrCat("Unknown token: ", token));
  }
  return id;
}

// Decodes the given TensorBuffer of token ids into a string.
absl::StatusOr<std::string> SentencePieceTokenizer::TokenIdsToText(
    const std::vector<int>& token_ids) {
  std::string text = "";
  for (const auto& token_id : token_ids) {
    if (processor_->IsByte(token_id)) {
      std::string piece = processor_->IdToPiece(token_id);
      int utf8_length = GetUtf8LengthFromByteToken(piece);
      // If the token is a single byte or invalid/continuation byte and not
      // bundled with other tokens, decode it immediately.
      if (size_of_token_chunk_ == 0 && utf8_length <= 1) {
        text += processor_->DecodeIds({token_id});
        continue;
      }

      // Update the expected chunk size based on the new byte.
      size_of_token_chunk_ = std::max(size_of_token_chunk_, utf8_length);
      buffered_token_ids_.push_back(token_id);

      // If the buffer satisfies the expected chunk size, decode chunk.
      // Clear the buffer and reset the expected chunk size.
      if (buffered_token_ids_.size() >= size_of_token_chunk_) {
        text += processor_->DecodeIds(buffered_token_ids_);
        buffered_token_ids_.clear();
        size_of_token_chunk_ = 0;
      }
    } else {
      // We are forced to use IdToPiece to account for leading whitespace.
      // Otherwise, the normalizer (depending on the configuration) would
      // remove that which makes streaming decoding impossible.
      // e.g., [[change], [_volume]] -> "change volume" vs.
      //       [[change], [volume]] -> "changevolume"
      text += processor_->IdToPiece(token_id);
    }
  }
  return text;
}

std::vector<std::string> SentencePieceTokenizer::GetTokens() const {
  std::vector<std::string> tokens;
  for (const auto& piece : processor_->model_proto().pieces()) {
    tokens.push_back(piece.piece());
  }
  return tokens;
}

}  // namespace litert::lm
