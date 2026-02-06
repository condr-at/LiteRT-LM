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

#include <fcntl.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetSentencePieceModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "sentencepiece.model")
      .string();
}

std::string GetGemma3TokenizerModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "gemma3_sentencepiece.model")
      .string();
}

absl::StatusOr<std::string> GetContents(absl::string_view path) {
#ifdef _WIN32
  int fd = open(path.data(), O_RDONLY | O_BINARY);
#else
  int fd = open(path.data(), O_RDONLY);
#endif
  if (fd < 0) {
    return absl::NotFoundError(absl::StrCat("File not found: ", path));
  }

  absl::Cleanup fd_closer = [fd]() { close(fd); };  // Called on return.

  int64_t contents_length = lseek(fd, 0, SEEK_END);
  if (contents_length < 0) {
    return absl::InternalError(absl::StrCat("Failed to get length: ", path));
  }

  std::string contents(contents_length, '\0');
  lseek(fd, 0, SEEK_SET);
  char* contents_ptr = contents.data();
  while (contents_length > 0) {
    int read_bytes = read(fd, contents_ptr, contents_length);
    if (read_bytes < 0) {
      return absl::InternalError(absl::StrCat("Failed to read: ", path));
    } else if (read_bytes == 0) {
      return absl::InternalError(absl::StrCat("File is empty: ", path));
    }
    contents_ptr += read_bytes;
    contents_length -= read_bytes;
  }

  return std::move(contents);
}

TEST(SentencePieceTokenizerTest, CreateFromFile) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, CreateFromBuffer) {
  auto model_buffer_or = GetContents(GetSentencePieceModelPath());
  EXPECT_TRUE(model_buffer_or.ok());
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromBuffer(*model_buffer_or);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, Create) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, GetTokenizerType) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_EQ(tokenizer_or.value()->GetTokenizerType(),
            TokenizerType::kSentencePiece);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, TextToTokenIds) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "How's it going?";
  auto ids_or = tokenizer->TextToTokenIds(text);
  EXPECT_TRUE(ids_or.ok());

  EXPECT_THAT(ids_or.value(),
              ::testing::ElementsAre(224, 24, 8, 66, 246, 18, 2295));
}

TEST(SentencePieceTokenizerTest, TokenToId) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  EXPECT_THAT(tokenizer->TokenToId("X"), IsOkAndHolds(882));
}

TEST(SentencePieceTokenizerTest, TokenToIdUnknownTokenReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  EXPECT_THAT(tokenizer->TokenToId("unknown_token"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(SentencePieceTokenizerTest, TokenIdsToText) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  const std::vector<int> ids = {90, 547, 58, 735, 210, 466, 2294};
  auto text_or = tokenizer->TokenIdsToText(ids);
  EXPECT_TRUE(text_or.ok());

  EXPECT_EQ(text_or.value(), "▁Hello▁World!");
}

TEST(SentencePieceTokenizerTest, TokenIdsToTextConsecutiveByteTokens) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetGemma3TokenizerModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 262144);

  // Consecutive byte token combination
  // <0xC2><0xB0> --> °
  EXPECT_EQ(tokens[432], "<0xC2>");
  EXPECT_EQ(tokens[414], "<0xB0>");

  // Pass to tokenizer in two separate calls to TokenIdsToText.
  auto text_id283_or = tokenizer->TokenIdsToText({432});
  EXPECT_EQ(text_id283_or.value(), "");
  auto text_id414_or = tokenizer->TokenIdsToText({414});
  EXPECT_EQ(text_id414_or.value(), "°");
}

TEST(SentencePieceTokenizerTest,
     TokenIdsToTextConsecutiveByteTokensWithNonByteTokens) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetGemma3TokenizerModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 262144);

  for (int i = 0; i < tokens.size(); ++i) {
    auto text_or = tokenizer->TokenIdsToText({i});
  }

  // Consecutive byte token combination
  // <0x6B><0x6D><0xC2><0xB2> --> km²
  EXPECT_EQ(tokens[345], "<0x6B>");
  EXPECT_EQ(tokens[347], "<0x6D>");
  EXPECT_EQ(tokens[432], "<0xC2>");
  EXPECT_EQ(tokens[416], "<0xB2>");

  // Pass as streaming mode with separate calls to TokenIdsToText.
  auto km_token_ids = {345, 347};
  auto text_km_or = tokenizer->TokenIdsToText(km_token_ids);
  EXPECT_EQ(text_km_or.value(), "km");
  auto text_id283_or = tokenizer->TokenIdsToText({432});
  EXPECT_EQ(text_id283_or.value(), "");
  auto text_id416_or = tokenizer->TokenIdsToText({416});
  EXPECT_EQ(text_id416_or.value(), "²");

  // Pass all as a single call with a single call to TokenIdsToText.
  auto combined_text_or = tokenizer->TokenIdsToText({345, 347, 432, 416});
  EXPECT_EQ(combined_text_or.value(), "km²");
}

TEST(SentencePieceTokenizerTest, GetTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 4000);

  // Verify 5 different tokens.
  EXPECT_EQ(tokens[0], "<unk>");
  EXPECT_EQ(tokens[1], "<s>");
  EXPECT_EQ(tokens[2], "</s>");
  EXPECT_EQ(tokens[224], "▁How");
  EXPECT_EQ(tokens[2295], "?");
}

}  // namespace
}  // namespace litert::lm
