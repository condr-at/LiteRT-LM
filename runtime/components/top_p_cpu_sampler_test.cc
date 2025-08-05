#include "runtime/components/top_p_cpu_sampler.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

TEST(TopPSamplerTest, Create) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*seed=*/1,
                                        /*is_perplexity_computed=*/false);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest,
     SampleToIdAndScoreBuffer_IdsOnly_BatchSize2_NoPerplexity) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1,
                                        /*is_perplexity_computed=*/false);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));
  EXPECT_FALSE(sampler->GetPerplexity().ok());
}

TEST(TopPSamplerTest,
     SampleToIdAndScoreBuffer_IdsOnly_BatchSize2_WithPerplexity) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1,
                                        /*is_perplexity_computed=*/true);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));
  // Manual softmax calculation for probabilities:
  // 1 / (1 + 2 * exp(-10.0f)) for the first batch.
  // 1 / (exp(-1.0f) + 1 + exp(-11.0f) + exp(-10.0f)) for the second
  // batch.
  auto perplexity = sampler->GetPerplexity();
  EXPECT_TRUE(perplexity.ok());
  EXPECT_NEAR(
      *perplexity,
      -1 * std::log(1 / (1 + 2 * exp(-10.0f))) +
          -1 * std::log(1 / (exp(-1.0f) + 1 + exp(-11.0f) + exp(-10.0f))),
      1e-3f);
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2_NoPerplexity) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1,
                                        /*is_perplexity_computed=*/false);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  EXPECT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, testing::ElementsAre(std::log(1.0f), std::log(1.0f)));
  // Perplexity is not computed.
  EXPECT_FALSE(sampler->GetPerplexity().ok());
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2_WithPerplexity) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1,
                                        /*is_perplexity_computed=*/true);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  EXPECT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, testing::ElementsAre(std::log(1.0f), std::log(1.0f)));
  EXPECT_TRUE(sampler->GetPerplexity().ok());
  auto perplexity = sampler->GetPerplexity();
  EXPECT_TRUE(perplexity.ok());
  EXPECT_NEAR(*perplexity, 0.0f, 1e-6f);
}

}  // namespace
}  // namespace litert::lm
