# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "mel_filterbank",
    srcs = [
        "mel_filterbank.cc",
    ],
    hdrs = [
        "mel_filterbank.h",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mel_filterbank_test",
    srcs = ["mel_filterbank_test.cc"],
    deps = [
        ":mel_filterbank",
        ":signal_vector_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/random",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "signal_vector_util",
    hdrs = ["signal_vector_util.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_test(
    name = "signal_vector_util_test",
    srcs = ["signal_vector_util_test.cc"],
    deps = [
        ":signal_vector_util",
        "@com_google_googletest//:gtest_main",
        "//runtime/util:test_utils",
    ],
)
