# LiteRT-LM

A C++ library to efficiently run language models across edge platforms.

## ⚠️ IMPORTANT: Repository Migration Notice (Jan 31, 2026)

**The LiteRT-LM repository has been migrated to use
[Git LFS (Large File Storage)](https://git-lfs.com) for all prebuilt binaries.**
Because this involved a history rewrite to shrink the repository size, all
previous commit hashes are now invalid.

### Action Required:
If you have a local copy of this repository from before **January 31, 2026**,
your local history is now incompatible with the remote. **Please do not attempt
to `git pull`.**

To fix your local environment, please perform a fresh clone:

```bash
# 1. Remove your old directory (or move it to a backup).
rm -rf LiteRT-LM

# 2. Re-clone the repository.
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM

# 3. Ensure LFS is initialized. If it's the first time to install LFS, download
#    LFS from https://git-lfs.com.
git lfs install
git lfs pull
```

## Description

Language models are no longer a single model but really a pipeline of models and
components working together. LiteRT-LM builds on top of
[LiteRT](https://github.com/google-ai-edge/LiteRT) to enable these pipelines
including:

*   **C++ api** to efficiently run language models
*   **Cross-Platform** support via portable C++ for broad deployment scenarios
*   **Flexible** so you can customize for your specific feature
*   **Hardware Acceleration** to unlock the full potential of your device's
    hardware

### Status: Early Preview
Full release is coming soon.
We heard the community feedback regarding Google AI Edge's Gemma 3n LiteRT
preview. You want access on more platforms, more visibility into the underlying
stack, and more flexibility. LiteRT-LM can help with all three.

### 🚀 What's New

*   ***Nov 2025*** **: Desktop GPU support and more (`v0.8.0`)**

    -   Desktop GPU support.
    -   Simple CLI for Desktop: [Link to Quick Start section](#quick_start)
    -   Multi-Modality support: Vision and Audio input are supported when models
        support it.
        [See more details here](./docs/api/cpp/conversation.md#multimodal-data-content)
    -   Kotlin API for Android and JVM (Linux, MacOS, Windows):
        [Link to LiteRT-LM Kotlin API](./docs/api/kotlin.md)
    -   Conversation API:
        [Link to Conversation API](./docs/api/cpp/conversation.md)
    -   Function calling support: [Link to Tool Use](./docs/api/cpp/tool-use.md)

*   ***June 24, 2025*** **: Run Gemma models with NPU Support (`v0.7.0`)**
    Unlock significant performance gains! Our latest release leverages the power
    of Neural Processing Units (NPUs) on devices with Qualcomm and MediaTek
    chipsets to run the Gemma3 1B model with incredible efficiency.

    **Note:** LiteRT-LM NPU acceleration is only available through an Early
    Access Program. Please check out
    [this page](https://ai.google.dev/edge/litert/next/npu) for more information
    about how to sign it up.

*   ***June 10, 2025*** **: The Debut of LiteRT-LM: A New Framework for
    On-Device LLMs** We're proud to release an early preview (`v0.6.1`) of the
    LiteRT-LM codebase! This foundational release enables you to run the latest
    Gemma series models across a wide of devices with initial support for CPU
    execution and powerful GPU acceleration on Android.

### Supported Backends & Platforms

Platform     | CPU Support | GPU Support | NPU Support |
:----------- | :---------: | :-----------: | :-----------:
**Android**  | ✅           | ✅            | ✅ |
**macOS**    | ✅           | ✅            | - |
**Windows**  | ✅           | ✅            | - |
**Linux**    | ✅           | ✅            | - |
**Embedded** | ✅           | -             | - |

### Supported Models and Performance

Currently supported models during our Preview (as `.litertlm` format).

| Model               | Usage Type                     | Quantization      | Context size | Model Size (Mb) | Give it a try                                                                                                                                                             |
| :------------------ | :----------------------------- | :---------------- | :----------- | :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gemma3-1B           | Chat Ready                     | 4-bit per-channel | 4096         | 557             | [Download](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm)                                             |
| Gemma-3n-E2B        | Chat Ready                     | 4-bit per-channel | 4096         | 2965            | [Download](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm-preview)                                                                                               |
| Gemma-3n-E4B        | Chat Ready                     | 4-bit per-channel | 4096         | 4235            | [Download](https://huggingface.co/google/gemma-3n-E4B-it-litert-lm-preview)                                                                                               |
| phi-4-mini          | Chat Ready                     | 8-bit per-channel | 4096         | 3728            | [Download](https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm)                            |
| qwen2.5-1.5b        | Chat Ready                     | 8-bit per-channel | 4096         | 1524            | [Download](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm)                        |
| FunctionGemma-270M  | Base (Fine-tuning required)    | 8-bit per-channel | 1024         | 288             | [Fine-tuning Guide](https://ai.google.dev/gemma/docs/mobile-actions)                                                                                                      |
| ↪ TinyGarden-270M   | Demo                           | 8-bit per-channel | 1024         | 288             | [Download](https://huggingface.co/google/functiongemma-270m-it/blob/main/tiny_garden.litertlm) / [Try App](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&hl=en_US) |

Below are the performance numbers of running each model on various devices. Note
that the benchmark is measured with 1024 tokens prefill and 256 tokens decode (
with performance lock on Android devices).

| Model | Device | Backend | Prefill (tokens/sec) | Decode (tokens/sec) | Context size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3-1B | MacBook Pro<br>(2023 M3) | CPU | 422.98 | 66.89 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | CPU | 243.24 | 43.56 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | GPU | 1876.5 | 44.57 | 4096 |
| Gemma3-1B | Samsung S25<br>(Ultra) | NPU | 5836.6 | 84.8 | 1280 |
| Gemma-3n-E2B | MacBook Pro<br>(2023 M3) | CPU | 232.5 | 27.6 | 4096 |
| Gemma-3n-E2B | Samsung S24<br>(Ultra) | CPU | 110.5 | 16.1 | 4096 |
| Gemma-3n-E2B | Samsung S24<br>(Ultra) | GPU | 816.4 | 15.6 | 4096 |
| Gemma-3n-E4B | MacBook Pro<br>(2023 M3) | CPU | 170.1 | 20.1 | 4096 |
| Gemma-3n-E4B | Samsung S24<br>(Ultra) | CPU | 73.5 | 9.2 | 4096 |
| Gemma-3n-E4B | Samsung S24<br>(Ultra) | GPU | 548.0 | 9.4 | 4096 |
| FunctionGemma | Samsung S25<br>(Ultra) | CPU | 1718.4 | 125.9 | 1024 |

## Quick Start <span id="quick_start"></span>

**Want to try it out first?** Before proceeding with the full setup, you can use
the pre-built binary below to run the LiteRT-LM immediately.
### Desktop CLI (LIT)

-   [MacOS ARM64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.macos_arm64)
-   [Linux x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.linux_x86_64)
-   [Linux ARM64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.linux_arm64)
-   [Windows x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.windows_x86_64.exe)

After the download the `lit` binary, just run `lit` to see the options.
Simple use case is like:

```shell
# Set the HuggingFace token in the HUGGING_FACE_HUB_TOKEN environment variable
# so that lit can pull the model from HuggingFace.

# On Linux or MacOS
export HUGGING_FACE_HUB_TOKEN="**your huggingface token**"

# On Windows Command Prompt
set HUGGING_FACE_HUB_TOKEN=your_huggingface_token

# On Windows Powershell
$env:HUGGING_FACE_HUB_TOKEN = "your_huggingface_token"
```

```shell
lit list --show_all
lit pull gemma3-1b
lit run gemma3-1b [--backend=<cpu|gpu>]
```

Note: **Running GPU on Windows needs DirectXShaderCompiler**
 <span id="windows_gpu"></span>
 Download the dxc_2025_07_14.zip or the latest zip file from
 https://github.com/microsoft/DirectXShaderCompiler/releases, unzip the file and
 locate the right architecture directory under `bin`, copy the `dxil.dll` and
 `dxcompiler.dll` into the same directory as the executable like `lit` or
 `litert_lm_main`.

Tip: For more functionality, use `lit --help` or `lit <command> --help`

Tip: Follow this [link](https://huggingface.co/docs/hub/en/security-tokens) to
get your own hugging face token

Tip: you may have to `chmod +x lit` and explicitly approve the usage of
pre-built binaries. For example, in MacOS, you should go to **System Settings >
Privacy & Security > Security** to approve the binary.

### Mobile Apps

-   [Android AI Edge Gallery App](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&hl=en_US&pli=1)
-   iOS (Coming soon)

Note that the LiteRT-LM runtime is designed to work
with models in the `.litertlm` format. You can find and download compatible
models in the [Supported Models and
Performance](#supported-models-and-performance) section.

Note: that the first time a given model is loaded on a given device, it will
take longer to load. This is because the model weights are being arranged to run
optimally on your particular device. Subsequent loads will be faster
because the optimized weights are cached on your device.

## Documentation

For detailed documentation, please visit the [docs](./docs/README.md) directory.

## Model Hosting and Deployment

When a model exceeds 1.5GB, it often surpasses the "over-the-air" download
limits of cellular networks or the internal limits of standard app bundles.
A remote fetch strategy is required.

Host your model file, then have your app fetch the latest version of your
model URL for download. [Firebase](https://firebase.google.com/) provides
solutions for downloading large files on [Android](https://firebase.google.com/docs/storage/android/download-files)
and [iOS](https://firebase.google.com/docs/storage/ios/download-files).

Alternatively, you can fetch a model directly from HuggingFace by using the
[HuggingFace API](https://huggingface.co/docs/huggingface_hub/guides/download).
For private or gated models, you will need to include a Hugging Face User
Access Token in the `Authorization: Bearer <TOKEN>` header of your download
request.

## FAQ

### LiteRT vs LiteRT-LM vs MediaPipe GenAI Tasks

LiteRT, LiteRT-LM, and MediaPipe GenAI Tasks are three libraries within the
Google AI Edge stack that build on each other. By exposing functionality at
different abstraction layers, we hope to enable developers to balance their
respective needs between flexibility and complexity.

[LiteRT](https://ai.google.dev/edge/litert) is Google AI Edge's underlying
on-device runtime. Developer can convert individual PyTorch, TensorFlow, and JAX
models to LiteRT and run them on-device.

**LiteRT-LM** gives developers the pipeline framework to stitch together
multiple LiteRT models with pre and post processing components (e.g. tokenizer,
vision encoder, text decoder).

[MediaPipe GenAI Tasks](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)
are out-of-the-box native APIs (Kotlin, Swift, JS) to run language models by
just setting a few parameters such as temperature and topK.

### .litertlm vs .task

MediaPipe GenAI Tasks currently use `.task` files to represent language models.
Task files are a zip of multiple LiteRT files, components, and metadata.
`.litertlm` is an evolution of the `.task` file format to include additional
metadata and enable better compression.

During our LiteRT-LM preview, we will release a small number of `.litertlm`
files. MediaPipe APIs will continue to use `.task` files. Once we have the first
full release of LiteRT-LM, we will migrate MediaPipe APIs to use the new
`.litertlm` files and release a wider collection of `.litertlm` files on the
[LiteRT Hugging Face Community](https://huggingface.co/litert-community)

## Reporting Issues

If you encounter a bug or have a feature request, we encourage you to use the
[GitHub Issues](https://github.com/google-ai-edge/LiteRT-LM/issues/new) page to
report it.

Before creating a new issue, please search the existing issues to avoid
duplicates. When filing a new issue, please provide a clear title and a detailed
description of the problem, including steps to reproduce it. The more
information you provide, the easier it will be for us to help you.
